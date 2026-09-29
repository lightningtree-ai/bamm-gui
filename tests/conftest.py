from __future__ import annotations

import copy
import math
import warnings
from pathlib import Path
from typing import Any

import bpx
import numpy as np
import pybamm
import pytest

EXAMPLES = Path(__file__).parent / "data" / "bpx_examples"
EXAMPLE_FILES = sorted(p.name for p in EXAMPLES.glob("*.json"))


def parse_example(name: str) -> bpx.BPX:
    """Parse a vendored BPX example (they are v0.x files, converted with a legacy warning)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        return bpx.parse_bpx_file(EXAMPLES / name)


def pybamm_from_bpx(data: dict[str, Any]) -> pybamm.ParameterValues:
    """``ParameterValues.create_from_bpx_obj``, working around a PyBaMM importer bug.

    The importer computes the Bruggeman exponent as ln(transport efficiency) / ln(porosity), which
    divides by zero for a porosity of exactly 1 (Marquis2019's separator), where the exponent is
    irrelevant. The next float below 1 gives the same transport efficiency (1) and model.
    """
    data = copy.deepcopy(data)
    for section in data["Parameterisation"].values():
        if isinstance(section, dict) and section.get("Porosity") == 1.0:
            section["Porosity"] = math.nextafter(1.0, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pybamm.ParameterValues.create_from_bpx_obj(data)


def is_composite(pv: pybamm.ParameterValues) -> bool:
    return "Secondary: Maximum concentration in negative electrode [mol.m-3]" in pv.keys()


def discharge_voltage(pv: pybamm.ParameterValues, temperature: float, t_end: float = 3600.0):
    """Isothermal DFN 1C discharge at ``temperature``: (time, voltage) on a fixed 10 s grid.

    Tight solver tolerances and a common output grid, so that two equivalent parameter sets (built
    as different expression trees) are compared below the default solver noise (~0.1 mV near the
    end of discharge) and without interpolating between adaptive steps.
    """
    pv = pv.copy()
    pv.update({"Ambient temperature [K]": temperature, "Initial temperature [K]": temperature})
    pv["Current function [A]"] = pv["Nominal cell capacity [A.h]"]
    options = {"particle phases": ("2", "1")} if is_composite(pv) else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
        sim = pybamm.Simulation(pybamm.lithium_ion.DFN(options), parameter_values=pv, solver=solver)
        sol = sim.solve([0, t_end], t_interp=np.arange(0, t_end + 1, 10.0))
    return sol["Time [s]"].entries, sol["Voltage [V]"].entries


def voltage_rmse(a, b) -> float:
    (t1, v1), (t2, v2) = a, b
    n = min(len(t1), len(t2))  # a run may stop early at the voltage cut-off
    np.testing.assert_allclose(t1[:n], t2[:n])
    return float(np.sqrt(np.mean((v1[:n] - v2[:n]) ** 2)))


@pytest.fixture(params=EXAMPLE_FILES)
def example_name(request) -> str:
    return request.param
