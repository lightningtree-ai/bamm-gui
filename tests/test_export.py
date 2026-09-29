import math
import warnings

import bpx
import numpy as np
import pybamm
import pytest
from conftest import discharge_voltage, parse_example, pybamm_from_bpx, voltage_rmse

from bamm_gui.export import BPXExportError, BPXExportWarning, pybamm_to_bpx
from bamm_gui.io import bpx_to_dict, write_bpx_file

F = pybamm.constants.F.value


def export(pv, **kwargs) -> bpx.BPX:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BPXExportWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        return pybamm_to_bpx(pv, **kwargs)


@pytest.fixture(scope="module")
def chen2020():
    return pybamm.ParameterValues("Chen2020")


@pytest.fixture(scope="module")
def chen2020_bpx(chen2020):
    return bpx_to_dict(export(chen2020, title="Chen2020"))


# -- mapping ---------------------------------------------------------------------------------------


def test_header(chen2020_bpx):
    header = chen2020_bpx["Header"]
    assert header["Model"] == "DFN" and header["Title"] == "Chen2020"
    assert header["BPX"] == bpx.__version__
    assert "Chen2020" in header["References"]


@pytest.mark.parametrize(("model", "header"), [("SPM", "SPMe"), ("SPMe", "SPMe"), ("DFN", "DFN")])
def test_model_header(chen2020, model, header):
    assert export(chen2020, model=model).header.model == header


def test_model_header_accepts_model_instance(chen2020):
    assert export(chen2020, model=pybamm.lithium_ion.SPM()).header.model == "SPMe"
    with pytest.raises(ValueError, match="model must be one of"):
        pybamm_to_bpx(chen2020, model="ECM")


def test_cell_and_geometry(chen2020, chen2020_bpx):
    p = chen2020_bpx["Parameterisation"]
    cell = p["Cell"]
    assert cell["Electrode area [m2]"] == pytest.approx(
        chen2020["Electrode height [m]"] * chen2020["Electrode width [m]"], rel=1e-15
    )
    assert cell["Number of electrode pairs connected in parallel to make a cell"] == 1
    assert cell["External surface area [m2]"] == chen2020["Cell cooling surface area [m2]"]
    assert cell["Volume [m3]"] == chen2020["Cell volume [m3]"]
    for side in ("Negative", "Positive"):
        electrode = p[f"{side} electrode"]
        eps = chen2020[f"{side} electrode porosity"]
        b = chen2020[f"{side} electrode Bruggeman coefficient (electrolyte)"]
        assert electrode["Transport efficiency"] == pytest.approx(eps**b, rel=1e-15)
        assert electrode["Surface area per unit volume [m-1]"] == pytest.approx(
            3 * chen2020[f"{side} electrode active material volume fraction"] / chen2020[f"{side} particle radius [m]"],
            rel=1e-15,
        )


def test_lumped_thermal_properties_preserve_volumetric_heat_capacity(chen2020, chen2020_bpx):
    layers = [
        "Negative current collector",
        "Negative electrode",
        "Separator",
        "Positive electrode",
        "Positive current collector",
    ]
    L = [chen2020[f"{layer} thickness [m]"] for layer in layers]
    rho_cp = [
        chen2020[f"{layer} density [kg.m-3]"] * chen2020[f"{layer} specific heat capacity [J.kg-1.K-1]"]
        for layer in layers
    ]
    cell = chen2020_bpx["Parameterisation"]["Cell"]
    expected = sum(r * t for r, t in zip(rho_cp, L, strict=True)) / sum(L)
    assert cell["Density [kg.m-3]"] * cell["Specific heat capacity [J.K-1.kg-1]"] == pytest.approx(expected, rel=1e-14)


def test_reaction_rate_follows_importer_normalisation(chen2020, chen2020_bpx):
    negative = chen2020_bpx["Parameterisation"]["Negative electrode"]
    c_max = chen2020["Maximum concentration in negative electrode [mol.m-3]"]
    c_e0 = chen2020["Initial concentration in electrolyte [mol.m-3]"]
    j0 = chen2020["Negative electrode exchange-current density [A.m-2]"]
    c_s, c_e, T = 0.4 * c_max, c_e0, pybamm.Scalar(298.15)
    j0_ref = float(j0(pybamm.Scalar(c_e), pybamm.Scalar(c_s), pybamm.Scalar(c_max), T).evaluate())
    # BPX: j0 = F * k_norm * sqrt(c_e / c_e0 * c_s / c_max * (1 - c_s / c_max))
    k_norm = negative["Reaction rate constant [mol.m-2.s-1]"]
    assert F * k_norm * math.sqrt(c_e / c_e0 * c_s / c_max * (1 - c_s / c_max)) == pytest.approx(j0_ref, rel=1e-12)
    assert negative["Reaction rate constant activation energy [J.mol-1]"] == 35000


def test_state_is_written(chen2020_bpx):
    state = chen2020_bpx["State"]
    assert 0.98 < state["Initial conditions"]["Initial state-of-charge"] <= 1
    assert state["Initial conditions"]["Initial electrolyte concentration [mol.m-3]"] == 1000
    assert state["Thermal environment"] == {
        "Ambient temperature [K]": 298.15,
        "Heat transfer coefficient [W.m-2.K-1]": 10,
    }


def test_include_state_false_keeps_electrolyte_concentration(chen2020):
    state = bpx_to_dict(export(chen2020, include_state=False))["State"]
    assert state == {"Initial conditions": {"Initial electrolyte concentration [mol.m-3]": 1000}}


def test_user_defined_extras(chen2020):
    data = bpx_to_dict(
        export(
            chen2020,
            user_defined={
                "My constant": 2.5,
                "My function": "1 + x",
                "My table": {"x": [0, 1], "y": [1, 2]},
                "Negative electrode OCP [V]": None,  # converted from the parameter set
            },
        )
    )
    extras = data["Parameterisation"]["User-defined"]
    assert extras["My constant"] == 2.5 and extras["My function"] == "1 + x"
    assert extras["My table"] == {"x": [0.0, 1.0], "y": [1.0, 2.0]}
    assert extras["Negative electrode OCP [V]"] == data["Parameterisation"]["Negative electrode"]["OCP [V]"]
    assert extras["Contact resistance [Ohm]"] == 0
    assert "Thermal conductivity [W.m-1.K-1]" in extras


def test_dropped_parameters_warn(chen2020):
    with pytest.warns(BPXExportWarning, match="no BPX field") as record:
        pybamm_to_bpx(chen2020)
    message = str(record[0].message)
    assert "SEI kinetic rate constant [m.s-1]" in message
    assert "Electrolyte diffusivity" not in message


def test_ocv_limits_differing_from_cutoffs_are_kept(chen2020):
    pv = chen2020.copy()
    pv["Open-circuit voltage at 0% SOC [V]"] = 2.6
    extras = bpx_to_dict(export(pv))["Parameterisation"]["User-defined"]
    assert extras["Open-circuit voltage at 0% SOC [V]"] == 2.6
    assert "Open-circuit voltage at 100% SOC [V]" not in extras


def test_non_integer_electrode_pairs_is_an_error(chen2020):
    pv = chen2020.copy()
    pv["Number of electrodes connected in parallel to make a cell"] = 1.5
    with pytest.raises(BPXExportError, match="integer"):
        export(pv)


def test_blended_material_names():
    data = bpx_to_dict(
        export("Chen2020_composite", function_mode="allow_tables", material_names={"negative": ("Graphite", "Silicon")})
    )
    assert list(data["Parameterisation"]["Negative electrode"]["Particle"]) == ["Graphite", "Silicon"]
    with pytest.raises(ValueError, match="needs 2 names"):
        export("Chen2020_composite", function_mode="allow_tables", material_names={"negative": ("Graphite",)})


def test_half_cell_is_rejected():
    with pytest.raises(BPXExportError, match="lithium metal"):
        pybamm_to_bpx("Xu2019")


@pytest.mark.filterwarnings("ignore:.*have no BPX field")
def test_allow_tables_warns_about_each_approximation():
    with pytest.warns(BPXExportWarning, match="Primary: Negative electrode OCP.*approximated.*sampled 101 points"):
        pybamm_to_bpx("Chen2020_composite", function_mode="allow_tables")


def test_invalid_function_mode(chen2020):
    with pytest.raises(ValueError, match="function_mode"):
        pybamm_to_bpx(chen2020, function_mode="sampled")


# -- exact mode across the built-in sets (evidence for a BPX spec proposal) -------------------------

#: Built-in PyBaMM lithium-ion sets that export exactly.
EXACT_SETS = ["Chen2020", "Marquis2019", "Mohtat2020", "Prada2013", "Ramadass2004"]

#: Sets that cannot be exported exactly, with the parameters that block them and why. BPX lacks:
#: cubic interpolants, non-Arrhenius (e.g. polynomial-in-T or 1/T) temperature dependence,
#: concentration-dependent transference numbers and thermodynamic factors, exchange-current
#: densities not of the form m(T) * sqrt(c_e c_s (c_max - c_s)), temperature-dependent thermal
#: properties, and stoichiometry-dependent electrode conductivity.
EXACT_FAILURES = {
    "Ai2020": {
        "Negative electrode OCP [V]": "cubic",
        "Positive electrode OCP [V]": "cubic",
        "Electrolyte diffusivity [m2.s-1]": "not Arrhenius",
        "Electrolyte conductivity [S.m-1]": "not Arrhenius",
        "Thermodynamic factor": "is a function",
    },
    "Bonkile2024": {"Primary: Negative electrode OCP [V]": "cubic"},
    "Chen2020_composite": {"Primary: Negative electrode OCP [V]": "cubic"},
    "Ecker2015": {"Electrolyte conductivity [S.m-1]": "not Arrhenius"},
    "NCA_Kim2011": {
        "Electrolyte diffusivity [m2.s-1]": "not Arrhenius",
        "Electrolyte conductivity [S.m-1]": "not Arrhenius",
    },
    "OKane2022": {"Negative electrode OCP [V]": "cubic"},
    "ORegan2022": {
        "Cation transference number": "function of c_e",
        "Electrolyte diffusivity [m2.s-1]": "not Arrhenius",
        "Electrolyte conductivity [S.m-1]": "not Arrhenius",
        "Negative electrode exchange-current density [A.m-2]": "not of the form",
        "Positive electrode exchange-current density [A.m-2]": "not of the form",
        "Positive electrode conductivity [S.m-1]": "function of sto",
        "Negative current collector specific heat capacity [J.kg-1.K-1]": "depends on temperature",
        "Negative current collector thermal conductivity [W.m-1.K-1]": "depends on temperature",
        "Thermodynamic factor": "is a function",
    },
}


@pytest.mark.parametrize("name", EXACT_SETS)
def test_exact_sets(name):
    model = export(name)
    assert isinstance(model, bpx.BPX)
    assert bpx.parse_bpx_obj(bpx_to_dict(model)) == model


@pytest.mark.parametrize("name", sorted(EXACT_FAILURES))
def test_exact_failures_list_their_reasons(name):
    with pytest.raises(BPXExportError) as info, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pybamm_to_bpx(name)
    problems = info.value.problems
    expected = EXACT_FAILURES[name]
    assert set(expected) <= set(problems), problems
    for key, reason in expected.items():
        assert reason in problems[key], (key, problems[key])


# -- PyBaMM -> BPX -> PyBaMM --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Chen2020", "Marquis2019", "Ecker2015", "Chen2020_composite"])
@pytest.mark.parametrize("temperature", [298.15, 318.15])
def test_pybamm_bpx_pybamm_voltage(name, temperature):
    pv = pybamm.ParameterValues(name)
    exact = name not in EXACT_FAILURES
    model = export(pv, function_mode="exact" if exact else "allow_tables")
    pv2 = pybamm_from_bpx(bpx_to_dict(model))
    rmse = voltage_rmse(discharge_voltage(pv, temperature), discharge_voltage(pv2, temperature))
    # exact sets agree to solver precision; the plan requires < 5 mV for all
    assert rmse < (1e-6 if exact else 5e-3), f"{name} at {temperature} K: RMSE {rmse * 1e3:.4f} mV"


# -- BPX -> PyBaMM -> BPX ----------------------------------------------------------------------------

ROUND_TRIP_EXAMPLES = [
    "nmc_pouch_cell_BPX.json",
    "nmc_pouch_cell_BPX_blended_electrode.json",
    "lfp_18650_cell_BPX.json",
    "nmc_pouch_cell_BPX_user-defined_hysteresis.json",
]

#: v0.x user-defined hysteresis branches are proper BPX v1.1 fields once they have been through PyBaMM.
RELOCATED = {
    "Parameterisation/User-defined/Negative electrode delithiation OCP [V]": (
        "Parameterisation/Negative electrode/OCP (delithiation) [V]"
    ),
    "Parameterisation/User-defined/Negative electrode lithiation OCP [V]": (
        "Parameterisation/Negative electrode/OCP (lithiation) [V]"
    ),
}
#: Written explicitly although the source leaves it to the importer's default (0: adiabatic).
ADDED = {"State/Thermal environment/Heat transfer coefficient [W.m-2.K-1]"}


def flatten(data: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in data.items():
        if isinstance(value, dict) and set(value) != {"x", "y"}:
            out.update(flatten(value, f"{prefix}{key}/"))
        else:
            out[f"{prefix}{key}"] = value
    return out


def assert_same_value(path: str, original, exported):
    if isinstance(original, dict):  # table
        assert isinstance(exported, dict), path
        np.testing.assert_allclose(sorted(exported["x"]), sorted(original["x"]), rtol=1e-12, err_msg=path)
        order_o, order_e = np.argsort(original["x"]), np.argsort(exported["x"])
        np.testing.assert_allclose(
            np.asarray(exported["y"])[order_e], np.asarray(original["y"])[order_o], rtol=1e-12, err_msg=path
        )
    elif isinstance(original, str):  # function: compare on a grid
        xs = np.linspace(1, 3000, 301) if "Electrolyte" in path else np.linspace(0.001, 0.999, 301)
        f_orig = bpx.Function(original).to_python_function(preamble="from numpy import exp, tanh, cosh")
        f_exp = bpx.Function(exported).to_python_function(preamble="from numpy import exp, tanh, cosh")
        np.testing.assert_allclose(f_exp(xs), f_orig(xs), rtol=1e-9, atol=1e-15, err_msg=path)
    else:
        assert isinstance(exported, int | float), path
        assert exported == pytest.approx(original, rel=1e-9, abs=1e-300), path


@pytest.mark.parametrize("name", ROUND_TRIP_EXAMPLES)
def test_bpx_pybamm_bpx(name):
    original = bpx_to_dict(parse_example(name))
    positive = original["Parameterisation"]["Positive electrode"]
    names = {"positive": list(positive["Particle"])} if "Particle" in positive else None
    exported = bpx_to_dict(export(pybamm_from_bpx(original), material_names=names))

    fo = flatten({k: original[k] for k in ("Parameterisation", "State")})
    fe = flatten({k: exported[k] for k in ("Parameterisation", "State")})
    for path, value in fo.items():
        target = RELOCATED.get(path, path)
        assert target in fe, f"{path} missing after the round trip"
        assert_same_value(path, value, fe[target])
    extra = set(fe) - {RELOCATED.get(p, p) for p in fo}
    assert extra <= ADDED, extra


# -- manual verification step from the plan, as a test ---------------------------------------------


def test_written_file_loads_in_pybamm(tmp_path, chen2020):
    path = write_bpx_file(export(chen2020, title="Chen2020"), tmp_path / "chen.json")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pv = pybamm.ParameterValues.create_from_bpx(path)
    assert pv["Nominal cell capacity [A.h]"] == 5.0
    c_n = pv["Initial concentration in negative electrode [mol.m-3]"]
    assert c_n == pytest.approx(chen2020["Initial concentration in negative electrode [mol.m-3]"], rel=1e-6)
