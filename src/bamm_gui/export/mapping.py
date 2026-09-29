"""Map PyBaMM parameter values onto the BPX schema.

This is the inverse of ``pybamm/parameters/bpx.py::bpx_to_param_dict`` (PyBaMM's BPX importer):
every value is written so that ``pybamm.ParameterValues.create_from_bpx_obj`` reconstructs the
original parameter, including the importer's conventions (effective electrode conductivity with a
zero solid Bruggeman coefficient, the reaction rate normalised by the initial electrolyte
concentration, Arrhenius factors around the BPX reference temperature, lumped thermal properties).
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np
import pybamm

from .functions import C_E, STO, Converted, FunctionConverter, NotRepresentableError

F = pybamm.constants.F.value

PHASE_PREFIXES = ("Primary: ", "Secondary: ")
DEFAULT_MATERIAL_NAMES = ("Primary", "Secondary")

#: Keys that carry no physics BPX could hold, or that the BPX importer sets by convention.
_IGNORED_KEYS = frozenset({"citations", "description"})

#: Keys the BPX importer writes into ParameterValues under BPX names; exporting a set that came from
#: BPX re-derives them from the PyBaMM parameters, so they are not reported as dropped.
_BPX_IMPORT_ARTEFACTS = re.compile(
    r"^((Primary|Secondary): )?("
    r"Electrode area \[m2\]|External surface area \[m2\]|Volume \[m3\]|Density \[kg\.m-3\]"
    r"|Specific heat capacity \[J\.K-1\.kg-1\]"
    r"|Number of electrode pairs connected in parallel to make a cell"
    r"|(Negative|Positive) electrode (transport efficiency|surface area per unit volume \[m-1\]"
    r"|reaction rate constant( activation energy)? \[.*\]|(minimum|maximum) stoichiometry"
    r"|OCP hysteresis decay constant)"
    r"|Separator transport efficiency|Porosity|Thickness \[m\]|Transport efficiency"
    r"|(Negative|Positive) (electrode|particle) diffusivity activation energy \[J\.mol-1\]"
    r"|Electrolyte (diffusivity|conductivity) activation energy \[J\.mol-1\]"
    r")$"
)

#: Defaults the BPX importer applies; a matching value round-trips without being written.
_IMPORT_DEFAULTS = {
    "Number of cells connected in series to make a battery": 1,
    "Thermodynamic factor": 1.0,
}
#: Values the BPX importer sets arbitrarily (current collectors); not reported as dropped when equal.
_IMPORT_PLACEHOLDERS = {
    "Negative current collector conductivity [S.m-1]": 4e7,
    "Positive current collector conductivity [S.m-1]": 4e7,
}


def tidy(value: float, ulps: int = 4) -> float:
    """The shortest decimal within a few ulps of a derived value (0.10270000000000001 -> 0.1027).

    Derived quantities (products, averages) already carry a few ulps of rounding, so this loses
    nothing, and values that went through BPX once come back as they were written.
    """
    if not math.isfinite(value) or value == 0:
        return value
    tolerance = ulps * math.ulp(value)
    for digits in range(1, 18):
        candidate = float(f"{value:.{digits}g}")
        if abs(candidate - value) <= tolerance:
            return candidate
    return value


@dataclass
class ExportReport:
    """What happened while exporting: errors (exact form impossible), approximations and dropped keys."""

    problems: dict[str, str] = field(default_factory=dict)
    approximations: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


def _cap(domain: str) -> str:
    return domain[:1].upper() + domain[1:]


class Mapper:
    """Builds the plain BPX dict for one ParameterValues object; see :func:`bamm_gui.export.pybamm_to_bpx`."""

    def __init__(
        self,
        parameter_values: pybamm.ParameterValues,
        *,
        allow_tables: bool,
        n_samples: int,
        material_names: Mapping[str, tuple[str, ...] | list[str]] | None,
    ):
        self.pv = parameter_values
        self.report = ExportReport()
        self.used: set[str] = set()
        self.to_user_defined: list[str] = []  # PyBaMM-named parameters exported verbatim
        self.material_names = {k.lower(): tuple(v) for k, v in (material_names or {}).items()}
        self.T_ref = float(self.get("Reference temperature [K]"))
        self.conv = FunctionConverter(
            self.T_ref, allow_tables=allow_tables, n_samples=n_samples, parameter_values=parameter_values
        )
        self.phases = {d: self._phase_prefixes(d) for d in ("negative", "positive")}
        self.c_e0 = self.scalar("Initial concentration in electrolyte [mol.m-3]")

    # -- lookup & conversion helpers --------------------------------------------------------------

    def has(self, name: str) -> bool:
        return name in self.pv.keys()

    def get(self, name: str) -> Any:
        self.used.add(name)
        return self.pv[name]

    def first(self, *names: str) -> str:
        """The first of ``names`` present in the parameter values (for renamed PyBaMM parameters)."""
        for name in names:
            if self.has(name):
                return name
        return names[0]

    def _record(self, name: str, fn: Callable[[], Converted]) -> Converted | None:
        try:
            converted = fn()
        except KeyError:
            self.report.problems[name] = "is missing from the parameter values"
            return None
        except NotRepresentableError as err:
            self.report.problems[name] = str(err)
            return None
        if converted.notes:
            self.report.approximations[name] = "; ".join(converted.notes)
        return converted

    def scalar(self, name: str, *, fallback: Mapping[str, float] | None = None) -> float | None:
        """A BPX number. With tables allowed, a function is evaluated at T_ref and the ``fallback`` arguments."""

        def convert() -> Converted:
            try:
                return self.conv.scalar(self.get(name))
            except NotRepresentableError as err:
                if not (self.conv.allow_tables and fallback):
                    raise
                value = self.conv.scalar(self.get(name), bindings=fallback).value
                at = ", ".join(f"{k} = {v}" for k, v in fallback.items())
                return Converted(value, notes=[f"{err}; used its value at {at} and {self.T_ref} K"])

        converted = self._record(name, convert)
        return None if converted is None else converted.value

    def function(self, name: str, domain, *, temperature: bool = False, **bindings: float) -> Converted | None:
        return self._record(
            name, lambda: self.conv.function(self.get(name), domain, temperature=temperature, bindings=bindings)
        )

    def _phase_prefixes(self, domain: str) -> tuple[str, ...]:
        if self.has(f"Secondary: Maximum concentration in {domain} electrode [mol.m-3]"):
            return PHASE_PREFIXES
        return ("",)

    def _material_names(self, domain: str) -> tuple[str, ...]:
        if self.phases[domain] == ("",):
            return ("",)
        names = self.material_names.get(f"{domain} electrode", self.material_names.get(domain, DEFAULT_MATERIAL_NAMES))
        if len(names) != len(self.phases[domain]):
            raise ValueError(f"material_names for the {domain} electrode needs {len(self.phases[domain])} names")
        return names

    # -- sections ---------------------------------------------------------------------------------

    def cell(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        height, width = self.scalar("Electrode height [m]"), self.scalar("Electrode width [m]")
        if height is not None and width is not None:
            out["Electrode area [m2]"] = tidy(height * width)
        if self.has("Cell cooling surface area [m2]"):
            out["External surface area [m2]"] = self.scalar("Cell cooling surface area [m2]")
        if self.has("Cell volume [m3]"):
            out["Volume [m3]"] = self.scalar("Cell volume [m3]")
        name = "Number of electrodes connected in parallel to make a cell"
        n = self.scalar(name)
        if n is not None:
            if float(n).is_integer():
                out["Number of electrode pairs connected in parallel to make a cell"] = int(n)
            else:
                self.report.problems[name] = f"is {n}; BPX needs an integer"
        for key in ("Lower voltage cut-off [V]", "Upper voltage cut-off [V]", "Nominal cell capacity [A.h]"):
            out[key] = self.scalar(key)
        out["Reference temperature [K]"] = self.T_ref
        out.update(self._lumped_thermal())
        return out

    def _thermal_layers(self) -> list[tuple[str, str]]:
        """(PyBaMM name prefix, thickness key) of each layer, in the order PyBaMM averages them."""
        return [
            ("Negative current collector", "Negative current collector thickness [m]"),
            ("Negative electrode", "Negative electrode thickness [m]"),
            ("Separator", "Separator thickness [m]"),
            ("Positive electrode", "Positive electrode thickness [m]"),
            ("Positive current collector", "Positive current collector thickness [m]"),
        ]

    def _lumped(self, quantity: str) -> list[tuple[float, float]] | None:
        """(value, thickness) per layer at T_ref, or None when a layer lacks the quantity."""
        pairs = []
        for prefix, thickness in self._thermal_layers():
            name = f"{prefix} {quantity}"
            if not self.has(name) or not self.has(thickness):
                return None
            value, L = self.scalar(name), self.scalar(thickness)
            if value is None or L is None:
                return None
            pairs.append((value, L))
        return pairs

    def _lumped_thermal(self) -> dict[str, float]:
        """Thickness-weighted density and specific heat, as PyBaMM's lumped thermal model averages them."""
        rho = self._lumped("density [kg.m-3]")
        cp = self._lumped("specific heat capacity [J.kg-1.K-1]")
        if rho is None or cp is None:
            return {}
        L = sum(t for _, t in rho)
        rho_av = sum(r * t for r, t in rho) / L
        rho_cp_av = sum(r * c * t for (r, t), (c, _) in zip(rho, cp, strict=True)) / L
        return {"Density [kg.m-3]": tidy(rho_av), "Specific heat capacity [J.K-1.kg-1]": tidy(rho_cp_av / rho_av)}

    def lumped_thermal_conductivity(self) -> float | None:
        pairs = self._lumped("thermal conductivity [W.m-1.K-1]")
        if pairs is None:
            return None
        return tidy(sum(k * t for k, t in pairs) / sum(t for _, t in pairs))

    def electrolyte(self) -> dict[str, Any]:
        out: dict[str, Any] = {"Cation transference number": self.scalar("Cation transference number")}
        domain = C_E.scaled(self.c_e0 or 1000.0)
        for field_name, key in (
            ("Diffusivity [m2.s-1]", "Electrolyte diffusivity [m2.s-1]"),
            ("Conductivity [S.m-1]", "Electrolyte conductivity [S.m-1]"),
        ):
            converted = self.function(key, domain, temperature=True)
            if converted is not None:
                out[field_name] = converted.value
                if converted.activation_energy is not None:
                    out[field_name.split(" [")[0] + " activation energy [J.mol-1]"] = converted.activation_energy
        return out

    def _contact(self, prefix: str) -> dict[str, Any]:
        porosity = self.scalar(f"{prefix} porosity")
        brug = self.scalar(f"{prefix} Bruggeman coefficient (electrolyte)")
        out = {"Thickness [m]": self.scalar(f"{prefix} thickness [m]"), "Porosity": porosity}
        if porosity is not None and brug is not None:
            out["Transport efficiency"] = tidy(porosity**brug)
        return out

    def separator(self) -> dict[str, Any]:
        return self._contact("Separator")

    def electrode(self, domain: str, stoichiometries: dict[str, tuple[float, float]]) -> dict[str, Any]:
        Dom = _cap(domain)
        out = self._contact(f"{Dom} electrode")
        # BPX holds the effective conductivity; the importer sets the solid Bruggeman coefficient to 0.
        sigma = self.scalar(f"{Dom} electrode conductivity [S.m-1]", fallback={"sto": 0.5})
        b_s = (
            self.scalar(f"{Dom} electrode Bruggeman coefficient (electrode)")
            if self.has(f"{Dom} electrode Bruggeman coefficient (electrode)")
            else 0.0
        )
        if sigma is not None and b_s is not None and out.get("Porosity") is not None:
            out["Conductivity [S.m-1]"] = tidy(sigma * (1 - out["Porosity"]) ** b_s)

        particles = {phase: self.particle(domain, phase, stoichiometries.get(phase)) for phase in self.phases[domain]}
        if self.phases[domain] == ("",):
            out.update(particles[""])
        else:
            out["Particle"] = dict(zip(self._material_names(domain), particles.values(), strict=True))
        return out

    def particle(self, domain: str, phase: str, sto_limits: tuple[float, float] | None) -> dict[str, Any]:
        Dom = _cap(domain)
        out: dict[str, Any] = {}
        if sto_limits is not None:
            out["Minimum stoichiometry"], out["Maximum stoichiometry"] = sto_limits
        c_max = self.scalar(f"{phase}Maximum concentration in {domain} electrode [mol.m-3]")
        radius = self.scalar(f"{phase}{Dom} particle radius [m]")
        eps_am = self.scalar(f"{phase}{Dom} electrode active material volume fraction")
        out["Maximum concentration [mol.m-3]"] = c_max
        out["Particle radius [m]"] = radius
        if radius is not None and eps_am is not None:
            out["Surface area per unit volume [m-1]"] = tidy(3 * eps_am / radius)

        D_name = self.first(
            f"{phase}{Dom} particle diffusivity [m2.s-1]", f"{phase}{Dom} electrode diffusivity [m2.s-1]"
        )
        D = self.function(D_name, STO, temperature=True)
        if D is not None:
            out["Diffusivity [m2.s-1]"] = D.value
            if D.activation_energy is not None:
                out["Diffusivity activation energy [J.mol-1]"] = D.activation_energy

        bindings = {"c_s_max": c_max} if c_max is not None else {}
        for field_name, key in (
            ("OCP [V]", f"{phase}{Dom} electrode OCP [V]"),
            ("OCP (delithiation) [V]", f"{phase}{Dom} electrode delithiation OCP [V]"),
            ("OCP (lithiation) [V]", f"{phase}{Dom} electrode lithiation OCP [V]"),
            ("Entropic change coefficient [V.K-1]", f"{phase}{Dom} electrode OCP entropic change [V.K-1]"),
        ):
            if field_name == "OCP [V]" or self.has(key):
                converted = self.function(key, STO, **bindings)
                if converted is not None:
                    out[field_name] = converted.value

        rates = [f"{phase}{Dom} particle {branch} hysteresis decay rate" for branch in ("lithiation", "delithiation")]
        if all(self.has(r) for r in rates):
            values = [self.scalar(r) for r in rates]
            if values[0] == values[1] and values[0] is not None:
                out["OCP hysteresis decay constant"] = values[0]
            else:  # BPX has one decay constant; keep both branches under their PyBaMM names
                self.to_user_defined.extend(rates)

        out.update(self._reaction_rate(domain, phase, c_max))
        return out

    def _reaction_rate(self, domain: str, phase: str, c_max: float | None) -> dict[str, float]:
        """BPX k_norm from PyBaMM j0 = m(T) sqrt(c_e c_s (c_max - c_s)), checked on a (c_e, c_s) grid."""
        name = f"{phase}{_cap(domain)} electrode exchange-current density [A.m-2]"
        if c_max is None or self.c_e0 is None:
            return {}
        try:
            j0 = self.get(name)
        except KeyError:
            self.report.problems[name] = "is missing from the parameter values"
            return {}

        c_e = np.repeat(np.array([0.25, 0.5, 1.0, 1.5, 2.0]) * self.c_e0, 9)
        c_s = np.tile(np.linspace(0.1, 0.9, 9) * c_max, 5)
        sqrt_term = np.sqrt(c_e * c_s * (c_max - c_s))

        # m(x) with x indexing the grid; FunctionConverter's Arrhenius check then runs on m.
        def m(index: pybamm.Symbol, T: pybamm.Symbol) -> pybamm.Symbol:
            idx = np.asarray(index.evaluate(), dtype=int).ravel() if isinstance(index, pybamm.Vector) else None
            if idx is None:
                raise NotRepresentableError("is not of the form m(T) * sqrt(c_e * c_s * (c_max - c_s))")
            value = (
                j0(pybamm.Vector(c_e[idx]), pybamm.Vector(c_s[idx]), pybamm.Scalar(c_max), T) if callable(j0) else j0
            )
            return value / pybamm.Vector(sqrt_term[idx])

        try:
            evaluate = self.conv._numeric(m, ["index", "T"], "index", {})
            grid = np.arange(c_e.size, dtype=float)
            values = evaluate(grid, self.T_ref)
            if not np.allclose(values, values[0], rtol=1e-9, atol=0):
                raise NotRepresentableError(
                    "is not of the form m(T) * sqrt(c_e * c_s * (c_max - c_s)) "
                    f"(m varies from {values.min():.6g} to {values.max():.6g} at {self.T_ref} K)"
                )
            Ea = self.conv._activation_energy(evaluate, grid) if callable(j0) else 0.0
        except NotRepresentableError as err:
            self.report.problems[name] = str(err)
            return {}

        out = {"Reaction rate constant [mol.m-2.s-1]": tidy(float(np.mean(values)) * c_max * math.sqrt(self.c_e0) / F)}
        if Ea:
            out["Reaction rate constant activation energy [J.mol-1]"] = Ea
        return out

    # -- stoichiometry & state --------------------------------------------------------------------

    def stoichiometries(self) -> tuple[dict[str, dict[str, tuple[float, float]]], float | None]:
        """Per-electrode, per-phase (min, max) stoichiometry and the initial SOC.

        The limits written to BPX are the stored ones when the set carries them (e.g. it came from
        BPX), else PyBaMM's electrode SOH solution. The initial SOC is always measured against the
        electrode SOH limits, because that is how PyBaMM's importer interprets it
        (``ParameterValues.set_initial_state``); both use the cyclable lithium of the initial state.
        """
        stored = self._stored_stoichiometries()
        try:
            with warnings.catch_warnings():  # PyBaMM's own ESOH internals warn about renamed inputs
                warnings.simplefilter("ignore")
                esoh = self._esoh_stoichiometries()
        except Exception as err:  # any ESOH failure is reported, and must not hide the other problems
            esoh = None
            if stored is None:
                self.report.problems["Minimum/Maximum stoichiometry"] = f"electrode SOH could not be solved ({err})"
                return {"negative": {}, "positive": {}}, None
            self.report.notes.append(
                f"electrode SOH could not be solved ({err}); the initial SOC uses the stored limits"
            )
        limits = stored if stored is not None else esoh
        return limits, self._initial_soc(esoh or stored)

    def _initial_soc(self, limits: dict[str, dict[str, tuple[float, float]]]) -> float | None:
        """SOC from the negative-electrode lithium content, weighting phases by their capacity (eps * c_max)."""
        num = den = 0.0
        for phase in self.phases["negative"]:
            key = f"{phase}Initial concentration in negative electrode [mol.m-3]"
            if not self.has(key) or phase not in limits["negative"]:
                return None
            c_max = self.scalar(f"{phase}Maximum concentration in negative electrode [mol.m-3]")
            eps = self.scalar(f"{phase}Negative electrode active material volume fraction")
            c_init = self.scalar(key)
            if None in (c_max, eps, c_init):
                return None
            x_0, x_100 = limits["negative"][phase]
            num += c_max * eps * (c_init / c_max - x_0)
            den += c_max * eps * (x_100 - x_0)
        if den <= 0:
            return None
        soc = num / den
        if not -1e-9 <= soc <= 1 + 1e-9:
            self.report.notes.append(
                f"the initial negative electrode concentration corresponds to SOC = {soc:.6g}, outside the "
                "stoichiometry limits; the BPX initial state-of-charge was clamped to [0, 1]"
            )
        return tidy(min(max(soc, 0.0), 1.0))

    def _stored_stoichiometries(self) -> dict[str, dict[str, tuple[float, float]]] | None:
        limits: dict[str, dict[str, tuple[float, float]]] = {}
        for domain in ("negative", "positive"):
            limits[domain] = {}
            for phase in self.phases[domain]:
                keys = [f"{phase}{_cap(domain)} electrode {b} stoichiometry" for b in ("minimum", "maximum")]
                if not all(self.has(k) for k in keys):
                    return None
                limits[domain][phase] = (self.scalar(keys[0]), self.scalar(keys[1]))
        return limits

    def _esoh_stoichiometries(self) -> dict[str, dict[str, tuple[float, float]]]:
        n_phases = (str(len(self.phases["negative"])), str(len(self.phases["positive"])))
        if n_phases == ("1", "1"):
            x_0, x_100, y_100, y_0 = pybamm.lithium_ion.get_min_max_stoichiometries(self.pv)
            return {"negative": {"": (float(x_0), float(x_100))}, "positive": {"": (float(y_100), float(y_0))}}

        from pybamm.models.full_battery_models.lithium_ion.electrode_soh_composite import (
            get_initial_stoichiometries_composite,
        )

        options = pybamm.BatteryModelOptions({"particle phases": n_phases})
        sol = get_initial_stoichiometries_composite(1.0, self.pv, options=options)
        limits: dict[str, dict[str, tuple[float, float]]] = {"negative": {}, "positive": {}}
        for domain, letter in (("negative", "x"), ("positive", "y")):
            for i, phase in enumerate(self.phases[domain], start=1):
                a, b = float(sol[f"{letter}_0_{i}"]), float(sol[f"{letter}_100_{i}"])
                limits[domain][phase] = (min(a, b), max(a, b))
        return limits

    def state(self, soc: float | None, include_state: bool) -> dict[str, Any]:
        """The BPX State section."""
        initial: dict[str, Any] = {}
        thermal: dict[str, Any] = {}
        if include_state:
            if soc is not None:
                initial["Initial state-of-charge"] = soc
            if self.has("Initial temperature [K]"):
                initial["Initial temperature [K]"] = self.scalar("Initial temperature [K]")
        # Always written: the importer normalises the reaction rate by this concentration.
        initial["Initial electrolyte concentration [mol.m-3]"] = self.c_e0
        if include_state:
            for domain in ("negative", "positive"):
                values = {}
                for phase, material in zip(self.phases[domain], self._material_names(domain), strict=True):
                    key = f"{phase}Initial hysteresis state in {domain} electrode"
                    if self.has(key):
                        values[material] = self.scalar(key)
                field_name = f"Initial hysteresis state: {_cap(domain)} electrode"
                if self.phases[domain] == ("",) and values:
                    initial[field_name] = next(iter(values.values()))
                elif values and len(values) == len(self.phases[domain]):
                    initial[field_name] = values
                elif values:  # partial per-material state: BPX needs every material, so keep PyBaMM names
                    for phase in self.phases[domain]:
                        key = f"{phase}Initial hysteresis state in {domain} electrode"
                        if self.has(key):
                            self.to_user_defined.append(key)
            if self.has("Ambient temperature [K]"):
                thermal["Ambient temperature [K]"] = self.scalar("Ambient temperature [K]")
            if self.has("Total heat transfer coefficient [W.m-2.K-1]"):
                thermal["Heat transfer coefficient [W.m-2.K-1]"] = self.scalar(
                    "Total heat transfer coefficient [W.m-2.K-1]"
                )
        state: dict[str, Any] = {"Initial conditions": initial}
        if thermal:
            state["Thermal environment"] = thermal
        return state

    # -- user-defined & dropped -------------------------------------------------------------------

    def user_defined(self, extras: Mapping[str, Any] | None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in ("Contact resistance [Ohm]", "Cell thermal expansion coefficient [m.K-1]"):
            if self.has(key):
                out[key] = self.scalar(key)
        conductivity = self.lumped_thermal_conductivity()
        if conductivity is not None:
            out["Thermal conductivity [W.m-1.K-1]"] = conductivity
        # The importer falls back to the voltage cut-offs for these, so only differing values are written.
        for ocv, cutoff in (
            ("Open-circuit voltage at 0% SOC [V]", "Lower voltage cut-off [V]"),
            ("Open-circuit voltage at 100% SOC [V]", "Upper voltage cut-off [V]"),
        ):
            if self.has(ocv):
                value = self.scalar(ocv)
                if value is not None and value != self.pv[cutoff]:
                    out[ocv] = value
        for key in self.to_user_defined:
            out[key] = self.scalar(key)
        for key, value in _IMPORT_DEFAULTS.items():
            if self.has(key):
                v = self.get(key)
                if not (isinstance(v, Real) and v == value):
                    if isinstance(v, Real):
                        out[key] = float(v)
                    else:
                        self.report.problems[key] = "is a function; BPX (and PyBaMM's importer) assume a constant"
        for key, value in (extras or {}).items():
            if value is None:
                converted = self._record(key, lambda k=key: self.conv.function(self.get(k), STO))
                if converted is not None:
                    out[key] = converted.value
            else:
                out[key] = value
        return out

    def dropped(self) -> list[str]:
        skip_implied = {
            "Current function [A]",
            "Negative electrode Bruggeman coefficient (electrode)",
            "Positive electrode Bruggeman coefficient (electrode)",
        }
        dropped = []
        for key in self.pv.keys():
            if key in self.used or key in _IGNORED_KEYS or key in skip_implied or _BPX_IMPORT_ARTEFACTS.match(key):
                continue
            if re.match(r"^((Primary|Secondary): )?Initial concentration in (negative|positive) electrode", key):
                continue  # represented by the initial state-of-charge
            if key in _IMPORT_PLACEHOLDERS and self.pv[key] == _IMPORT_PLACEHOLDERS[key]:
                continue
            dropped.append(key)
        return sorted(dropped)
