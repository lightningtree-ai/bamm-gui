"""Export PyBaMM parameter sets to BPX.

>>> import pybamm
>>> from bamm_gui.export import pybamm_to_bpx
>>> model = pybamm_to_bpx(pybamm.ParameterValues("Chen2020"), title="Chen2020")  # doctest: +SKIP

In ``"exact"`` mode (the default) the result reproduces the parameter set when read back with
``pybamm.ParameterValues.create_from_bpx_obj``, or :class:`BPXExportError` lists every parameter
that has no exact BPX form. ``"allow_tables"`` samples such functions onto tables instead and
reports each approximation as a :class:`BPXExportWarning`. Parameters that BPX has no field for are
always reported with a :class:`BPXExportWarning`.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping
from typing import Any, Literal

import bpx
import pybamm

from .mapping import ExportReport, Mapper

__all__ = ["BPXExportError", "BPXExportWarning", "ExportReport", "pybamm_to_bpx"]

_MODEL_HEADERS = {"SPM": "SPMe", "SPMe": "SPMe", "DFN": "DFN"}


class BPXExportError(ValueError):
    """Some parameters have no exact BPX representation; ``problems`` maps each name to the reason."""

    def __init__(self, problems: Mapping[str, str]):
        self.problems = dict(problems)
        lines = "\n".join(f"  - {name}: {reason}" for name, reason in self.problems.items())
        super().__init__(f"{len(self.problems)} parameter(s) cannot be exported to BPX exactly:\n{lines}")


class BPXExportWarning(UserWarning):
    """A parameter was approximated or dropped while exporting to BPX."""


def _model_header(model: str | pybamm.BaseModel) -> str:
    name = model if isinstance(model, str) else type(model).__name__
    try:
        return _MODEL_HEADERS[name]
    except KeyError:
        raise ValueError(f"model must be one of {sorted(_MODEL_HEADERS)}, got {name!r}") from None


def _prune(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _prune(v) for k, v in value.items() if v is not None}
    return value


def pybamm_to_bpx(
    parameter_values: pybamm.ParameterValues | str,
    model: str | pybamm.BaseModel = "DFN",
    *,
    title: str | None = None,
    description: str | None = None,
    references: str | None = None,
    material_names: Mapping[str, tuple[str, ...] | list[str]] | None = None,
    function_mode: Literal["exact", "allow_tables"] = "exact",
    n_samples: int = 101,
    include_state: bool = True,
    user_defined: Mapping[str, Any] | None = None,
) -> bpx.BPX:
    """Convert PyBaMM parameter values into a validated :class:`bpx.BPX` model.

    Parameters
    ----------
    parameter_values
        A :class:`pybamm.ParameterValues`, or the name of a built-in PyBaMM parameter set.
    model
        ``"SPM"``, ``"SPMe"`` or ``"DFN"`` (or a PyBaMM model instance). SPM and SPMe are both
        written as ``"SPMe"``: the reduced BPX ``"SPM"`` layout drops the electrolyte and separator.
    title, description, references
        Header fields. ``references`` defaults to the parameter set's PyBaMM citation keys.
    material_names
        Names of the materials in blended electrodes, e.g. ``{"negative": ("Graphite", "Silicon")}``,
        in PyBaMM phase order (Primary, Secondary). Defaults to ``("Primary", "Secondary")``.
    function_mode
        ``"exact"`` raises :class:`BPXExportError` for anything without an exact BPX form;
        ``"allow_tables"`` samples it (``n_samples`` points) and warns instead.
    include_state
        Write the initial state (SOC, temperatures, hysteresis) and thermal environment. The initial
        electrolyte concentration is always written, because the BPX reaction rate is normalised by it.
    user_defined
        Extra ``User-defined`` entries: a number, BPX function string or ``{"x": [...], "y": [...]}``
        table, or ``None`` to convert the PyBaMM parameter of that name (a function of one variable).
    """
    if function_mode not in ("exact", "allow_tables"):
        raise ValueError(f"function_mode must be 'exact' or 'allow_tables', got {function_mode!r}")
    header_model = _model_header(model)
    if isinstance(parameter_values, str):
        parameter_values = pybamm.ParameterValues(parameter_values)
    keys = parameter_values.keys()
    if "Exchange-current density for lithium metal electrode [A.m-2]" in keys and not any(
        k.endswith("Maximum concentration in negative electrode [mol.m-3]") for k in keys
    ):
        raise BPXExportError({"Negative electrode": "is lithium metal (a half-cell set); BPX describes full cells"})

    mapper = Mapper(
        parameter_values,
        allow_tables=function_mode == "allow_tables",
        n_samples=n_samples,
        material_names=material_names,
    )
    limits, soc = mapper.stoichiometries()
    parameterisation: dict[str, Any] = {
        "Cell": mapper.cell(),
        "Electrolyte": mapper.electrolyte(),
        "Negative electrode": mapper.electrode("negative", limits["negative"]),
        "Positive electrode": mapper.electrode("positive", limits["positive"]),
        "Separator": mapper.separator(),
    }
    state = mapper.state(soc, include_state)
    extras = mapper.user_defined(user_defined)
    if extras:
        parameterisation["User-defined"] = extras

    report = mapper.report
    report.dropped = mapper.dropped()
    if report.problems:
        raise BPXExportError(report.problems)
    for name, reason in report.approximations.items():
        warnings.warn(f"{name!r} was approximated: {reason}", BPXExportWarning, stacklevel=2)
    for note in report.notes:
        warnings.warn(note, BPXExportWarning, stacklevel=2)
    if report.dropped:
        warnings.warn(
            f"{len(report.dropped)} parameter(s) have no BPX field and were not exported: " + ", ".join(report.dropped),
            BPXExportWarning,
            stacklevel=2,
        )

    if references is None and "citations" in parameter_values.keys():
        citations = parameter_values["citations"]
        references = "PyBaMM parameter set citations: " + ", ".join(citations) if citations else None
    header = {
        "BPX": bpx.__version__,
        "Title": title,
        "Description": description,
        "References": references,
        "Model": header_model,
    }
    data = _prune({"Header": header, "Parameterisation": parameterisation, "State": state})
    # BPX validation mutates its input, so validate a copy.
    return bpx.parse_bpx_obj(copy.deepcopy(data))
