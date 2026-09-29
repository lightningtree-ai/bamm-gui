# bamm-gui

A local Python web app that opens, edits, validates and plots BPX parameter sets, runs PyBaMM
simulations on them, and imports PyBaMM parameter sets through an exact PyBaMM→BPX exporter.
BPX is the file format; PyBaMM is the engine. Both come from PyPI.

## Build order

1. **M1 (done):** exporter library: `bamm_gui.io` (BPX writer) and `bamm_gui.export` (PyBaMM→BPX).
2. **M2 (next):** NiceGUI app in `src/bamm_gui/app/`, `bamm-gui [file.json]` console script. Add the
   deps with `uv add --optional gui nicegui plotly`. Build the UI from `bpx.BPX.model_json_schema()`
   rather than hand-written forms: section tree, number/Function/table field editors, live Plotly
   plots (tables drawn with linear interpolation), live validation (`to_python_function`,
   `check_sto_limits`), add/remove blended materials and User-defined parameters, and save via
   `write_bpx_file`. Simulate SPMe/DFN via `pybamm.ParameterValues.create_from_bpx_obj` against the
   file's `Validation` data. Import built-in or custom PyBaMM sets via `pybamm_to_bpx`.

## Commands

```sh
uv sync --all-extras
uv run ruff check && uv run ruff format --check
uv run pytest --cov=bamm_gui
```

`pyproject.toml` points `bpx` at the sibling checkout `../BPX` (editable) through
`[tool.uv.sources]`, so BPX fixes can be developed and upstreamed. CI sets `UV_NO_SOURCES=1` to
use PyPI instead. **Remove that source before release.**

## Exporter decisions

- **Model header:** PyBaMM `SPM`/`SPMe` → `"SPMe"`, `DFN` → `"DFN"`. The reduced BPX `"SPM"`
  layout (no electrolyte/separator) is never written.
- **Modes:** `function_mode="exact"` collects every non-representable value and raises
  `BPXExportError` listing all of them. `"allow_tables"` samples functions onto tables (or fits an
  Arrhenius energy) and warns with `BPXExportWarning`. Parameters BPX has no field for always warn.
- **Mapping** is the inverse of PyBaMM's `pybamm/parameters/bpx.py::bpx_to_param_dict`, including
  its conventions: effective electrode conductivity σ(1−ε)^b_s (the importer sets b_s = 0);
  transport efficiency ε^b; a = 3ε_am/R; k_norm = m·c_max·√c_e0/F, with j0 checked to be
  m(T)·√(c_e c_s (c_max−c_s)) on a grid; thickness-weighted lumped ρ and ρc_p/ρ over all five
  layers (matches PyBaMM's `rho_c_p_eff`).
- **Function converter** (`export/functions.py`): trace the callable under PyBaMM's
  `pybamm.expression_tree.tracing.tracing()` and print it **directly from the expression tree**,
  folding constant subtrees with `repr` floats. This deviates from the original plan
  (`to_equation()` → sympy): sympy expands `a*(x-b)` to `a*x - a*b`, which changes floating-point
  results. The printed string keeps the tree's association, so it evaluates like PyBaMM. Only
  exact IEEE identities are applied (x·1, x+0, a+(−c)y → a−cy). Rewrites: sqrt → `**0.5`,
  sinh → `tanh*cosh` (keeps relative accuracy near 0), abs → `(u**2)**0.5`. Every result is checked
  against PyBaMM at rtol 1e-12. Linear 1D interpolants (also scaled, or of `a*x`) become
  `InterpolatedTable`s. Temperature: evaluate at T_ref, T_ref±10 K; Ea must be constant in x and
  consistent at both temperatures (rtol 1e-9), then it is snapped to its shortest decimal.
  References to other parameters inside a function body (`pybamm.Parameter`/`FunctionParameter`)
  are expanded from the ParameterValues.
- **Stoichiometry and SOC:** min/max stoichiometry comes from stored keys when present (sets that
  came from BPX), else from PyBaMM's electrode SOH (composite solver for blends). The initial SOC is
  **always** measured against the ESOH limits, because that is how the importer's
  `set_initial_state` interprets it. Otherwise SOC drifts on every round trip.
- **State:** `Initial electrolyte concentration` is always written, because the importer
  normalises the reaction rate by it. `include_state=False` writes only that.
- **User-defined:** contact resistance, thermal expansion, weighted thermal conductivity, OCV at
  0%/100% SOC when they differ from the cut-offs, unequal hysteresis decay rates and partial
  per-phase initial hysteresis states under their PyBaMM names, plus caller extras.
- Derived floats are tidied to the shortest decimal within 4 ulps (`mapping.tidy`).

## Verification status

- PyBaMM→BPX→PyBaMM isothermal DFN 1C at 25 °C and 45 °C: Chen2020 and Marquis2019 agree to
  ~10 nV (tight solver, common time grid). Ecker2015 and Chen2020_composite need `allow_tables`
  and stay under 2 mV.
- BPX→PyBaMM→BPX on the BPX examples: scalars within rel 1e-9, functions equal on a grid, SOC
  preserved.
- Which built-in sets fail exact mode, and why, is recorded in `EXACT_FAILURES` in
  `tests/test_export.py`. This is the evidence for a BPX spec proposal (cubic interpolants,
  non-Arrhenius temperature dependence, concentration-dependent t+ and thermodynamic factor,
  non-standard j0, T-dependent thermal properties, sto-dependent electrode conductivity).

## Upstream candidates

BPX (`../BPX`; fork it when the first PR is ready):
- `bamm_gui/io.py` (`bpx_to_dict`, `write_bpx_file`): BPX has no writer.
- `Function.to_python_function` writes a temp file per call with `delete=False` and never removes it.
- `expression_parser.py` uses pyparsing's deprecated camelCase API (`delimitedList`, `setParseAction`).
- The Function grammar parses `-x**2` as `(-x)**2`, but evaluation is Python (`-(x**2)`). The
  exporter always parenthesises to avoid the ambiguity.

PyBaMM BPX importer (`pybamm/parameters/bpx.py`):
- Porosity exactly 1 (Marquis2019's separator) → `ZeroDivisionError` in the Bruggeman
  ln(τ)/ln(ε). Tests work around it (`tests/conftest.py::pybamm_from_bpx`).
- A User-defined lumped `Thermal conductivity [W.m-1.K-1]` is added after the lumping loop, so it
  never reaches the per-layer parameters.
- The separator section is processed twice (once as the `experiment` domain), which leaves bare
  `Porosity`, `Thickness [m]` and `Transport efficiency` keys in the ParameterValues.
- User-defined functions are wrapped as one-argument callables only.
