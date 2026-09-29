import math
from functools import partial

import bpx
import numpy as np
import pybamm
import pytest
from pybamm.expression_tree.tracing import tracing

from bamm_gui.export.functions import (
    C_E,
    STO,
    FunctionConverter,
    NotRepresentableError,
    symbol_to_expression,
)

R = pybamm.constants.R.value
T_REF = 298.15
X = pybamm.Parameter("x")


def pybamm_eval(expression: str, xs: np.ndarray) -> np.ndarray:
    """Evaluate a BPX function the way PyBaMM's importer does (pybamm preamble, Symbol input)."""
    f = bpx.Function(expression).to_python_function(preamble="from pybamm import exp, tanh, cosh")
    return np.asarray(f(pybamm.Vector(xs)).evaluate()).ravel()


def exact() -> FunctionConverter:
    return FunctionConverter(T_REF)


# -- printer --------------------------------------------------------------------------------------


def traced(build):
    """Build an expression the way the converter does: under tracing, so it is not re-associated."""
    with tracing():
        return build()


def test_printer_keeps_association():
    # a + (b + c) differs from (a + b) + c in floating point, so the parentheses must survive
    assert symbol_to_expression(traced(lambda: X + (0.1 + X * 3))) == "x + (0.1 + 3.0 * x)"
    assert symbol_to_expression(traced(lambda: X * (X * 0.1))) == "x * (0.1 * x)"
    assert symbol_to_expression(traced(lambda: X / (X / 2))) == "x / (x / 2.0)"
    assert symbol_to_expression(traced(lambda: X - (2 * X - 1))) == "x - (2.0 * x - 1.0)"


def test_printer_folds_constants_and_signs():
    F = pybamm.constants.F  # a named Constant, folded to its value
    assert (
        symbol_to_expression(traced(lambda: 2.0 * pybamm.exp(-X / (pybamm.Scalar(4.0) * 2)))) == "2.0 * exp(-x / 8.0)"
    )
    assert symbol_to_expression(traced(lambda: 1.0 - pybamm.Scalar(-0.5) * X)) == "1.0 + 0.5 * x"
    assert symbol_to_expression(traced(lambda: X * 1.0 + 0.0)) == "x"
    assert symbol_to_expression(traced(lambda: X / F)) == f"x / {float(F.value)!r}"


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda x: pybamm.sinh(x), "tanh(x) * cosh(x)"),
        (lambda x: pybamm.sqrt(x + 1), "(1.0 + x) ** 0.5"),
        (lambda x: abs(x - 0.5), "((x - 0.5) ** 2) ** 0.5"),
        (lambda x: x ** (-x), "x ** (-x)"),
        (lambda x: -(x**2), "-(x ** 2.0)"),
    ],
)
def test_printer_rewrites(build, expected):
    expression = symbol_to_expression(traced(lambda: build(X)))
    assert expression == expected
    xs = np.linspace(-0.9, 0.9, 19) if "- 0.5" in expected else np.linspace(0.1, 0.9, 9)
    reference = np.asarray(build(pybamm.Vector(xs)).evaluate()).ravel()
    np.testing.assert_allclose(pybamm_eval(expression, xs), reference, rtol=1e-14)


@pytest.mark.parametrize("symbol", [pybamm.log(X), pybamm.arctan(X), pybamm.maximum(X, 0.5)])
def test_printer_rejects_unsupported(symbol):
    with pytest.raises(NotRepresentableError):
        symbol_to_expression(symbol)


# -- plain values ---------------------------------------------------------------------------------


def test_numbers_and_data_tables():
    assert exact().function(3, STO).value == 3.0
    table = exact().function(("OCP", ([np.array([1.0, 0.0, 0.5])], np.array([0.1, 1.0, 0.4]))), STO).value
    assert isinstance(table, bpx.InterpolatedTable)
    assert table.x == [0.0, 0.5, 1.0] and table.y == [1.0, 0.4, 0.1]


def test_scalar_rejects_real_functions():
    with pytest.raises(NotRepresentableError, match="function of sto"):
        exact().scalar(lambda sto: 2 * sto)
    with pytest.raises(NotRepresentableError, match="temperature"):
        exact().scalar(lambda T: T / 300)
    assert FunctionConverter(T_REF, allow_tables=True).scalar(lambda T: T / 300).value == T_REF / 300
    assert exact().scalar(lambda T: pybamm.Scalar(5.0)).value == 5.0


# -- callables ------------------------------------------------------------------------------------


def test_expression_matches_pybamm_exactly():
    def ocp(sto):
        return 0.2482 + 1.9793 * pybamm.exp(-39.3631 * sto) - 0.0909 * pybamm.tanh(29.8538 * (sto - 0.1234))

    converted = exact().function(ocp, STO)
    assert isinstance(converted.value, bpx.Function) and converted.activation_energy is None
    xs = np.linspace(0.001, 0.999, 57)
    expected = np.asarray(ocp(pybamm.Vector(xs)).evaluate()).ravel()
    np.testing.assert_allclose(pybamm_eval(converted.value, xs), expected, rtol=1e-14)


def test_arrhenius_is_extracted_and_snapped():
    def diffusivity(sto, T):
        return (1e-14 + 2e-14 * sto) * pybamm.exp(35000 / pybamm.constants.R * (1 / 296.15 - 1 / T))

    converted = exact().function(diffusivity, STO, temperature=True)
    assert converted.activation_energy == 35000.0
    xs = np.linspace(0.01, 0.99, 11)
    expected = np.asarray(diffusivity(pybamm.Vector(xs), pybamm.Scalar(T_REF)).evaluate()).ravel()
    np.testing.assert_allclose(pybamm_eval(converted.value, xs), expected, rtol=1e-13)


def test_constant_with_arrhenius_becomes_number():
    def diffusivity(sto, T):
        return 3.9e-14 * pybamm.exp(42770 / pybamm.constants.R * (1 / T_REF - 1 / T))

    converted = exact().function(diffusivity, STO, temperature=True)
    assert converted.value == 3.9e-14 and converted.activation_energy == 42770.0


def test_temperature_independent_function_has_no_activation_energy():
    converted = exact().function(lambda c_e, T: 1e-10 + 0 * c_e, C_E.scaled(1000), temperature=True)
    assert converted.activation_energy is None


def test_non_arrhenius_raises_or_is_fitted():
    def conductivity(c_e, T):  # the 1/T prefactor is not Arrhenius
        return 0.1 * c_e / 1000 * pybamm.exp(-1000 / T) / T

    with pytest.raises(NotRepresentableError, match="not Arrhenius"):
        exact().function(conductivity, C_E.scaled(1000), temperature=True)
    approx = FunctionConverter(T_REF, allow_tables=True).function(conductivity, C_E.scaled(1000), temperature=True)
    assert approx.notes and "fitted" in approx.notes[0]
    assert approx.activation_energy == pytest.approx(R * 1000 - R * T_REF, rel=1e-2)
    assert isinstance(approx.value, bpx.Function)  # the x-dependence at T_ref is still exact


def test_temperature_dependence_rejected_where_not_allowed():
    with pytest.raises(NotRepresentableError, match="no activation energy"):
        exact().function(lambda sto, T: sto * pybamm.exp(-1000 / T), STO)


def test_linear_interpolant_becomes_table():
    x, y = np.linspace(0, 1, 11), np.linspace(1, 0, 11) ** 2

    def ocp(sto):
        return 2.0 * pybamm.Interpolant(x, y, sto, interpolator="linear")

    table = exact().function(ocp, STO).value
    assert isinstance(table, bpx.InterpolatedTable)
    np.testing.assert_allclose(table.y, 2 * y, rtol=1e-15)


def test_scaled_argument_interpolant_becomes_table():
    x, y = np.linspace(0, 30000, 7), np.linspace(1e-14, 2e-14, 7)
    table = exact().function(lambda sto: pybamm.Interpolant(x, y, sto * 30000.0), STO).value
    np.testing.assert_allclose(table.x, x / 30000.0)


def test_cubic_interpolant_raises_or_is_sampled():
    x, y = np.linspace(0, 1, 11), np.sin(np.linspace(0, 1, 11))

    def ocp(sto):
        return pybamm.Interpolant(x, y, sto, interpolator="cubic")

    with pytest.raises(NotRepresentableError, match="cubic"):
        exact().function(ocp, STO)
    approx = FunctionConverter(T_REF, allow_tables=True, n_samples=21).function(ocp, STO)
    assert isinstance(approx.value, bpx.InterpolatedTable) and len(approx.value.x) == 21
    assert "sampled 21 points" in approx.notes[0]


def test_function_referencing_other_parameters():
    pv = pybamm.ParameterValues("Ecker2015")
    converter = FunctionConverter(pv["Reference temperature [K]"], parameter_values=pv)
    # Ecker's electrolyte diffusivity calls the conductivity FunctionParameter internally
    converted = converter.function(pv["Electrolyte diffusivity [m2.s-1]"], C_E.scaled(1000), temperature=True)
    assert isinstance(converted.value, bpx.Function) and converted.activation_energy == 17100.0


def test_bound_partials_and_extra_bindings():
    def entropic(sto, c_s_max, scale):
        return scale * sto / c_s_max

    converted = exact().function(partial(entropic, scale=2.0), STO, bindings={"c_s_max": 4.0})
    assert pybamm_eval(converted.value, np.array([0.5]))[0] == 0.25
    with pytest.raises(NotRepresentableError, match="c_s_max"):
        exact().function(partial(entropic, scale=2.0), STO)


@pytest.mark.parametrize("name", ["Chen2020", "Marquis2019", "Mohtat2020", "Ramadass2004", "NCA_Kim2011"])
def test_builtin_ocps_convert_exactly(name):
    pv = pybamm.ParameterValues(name)
    xs = np.linspace(0.02, 0.98, 49)
    for domain in ("Negative", "Positive"):
        f = pv[f"{domain} electrode OCP [V]"]
        converted = FunctionConverter(pv["Reference temperature [K]"], parameter_values=pv).function(f, STO)
        expected = np.asarray(f(pybamm.Vector(xs)).evaluate()).ravel()
        got = pybamm_eval(converted.value, xs)
        assert np.allclose(got, expected, rtol=1e-12, atol=0), f"{name} {domain}: {converted.value}"
        assert math.isfinite(got.sum())
