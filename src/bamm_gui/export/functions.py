"""Exact conversion of PyBaMM parameter values into BPX ``FloatFunctionTable`` values.

A PyBaMM parameter is a number, a data tuple (a linear 1D interpolant) or a Python callable that
builds an expression tree. BPX can hold a number, an :class:`bpx.InterpolatedTable` (linearly
interpolated) or a :class:`bpx.Function` string in one variable ``x`` using ``exp``/``tanh``/``cosh``,
optionally with an Arrhenius activation energy on the side.

Callables are traced with :func:`pybamm.expression_tree.tracing.tracing` so the tree records the
function as written, then printed directly from the tree. Constant subtrees are folded to floats
printed with ``repr`` (shortest round-trip), so the printed string evaluates in the same order and
to the same values as PyBaMM. Every result is re-parsed and compared against PyBaMM before use.
Anything that cannot be written exactly raises :class:`NotRepresentableError`, unless sampling is
allowed, in which case it is tabulated and the approximation is reported in ``Converted.notes``.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import bpx
import numpy as np
import pybamm
from pybamm.expression_tree.tracing import tracing

__all__ = [
    "C_E",
    "STO",
    "Converted",
    "FunctionConverter",
    "NotRepresentableError",
    "XDomain",
    "symbol_to_expression",
]

R = pybamm.constants.R.value

#: Relative tolerance for the numerical check of converted expressions and tables.
EXACT_RTOL = 1e-12
#: Relative tolerance for activation-energy consistency (Arrhenius form) checks.
ARRHENIUS_RTOL = 1e-9
#: Temperature offset [K] used to probe temperature dependence around the reference temperature.
DELTA_T = 10.0

_TEMPERATURE_ARGS = frozenset({"T", "T_dim", "T_K", "temperature"})
_X = "x"


class NotRepresentableError(ValueError):
    """A PyBaMM value has no exact BPX representation."""


@dataclass(frozen=True)
class XDomain:
    """The independent variable ``x`` of a BPX function, with the range used to check and sample it."""

    name: str
    lo: float
    hi: float

    def scaled(self, factor: float) -> XDomain:
        return XDomain(self.name, self.lo * factor, self.hi * factor)

    def check_grid(self, n: int = 201) -> np.ndarray:
        """Interior grid for numerical checks (avoids singular end points such as sto = 0)."""
        return np.linspace(self.lo, self.hi, n + 2)[1:-1]

    def sample_grid(self, n: int) -> np.ndarray:
        return np.linspace(self.lo, self.hi, n)


#: Stoichiometry, for particle functions.
STO = XDomain("stoichiometry", 0.0, 1.0)
#: Electrolyte concentration as a multiple of 1 mol/m3; scale by the reference concentration range.
C_E = XDomain("electrolyte concentration [mol.m-3]", 0.0, 3.0)


@dataclass
class Converted:
    """A converted value.

    ``activation_energy`` is ``None`` when the value does not depend on temperature. ``notes`` lists
    approximations (non-empty only when sampling was allowed and needed).
    """

    value: float | bpx.Function | bpx.InterpolatedTable
    activation_energy: float | None = None
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------------------
# Expression printer
# --------------------------------------------------------------------------------------------------

_ADD, _MUL, _NEG, _POW, _ATOM = 1, 2, 3, 4, 5
_NATIVE_FUNCTIONS = {pybamm.Exp: "exp", pybamm.Tanh: "tanh", pybamm.Cosh: "cosh"}


def _is_x(node: pybamm.Symbol) -> bool:
    return isinstance(node, pybamm.Parameter) and node.name == _X


def _format_number(value: Any) -> tuple[str, int]:
    arr = np.asarray(value, dtype=float)
    if arr.size != 1:
        raise NotRepresentableError(f"constant subexpression is not a scalar (shape {arr.shape})")
    v = float(arr.reshape(()))
    if not math.isfinite(v):
        raise NotRepresentableError(f"constant subexpression is not finite ({v})")
    s = repr(v)
    return s, (_NEG if s.startswith("-") else _ATOM)


class _Printer:
    def __init__(self, evaluate: Callable[[pybamm.Symbol], Any]):
        self._evaluate = evaluate
        self._depends: dict[int, bool] = {}

    def depends_on_x(self, node: pybamm.Symbol) -> bool:
        key = id(node)
        if key not in self._depends:
            self._depends[key] = _is_x(node) or any(self.depends_on_x(c) for c in node.children)
        return self._depends[key]

    def constant(self, node: pybamm.Symbol) -> float | None:
        """The value of a constant scalar subtree, else None."""
        if self.depends_on_x(node):
            return None
        value = np.asarray(self._evaluate(node), dtype=float)
        return float(value.reshape(())) if value.size == 1 else None

    def negated(self, node: pybamm.Symbol) -> str | None:
        """``node`` printed with its sign flipped, when that is exact and reads better (-c, -c * y, -y)."""
        c = self.constant(node)
        if c is not None:
            return _format_number(-c)[0] if c < 0 else None
        if isinstance(node, pybamm.Negate):
            return self.wrap(node.children[0], _ADD + 1)
        if isinstance(node, pybamm.Multiplication | pybamm.Division):
            left, right = node.children
            c = self.constant(left)
            if c is not None and c < 0:
                op = "*" if isinstance(node, pybamm.Multiplication) else "/"
                return f"{_format_number(-c)[0]} {op} {self.wrap(right, _MUL + 1)}"
        return None

    def wrap(self, node: pybamm.Symbol, min_prec: int) -> str:
        s, prec = self.print(node)
        return s if prec >= min_prec else f"({s})"

    def print(self, node: pybamm.Symbol) -> tuple[str, int]:
        if not self.depends_on_x(node):
            return _format_number(self._evaluate(node))
        if _is_x(node):
            return _X, _ATOM
        # Binary operators: keep the tree's association (a + (b + c) is not (a + b) + c in floating
        # point), so the right operand of a left-associative operator is wrapped at equal precedence.
        # The identities below are exact in IEEE arithmetic, so they only tidy the output.
        if isinstance(node, pybamm.Addition | pybamm.Subtraction):
            left, right = node.children
            plus = isinstance(node, pybamm.Addition)
            if self.constant(right) == 0.0:
                return self.print(left)
            if plus and self.constant(left) == 0.0:
                return self.print(right)
            negated = self.negated(right)  # a + (-y) == a - y and a - (-y) == a + y
            if negated is not None:
                return f"{self.wrap(left, _ADD)} {'-' if plus else '+'} {negated}", _ADD
            return f"{self.wrap(left, _ADD)} {'+' if plus else '-'} {self.wrap(right, _ADD + 1)}", _ADD
        if isinstance(node, pybamm.Multiplication | pybamm.Division):
            left, right = node.children
            times = isinstance(node, pybamm.Multiplication)
            if self.constant(right) == 1.0:
                return self.print(left)
            if times and self.constant(left) == 1.0:
                return self.print(right)
            return f"{self.wrap(left, _MUL)} {'*' if times else '/'} {self.wrap(right, _MUL + 1)}", _MUL
        if isinstance(node, pybamm.Power):
            left, right = node.children
            return f"{self.wrap(left, _ATOM)} ** {self.wrap(right, _ATOM)}", _POW
        if isinstance(node, pybamm.Negate):
            return f"-{self.wrap(node.children[0], _ATOM)}", _NEG
        for cls, name in _NATIVE_FUNCTIONS.items():
            if type(node) is cls:
                return f"{name}({self.print(node.children[0])[0]})", _ATOM
        # Exact (to rounding) rewrites into the BPX function set.
        if type(node) is pybamm.Sinh:  # sinh = tanh * cosh keeps full relative accuracy near 0
            arg = self.print(node.children[0])[0]
            return f"tanh({arg}) * cosh({arg})", _MUL
        if type(node) is pybamm.Sqrt:
            return f"{self.wrap(node.children[0], _ATOM)} ** 0.5", _POW
        if type(node) is pybamm.AbsoluteValue:
            return f"({self.wrap(node.children[0], _ATOM)} ** 2) ** 0.5", _POW
        raise NotRepresentableError(f"uses {_describe(node)}, which BPX functions cannot express")


def _describe(node: pybamm.Symbol) -> str:
    if isinstance(node, pybamm.Interpolant):
        return f"a {node.interpolator} {node.dimension}D interpolant"
    if isinstance(node, pybamm.Parameter):
        return f"the unbound parameter {node.name!r}"
    return f"'{type(node).__name__}'"


def symbol_to_expression(symbol: pybamm.Symbol, evaluate: Callable[[pybamm.Symbol], Any] | None = None) -> str:
    """Print a PyBaMM expression in ``Parameter("x")`` as a BPX function string.

    ``evaluate`` folds constant subtrees to numbers (default ``symbol.evaluate()``).
    """
    return _Printer(evaluate or (lambda s: s.evaluate())).print(symbol)[0]


# --------------------------------------------------------------------------------------------------
# Converter
# --------------------------------------------------------------------------------------------------


def _free_arguments(func: Callable) -> list[str]:
    """Argument names a PyBaMM function parameter is called with (unbound arguments of a partial)."""
    params = inspect.signature(func).parameters.items()
    return [
        name
        for name, p in params
        if p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


def _python_function(expression: str) -> Callable[[np.ndarray], np.ndarray]:
    """Vectorised equivalent of ``bpx.Function(expression).to_python_function()``, without temp files."""
    code = compile(expression, "<bpx function>", "eval")
    namespace = {"__builtins__": {}, "exp": np.exp, "tanh": np.tanh, "cosh": np.cosh}

    def f(x: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            return np.broadcast_to(np.asarray(eval(code, namespace, {"x": x}), dtype=float), np.shape(x))

    return f


def _close(actual: np.ndarray, expected: np.ndarray, rtol: float) -> bool:
    finite = np.isfinite(expected)
    if not np.array_equal(finite, np.isfinite(actual)):
        return False
    scale = np.max(np.abs(expected[finite]), initial=0.0)
    return bool(np.allclose(actual[finite], expected[finite], rtol=rtol, atol=rtol * scale))


def _snap(value: float, accept: Callable[[float], bool]) -> float:
    """The shortest decimal rounding of ``value`` that still passes ``accept`` (e.g. 34999.99999999 -> 35000)."""
    for digits in range(1, 17):
        candidate = float(f"{value:.{digits}g}")
        if accept(candidate):
            return candidate
    return value


class FunctionConverter:
    """Convert PyBaMM parameter values to BPX values at a reference temperature.

    Parameters
    ----------
    reference_temperature
        BPX "Reference temperature [K]": the temperature at which functions are evaluated and the
        Arrhenius reference point.
    allow_tables
        Sample values that have no exact form onto an :class:`bpx.InterpolatedTable` instead of
        raising :class:`NotRepresentableError`.
    n_samples
        Number of samples for such tables.
    parameter_values
        Used to resolve other parameters referenced inside a function body.
    """

    def __init__(
        self,
        reference_temperature: float,
        *,
        allow_tables: bool = False,
        n_samples: int = 101,
        parameter_values: pybamm.ParameterValues | None = None,
    ):
        self.T_ref = float(reference_temperature)
        self.allow_tables = allow_tables
        self.n_samples = n_samples
        self._pv = parameter_values

    # -- evaluation helpers -----------------------------------------------------------------------

    @staticmethod
    def _evaluate_symbol(symbol: Any) -> Any:
        return symbol.evaluate() if isinstance(symbol, pybamm.Symbol) else symbol

    def _expand(self, node: Any) -> Any:
        """Replace references to other parameters inside a function body by their values/functions."""
        if not isinstance(node, pybamm.Symbol):
            return node
        if isinstance(node, pybamm.FunctionParameter | pybamm.Parameter) and not _is_x(node):
            if self._pv is None or node.name not in self._pv.keys():
                raise NotRepresentableError(f"refers to the parameter {node.name!r}, which has no value")
            value = self._pv[node.name]
            if isinstance(value, Real) and not isinstance(value, bool):
                return pybamm.Scalar(float(value))
            if isinstance(node, pybamm.FunctionParameter) and callable(value):
                return self._expand(value(*[self._expand(c) for c in node.children]))
            raise NotRepresentableError(f"refers to the parameter {node.name!r} of unsupported type")
        if not node.children:
            return node
        children = [self._expand(c) for c in node.children]
        if all(new is old for new, old in zip(children, node.children, strict=True)):
            return node
        return node.create_copy(new_children=children)

    def _call(self, func: Callable, args: list[str], x: Any, x_arg: str, T: Any, bindings: Mapping[str, float]):
        kwargs = {}
        for name in args:
            if name == x_arg:
                kwargs[name] = x
            elif name in bindings:
                kwargs[name] = pybamm.Scalar(bindings[name])
            elif name in _TEMPERATURE_ARGS:
                kwargs[name] = T
            else:
                raise NotRepresentableError(f"has an argument {name!r} that cannot be fixed or mapped to x")
        with tracing():
            return self._expand(func(**kwargs))

    def _numeric(self, func, args, x_arg, bindings) -> Callable[[np.ndarray, float], np.ndarray]:
        def evaluate(xs: np.ndarray, T: float) -> np.ndarray:
            out = self._evaluate_symbol(self._call(func, args, pybamm.Vector(xs), x_arg, pybamm.Scalar(T), bindings))
            return np.broadcast_to(np.asarray(out, dtype=float).reshape(-1), xs.shape).copy()

        return evaluate

    # -- public API -------------------------------------------------------------------------------

    def scalar(self, value: Any, *, bindings: Mapping[str, float] | None = None) -> Converted:
        """Convert a value BPX stores as a plain number (a callable must be constant at ``T_ref``)."""
        if isinstance(value, Real) and not isinstance(value, bool):
            return Converted(float(value))
        if isinstance(value, pybamm.Symbol):
            return Converted(float(np.asarray(self._expand(value).evaluate(), dtype=float).reshape(())))
        if callable(value):
            args = _free_arguments(value)
            bindings = dict(bindings or {})
            unknown = [a for a in args if a not in bindings and a not in _TEMPERATURE_ARGS]
            if unknown:
                reason = f"is a function of {', '.join(unknown)}; BPX needs a constant"
            else:
                evaluate = self._numeric(value, args, x_arg="", bindings=bindings)
                grid = np.array([0.0])
                v0 = evaluate(grid, self.T_ref)[0]
                probes = [evaluate(grid, self.T_ref + d)[0] for d in (-DELTA_T, DELTA_T)]
                if all(p == v0 for p in probes):
                    return Converted(float(v0))
                reason = "depends on temperature; BPX needs a constant"
            if self.allow_tables:
                if unknown:
                    raise NotRepresentableError(reason)
                return Converted(float(v0), notes=[f"{reason}, used its value at {self.T_ref} K"])
            raise NotRepresentableError(reason)
        raise NotRepresentableError(f"has unsupported type {type(value).__name__}")

    def function(
        self,
        value: Any,
        domain: XDomain,
        *,
        temperature: bool = False,
        bindings: Mapping[str, float] | None = None,
    ) -> Converted:
        """Convert a value BPX stores as a ``FloatFunctionTable`` of ``x`` in ``domain``.

        ``temperature=True`` allows an Arrhenius temperature dependence, returned as the activation
        energy; otherwise the value must not depend on temperature.
        """
        bindings = dict(bindings or {})
        if isinstance(value, Real) and not isinstance(value, bool):
            return Converted(float(value))
        if isinstance(value, tuple):
            return Converted(self._data_table(value))
        if isinstance(value, pybamm.Symbol):
            return self.scalar(value)
        if not callable(value):
            raise NotRepresentableError(f"has unsupported type {type(value).__name__}")

        args = _free_arguments(value)
        if not args:
            raise NotRepresentableError("is a function with no arguments")
        x_arg = args[0]
        evaluate = self._numeric(value, args, x_arg, bindings)
        grid = domain.check_grid()
        has_T = any(a in _TEMPERATURE_ARGS and a not in bindings for a in args[1:])

        notes: list[str] = []
        try:
            Ea = self._activation_energy(evaluate, grid) if has_T else None
            if Ea is not None and not temperature:
                raise NotRepresentableError("depends on temperature, but this BPX field has no activation energy")
        except NotRepresentableError as err:
            if not self.allow_tables:
                raise
            Ea = self._fitted_activation_energy(evaluate, grid) if temperature else None
            fitted = f"fitted an Arrhenius activation energy of {Ea:.6g} J/mol and " if temperature else ""
            notes.append(f"{err}; {fitted}used its x-dependence at {self.T_ref} K")

        try:
            converted = self._exact_x_dependence(value, args, x_arg, bindings, evaluate, grid)
        except NotRepresentableError as err:
            if not self.allow_tables:
                raise
            xs = domain.sample_grid(self.n_samples)
            ys = evaluate(xs, self.T_ref)
            if not np.all(np.isfinite(ys)):
                xs = domain.check_grid(self.n_samples - 2)
                ys = evaluate(xs, self.T_ref)
            converted = bpx.InterpolatedTable(x=xs.tolist(), y=ys.tolist())
            notes.append(f"{err}; sampled {len(xs)} points over {domain.name} [{xs[0]:.6g}, {xs[-1]:.6g}]")
        return Converted(converted, activation_energy=Ea or None, notes=notes)

    # -- internals --------------------------------------------------------------------------------

    @staticmethod
    def _data_table(value: tuple) -> bpx.InterpolatedTable:
        """``(name, (x, y))`` data, which PyBaMM interpolates linearly."""
        _, (xs, ys) = value
        xs = [np.asarray(a, dtype=float).ravel() for a in (xs if isinstance(xs, list | tuple) else [xs])]
        if len(xs) != 1:
            raise NotRepresentableError(f"is {len(xs)}D data; BPX tables are 1D")
        x = xs[0]
        y = np.asarray(ys, dtype=float).ravel()
        order = np.argsort(x, kind="stable")
        return bpx.InterpolatedTable(x=x[order].tolist(), y=y[order].tolist())

    def _exact_x_dependence(
        self, func, args, x_arg, bindings, evaluate, grid
    ) -> float | bpx.Function | bpx.InterpolatedTable:
        symbol = self._call(func, args, pybamm.Parameter(_X), x_arg, pybamm.Scalar(self.T_ref), bindings)
        expected = evaluate(grid, self.T_ref)
        if not isinstance(symbol, pybamm.Symbol) or not any(_is_x(n) for n in symbol.pre_order()):
            return float(np.asarray(self._evaluate_symbol(symbol), dtype=float).reshape(()))

        table = self._as_table(symbol)
        if table is not None:
            got = np.interp(grid, table.x, table.y)
            inside = (grid >= table.x[0]) & (grid <= table.x[-1])
            if not _close(got[inside], expected[inside], EXACT_RTOL):
                raise NotRepresentableError("its interpolant does not reproduce PyBaMM's values")
            return table

        expression = _Printer(self._evaluate_symbol).print(symbol)[0]
        try:
            function = bpx.Function.validate(expression)
        except ValueError as err:
            raise NotRepresentableError(f"printed expression is not valid BPX ({err})") from err
        if not _close(_python_function(function)(grid), expected, EXACT_RTOL):
            raise NotRepresentableError("printed expression does not reproduce PyBaMM's values")
        return function

    def _as_table(self, symbol: pybamm.Symbol) -> bpx.InterpolatedTable | None:
        """A linear 1D interpolant of x, optionally scaled by constants, as a BPX table."""
        scale = 1.0
        node = symbol
        while isinstance(node, pybamm.Multiplication | pybamm.Division):
            left, right = node.children
            const_left = not any(_is_x(n) for n in left.pre_order())
            const_right = not any(_is_x(n) for n in right.pre_order())
            if isinstance(node, pybamm.Multiplication) and const_left:
                scale *= float(self._evaluate_symbol(left))
                node = right
            elif const_right:
                c = float(self._evaluate_symbol(right))
                scale = scale * c if isinstance(node, pybamm.Multiplication) else scale / c
                node = left
            else:
                return None
        if not isinstance(node, pybamm.Interpolant):
            return None
        x_scale = self._x_scale(node.children[0]) if node.children else None
        if node.dimension != 1 or node.interpolator != "linear" or x_scale is None:
            raise NotRepresentableError(f"uses {_describe(node)} (only linear 1D interpolants of x are exact)")
        x = np.asarray(node.x[0] if isinstance(node.x, list | tuple) else node.x, dtype=float).ravel() / x_scale
        y = np.asarray(node.y, dtype=float).ravel() * scale
        order = np.argsort(x, kind="stable")
        return bpx.InterpolatedTable(x=x[order].tolist(), y=y[order].tolist())

    def _x_scale(self, node: pybamm.Symbol) -> float | None:
        """``a`` if ``node`` is ``x``, ``a * x``, ``x * a`` or ``x / (1/a)`` for a constant ``a``, else None."""
        if _is_x(node):
            return 1.0
        if isinstance(node, pybamm.Multiplication | pybamm.Division) and any(_is_x(c) for c in node.children):
            left, right = node.children
            other = right if _is_x(left) else left
            if any(_is_x(n) for n in other.pre_order()):
                return None
            a = float(self._evaluate_symbol(other))
            if isinstance(node, pybamm.Division):
                return 1 / a if _is_x(left) else None
            return a
        return None

    def _activation_energy(self, evaluate, grid) -> float:
        """Exact Arrhenius activation energy, 0 if temperature-independent; raises if not Arrhenius."""
        T0, T1, T2 = self.T_ref, self.T_ref + DELTA_T, self.T_ref - DELTA_T
        f0, f1, f2 = evaluate(grid, T0), evaluate(grid, T1), evaluate(grid, T2)
        if np.array_equal(f0, f1) and np.array_equal(f0, f2):
            return 0.0
        ok = np.isfinite(f0) & (f0 != 0)
        if np.any(~ok & ((f1 != f0) | (f2 != f0))) or not np.any(ok):
            raise NotRepresentableError("has a temperature dependence that is not Arrhenius (zero crossing)")
        ratio = f1[ok] / f0[ok]
        if np.any(ratio <= 0):
            raise NotRepresentableError("has a temperature dependence that is not Arrhenius (sign change)")
        Ea = R * np.log(ratio) / (1 / T0 - 1 / T1)
        Ea_ref = float(np.median(Ea))

        def consistent(E: float) -> bool:
            for T, f in ((T1, f1), (T2, f2)):
                if not _close(f0[ok] * np.exp(E / R * (1 / T0 - 1 / T)), f[ok], ARRHENIUS_RTOL):
                    return False
            return True

        if not consistent(Ea_ref):
            if not np.allclose(Ea, Ea_ref, rtol=ARRHENIUS_RTOL):
                detail = f"its apparent activation energy varies with x from {Ea.min():.6g} to {Ea.max():.6g} J/mol"
            else:
                Ea_below = float(np.median(R * np.log(f2[ok] / f0[ok]) / (1 / T0 - 1 / T2)))
                detail = (
                    f"its apparent activation energy is {Ea_ref:.6g} J/mol above {T0} K but {Ea_below:.6g} J/mol below"
                )
            raise NotRepresentableError(f"has a temperature dependence that is not Arrhenius ({detail})")
        return _snap(Ea_ref, consistent)

    def _fitted_activation_energy(self, evaluate, grid) -> float:
        T0, T1, T2 = self.T_ref, self.T_ref + DELTA_T, self.T_ref - DELTA_T
        f0 = evaluate(grid, T0)
        with np.errstate(all="ignore"):
            ln1, ln2 = np.log(evaluate(grid, T1) / f0), np.log(evaluate(grid, T2) / f0)
        Ea = np.concatenate([R * ln1 / (1 / T0 - 1 / T1), R * ln2 / (1 / T0 - 1 / T2)])
        Ea = Ea[np.isfinite(Ea)]
        return float(np.mean(Ea)) if Ea.size else 0.0
