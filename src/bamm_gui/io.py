"""Serialise :class:`bpx.BPX` models back to BPX JSON/YAML.

``bpx`` ships parsers (``parse_bpx_file``/``parse_bpx_obj``) but no writer. These helpers are the
inverse, written so they can be upstreamed to BPX unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import bpx
import yaml

__all__ = ["bpx_to_dict", "write_bpx_file"]


def _plain(value: Any) -> Any:
    """Recursively convert pydantic models and BPX value types into JSON-compatible builtins."""
    if isinstance(value, bpx.InterpolatedTable):
        return {"x": list(value.x), "y": list(value.y)}
    if isinstance(value, bpx.Function):
        return str(value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items() if v is not None}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def bpx_to_dict(model: bpx.BPX) -> dict[str, Any]:
    """Convert a :class:`bpx.BPX` model into a plain dict keyed by the BPX field aliases.

    Absent optional fields are omitted rather than written as ``null``, so the result is a valid
    BPX object: ``bpx.parse_bpx_obj(bpx_to_dict(m)) == m``.
    """
    # User-defined entries are pydantic "extra" fields holding Function/InterpolatedTable
    # instances (possibly nested in dicts), so post-process the dump with _plain.
    return _plain(model.model_dump(by_alias=True, exclude_none=True, mode="python"))


def write_bpx_file(model: bpx.BPX | dict[str, Any], filename: str | Path, *, indent: int = 4) -> Path:
    """Write a BPX model (or an already-plain BPX dict) to ``filename``.

    The format follows the extension: ``.yml``/``.yaml`` writes YAML, anything else JSON. Floats are
    written with Python's shortest round-trip representation, so re-parsing gives identical values.
    """
    data = bpx_to_dict(model) if isinstance(model, bpx.BPX) else _plain(model)
    path = Path(filename)
    with path.open("w", encoding="utf-8") as f:
        if path.suffix.lower() in {".yml", ".yaml"}:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        else:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            f.write("\n")
    return path
