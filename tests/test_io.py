import json

import bpx
import pytest
import yaml
from conftest import parse_example

from bamm_gui.io import bpx_to_dict, write_bpx_file


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_write_then_parse_gives_equal_model(example_name, tmp_path, suffix):
    model = parse_example(example_name)
    path = write_bpx_file(model, tmp_path / f"out{suffix}")
    assert bpx.parse_bpx_file(path) == model


def test_dict_has_aliases_and_no_nulls(example_name):
    data = bpx_to_dict(parse_example(example_name))
    assert set(data) <= {"Header", "Parameterisation", "State", "Validation"}
    assert "BPX" in data["Header"]

    def walk(value):
        assert value is not None
        if isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(data)
    json.dumps(data)  # plain builtins only


def test_functions_and_tables_are_plain(tmp_path):
    model = parse_example("lfp_18650_cell_BPX.json")
    data = bpx_to_dict(model)
    positive = data["Parameterisation"]["Positive electrode"]
    assert type(positive["OCP [V]"]) is str
    assert set(positive["Entropic change coefficient [V.K-1]"]) == {"x", "y"}
    path = write_bpx_file(model, tmp_path / "out.yml")
    assert yaml.safe_load(path.read_text())["Parameterisation"]["Positive electrode"]["OCP [V]"] == positive["OCP [V]"]


def test_user_defined_tables_round_trip(tmp_path):
    model = parse_example("nmc_pouch_cell_BPX_user-defined_hysteresis.json")
    data = bpx_to_dict(model)
    table = data["Parameterisation"]["User-defined"]["Negative electrode lithiation OCP [V]"]
    assert set(table) == {"x", "y"} and len(table["x"]) == len(table["y"]) > 100
    assert bpx.parse_bpx_file(write_bpx_file(model, tmp_path / "out.json")) == model


def test_write_accepts_plain_dict(tmp_path):
    data = bpx_to_dict(parse_example("nmc_pouch_cell_BPX.json"))
    path = write_bpx_file(data, tmp_path / "out.json")
    assert json.loads(path.read_text()) == data


def test_floats_are_written_exactly(tmp_path):
    model = parse_example("nmc_pouch_cell_BPX.json")
    data = bpx_to_dict(model)
    data["Parameterisation"]["Cell"]["Nominal cell capacity [A.h]"] = 0.1 + 0.2  # 0.30000000000000004
    path = write_bpx_file(data, tmp_path / "out.json")
    assert bpx.parse_bpx_file(path).parameterisation.cell.nominal_cell_capacity == 0.1 + 0.2
