"""
Tests for the params module.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from zembeddings.params import (
    PARAMS,
    get_params,
    load_params,
    save_params,
    describe_params,
)


class TestParams:
    def test_params_is_dict(self):
        assert isinstance(PARAMS, dict)

    def test_all_top_level_keys(self):
        expected = {
            "model", "window", "ema", "boundary", "kalman",
            "semantic_cloud", "derivatives", "database", "paths",
        }
        assert expected == set(PARAMS.keys())

    def test_get_params_returns_copy(self):
        p = get_params()
        assert p is not PARAMS
        p["window"]["size"] = 999
        assert PARAMS["window"]["size"] == 10  # unchanged

    def test_get_params_dot_override(self):
        p = get_params(**{"window.size": 42, "ema.alpha": 0.9})
        assert p["window"]["size"] == 42
        assert p["ema"]["alpha"] == 0.9
        # Other values untouched
        assert p["window"]["stride"] == 1

    def test_save_and_load(self, tmp_path):
        p = get_params(**{"window.size": 77})
        yaml_path = tmp_path / "test_params.yaml"
        save_params(p, str(yaml_path))
        loaded = load_params(str(yaml_path))
        assert loaded["window"]["size"] == 77

    def test_describe_params(self):
        desc = describe_params()
        assert "window" in desc
        assert "model" in desc
        assert isinstance(desc, str)
