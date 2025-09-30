import importlib
import json
import os
from pathlib import Path

import pytest


def reload_config_with(path: Path):
    os.environ["BOT_CONFIG_JSON"] = str(path)
    if "config" in globals():
        import sys as _sys

        _sys.modules.pop("config", None)
    import config as cfg

    importlib.reload(cfg)
    return cfg


def test_config_json_validation_rejects_unknown_key(tmp_path):
    try:
        import jsonschema  # noqa: F401
    except Exception:
        pytest.skip("jsonschema not installed; skip validation error check")
    bad = {"unknown_key": 123}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    try:
        try:
            reload_config_with(p)
            assert False, "Expected validation failure for unknown key"
        except RuntimeError as e:
            assert "validation" in str(e).lower() or "Failed to parse" in str(e)
    finally:
        os.environ.pop("BOT_CONFIG_JSON", None)


def test_config_json_valid_minimal(tmp_path):
    good = {"testnet": True, "universe": ["BTC/USDT", "ETH/USDT"]}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(good), encoding="utf-8")
    try:
        cfg = reload_config_with(p)
        assert cfg.TESTNET is True
        assert "BTC/USDT" in cfg.UNIVERSE
    finally:
        os.environ.pop("BOT_CONFIG_JSON", None)
