from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

TAG_KEYS = (
    "strategy_name",
    "strategy_version",
    "strategy_params_hash",
)


def params_hash(params: Dict[str, Any]) -> str:
    """Return a short deterministic hash of the given parameters."""
    normalized = json.dumps(params, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return digest[:8]


def strategy_tags(name: str, version: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Build standard strategy tags payload."""
    params_copy = dict(params)
    return {
        "strategy_name": name,
        "strategy_version": version,
        "strategy_params_hash": params_hash(params_copy),
        "strategy_params": params_copy,
    }


def extract_strategy_tags(meta: Dict[str, Any]) -> Dict[str, Any]:
    tags = {key: meta.get(key) for key in TAG_KEYS}
    if any(value is None for value in tags.values()):
        return {}
    params = meta.get("strategy_params")
    if params is not None:
        tags["strategy_params"] = dict(params)
    return tags
