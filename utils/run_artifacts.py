from __future__ import annotations

import datetime as _dt
import json as _json
import os as _os
import platform as _platform
import sys as _sys
from typing import Any, Dict, Optional


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def build_run_config(cfg: Any, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "run": {
            "script": _os.path.basename(_sys.argv[0]) if _sys.argv else None,
            "argv": _sys.argv[1:] if len(_sys.argv) > 1 else [],
        },
        "system": {
            "python": _sys.version.split()[0],
            "platform": _platform.platform(),
        },
        "config": {k: _jsonable(v) for k, v in vars(cfg).items()},
    }
    if extra:
        data["extra"] = _jsonable(extra)
    return data


def save_run_config(
    log_dir: str,
    cfg: Any,
    *,
    extra: Optional[Dict[str, Any]] = None,
    json_name: str = "config.json",
    yaml_name: Optional[str] = None,
) -> None:
    _os.makedirs(log_dir, exist_ok=True)
    data = build_run_config(cfg, extra=extra)

    json_path = _os.path.join(log_dir, json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump(data, f, ensure_ascii=False, indent=2)


def save_metrics(
    log_dir: str,
    metrics: Dict[str, Any],
    *,
    json_name: str = "metrics.json",
    yaml_name: Optional[str] = None,
) -> None:
    _os.makedirs(log_dir, exist_ok=True)
    json_path = _os.path.join(log_dir, json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump(metrics, f, ensure_ascii=False, indent=2)
