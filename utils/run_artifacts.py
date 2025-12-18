from __future__ import annotations

import datetime as _dt
import json as _json
import os as _os
import platform as _platform
import subprocess as _subprocess
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


def _safe_git(cmd: list[str]) -> Optional[str]:
    repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
    try:
        out = _subprocess.check_output(cmd, cwd=repo_root, stderr=_subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return None


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
        "git": {
            "commit": _safe_git(["git", "rev-parse", "HEAD"]),
            "status": _safe_git(["git", "status", "--porcelain"]),
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
    yaml_name: str = "config.yaml",
) -> None:
    _os.makedirs(log_dir, exist_ok=True)
    data = build_run_config(cfg, extra=extra)

    json_path = _os.path.join(log_dir, json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump(data, f, ensure_ascii=False, indent=2)

    try:
        import yaml  # type: ignore

        yaml_path = _os.path.join(log_dir, yaml_name)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    except Exception:
        # YAML 依赖不是必需：没有 PyYAML 时自动跳过，不影响训练。
        pass
