from __future__ import annotations

import os
import random
from typing import Any, Optional

import numpy as np
import torch

def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    val = val.strip().lower()
    return val not in ("0", "false", "no", "off", "")


def seed_everything(
    seed: int,
    *,
    env: Optional[Any] = None,
    deterministic_torch: Optional[bool] = None,
) -> None:
    """
    Seed Python/NumPy/Torch (+ optional gymnasium env spaces) for reproducibility.

    Notes
    - `PYTHONHASHSEED` 需要在解释器启动前设置才完全生效；这里仍会设置它，方便子进程继承。
    - 若要强制 torch 的确定性，可传 deterministic_torch=True 或设置环境变量 TORCH_DETERMINISTIC=1。
    """
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch is None:
        deterministic_torch = _env_flag("TORCH_DETERMINISTIC", default=False)
    if deterministic_torch:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # 部分版本/算子可能不支持严格确定性，保持尽可能确定即可
            pass

    if env is not None:
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
        try:
            env.observation_space.seed(seed)
        except Exception:
            pass


def episode_seed(base_seed: int, episode_idx: int) -> int:
    """
    Compute a deterministic per-episode seed.

    Default keeps backward compatibility with existing code: `base_seed + episode_idx`.
    You can switch to a collision-resistant scheme via env var:
      EPISODE_SEED_MODE=ss
    """
    base_seed = int(base_seed)
    episode_idx = int(episode_idx)
    mode = os.environ.get("EPISODE_SEED_MODE", "add").strip().lower()
    if mode in ("ss", "seedsequence", "seed_sequence"):
        ss = np.random.SeedSequence([base_seed, episode_idx])
        return int(ss.generate_state(1, dtype=np.uint32)[0])
    # default: backward compatible
    return base_seed + episode_idx

