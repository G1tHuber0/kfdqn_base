"""
顺序运行仓库内训练脚本，并用不同随机种子重复执行多次。
Ctrl+C 可中断，脚本会尽量终止当前子进程。
"""

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


# 可按需调整执行清单
TRAIN_SCRIPTS = [
    "train_MountainCar/train_dqn.py",
    "train_MountainCar/train_double_dqn.py",
    "train_MountainCar/train_dueling_dqn.py",
    "train_MountainCar/train_reinforce.py",
    "train_MountainCar/train_ac.py",
    "train_MountainCar/train_kfdqn.py",
]

# 说明：
# - 仓库内各 train_*.py 默认使用 Config.seed=42 且 log_dir 只精确到分钟/秒，直接重复运行会目录冲突。
# - 为了不改动各训练脚本/Config，这里用子进程 wrapper 动态注入：
#   1) 覆盖 Config.__init__ 中的 self.seed
#   2) 将 datetime.datetime.strftime("%Y%m%d_%H%M") 固定为 RUN_TIMESTAMP（run_all 启动时刻）
#      同时在时间前追加 RUN_TAG（默认 seedXXX）用于区分不同随机种子
WRAPPER_CODE = r"""
import os
import runpy
import datetime

seed = int(os.environ.get("TRAIN_SEED", "42"))
run_tag = os.environ.get("RUN_TAG", f"seed{seed}")
run_timestamp = os.environ.get("RUN_TIMESTAMP")
if not run_timestamp:
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

import config
_orig_init = config.Config.__init__

def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self.seed = seed

config.Config.__init__ = _patched_init

_orig_dt = datetime.datetime

class _PatchedDatetime(_orig_dt):
    def strftime(self, fmt):
        base = super().strftime(fmt)
        if fmt == "%Y%m%d_%H%M":
            return f"{run_tag}_{run_timestamp}"
        return base

datetime.datetime = _PatchedDatetime

script = os.environ["TRAIN_SCRIPT"]
runpy.run_path(script, run_name="__main__")
"""


default_runs = 1
default_seed = 66
default_seed_step = 1

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all training scripts multiple times with different seeds.")
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated runs (default: 50).")
    parser.add_argument("--base-seed", type=int, default=default_seed, help="Base seed for run 0 (default: 42).")
    parser.add_argument("--seed-step", type=int, default=1, help="Seed increment per run (default: 1).")
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Continue remaining runs even if a script fails (default: stop on first failure).",
    )
    return parser.parse_args()


def _run_one_script(root: Path, python_exec: str, script: str, *, seed: int, run_timestamp: str) -> int:
    script_path = root / script
    if not script_path.exists():
        print(f"[skip] {script} 不存在，跳过")
        return 0

    env = os.environ.copy()
    env["TRAIN_SCRIPT"] = str(script_path)
    env["TRAIN_SEED"] = str(seed)
    env["RUN_TAG"] = f"seed{seed}"
    env["RUN_TIMESTAMP"] = run_timestamp
    env["TRAIN_SILENT"] = "1"   # 批量运行时屏蔽子脚本的常规日志打印
    env["TQDM_POSITION"] = "0"  # 顺序运行时固定一行显示

    cmd = [python_exec, "-c", WRAPPER_CODE]
    print(f"[start] {script} | seed={seed}")
    proc = subprocess.Popen(cmd, cwd=root, env=env)
    try:
        ret = proc.wait()
    except KeyboardInterrupt:
        print("\n捕获到中断，尝试终止当前子进程...")
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        return 1

    status = "ok" if ret == 0 else f"fail({ret})"
    print(f"[done] {script}: {status}")
    return ret


def main() -> int:
    args = _parse_args()
    root = Path(__file__).resolve().parent
    python_exec = sys.executable or "python3"

    if args.runs <= 0:
        print("runs 必须为正整数。")
        return 2

    # 固定为 run_all 启动时刻：所有脚本都使用同一个时间后缀
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[run_all] timestamp={run_timestamp}")

    for run_idx in range(args.runs):
        seed = args.base_seed + run_idx * args.seed_step
        print(f"\n[run {run_idx + 1}/{args.runs}] seed={seed}")
        for script in TRAIN_SCRIPTS:
            ret = _run_one_script(root, python_exec, script, seed=seed, run_timestamp=run_timestamp)
            if ret != 0 and not args.continue_on_fail:
                print(f"\n在 run={run_idx + 1} 时失败，已停止（可用 --continue-on-fail 继续）。")
                return ret

    print("\n所有 runs 执行完毕。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
