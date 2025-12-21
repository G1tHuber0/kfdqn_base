"""
顺序运行仓库内训练脚本，并用不同随机种子重复执行多次。
包含进度提示与参数配置。
"""

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# ================= 配置区域 =================

# 1.在此处调整需要执行的脚本清单
TRAIN_SCRIPTS = [
    "train_MountainCar/train_dqn.py",
    "train_MountainCar/train_double_dqn.py",
    "train_MountainCar/train_dueling_dqn.py",
    "train_MountainCar/train_reinforce.py",
    "train_MountainCar/train_ac.py",
    "train_MountainCar/train_kfdqn.py",
]

# 2.在此处修改默认运行参数（修改这里即可生效）
default_runs = 3        # 默认运行轮数
default_seed = 46       # 默认起始种子
default_seed_step = 2   # 默认种子递增步长

# ===========================================

# 说明：
# - 仓库内各 train_*.py 默认使用 Config.seed=42 且 log_dir 只精确到分钟/秒。
# - 使用 WRAPPER_CODE 动态注入，强制修改 seed 并重定向日志目录。
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all training scripts multiple times with different seeds.")
    
    # 修正：default值现在正确引用顶部的全局变量
    parser.add_argument(
        "--runs", 
        type=int, 
        default=default_runs, 
        help=f"Number of repeated runs (default: {default_runs})."
    )
    parser.add_argument(
        "--base-seed", 
        type=int, 
        default=default_seed, 
        help=f"Base seed for run 0 (default: {default_seed})."
    )
    parser.add_argument(
        "--seed-step", 
        type=int, 
        default=default_seed_step, 
        help=f"Seed increment per run (default: {default_seed_step})."
    )
    
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Continue remaining runs even if a script fails (default: stop on first failure).",
    )
    return parser.parse_args()


def _run_one_script(
    root: Path, 
    python_exec: str, 
    script: str, 
    *, 
    seed: int, 
    run_timestamp: str,
    progress_info: str = "" 
) -> int:
    script_path = root / script
    if not script_path.exists():
        print(f"[skip] {script} 不存在，跳过")
        return 0

    env = os.environ.copy()
    env["TRAIN_SCRIPT"] = str(script_path)
    env["TRAIN_SEED"] = str(seed)
    env["RUN_TAG"] = f"seed{seed}"
    env["RUN_TIMESTAMP"] = run_timestamp
    env["TRAIN_SILENT"] = "1"   # 屏蔽子脚本常规输出
    env["TQDM_POSITION"] = "0"

    cmd = [python_exec, "-c", WRAPPER_CODE]
    
    print(f">>>>>>[Start] | {progress_info} | {script} | seed={seed}")
    
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
    print(f"[Done] | {script}: {status}")
    return ret


def main() -> int:
    args = _parse_args()
    root = Path(__file__).resolve().parent
    python_exec = sys.executable or "python3"

    if args.runs <= 0:
        print("Runs must be > 0.")
        return 2

    # 生成统一的时间戳后缀
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 打印本次任务的配置摘要
    print(f"[Run All] Timestamp: {run_timestamp}")
    print(f"[Config]  Runs: {args.runs} | Base Seed: {args.base_seed} | Step: {args.seed_step}")
    print(f"[Scripts] Total: {len(TRAIN_SCRIPTS)}")

    total_scripts = len(TRAIN_SCRIPTS)

    for run_idx in range(args.runs):
        # 计算当前轮次的种子
        seed = args.base_seed + run_idx * args.seed_step
        
        print(f"\n{'='*25} [Run {run_idx + 1}/{args.runs}] seed={seed} {'='*25}")
        
        for script_idx, script in enumerate(TRAIN_SCRIPTS):
            # 进度提示字符串
            progress_str = f"[Run {run_idx + 1}/{args.runs}][Script {script_idx + 1}/{total_scripts}]"
            
            ret = _run_one_script(
                root, 
                python_exec, 
                script, 
                seed=seed, 
                run_timestamp=run_timestamp,
                progress_info=progress_str
            )
            
            if ret != 0 and not args.continue_on_fail:
                print(f"\nError occurred at {progress_str}. Stopping.")
                return ret

    print("\nAll runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())