"""
一键运行导航/避障训练脚本（仿照 MountainCar_run_all.py）
- 支持多脚本顺序执行
- 支持多轮 runs + 不同 seed
- 动态注入覆盖 Config.seed / env_id，并重定向日志时间戳
- 可选启动 roscore + roslaunch，并等待 /scan /odom 出消息后再训练
"""

from __future__ import annotations

import argparse
import datetime
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


# ================= 配置区域 =================

# 1) 在此处调整需要执行的脚本清单（按你的仓库实际路径修改）
#    建议你把导航相关训练脚本都列在这里，例如 DQN / DoubleDQN / Dueling / KFDQN 等
TRAIN_SCRIPTS = [
    # 示例（请按你仓库实际存在的脚本改路径）
    "train_GoalReachROS/train_dqn.py",
    "train_GoalReachROS/train_double_dqn.py",
    "train_GoalReachROS/train_dueling_dqn.py",
    "train_GoalReachROS/train_kfdqn.py",
]

# 2) 默认运行参数
default_runs = 3
default_seed = 46
default_seed_step = 2

# 3) 默认环境 ID（与你注册的 Gymnasium env id 一致）
default_env_id = "GoalReachROS-v0"       # 纯导航
# default_env_id = "ObstacleAvoidROS-v0" # 避障导航

# 4) 可选：默认等待的 ROS topics
DEFAULT_WAIT_TOPICS = ["/scan", "/odom"]

# ===========================================


# 动态注入：强制改 seed/env_id，并让日志时间戳带唯一后缀
WRAPPER_CODE = r"""
import os
import runpy
import datetime

seed = int(os.environ.get("TRAIN_SEED", "42"))
env_id = os.environ.get("TRAIN_ENV_ID", "")
run_tag = os.environ.get("RUN_TAG", f"seed{seed}")
run_timestamp = os.environ.get("RUN_TIMESTAMP")
if not run_timestamp:
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 1) patch Config.seed / env_id（尽量兼容不同字段命名）
import config
_orig_init = config.Config.__init__

def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)

    # seed
    if hasattr(self, "seed"):
        self.seed = seed

    # env id (尽量覆盖常见字段名，不存在则忽略)
    if env_id:
        for k in ("env_id", "env_name", "env", "gym_id"):
            if hasattr(self, k):
                try:
                    setattr(self, k, env_id)
                except Exception:
                    pass

config.Config.__init__ = _patched_init

# 2) patch datetime 格式化：让 log_dir 唯一
_orig_dt = datetime.datetime

class _PatchedDatetime(_orig_dt):
    def strftime(self, fmt):
        base = super().strftime(fmt)
        if fmt in ("%Y%m%d_%H%M", "%Y%m%d_%H%M%S"):
            return f"{run_tag}_{run_timestamp}"
        return base

datetime.datetime = _PatchedDatetime

# 3) 运行训练脚本
script = os.environ["TRAIN_SCRIPT"]
runpy.run_path(script, run_name="__main__")
"""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run navigation training scripts with multiple seeds.")

    p.add_argument("--runs", type=int, default=default_runs, help=f"Number of repeated runs (default: {default_runs}).")
    p.add_argument("--base-seed", type=int, default=default_seed, help=f"Base seed (default: {default_seed}).")
    p.add_argument("--seed-step", type=int, default=default_seed_step, help=f"Seed increment (default: {default_seed_step}).")
    p.add_argument("--env-id", type=str, default=default_env_id, help=f"Gym env id (default: {default_env_id}).")

    p.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Continue remaining runs even if a script fails (default: stop on first failure).",
    )

    # 可选：一键拉起 ROS/Gazebo
    p.add_argument(
        "--roslaunch",
        type=str,
        default="",
        help='Optional roslaunch command, e.g. "roslaunch your_pkg your.launch gui:=false". If empty, do not launch.',
    )
    p.add_argument(
        "--no-roscore",
        action="store_true",
        help="Do not auto-start roscore (default: auto-start if needed).",
    )
    p.add_argument(
        "--wait-topics",
        type=str,
        default=",".join(DEFAULT_WAIT_TOPICS),
        help=f"Comma-separated topics to wait for messages before training (default: {DEFAULT_WAIT_TOPICS}).",
    )
    p.add_argument(
        "--wait-timeout",
        type=float,
        default=30.0,
        help="Wall-time seconds to wait for ROS topics to produce at least one message (default: 30s).",
    )

    return p.parse_args()


def _run_cmd_background(cmd: List[str], cwd: Path, env: dict, stdout_to_null: bool = True) -> subprocess.Popen:
    stdout = subprocess.DEVNULL if stdout_to_null else None
    stderr = subprocess.STDOUT if stdout_to_null else None
    return subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=stdout, stderr=stderr)


def _ros_master_is_up(env: dict) -> bool:
    # 用 rosnode list 判定 master 是否可用
    try:
        r = subprocess.run(["rosnode", "list"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
        return r.returncode == 0
    except Exception:
        return False


def _wait_for_topic_message(topic: str, env: dict, timeout_wall: float) -> bool:
    # rostopic echo -n 1 会在拿到一条消息后退出
    # 这里用 wall-time timeout，避免 sim-time 卡住
    t0 = time.time()
    while time.time() - t0 < timeout_wall:
        try:
            r = subprocess.run(
                ["rostopic", "echo", "-n", "1", topic],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            if r.returncode == 0:
                return True
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass
        time.sleep(0.2)
    return False


def _run_one_script(
    root: Path,
    python_exec: str,
    script: str,
    *,
    seed: int,
    env_id: str,
    run_timestamp: str,
    progress_info: str = "",
) -> int:
    script_path = root / script
    if not script_path.exists():
        print(f"[skip] {script} 不存在，跳过")
        return 0

    env = os.environ.copy()
    env["TRAIN_SCRIPT"] = str(script_path)
    env["TRAIN_SEED"] = str(seed)
    env["TRAIN_ENV_ID"] = env_id
    env["RUN_TAG"] = f"seed{seed}"
    env["RUN_TIMESTAMP"] = run_timestamp

    # 可选：给子脚本做静默（如果你脚本支持该环境变量）
    env["TRAIN_SILENT"] = "1"
    env["TQDM_POSITION"] = "0"

    cmd = [python_exec, "-c", WRAPPER_CODE]

    print(f">>>>>>[Start] | {progress_info} | {script} | env={env_id} | seed={seed}")
    proc = subprocess.Popen(cmd, cwd=str(root), env=env)
    try:
        ret = proc.wait()
    except KeyboardInterrupt:
        print("\n捕获到中断，尝试终止当前训练子进程...")
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

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[Run All - Navigation] Timestamp: {run_timestamp}")
    print(f"[Config] Runs: {args.runs} | Base Seed: {args.base_seed} | Step: {args.seed_step} | Env: {args.env_id}")
    print(f"[Scripts] Total: {len(TRAIN_SCRIPTS)}")

    # ---- 可选：启动 roscore / roslaunch ----
    env = os.environ.copy()
    roscore_proc: Optional[subprocess.Popen] = None
    roslaunch_proc: Optional[subprocess.Popen] = None

    try:
        if not args.no_roscore:
            if not _ros_master_is_up(env):
                print("[ROS] roscore not found, starting roscore ...")
                roscore_proc = _run_cmd_background(["roscore"], cwd=root, env=env, stdout_to_null=True)
                # 等 master up
                for _ in range(60):
                    if _ros_master_is_up(env):
                        break
                    time.sleep(0.2)
                if not _ros_master_is_up(env):
                    print("[ROS] roscore failed to start.")
                    return 3
                print("[ROS] roscore ready.")

        if args.roslaunch.strip():
            roslaunch_cmd = shlex.split(args.roslaunch.strip())
            print(f"[ROS] starting roslaunch: {args.roslaunch}")
            roslaunch_proc = _run_cmd_background(roslaunch_cmd, cwd=root, env=env, stdout_to_null=True)

        # 等 topics 出消息（如果用户配置了）
        wait_topics = [t.strip() for t in args.wait_topics.split(",") if t.strip()]
        if wait_topics:
            print(f"[ROS] waiting for topics to publish messages: {wait_topics} (timeout={args.wait_timeout}s)")
            for t in wait_topics:
                ok = _wait_for_topic_message(t, env=env, timeout_wall=args.wait_timeout)
                if not ok:
                    print(f"[ROS] timeout waiting for topic message: {t}")
                    return 4
            print("[ROS] topics ready.")

        # ---- 顺序运行训练脚本 ----
        total_scripts = len(TRAIN_SCRIPTS)

        for run_idx in range(args.runs):
            seed = args.base_seed + run_idx * args.seed_step
            print(f"\n{'='*25} [Run {run_idx + 1}/{args.runs}] seed={seed} {'='*25}")

            for script_idx, script in enumerate(TRAIN_SCRIPTS):
                progress = f"[Run {run_idx + 1}/{args.runs}][Script {script_idx + 1}/{total_scripts}]"
                ret = _run_one_script(
                    root,
                    python_exec,
                    script,
                    seed=seed,
                    env_id=args.env_id,
                    run_timestamp=run_timestamp,
                    progress_info=progress,
                )
                if ret != 0 and not args.continue_on_fail:
                    print(f"\nError occurred at {progress}. Stopping.")
                    return ret

        print("\nAll runs completed.")
        return 0

    finally:
        # ---- 清理 ROS 进程 ----
        def _terminate(proc: Optional[subprocess.Popen], name: str) -> None:
            if proc is None:
                return
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            print(f"[ROS] {name} terminated.")

        _terminate(roslaunch_proc, "roslaunch")
        _terminate(roscore_proc, "roscore")


if __name__ == "__main__":
    raise SystemExit(main())
