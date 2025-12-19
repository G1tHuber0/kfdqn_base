"""
并行运行仓库内所有训练脚本，互不阻塞。
Ctrl+C 可中断，脚本会尽量终止所有子进程。
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


# 可按需调整执行清单
TRAIN_SCRIPTS = [
    "train_dqn.py",
    "train_double_dqn.py",
    "train_dueling_dqn.py",
    "train_reinforce.py",
    "train_ac.py",
    "train_kfdqn.py",
]


def main() -> int:
    root = Path(__file__).resolve().parent
    python_exec = sys.executable or "python3"

    procs: Dict[str, subprocess.Popen] = {}
    skipped: List[str] = []

    for idx, script in enumerate(TRAIN_SCRIPTS):
        script_path = root / script
        if not script_path.exists():
            skipped.append(script)
            print(f"[skip] {script} 不存在，跳过")
            continue
        cmd = [python_exec, str(script_path)]
        env = os.environ.copy()
        env["TQDM_POSITION"] = str(idx)  # 为 tqdm 分配独立行
        env["TRAIN_SILENT"] = "1"       # 并行时屏蔽子脚本的常规日志打印
        print(f"[start] {script} (tqdm position={idx})")
        proc = subprocess.Popen(cmd, cwd=root, env=env)
        procs[script] = proc

    if not procs:
        print("没有可运行的脚本。")
        return 0

    try:
        while procs:
            finished = []
            for name, proc in procs.items():
                ret = proc.poll()
                if ret is not None:
                    finished.append(name)
                    status = "ok" if ret == 0 else f"fail({ret})"
                    print(f"[done] {name}: {status}")
            for name in finished:
                procs.pop(name, None)
            if procs:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n捕获到中断，尝试终止子进程...")
        for name, proc in procs.items():
            try:
                proc.terminate()
            except Exception:
                pass
        for name, proc in procs.items():
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        return 1

    print("所有脚本执行完毕。")
    if skipped:
        print(f"已跳过: {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
