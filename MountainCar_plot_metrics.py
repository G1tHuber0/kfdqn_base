"""
从各算法的所有运行目录读取 metrics.json，按 seed+timestamp 分组生成对比图：
- total_return（横向柱状图）
- efficiency_episodes_for_success_target（横向柱状图，完成 250 次成功所需回合数）
- normalized_reward_variance（横向柱状图，归一化稳定性）
- avg_return_success_only（横向柱状图，仅成功回合的平均回报）
- reward_bins（Fail=-200 / Success<=-100 / Success>-100 的饼图，每个算法一张）
输出目录：results_MountainCar/Summary/<seed+timestamp>/。
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


BASE_RESULTS = Path("results_MountainCar")
# 固定算法顺序
ALGO_ORDER = ["DQN", "DoubleDQN", "DuelingDQN", "Reinforce", "AC", "KFDQN"]
# 柱状图颜色
BAR_COLORS = {
    "DQN": "#003f5c",          # 深蓝
    "DoubleDQN": "#d7191c",    # 红
    "DuelingDQN": "#fdae61",   # 橙
    "Reinforce": "#756bb1",    # 紫
    "AC": "#1b9e77",           # 绿
    "KFDQN": "#66c2ff",        # 浅蓝
}
BIN_LEGEND_ORDER = ["Fail (-200)", "Success (Normal)", "Success (High Performance)"]
BIN_COLORS = {
    "Fail (-200)": "#c1121f",
    "Success (Normal)": "#fcbf49",
    "Success (High Performance)": "#2a9d8f",
}
FIGSIZE_BAR = (9, 5)
FIGSIZE_PIE = (12, 8)
FIG_DPI = 120
SAVE_DPI = 300

plt.rcParams.update(
    {
        "figure.dpi": FIG_DPI,
        "savefig.dpi": SAVE_DPI,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


def _parse_timestamp_from_name(name: str) -> Optional[datetime]:
    parts = name.split("_")
    if len(parts) < 2:
        return None
    date_part = parts[-2]
    time_part = parts[-1]
    if not (date_part.isdigit() and time_part.isdigit()):
        return None
    if len(date_part) != 8 or len(time_part) not in (4, 6):
        return None
    fmt = "%Y%m%d_%H%M" if len(time_part) == 4 else "%Y%m%d_%H%M%S"
    try:
        return datetime.strptime(f"{date_part}_{time_part}", fmt)
    except ValueError:
        return None


def _extract_suffix(dir_path: Path) -> Optional[str]:
    name = dir_path.name
    parts = name.split("_")
    if len(parts) >= 3 and parts[-3].startswith("seed"):
        return "_".join(parts[-3:])
    if len(parts) >= 2:
        return "_".join(parts[-2:])
    return name or None


def _collect_all_metrics() -> Dict[str, Dict[str, Dict]]:
    all_metrics: Dict[str, Dict[str, Dict]] = {}
    if not BASE_RESULTS.exists():
        return all_metrics
    for algo in ALGO_ORDER:
        algo_dir = BASE_RESULTS / algo
        if not algo_dir.exists() or not algo_dir.is_dir():
            continue
        for sub in algo_dir.iterdir():
            if not sub.is_dir():
                continue
            metrics_path = sub / "metrics.json"
            if not metrics_path.exists():
                continue
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            suf = _extract_suffix(sub)
            if not suf:
                continue
            all_metrics.setdefault(suf, {})[algo] = data
    return all_metrics


def _sorted_suffixes(suffixes: List[str]) -> List[str]:
    def sort_key(suf: str):
        ts = _parse_timestamp_from_name(suf)
        if ts is not None:
            return (0, ts)
        return (1, suf)

    return sorted(suffixes, key=sort_key)


def _barh_plot(
    output: Path,
    title: str,
    metrics: Dict[str, Dict],
    key: str,
    xlabel: str,
    skip_none: bool = True,
):
    items = []
    for algo in ALGO_ORDER:
        if algo not in metrics:
            continue
        val = metrics[algo].get(key)
        if val is None and skip_none:
            continue
        items.append((algo, val))
    if not items:
        return
    algos, vals_raw = zip(*items)
    vals = []
    text_vals = []
    for v in vals_raw:
        if key in ("total_return", "episodes_to_cumulative_target", "efficiency_episodes_for_success_target"):
            bar_val = int(round(v))
            text_val = bar_val
        elif key == "normalized_reward_variance":
            bar_val = float(v)
            text_val = round(v, 4)
        elif key == "avg_return_success_only":
            bar_val = float(v)
            text_val = round(v, 2)
        else:
            bar_val = v
            text_val = v
        vals.append(bar_val)
        text_vals.append(text_val)

    y_pos = np.arange(len(algos))
    plt.figure(figsize=FIGSIZE_BAR)
    colors = [BAR_COLORS.get(a, "skyblue") for a in algos]
    bars = plt.barh(y_pos, vals, color=colors, edgecolor="black", linewidth=1)
    plt.yticks(y_pos, algos)
    plt.xlabel(xlabel)
    plt.title(title)
    max_abs = max(abs(v) for v in vals) if vals else 0.0
    offset = max_abs * 0.01 if max_abs else 0.1
    for bar, bar_val, text_val in zip(bars, vals, text_vals):
        if bar_val >= 0:
            text_x = bar.get_width() + offset
            align = "left"
        else:
            text_x = bar.get_width() - offset
            align = "right"
        plt.text(text_x, bar.get_y() + bar.get_height() / 2, f"{text_val}", va="center", ha=align)
    plt.tight_layout()
    plt.savefig(output, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()


def _combined_pie(output: Path, metrics: Dict[str, Dict]):
    algos = [a for a in ALGO_ORDER if a in metrics]
    if not algos:
        return
    n = len(algos)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    plt.figure(figsize=FIGSIZE_PIE)
    for idx, algo in enumerate(algos):
        bins = metrics[algo].get("reward_bins", [])
        if not bins:
            continue
        labels = [b.get("range", "") for b in bins]
        ratios = [b.get("ratio", 0.0) for b in bins]
        if all(r == 0 for r in ratios):
            continue
        colors = [BIN_COLORS.get(lbl, "#cccccc") for lbl in labels]
        ax = plt.subplot(rows, cols, idx + 1)
        wedges, texts, autotexts = ax.pie(
            ratios,
            labels=["" for _ in labels],
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            counterclock=True,
            wedgeprops={"edgecolor": "black", "linewidth": 1},
            textprops={"fontsize": 12},
        )
        for t in autotexts:
            t.set_fontsize(12)
            t.set_color("black")
        # 调整百分比位置，使不相邻太近
        for w, t in zip(wedges, autotexts):
            ang = (w.theta2 + w.theta1) / 2.0
            x = math.cos(math.radians(ang))
            y = math.sin(math.radians(ang))
            # 稍微外移
            t.set_position((1.2 * x, 1.2 * y))
        ax.set_title(f"{algo} reward_bins (ratio)")
    # 统一图例
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BIN_COLORS.get(label, "#cccccc"), ec="black", lw=1, label=label)
        for label in BIN_LEGEND_ORDER
    ]
    plt.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=len(BIN_LEGEND_ORDER))
    plt.tight_layout()
    plt.savefig(output, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()


def main() -> int:
    all_metrics = _collect_all_metrics()
    if not all_metrics:
        print("未找到 metrics.json，请先运行训练脚本生成指标。")
        return 1

    output_dirs: List[Path] = []
    for suffix in _sorted_suffixes(list(all_metrics.keys())):
        metrics_map = all_metrics[suffix]
        if not metrics_map:
            continue
        out_dir = BASE_RESULTS / "Summary" / suffix
        out_dir.mkdir(parents=True, exist_ok=True)

        target_count = next((m.get("success_target_count") for m in metrics_map.values() if m.get("success_target_count")), None)
        efficiency_title = (
            f"Efficiency: Episodes to {target_count} Successes"
            if target_count
            else "Efficiency: Episodes to Success Target"
        )

        _barh_plot(
            out_dir / "efficiency_episodes_for_success_target.png",
            efficiency_title,
            metrics_map,
            "efficiency_episodes_for_success_target",
            "Episodes",
        )
        _barh_plot(
            out_dir / "stability_normalized_reward_variance.png",
            "Stability (Normalized Reward Variance)",
            metrics_map,
            "normalized_reward_variance",
            "Variance",
        )
        _barh_plot(
            out_dir / "avg_return_success_only.png",
            "Avg Return (Successful Episodes)",
            metrics_map,
            "avg_return_success_only",
            "Return",
        )
        _barh_plot(
            out_dir / "total_return.png",
            "Total Return",
            metrics_map,
            "total_return",
            "Total Return",
        )

        _combined_pie(out_dir / "reward_bins_combined.png", metrics_map)
        output_dirs.append(out_dir)

    if not output_dirs:
        print("未找到可用的 metrics.json，请检查结果目录。")
        return 1

    last_dir = output_dirs[-1]
    print(f"图像已保存至: {last_dir} 等目录")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
