"""
从各算法的所有运行目录读取 metrics.json，按 seed+timestamp 分组生成对比图：
- total_return（横向柱状图）
- episodes_to_cumulative_target（横向柱状图）
- avg_reward_50_mad_to_success_threshold（横向柱状图）
- reward_bins（按 ratio 的饼图，每个算法一张）
输出目录：results_cartpole/Summary/<seed+timestamp>/。

修改点：
1. 移除 explode：解决扇形外缘不圆、裂缝过大的问题。
2. 完美的正圆：通过 wedgeprops={'edgecolor': 'black'} 实现区域分隔。
3. 保持文字描边和内部显示。
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np


BASE_RESULTS = Path("results_cartpole")
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
        "axes.titleweight": "bold",
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
    text_strs = []
    
    for v in vals_raw:
        if key == "total_return":
            val_num = int(round(v))
            vals.append(val_num)
            text_strs.append(str(val_num))
        elif key == "avg_reward_50_mad_to_success_threshold" and v is not None:
            val_num = round(v, 4)
            vals.append(val_num)
            text_strs.append(f"{val_num:.4f}")
        else:
            vals.append(v)
            text_strs.append(str(v))

    y_pos = np.arange(len(algos))
    plt.figure(figsize=FIGSIZE_BAR)
    
    colors = [BAR_COLORS.get(a, "skyblue") for a in algos]
    
    ax = plt.gca()
    bars = ax.barh(y_pos, vals, color=colors, edgecolor="black", linewidth=1, alpha=0.9)
    
    plt.yticks(y_pos, algos)
    plt.xlabel(xlabel)
    plt.title(title)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    max_val = max(vals) if vals else 1.0
    threshold = max_val * 0.15 

    for bar, text_str in zip(bars, text_strs):
        width = bar.get_width()
        y_center = bar.get_y() + bar.get_height() / 2
        
        if width > threshold:
            txt = plt.text(width - (max_val * 0.01), y_center, text_str, 
                           va="center", ha="right", color="white", fontweight="bold")
            txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
        else:
            txt = plt.text(width + (max_val * 0.01), y_center, text_str, 
                           va="center", ha="left", color="black", fontweight="bold")
            txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
            
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
    
    fig, axes = plt.subplots(rows, cols, figsize=FIGSIZE_PIE)
    if n == 1:
        axes_flat = [axes]
    else:
        axes_flat = np.array(axes).reshape(-1)
        
    colors = ["#390080", "#2ec4b6", "yellow"]  # <=100, (100,200), ==200
    legend_labels = ["<=100", "(100,200)", "==200"]
    
    idx = 0
    for algo in algos:
        bins = metrics[algo].get("reward_bins", [])
        ax = axes_flat[idx]
        
        if not bins:
            ax.axis('off')
            idx += 1
            continue
            
        labels = ["", "", ""]
        ratios = [b.get("ratio", 0.0) for b in bins]
        
        if all(r == 0 for r in ratios):
            ax.axis('off')
            idx += 1
            continue
        
        # --- 修正区域：移除 explode，回归完美的圆 ---
        wedges, texts, autotexts = ax.pie(
            ratios,
            labels=labels,
            autopct=lambda p: f'{p:.1f}%' if p > 0.0 else '', 
            pctdistance=0.6, 
            startangle=90,
            colors=colors,
            counterclock=True,
            # 关键：linewidth=1.2 创建清晰的黑色分隔线，edgecolor='black' 满足黑色描边需求
            wedgeprops={"edgecolor": "black", "linewidth": 1.2, "antialiased": True}, 
            textprops={"fontsize": 11, "weight": "bold", "color": "white"},
        )
        
        for t in autotexts:
            t.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
            
        ax.set_title(f"{algo}", fontweight="bold")
        idx += 1
        
    for ax in axes_flat[idx:]:
        ax.axis("off")
        
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[i], ec="black", lw=1, label=legend_labels[i])
        for i in range(len(colors))
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=len(colors), frameon=False)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


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
        print(f"Generating charts for: {suffix}")

        _barh_plot(
            out_dir / "total_return.png",
            "Total Return",
            metrics_map,
            "total_return",
            "Total Return",
        )
        _barh_plot(
            out_dir / "episodes_to_cumulative_target.png",
            "Episodes to Reach Cumulative 50,000",
            metrics_map,
            "episodes_to_cumulative_target",
            "Episodes",
        )
        _barh_plot(
            out_dir / "avg_reward_50_mad_to_success_threshold.png",
            "MAD to 200 (50-episode avg)",
            metrics_map,
            "avg_reward_50_mad_to_success_threshold",
            "MAD",
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