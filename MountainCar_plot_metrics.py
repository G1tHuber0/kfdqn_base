"""
MountainCar 最终完美修正版：
1. 修复：当饼图为 100% 单一颜色时，移除圆内的垂直黑线，只保留完美的圆形外边框。
2. 保持：柱状图绝对值排序、负数向右、文字描边、饼图黑色分割线、智能标注。
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # <--- 新增：用于绘制完美的圆形外框
import matplotlib.patheffects as path_effects
import numpy as np

# --- 配置区域 ---
BASE_RESULTS = Path("results_MountainCar")

ALGO_ORDER = ["DQN", "DoubleDQN", "DuelingDQN", "Reinforce", "AC", "KFDQN"]

BAR_COLORS = {
    "DQN": "#003f5c",
    "DoubleDQN": "#d7191c",
    "DuelingDQN": "#fdae61",
    "Reinforce": "#756bb1",
    "AC": "#1b9e77",
    "KFDQN": "#66c2ff",
}

BIN_LEGEND_ORDER = ["Fail (-200)", "Success (Normal)", "Success (High Performance)"]
BIN_COLORS = {
    "Fail (-200)": "#c1121f",
    "Success (Normal)": "#fcbf49",
    "Success (High Performance)": "#2a9d8f",
}

FIGSIZE_BAR = (10, 6)
FIGSIZE_PIE = (12, 8)
FIG_DPI = 120
SAVE_DPI = 300

plt.rcParams.update({
    "figure.dpi": FIG_DPI,
    "savefig.dpi": SAVE_DPI,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "lines.linewidth": 1.5,
})


def _parse_timestamp_from_name(name: str) -> Optional[datetime]:
    parts = name.split("_")
    if len(parts) < 2: return None
    date_part, time_part = parts[-2], parts[-1]
    if not (date_part.isdigit() and time_part.isdigit() and len(date_part) == 8):
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
    all_metrics = {}
    if not BASE_RESULTS.exists(): return all_metrics
    for algo in ALGO_ORDER:
        algo_dir = BASE_RESULTS / algo
        if not algo_dir.is_dir(): continue
        for sub in algo_dir.iterdir():
            if not sub.is_dir(): continue
            metrics_path = sub / "metrics.json"
            if not metrics_path.exists(): continue
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                suf = _extract_suffix(sub)
                if suf: all_metrics.setdefault(suf, {})[algo] = data
            except Exception:
                continue
    return all_metrics


def _sorted_suffixes(suffixes: List[str]) -> List[str]:
    def sort_key(suf: str):
        ts = _parse_timestamp_from_name(suf)
        return (0, -ts.timestamp()) if ts else (1, suf)
    return sorted(suffixes, key=sort_key)


def _barh_plot_abs(output: Path, title: str, metrics: Dict[str, Dict], key: str, xlabel: str):
    items = []
    for algo in ALGO_ORDER:
        if algo in metrics and metrics[algo].get(key) is not None:
            items.append((algo, metrics[algo][key]))

    if not items: return

    algos, vals_raw = zip(*items)
    
    vals_abs = []
    text_strs = []

    for v in vals_raw:
        vals_abs.append(abs(v))
        if isinstance(v, float):
            if key == "normalized_reward_variance":
                text_strs.append(f"{v:.4f}")
            else:
                text_strs.append(f"{v:.2f}")
        else:
            text_strs.append(str(int(v)))

    y_pos = np.arange(len(algos))

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    colors = [BAR_COLORS.get(a, "#888888") for a in algos]

    bars = ax.barh(y_pos, vals_abs, color=colors, edgecolor="black", height=0.7, alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(algos)
    ax.set_xlabel(f"Magnitude (|{xlabel}|)")
    ax.set_title(title)
    
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    max_val = max(vals_abs) if vals_abs else 1.0
    threshold = max_val * 0.15

    for bar, text_str in zip(bars, text_strs):
        width = bar.get_width()
        y_center = bar.get_y() + bar.get_height() / 2
        
        if width > threshold:
            txt = ax.text(width - (max_val * 0.01), y_center, text_str, 
                          ha='right', va='center', color='white', fontweight='bold')
            txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
        else:
            txt = ax.text(width + (max_val * 0.01), y_center, text_str, 
                          ha='left', va='center', color='black', fontweight='bold')
            txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

    plt.tight_layout()
    plt.savefig(output, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()


def _combined_pie(output: Path, metrics: Dict[str, Dict]):
    algos = [a for a in ALGO_ORDER if a in metrics]
    if not algos: return

    n = len(algos)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=FIGSIZE_PIE)
    if n == 1: axes_flat = [axes]
    else: axes_flat = np.array(axes).reshape(-1)

    idx = 0
    valid = False
    for algo in algos:
        bins = metrics[algo].get("reward_bins", [])
        ax = axes_flat[idx]
        if not bins:
            ax.axis('off')
            idx += 1
            continue
            
        ratios = [b.get("ratio", 0.0) for b in bins]
        if sum(ratios) == 0:
            ax.axis('off')
            idx += 1
            continue
            
        valid = True
        colors = [BIN_COLORS.get(b.get("range"), "#ccc") for b in bins]
        
        # --- 核心逻辑修复 ---
        # 统计非 0 的块数，判断是否为 "100% 单一颜色"
        non_zero_count = sum(1 for r in ratios if r > 0)
        is_solid_circle = (non_zero_count == 1)

        if is_solid_circle:
            # 如果只有一块（100%），将线宽设为0，移除那条垂直的半径线
            current_wedgeprops = {"linewidth": 0}
        else:
            # 如果有多块，使用黑色边框来分隔区域
            current_wedgeprops = {"edgecolor": "black", "linewidth": 1.2, "antialiased": True}

        wedges, texts, autotexts = ax.pie(
            ratios, 
            labels=None, 
            autopct=lambda p: f'{p:.1f}%' if p > 0.0 else '',
            pctdistance=0.6,
            startangle=90, 
            colors=colors, 
            counterclock=False,
            wedgeprops=current_wedgeprops, # 应用动态属性
            textprops={'fontsize': 11, 'color': 'white', 'weight': 'bold'}
        )

        # 补救措施：如果刚才把线宽设为了0（去掉了竖线），现在需要把圆形外边框画回来
        if is_solid_circle:
            # 绘制一个纯圆环叠加在最上面，只画边框
            circle_border = mpatches.Circle(
                (0, 0), 1, 
                transform=ax.transData, 
                fill=False, 
                edgecolor='black', 
                linewidth=1.2
            )
            ax.add_patch(circle_border)

        # 给文字描边
        for t in autotexts:
            t.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])

        ax.set_title(algo, fontweight='bold')
        idx += 1

    if not valid:
        plt.close(fig)
        return

    for ax in axes_flat[idx:]: ax.axis("off")

    handles = [plt.Rectangle((0,0),1,1, color=BIN_COLORS.get(l), label=l) for l in BIN_LEGEND_ORDER]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output, dpi=SAVE_DPI)
    plt.close(fig)


def main() -> int:
    print(f"Reading from: {BASE_RESULTS.absolute()}")
    all_metrics = _collect_all_metrics()
    if not all_metrics:
        print("Error: No metrics.json found.")
        return 1

    for suffix in _sorted_suffixes(list(all_metrics.keys())):
        data = all_metrics[suffix]
        out_dir = BASE_RESULTS / "Summary" / suffix
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Generating charts for: {suffix}")

        target_cnt = next((m.get("success_target_count") for m in data.values() if m.get("success_target_count")), "Target")
        
        _barh_plot_abs(out_dir / "efficiency_episodes.png", 
                       f"Efficiency: Episodes to {target_cnt} Successes", 
                       data, "efficiency_episodes_for_success_target", "Episodes")

        _barh_plot_abs(out_dir / "stability_variance.png", 
                       "Stability (Normalized Reward Variance)", 
                       data, "normalized_reward_variance", "Variance")

        _barh_plot_abs(out_dir / "avg_return_success.png", 
                       "Avg Return (Successful Episodes Only)", 
                       data, "avg_return_success_only", "Return (Magnitude)")

        _barh_plot_abs(out_dir / "total_return.png", 
                       "Total Return (All Episodes)", 
                       data, "total_return", "Total Return (Magnitude)")

        _combined_pie(out_dir / "reward_distribution.png", data)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())