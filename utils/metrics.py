from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import math


def _variance(seq: Sequence[float]) -> Optional[float]:
    if not seq:
        return None
    mean = sum(seq) / len(seq)
    return sum((v - mean) ** 2 for v in seq) / len(seq)


def _normalize_curve(values: Sequence[float], value_range: Optional[Tuple[float, float]] = None) -> List[float]:
    if not values:
        return []
    if value_range is None:
        v_min, v_max = min(values), max(values)
    else:
        v_min, v_max = value_range
    if v_max - v_min <= 1e-9:
        return [0.0 for _ in values]
    normed = [(v - v_min) / (v_max - v_min) for v in values]
    return [min(1.0, max(0.0, n)) for n in normed]


def _episodes_to_success_target(returns: Sequence[float], cutoff: float, target: Optional[int]) -> Optional[int]:
    if target is None or target <= 0:
        return None
    success = 0
    for idx, r in enumerate(returns, 1):
        if r > cutoff:
            success += 1
        if success >= target:
            return idx
    return None


def _reward_bins(
    returns: Sequence[float],
    bins: Optional[List[float]] = None,
    custom_bins: Optional[List[Dict[str, float]]] = None,
):
    if custom_bins:
        total = len(returns) if returns else 1
        results = []
        for rule in custom_bins:
            label = rule.get("label", "")
            left = rule.get("left", -math.inf)
            right = rule.get("right", math.inf)
            left_open = rule.get("left_open", False)
            right_open = rule.get("right_open", True)
            count = 0
            for r in returns:
                cond_left = r > left if left_open else r >= left
                cond_right = r < right if right_open else r <= right
                if cond_left and cond_right:
                    count += 1
            results.append(
                {
                    "range": label,
                    "count": count,
                    "ratio": count / total,
                }
            )
        return results

    # 默认区间：<=100, (100, 200), ==200
    if bins is None:
        le_100 = sum(1 for r in returns if r <= 100)
        between = sum(1 for r in returns if (r > 100) and (r < 200))
        eq_200 = sum(1 for r in returns if r == 200)
        total = len(returns) if returns else 1
        return [
            {"range": "<=100", "count": le_100, "ratio": le_100 / total},
            {"range": "(100,200)", "count": between, "ratio": between / total},
            {"range": "==200", "count": eq_200, "ratio": eq_200 / total},
        ]

    # 自定义区间仍按边界列表处理（左闭右开，最后一段右闭）
    edges = bins
    labels = []
    counts = []
    ratios = []
    total = len(returns) if returns else 1
    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = [r >= left for r in returns]
        else:
            mask = [left <= r < right for r in returns]
        count = sum(mask)
        ratio = count / total
        labels.append(f"[{left},{right})" if right != math.inf else f"[{left},inf)")
        counts.append(count)
        ratios.append(ratio)
    return [{"range": labels[i], "count": counts[i], "ratio": ratios[i]} for i in range(len(labels))]


def _rolling_avg(seq: Sequence[float], window: int) -> List[float]:
    if window <= 0:
        return []
    buf: List[float] = []
    res: List[float] = []
    running = 0.0
    for i, v in enumerate(seq):
        running += v
        buf.append(v)
        if len(buf) > window:
            running -= buf.pop(0)
        if len(buf) == window:
            res.append(running / window)
    return res


def compute_training_metrics(
    returns: Sequence[float],
    avg_reward_50: Optional[Sequence[float]] = None,
    *,
    success_threshold: float = 200.0,
    cumulative_target: float = 50000.0,
    bins: Optional[List[float]] = None,
    custom_bins: Optional[List[Dict[str, float]]] = None,
    success_reward_cutoff: Optional[float] = None,
    success_target_count: Optional[int] = None,
    reward_norm_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    returns_list = list(returns)
    # 优先使用传入的 50 回合平均曲线，否则从原始回报重新按 50 回合滑窗计算
    if avg_reward_50 is not None:
        avg_list = list(avg_reward_50)
    else:
        avg_list = _rolling_avg(returns_list, window=50)

    total_return = float(sum(returns_list))
    metrics["total_return"] = int(round(total_return))
    metrics["episodes"] = len(returns_list)
    metrics["success_threshold"] = success_threshold

    if returns_list:
        successes = sum(1 for r in returns_list if r >= success_threshold)
        metrics["success_rate"] = successes / len(returns_list)

        cum_sum = 0.0
        episodes_to_target = None
        for idx, r in enumerate(returns_list, 1):
            cum_sum += r
            if cum_sum >= cumulative_target:
                episodes_to_target = idx
                break
        metrics["episodes_to_cumulative_target"] = episodes_to_target
    else:
        metrics["success_rate"] = 0.0
        metrics["episodes_to_cumulative_target"] = None

    success_cutoff = success_reward_cutoff if success_reward_cutoff is not None else success_threshold
    success_returns = [r for r in returns_list if success_cutoff is not None and r > success_cutoff]
    metrics["success_count_over_cutoff"] = len(success_returns)
    metrics["avg_return_success_only"] = (
        sum(success_returns) / len(success_returns) if success_returns else None
    )
    metrics["success_target_count"] = success_target_count
    metrics["efficiency_episodes_for_success_target"] = (
        _episodes_to_success_target(returns_list, success_cutoff, success_target_count)
        if success_cutoff is not None
        else None
    )

    curve_for_stability = avg_list if avg_list else returns_list
    if curve_for_stability:
        norm_curve = _normalize_curve(curve_for_stability, value_range=reward_norm_range)
        metrics["normalized_reward_variance"] = _variance(norm_curve)
    else:
        metrics["normalized_reward_variance"] = None

    if avg_list:
        mad = sum(abs(v - success_threshold) for v in avg_list) / len(avg_list)
        metrics["avg_reward_50_mad_to_success_threshold"] = mad
    else:
        metrics["avg_reward_50_mad_to_success_threshold"] = None

    metrics["reward_bins"] = _reward_bins(returns_list, bins=bins, custom_bins=custom_bins)
    return metrics
