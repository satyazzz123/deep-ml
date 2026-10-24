
import numpy as np

def track_gpu_utilization(timestamps_ms: list, gpu_util_pct: list, phase_labels: list) -> dict:
    """
    Analyze GPU utilization during RL training and compute per-phase statistics.
    """
    timestamps = np.asarray(timestamps_ms, dtype=float)
    utilization = np.asarray(gpu_util_pct, dtype=float)

    # Duration of each interval
    diff = timestamps[1:] - timestamps[:-1]

    # Total duration (works even if timestamps don't start at zero)
    total_time_ms = timestamps[-1] - timestamps[0]

    # Overall time-weighted average GPU utilization
    avg_gpu_util_pct = np.dot(diff, utilization) / total_time_ms

    # Aggregate statistics for each phase
    phases = {}

    for i, phase in enumerate(phase_labels):
        duration = diff[i]
        util = utilization[i]

        if phase not in phases:
            phases[phase] = {
                "total_time": 0.0,
                "weighted_util": 0.0,
                "wasted_gpu_time": 0.0,
                "first_index": i,
            }

        phases[phase]["total_time"] += duration
        phases[phase]["weighted_util"] += duration * util
        phases[phase]["wasted_gpu_time"] += duration * (100 - util)

    # Construct phase statistics in alphabetical order
    phase_stats = {}

    for phase in sorted(phases):
        stats = phases[phase]
        phase_time = stats["total_time"]

        phase_stats[phase] = {
            "total_time_ms": round(phase_time, 2),
            "avg_util_pct": round(
                stats["weighted_util"] / phase_time, 2
            ),
            "time_fraction_pct": round(
                phase_time / total_time_ms * 100, 2
            ),
        }

    # Highest wasted GPU-time; ties go to earliest appearing phase
    bottleneck_phase = min(
        phases,
        key=lambda phase: (
            -phases[phase]["wasted_gpu_time"],
            phases[phase]["first_index"],
        ),
    )

    # Overall unused GPU capacity
    gpu_idle_fraction_pct = 100 - avg_gpu_util_pct

    return {
        "total_time_ms": round(total_time_ms, 2),
        "avg_gpu_util_pct": round(avg_gpu_util_pct, 2),
        "phase_stats": phase_stats,
        "bottleneck_phase": bottleneck_phase,
        "gpu_idle_fraction_pct": round(gpu_idle_fraction_pct, 2),
    }
