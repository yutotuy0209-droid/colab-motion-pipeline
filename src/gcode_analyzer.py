"""G-code Motion & Dwell Timing Analyzer for Cloud & Local Execution.

Analytically parses CNC G-code programs to determine:
  - Total travel distances per axis (X, Y, Z)
  - Feed transitions & exact dwell rest intervals (G04 X...)
  - Vibration-free steady-state camera trigger windows
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# ==============================================================================
# [FOR CODEX REFACTOR]: G-code Analytical Timeline & Rest Interval Profiler
# [INPUT/OUTPUT SPEC]:
#   - Input: gcode_text: str, default_feed_mm_min: float = 60.0
#   - Output: GCodeSimulationResult with total duration, distances, rest intervals
# [ALGORITHM INTENT]:
#   - Enables automated cloud simulation of machine tool paths without physical CNC.
#   - Identifies steady-state windows where stage velocity is strictly zero.
# [TODO FOR CODEX]:
#   - Add S-curve acceleration profile modeling for high-speed CNC feeds.
# ==============================================================================

@dataclass
class MotionSegment:
    command: str
    target_pos_um: Dict[str, float]
    feed_mm_min: float
    start_time_s: float
    duration_s: float
    is_dwell: bool = False


@dataclass
class GCodeSimulationResult:
    total_duration_s: float
    total_distance_um: Dict[str, float]
    segments: List[MotionSegment]
    rest_intervals: List[Tuple[float, float, Dict[str, float]]]


def parse_gcode_timeline(
    gcode_text: str,
    default_feed_mm_min: float = 60.0,
    rapid_feed_mm_min: float = 1000.0
) -> GCodeSimulationResult:
    """Parse G-code and compute analytical time and position trajectory."""
    lines = gcode_text.splitlines()

    curr_pos = {"X": 0.0, "Y": 0.0, "Z": 0.0}
    total_dist = {"X": 0.0, "Y": 0.0, "Z": 0.0}
    curr_feed = default_feed_mm_min
    curr_time = 0.0

    segments: List[MotionSegment] = []
    rest_intervals: List[Tuple[float, float, Dict[str, float]]] = []

    cmd_pattern = re.compile(r"([GXYZFM])\s*([-\d\.]+)", re.IGNORECASE)

    for raw_line in lines:
        line = raw_line.split(";")[0].split("(")[0].strip()
        if not line:
            continue

        tokens = cmd_pattern.findall(line)
        if not tokens:
            continue

        g_cmd = None
        new_pos = dict(curr_pos)
        dwell_s = 0.0
        has_motion = False

        for code, val_str in tokens:
            code = code.upper()
            val = float(val_str)

            if code == "G":
                g_val = int(val)
                if g_val in (0, 1, 4):
                    g_cmd = f"G{g_val:02d}"
            elif code == "F":
                curr_feed = val
            elif code in ("X", "Y", "Z"):
                val_um = val * 1000.0
                if g_cmd == "G04" and code == "X":
                    dwell_s = val
                else:
                    new_pos[code] = val_um
                    has_motion = True

        if g_cmd == "G04" and dwell_s > 0:
            seg = MotionSegment(
                command="G04",
                target_pos_um=dict(curr_pos),
                feed_mm_min=0.0,
                start_time_s=curr_time,
                duration_s=dwell_s,
                is_dwell=True
            )
            segments.append(seg)
            rest_intervals.append((curr_time, curr_time + dwell_s, dict(curr_pos)))
            curr_time += dwell_s

        elif has_motion and g_cmd in ("G00", "G01"):
            dx = new_pos["X"] - curr_pos["X"]
            dy = new_pos["Y"] - curr_pos["Y"]
            dz = new_pos["Z"] - curr_pos["Z"]
            dist_um = math.sqrt(dx**2 + dy**2 + dz**2)
            dist_mm = dist_um / 1000.0

            feed = rapid_feed_mm_min if g_cmd == "G00" else curr_feed
            feed_mm_s = max(1e-4, feed / 60.0)
            duration_s = dist_mm / feed_mm_s

            total_dist["X"] += abs(dx)
            total_dist["Y"] += abs(dy)
            total_dist["Z"] += abs(dz)

            seg = MotionSegment(
                command=g_cmd,
                target_pos_um=dict(new_pos),
                feed_mm_min=feed,
                start_time_s=curr_time,
                duration_s=duration_s,
                is_dwell=False
            )
            segments.append(seg)
            curr_time += duration_s
            curr_pos = dict(new_pos)

    return GCodeSimulationResult(
        total_duration_s=curr_time,
        total_distance_um=total_dist,
        segments=segments,
        rest_intervals=rest_intervals
    )
