"""Autonomous Agent Experiment API for Colab & Local Execution.

Enables AI agents (Antigravity & Codex) to execute experiments,
profile G-code timing, and fit backlash with machine-readable JSON I/O.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict
import numpy as np

from src.gcode_analyzer import parse_gcode_timeline
from src.grid_extractor import extract_subpixel_dots
from src.hysteresis_fitter import fit_bidirectional_backlash
from src.poc_core import compute_displacement
from src.rotation_estimator import estimate_rotation_center_qr
from src.synthetic_data import generate_motion_sequence


def execute_action(action: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute target computational action."""
    try:
        if action == "analyze_gcode":
            gcode_text = params.get("gcode", "")
            feed = float(params.get("feed", 60.0))
            res = parse_gcode_timeline(gcode_text, default_feed_mm_min=feed)
            return {
                "status": "success",
                "total_duration_s": res.total_duration_s,
                "total_distance_um": res.total_distance_um,
                "rest_intervals": [
                    {"start_s": t0, "end_s": t1, "duration_s": t1 - t0, "pos_um": p}
                    for t0, t1, p in res.rest_intervals
                ]
            }

        elif action == "fit_backlash":
            cmd = np.array(params.get("cmd_positions", []))
            meas = np.array(params.get("meas_positions", []))
            fit_res = fit_bidirectional_backlash(cmd, meas)
            return {
                "status": "success",
                "backlash_um": fit_res.backlash_um,
                "hysteresis_area": fit_res.hysteresis_area,
                "alpha_forward": fit_res.alpha_forward,
                "beta_forward": fit_res.beta_forward,
                "alpha_backward": fit_res.alpha_backward,
                "beta_backward": fit_res.beta_backward,
                "three_sigma": fit_res.three_sigma
            }

        elif action == "estimate_rotation":
            pts = np.array(params.get("points", []))
            res = estimate_rotation_center_qr(pts)
            return {
                "status": "success",
                "center_x": res.center_x,
                "center_y": res.center_y,
                "radius": res.radius,
                "peak_to_valley_runout": res.peak_to_valley_runout,
                "circularity_rms": res.circularity_rms
            }

        elif action == "synthetic_benchmark":
            frames_count = int(params.get("frames", 15))
            step_dx = float(params.get("step_dx", 0.20))
            step_dy = float(params.get("step_dy", 0.08))
            frames, tx, ty = generate_motion_sequence(num_frames=frames_count, step_dx=step_dx, step_dy=step_dy)
            meas_x, meas_y = [], []
            for i in range(len(frames)):
                dx, dy, p = compute_displacement(frames[0], frames[i], use_gpu=True)
                meas_x.append(dx)
                meas_y.append(dy)

            return {
                "status": "success",
                "frames": frames_count,
                "true_dx": list(tx),
                "true_dy": list(ty),
                "meas_dx": meas_x,
                "meas_dy": meas_y
            }

        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Colab Agent API")
    parser.add_argument("action", choices=["analyze_gcode", "fit_backlash", "estimate_rotation", "synthetic_benchmark"])
    parser.add_argument("--params", type=str, default="{}")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    params = json.loads(args.params)
    res = execute_action(args.action, params)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
