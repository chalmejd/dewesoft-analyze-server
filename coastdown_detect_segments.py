import json
import sys
from typing import Any, Dict, List

import numpy as np

from coastdown_preprocess import compute_coastdown_preprocessed_series


def _get_float(cfg: Dict[str, Any], key: str, default: float) -> float:
    v = cfg.get(key, default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _get_int(cfg: Dict[str, Any], key: str, default: int) -> int:
    v = cfg.get(key, default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _first_crossing_down(
    v: np.ndarray,
    start_idx: int,
    v_target: float,
    tol: float,
) -> int:
    """
    Return the first index >= start_idx where v enters the target band [v_target - tol, v_target + tol].
    """
    lo = v_target - tol
    hi = v_target + tol
    idxs = np.where((v[start_idx:] >= lo) & (v[start_idx:] <= hi))[0]
    if idxs.size == 0:
        return -1
    return start_idx + int(idxs[0])


def _last_crossing_down(
    v: np.ndarray,
    start_idx: int,
    end_idx: int,
    v_target: float,
    tol: float,
) -> int:
    """
    Return the last index in [start_idx, end_idx] where v is within target band.
    """
    lo = v_target - tol
    hi = v_target + tol
    idxs = np.where((v[start_idx : end_idx + 1] >= lo) & (v[start_idx : end_idx + 1] <= hi))[0]
    if idxs.size == 0:
        return -1
    return start_idx + int(idxs[-1])


def detect_coastdown_segments_from_preprocessed(
    preprocessed: Dict[str, Any],
    segment_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    New behavior:
      - "start_speed_mph" is a TARGET (ex: 45)
      - "end_speed_mph" is a TARGET (ex: 0)
      - We find segments that start when speed enters start target band (± tolerance)
        and only if locally decelerating.
      - Segment ends when speed enters end target band (± tolerance) AFTER the start.

    segment_config keys:
      - start_speed_mph (default 45)
      - end_speed_mph (default 0)
      - tolerance_mph (default 1.0)         # band width around targets
      - require_decelerating (default True) # require dv < 0 around the start
      - decel_lookahead_points (default 3)  # how many diffs to check for decel at start
      - min_duration_s (default 8)
      - min_points (default 20)
      - max_positive_accel_mps2 (default 0.5)
      - max_abs_accel_mps2 (default 8.0)
      - require_monotonic_fraction (default 0.4) # fraction of dv <= 0 within segment
    """
    rows = preprocessed.get("data") or []
    if len(rows) < 2:
        return {"segments": [], "debug": {"reason": "preprocessed data has <2 rows"}}

    # Target speeds (the new meaning)
    start_target_mph = _get_float(segment_config, "start_speed_mph", 45.0)
    end_target_mph = _get_float(segment_config, "end_speed_mph", 0.0)
    tol_mph = _get_float(segment_config, "tolerance_mph", 1.0)

    require_decel = bool(segment_config.get("require_decelerating", True))
    decel_lookahead = _get_int(segment_config, "decel_lookahead_points", 3)

    min_duration_s = _get_float(segment_config, "min_duration_s", 8.0)
    min_points = _get_int(segment_config, "min_points", 20)
    max_pos_accel = _get_float(segment_config, "max_positive_accel_mps2", 0.5)
    max_abs_accel = _get_float(segment_config, "max_abs_accel_mps2", 8.0)
    mono_frac_req = _get_float(segment_config, "require_monotonic_fraction", 0.4)

    t = np.array([r.get("t_s") for r in rows], dtype=float)
    v_mph = np.array([r.get("wheel_speed_mph") for r in rows], dtype=float)

    accel_list = []
    for r in rows:
        a = r.get("accel_ms2")
        accel_list.append(np.nan if a is None else float(a))
    accel = np.array(accel_list, dtype=float)

    finite_mask = np.isfinite(t) & np.isfinite(v_mph)
    if not np.any(finite_mask):
        return {"segments": [], "debug": {"reason": "no finite time/speed points"}}

    # Candidate points: allow everything above end target band bottom, and reject crazy accel.
    # We don't enforce start_target here; we locate it by crossing.
    end_lo = end_target_mph - tol_mph
    cand = (
        finite_mask
        & (v_mph >= end_lo)
        & (np.isnan(accel) | (accel <= max_pos_accel))
        & (np.isnan(accel) | (np.abs(accel) <= max_abs_accel))
    )

    v_valid = v_mph[finite_mask]
    t_valid = t[finite_mask]

    debug = {
        "n_rows": int(len(rows)),
        "speed_min_mph": float(np.nanmin(v_valid)),
        "speed_max_mph": float(np.nanmax(v_valid)),
        "duration_s": float(np.nanmax(t_valid) - np.nanmin(t_valid)),
        "start_speed_mph": float(start_target_mph),
        "end_speed_mph": float(end_target_mph),
        "tolerance_mph": float(tol_mph),
        "require_decelerating": bool(require_decel),
        "decel_lookahead_points": int(decel_lookahead),
        "min_duration_s": float(min_duration_s),
        "min_points": int(min_points),
        "max_positive_accel_mps2": float(max_pos_accel),
        "max_abs_accel_mps2": float(max_abs_accel),
        "require_monotonic_fraction": float(mono_frac_req),
        "candidate_fraction": float(np.mean(cand[finite_mask])) if np.any(finite_mask) else 0.0,
    }

    # If you literally can't reach the target bands, signal that.
    if float(np.nanmax(v_valid)) < (start_target_mph - tol_mph):
        debug["reason_start"] = "max speed never reaches start target band; lower start_speed_mph or tolerance_mph"
    if float(np.nanmin(v_valid)) > (end_target_mph + tol_mph):
        debug["reason_end"] = "min speed never reaches end target band; raise end_speed_mph or tolerance_mph"

    segments: List[Dict[str, Any]] = []
    seg_id = 1

    n = len(t)
    i = 0
    while i < n:
        # Find next start crossing (entering start target band)
        start_idx = _first_crossing_down(v_mph, i, start_target_mph, tol_mph)
        if start_idx < 0:
            break

        # Require deceleration at the start (local trend down)
        if require_decel:
            j0 = start_idx
            j1 = min(n - 1, start_idx + max(1, decel_lookahead))
            dv_local = np.diff(v_mph[j0 : j1 + 1])
            # Must have at least one diff and be mostly negative (or <=0)
            if dv_local.size == 0 or float(np.mean(dv_local <= 0)) < 0.75:
                i = start_idx + 1
                continue

        # Now find end crossing after start
        # We’ll scan forward until we hit end band while still in candidate mask,
        # allowing gaps where cand is False to break and restart.
        end_idx = -1
        k = start_idx + 1
        while k < n:
            if not cand[k]:
                # If signal becomes invalid or acceleration is too positive, stop searching this segment.
                # Move search start forward.
                break

            # If we enter end target band, mark end and stop
            if (v_mph[k] >= (end_target_mph - tol_mph)) and (v_mph[k] <= (end_target_mph + tol_mph)):
                end_idx = k
                break

            k += 1

        if end_idx < 0:
            # No valid end crossing found; move forward after start and keep searching
            i = start_idx + 1
            continue

        # Validate length/duration
        if (end_idx - start_idx + 1) < min_points:
            i = end_idx + 1
            continue

        t_start = float(t[start_idx])
        t_end = float(t[end_idx])
        if (t_end - t_start) < min_duration_s:
            i = end_idx + 1
            continue

        # Monotonic fraction inside segment
        v_seg = v_mph[start_idx : end_idx + 1]
        dv = np.diff(v_seg)
        if dv.size == 0:
            i = end_idx + 1
            continue
        mono_fraction = float(np.mean(dv <= 0))
        if mono_fraction < mono_frac_req:
            i = end_idx + 1
            continue

        # Optional: tighten end to LAST point in end band (sometimes you bounce near 0)
        end_idx2 = _last_crossing_down(v_mph, start_idx, end_idx, end_target_mph, tol_mph)
        if end_idx2 >= 0 and end_idx2 > start_idx:
            end_idx = end_idx2
            t_end = float(t[end_idx])

        segments.append(
            {
                "id": seg_id,
                "t_start": t_start,
                "t_end": t_end,
                "v_start_mph": float(v_mph[start_idx]),
                "v_end_mph": float(v_mph[end_idx]),
                "n": int(end_idx - start_idx + 1),
                "start_target_mph": float(start_target_mph),
                "end_target_mph": float(end_target_mph),
                "tolerance_mph": float(tol_mph),
            }
        )
        seg_id += 1

        # Continue searching after the end
        i = end_idx + 1

    debug["segments_found"] = int(len(segments))
    if len(segments) == 0:
        debug["hint"] = (
            "Try increasing tolerance_mph (e.g., 2.0–5.0), "
            "or set require_decelerating=false if your signal is noisy, "
            "or increase max_positive_accel_mps2, or enable smoothing/resampling in preprocess."
        )

    return {"segments": segments, "debug": debug}


def main() -> None:
    """
    Usage:
      python coastdown_detect_segments.py <mf4_path> <dbc_path> '<preprocess_config_json>' '<segment_config_json>'
    """
    if len(sys.argv) < 5:
        print(
            json.dumps(
                {
                    "error": (
                        "Usage: coastdown_detect_segments.py <mf4_path> <dbc_path> "
                        "'<preprocess_config_json>' '<segment_config_json>'"
                    )
                }
            )
        )
        sys.exit(1)

    mf4_path = sys.argv[1]
    dbc_path = sys.argv[2]

    try:
        preprocess_config = json.loads(sys.argv[3])
    except Exception as e:
        print(json.dumps({"error": f"Failed to parse preprocess_config JSON: {e}"}))
        sys.exit(1)

    try:
        segment_config = json.loads(sys.argv[4])
    except Exception as e:
        print(json.dumps({"error": f"Failed to parse segment_config JSON: {e}"}))
        sys.exit(1)

    try:
        preprocessed = compute_coastdown_preprocessed_series(mf4_path, dbc_path, preprocess_config)
        out = detect_coastdown_segments_from_preprocessed(preprocessed, segment_config)
        print(json.dumps(out))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
