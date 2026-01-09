import json
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

# Reuse preprocessing implementation (Step B)
from coastdown_preprocess import compute_coastdown_preprocessed_series

# Unit constants
N_TO_LBF = 0.2248089
M_TO_FT = 3.280839895
MPS_TO_MPH = 2.236936


def _as_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _segment_lookup(segments: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for s in segments or []:
        try:
            sid = int(s.get("id"))
            out[sid] = s
        except Exception:
            continue
    return out


def _filter_points_to_segments(
    t: np.ndarray,
    v_mps: np.ndarray,
    force_n: np.ndarray,
    segments: List[Dict[str, Any]],
    segment_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (v_mps_selected, force_n_selected) across all selected segments.
    """
    seg_map = _segment_lookup(segments)
    masks = []

    for sid in segment_ids:
        s = seg_map.get(int(sid))
        if not s:
            continue
        t0 = _as_float(s.get("t_start"))
        t1 = _as_float(s.get("t_end"))
        if not np.isfinite(t0) or not np.isfinite(t1):
            continue
        if t1 < t0:
            t0, t1 = t1, t0
        masks.append((t >= t0) & (t <= t1))

    if not masks:
        return np.array([], dtype=float), np.array([], dtype=float)

    mask = np.logical_or.reduce(masks)
    v_sel = v_mps[mask]
    f_sel = force_n[mask]

    finite = np.isfinite(v_sel) & np.isfinite(f_sel)
    return v_sel[finite], f_sel[finite]


def _convert_coefficients_si_to_us(A_N: float, B_Ns_per_m: float, C_Ns2_per_m2: float):
    """
    Convert coefficients from:
      F_N = A_N + B_Ns/m * v_m/s + C_Ns^2/m^2 * v^2

    to:
      F_lbf = A_lbf + B_lbf*s/ft * v_ft/s + C_lbf*s^2/ft^2 * v^2
    """
    A_lbf = A_N * N_TO_LBF
    B_lbf_s_per_ft = B_Ns_per_m * (N_TO_LBF / M_TO_FT)
    C_lbf_s2_per_ft2 = C_Ns2_per_m2 * (N_TO_LBF / (M_TO_FT ** 2))
    return A_lbf, B_lbf_s_per_ft, C_lbf_s2_per_ft2


def fit_coastdown_abc_both_units(v_mps: np.ndarray, force_n: np.ndarray) -> Dict[str, Any]:
    """
    Fit: F = A + B*v + C*v^2  (SI internally)
    Returns both SI and USCS coefficients + fit quality.
    """
    n = v_mps.size
    if n < 10:
        raise ValueError("Not enough points to fit coefficients (need at least ~10).")

    X = np.column_stack([np.ones(n), v_mps, v_mps ** 2])
    y = force_n

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    A_N, B_Ns_per_m, C_Ns2_per_m2 = beta.tolist()

    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse_N = float(np.sqrt(ss_res / n)) if n > 0 else float("nan")

    A_lbf, B_lbf_s_per_ft, C_lbf_s2_per_ft2 = _convert_coefficients_si_to_us(
        A_N, B_Ns_per_m, C_Ns2_per_m2
    )
    rmse_lbf = rmse_N * N_TO_LBF

    return {
        "coefficients_si": {
            "A_N": float(A_N),
            "B_Ns_per_m": float(B_Ns_per_m),
            "C_Ns2_per_m2": float(C_Ns2_per_m2),
        },
        "coefficients_us": {
            "A_lbf": float(A_lbf),
            "B_lbf_s_per_ft": float(B_lbf_s_per_ft),
            "C_lbf_s2_per_ft2": float(C_lbf_s2_per_ft2),
        },
        "fit_quality": {
            "r2": r2,
            "rmse_N": rmse_N,
            "rmse_lbf": float(rmse_lbf),
            "n_points": int(n),
        },
    }


def _apply_speed_range_filter_mph(v_mps: np.ndarray, f_n: np.ndarray, min_mph, max_mph):
    if min_mph is None and max_mph is None:
        return v_mps, f_n

    v_mph = v_mps * MPS_TO_MPH
    mask = np.ones_like(v_mph, dtype=bool)
    if min_mph is not None:
        mask &= v_mph >= float(min_mph)
    if max_mph is not None:
        mask &= v_mph <= float(max_mph)

    return v_mps[mask], f_n[mask]


def main() -> None:
    """
    Usage:
      python coastdown_fit_coefficients.py <mf4_path> <dbc_path> '<preprocess_config_json>' '<fit_config_json>'

    fit_config JSON expected:
      {
        "segments": [...],         # output from detect script (list of segment dicts)
        "segment_ids": [1,2,3],    # selected ids to include
        "min_speed_mph": 20,       # optional
        "max_speed_mph": 70        # optional
      }
    """
    if len(sys.argv) < 5:
        print(
            json.dumps(
                {
                    "error": (
                        "Usage: coastdown_fit_coefficients.py <mf4_path> <dbc_path> "
                        "'<preprocess_config_json>' '<fit_config_json>'"
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
        fit_config = json.loads(sys.argv[4])
    except Exception as e:
        print(json.dumps({"error": f"Failed to parse fit_config JSON: {e}"}))
        sys.exit(1)

    try:
        # Preprocess (units/resample/smoothing handled there)
        preprocessed = compute_coastdown_preprocessed_series(mf4_path, dbc_path, preprocess_config)
        rows = preprocessed.get("data") or []
        if len(rows) < 2:
            raise ValueError("Preprocessed data is empty.")

        # Arrays
        t = np.array([r.get("t_s") for r in rows], dtype=float)
        v_mps = np.array([r.get("wheel_speed_ms") for r in rows], dtype=float)

        force_list = []
        for r in rows:
            f = r.get("force_n")
            force_list.append(np.nan if f is None else float(f))
        force_n = np.array(force_list, dtype=float)

        segments = fit_config.get("segments") or []
        segment_ids = fit_config.get("segment_ids") or []
        if not isinstance(segment_ids, list) or len(segment_ids) == 0:
            raise ValueError("fit_config.segment_ids must be a non-empty list.")
        segment_ids = [int(x) for x in segment_ids]

        min_speed_mph = fit_config.get("min_speed_mph", None)
        max_speed_mph = fit_config.get("max_speed_mph", None)

        seg_map = _segment_lookup(segments)

        # --- Per-segment fits ---
        per_segment: List[Dict[str, Any]] = []
        for sid in segment_ids:
            s = seg_map.get(int(sid))
            if not s:
                per_segment.append(
                    {
                        "segment_id": sid,
                        "error": "Segment ID not found in provided segments list.",
                    }
                )
                continue

            v_one, f_one = _filter_points_to_segments(t, v_mps, force_n, segments, [sid])
            v_one, f_one = _apply_speed_range_filter_mph(v_one, f_one, min_speed_mph, max_speed_mph)

            if v_one.size < 10:
                per_segment.append(
                    {
                        "segment_id": sid,
                        "t_start": _as_float(s.get("t_start")),
                        "t_end": _as_float(s.get("t_end")),
                        "v_start_mph": _as_float(s.get("v_start_mph")),
                        "v_end_mph": _as_float(s.get("v_end_mph")),
                        "error": "Not enough points to fit (need at least ~10 after filtering).",
                        "fit_quality": {"n_points": int(v_one.size)},
                    }
                )
                continue

            fit_one = fit_coastdown_abc_both_units(v_one, f_one)
            per_segment.append(
                {
                    "segment_id": sid,
                    "t_start": _as_float(s.get("t_start")),
                    "t_end": _as_float(s.get("t_end")),
                    "v_start_mph": _as_float(s.get("v_start_mph")),
                    "v_end_mph": _as_float(s.get("v_end_mph")),
                    **fit_one,
                }
            )

        # --- Combined fit across all selected segments (optional) ---
        combined = None
        try:
            v_all, f_all = _filter_points_to_segments(t, v_mps, force_n, segments, segment_ids)
            v_all, f_all = _apply_speed_range_filter_mph(v_all, f_all, min_speed_mph, max_speed_mph)
            if v_all.size >= 10:
                combined = fit_coastdown_abc_both_units(v_all, f_all)
        except Exception:
            combined = None

        out = {
            "per_segment": per_segment,
            "combined": combined,
            "meta": {
                "segment_ids": segment_ids,
                "speed_channels": preprocessed.get("meta", {}).get("speed_channels"),
                "vehicle_mass_kg": preprocessed.get("meta", {}).get("vehicle_mass_kg"),
                "speed_units_mode": preprocessed.get("meta", {}).get("speed_units_mode"),
                "resample_interval_s": preprocessed.get("meta", {}).get("resample_interval_s"),
                "smoothing_window": preprocessed.get("meta", {}).get("smoothing_window"),
                "min_speed_mph": min_speed_mph,
                "max_speed_mph": max_speed_mph,
            },
        }

        print(json.dumps(out))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
