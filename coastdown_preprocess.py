import json
import sys
from pathlib import Path
from typing import Any, Optional, Literal

import numpy as np
from asammdf import MDF

# ---------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------
KMH_TO_MPS = 1.0 / 3.6
MPS_TO_KMH = 3.6
MPS_TO_MPH = 2.236936
MPH_TO_MPS = 1.0 / MPS_TO_MPH
N_TO_LBF = 0.2248089

SpeedUnitsMode = Literal["auto", "km/h", "kph", "m/s", "mph"]


# ---------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------
def _normalize_unit(unit: Optional[str]) -> Optional[str]:
    if not unit:
        return None
    u = str(unit).strip().lower().replace(" ", "")
    u = u.replace("kmh", "km/h").replace("kph", "km/h").replace("km/hr", "km/h")
    u = u.replace("mps", "m/s")

    if "km/h" in u:
        return "km/h"
    if "m/s" in u:
        return "m/s"
    if "mph" in u:
        return "mph"
    return None


def _infer_units(unit_field: Optional[str], values: np.ndarray) -> Optional[str]:
    """
    Infer units from MF4 metadata when possible, else use magnitude heuristic.
    Returns "km/h", "m/s", "mph", or None.
    """
    meta = _normalize_unit(unit_field)
    if meta in {"km/h", "m/s", "mph"}:
        return meta

    v = values[np.isfinite(values)]
    if v.size < 10:
        return None

    p95 = float(np.percentile(np.abs(v), 95))

    # Heuristic:
    # - >140 likely km/h
    # - <=70 likely m/s
    # - between could be mph or km/h (ambiguous) => None
    if p95 > 140:
        return "km/h"
    if p95 <= 70:
        return "m/s"
    return None


def _to_mps(values: np.ndarray, mode: SpeedUnitsMode, unit_field: Optional[str]) -> tuple[np.ndarray, str]:
    """
    Convert input values to m/s.
    Returns (values_mps, units_used_string).
    """
    m = mode.lower()

    if m in {"km/h", "kph"}:
        return values.astype(float) * KMH_TO_MPS, "km/h (user)"
    if m == "m/s":
        return values.astype(float), "m/s (user)"
    if m == "mph":
        return values.astype(float) * MPH_TO_MPS, "mph (user)"

    inferred = _infer_units(unit_field, values.astype(float))
    if inferred == "km/h":
        return values.astype(float) * KMH_TO_MPS, "km/h (auto)"
    if inferred == "m/s":
        return values.astype(float), "m/s (auto)"
    if inferred == "mph":
        return values.astype(float) * MPH_TO_MPS, "mph (auto)"

    raise ValueError(
        "Unable to determine speed units automatically for the selected channel(s). "
        "Set speed_units_mode explicitly to km/h, m/s, or mph."
    )


# ---------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------
def _get_mdf_with_dbc_if_possible(mf4_path: str, dbc_path: str, channel_name: str) -> MDF:
    """
    Try to DBC-decode CAN signals and return an MDF object that contains channel_name.
    If decoding doesn't yield the channel, returns the original MDF for raw-channel access.
    """
    mdf = MDF(mf4_path)
    candidate_bus_ids = [1, 2, 3, 4, 5, 6, 7, 8]

    for bus_id in candidate_bus_ids:
        try:
            extracted_try = mdf.extract_bus_logging(database_files={"CAN": [(dbc_path, bus_id)]})
            if channel_name in extracted_try.channels_db:
                return extracted_try
        except Exception:
            continue

    return mdf


def _extract_time_and_speed(
    mf4_path: str,
    dbc_path: str,
    channel_name: str,
    speed_units_mode: SpeedUnitsMode,
) -> tuple[np.ndarray, np.ndarray, str, Optional[str]]:
    """
    Returns (time_s, speed_mps, units_used, unit_field).
    """
    mf4_path = str(Path(mf4_path))
    dbc_path = str(Path(dbc_path))

    mdf = _get_mdf_with_dbc_if_possible(mf4_path, dbc_path, channel_name)
    sig = mdf.get(channel_name)

    time_s = np.asarray(sig.timestamps, dtype=float)
    raw = np.asarray(sig.samples, dtype=float)
    unit_field = getattr(sig, "unit", None)

    speed_mps, units_used = _to_mps(raw, speed_units_mode, unit_field)

    return time_s, speed_mps, units_used, unit_field


# ---------------------------------------------------------------------
# Resampling + smoothing
# ---------------------------------------------------------------------
def _resample_to_grid(time_s: np.ndarray, values: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """
    Interpolates values onto t_grid. Assumes time_s is increasing; sorts if needed.
    """
    order = np.argsort(time_s)
    t = time_s[order]
    v = values[order]

    # Remove duplicates in time (np.interp requires increasing x)
    unique_mask = np.concatenate(([True], np.diff(t) > 0))
    t = t[unique_mask]
    v = v[unique_mask]

    if t.size < 2:
        raise ValueError("Not enough points after removing duplicate timestamps to resample.")

    return np.interp(t_grid, t, v)


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return x
    if window % 2 == 0:
        # Prefer odd; if even provided, make it odd
        window += 1
    if window < 3:
        return x

    kernel = np.ones(window, dtype=float) / float(window)
    # "same" keeps length and aligns roughly centered
    return np.convolve(x, kernel, mode="same")


# ---------------------------------------------------------------------
# Main preprocessing compute
# ---------------------------------------------------------------------
def compute_coastdown_preprocessed_series(
    mf4_path: str,
    dbc_path: str,
    preprocess_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Preprocess for coastdown analysis: speed -> accel -> force.

    preprocess_config expected keys:
      - speed_channels: list[str] (required, >=1)
      - combine_method: "mean" (optional, default "mean")
      - vehicle_mass_kg: float (required)
      - speed_units_mode: "auto"|"km/h"|"m/s"|"mph" (optional, default "auto")
      - resample_interval_s: float|null (optional)
      - smoothing_window: int|null (optional)
    """
    speed_channels = preprocess_config.get("speed_channels") or []
    if not isinstance(speed_channels, list) or len(speed_channels) == 0:
        raise ValueError("preprocess_config.speed_channels must be a non-empty list.")

    combine_method = preprocess_config.get("combine_method", "mean")
    vehicle_mass_kg = float(preprocess_config["vehicle_mass_kg"])

    speed_units_mode: SpeedUnitsMode = preprocess_config.get("speed_units_mode", "auto")
    resample_interval_s = preprocess_config.get("resample_interval_s", None)
    smoothing_window = preprocess_config.get("smoothing_window", None)

    # Extract each channel in m/s
    series = []
    units_used_set = set()
    unit_fields = {}

    for ch in speed_channels:
        t, v_mps, units_used, unit_field = _extract_time_and_speed(
            mf4_path, dbc_path, ch, speed_units_mode
        )
        if t.size < 2:
            raise ValueError(f"Channel '{ch}' has fewer than 2 samples.")
        series.append((ch, t, v_mps))
        units_used_set.add(units_used)
        unit_fields[ch] = unit_field

    # Build a common time base
    # If resampling is requested, create a uniform grid over the overlapping time range.
    # Otherwise, use the first channel's native time base and interpolate others onto it.
    if resample_interval_s is not None:
        dt = float(resample_interval_s)
        if dt <= 0:
            raise ValueError("resample_interval_s must be > 0.")

        t_start = max(float(np.min(t)) for _, t, _ in series)
        t_end = min(float(np.max(t)) for _, t, _ in series)

        if t_end <= t_start:
            # No overlap; fall back to first channel range
            t_start = float(np.min(series[0][1]))
            t_end = float(np.max(series[0][1]))

        t_grid = np.arange(t_start, t_end, dt)
        if t_grid.size < 2:
            raise ValueError("Resample grid has fewer than 2 points; adjust resample_interval_s.")

        aligned = []
        for ch, t, v in series:
            aligned.append(_resample_to_grid(t, v, t_grid))

        speed_stack = np.vstack(aligned)
        time_base = t_grid

    else:
        # Native time base: use first channel timestamps
        time_base = np.asarray(series[0][1], dtype=float)
        aligned = [np.asarray(series[0][2], dtype=float)]
        for ch, t, v in series[1:]:
            aligned.append(_resample_to_grid(t, v, time_base))
        speed_stack = np.vstack(aligned)

    # Combine channels
    if speed_stack.shape[0] == 1:
        speed_mps = speed_stack[0]
    else:
        if combine_method != "mean":
            raise ValueError("Only combine_method='mean' is supported currently.")
        speed_mps = np.nanmean(speed_stack, axis=0)

    # Smoothing (on speed, before accel)
    if smoothing_window is not None:
        speed_mps = _moving_average(speed_mps, int(smoothing_window))

    # Acceleration: (Vn - Vn-1) / (Tn - Tn-1)
    dv = np.diff(speed_mps)
    dt = np.diff(time_base)

    accel = np.empty_like(speed_mps)
    accel[:] = np.nan
    valid = dt > 0
    accel[1:] = np.where(valid, dv / dt, np.nan)

    # Force
    force_n = accel * vehicle_mass_kg
    force_lbf = force_n * N_TO_LBF

    # Additional speed units for display
    speed_mph = speed_mps * MPS_TO_MPH
    speed_kmh = speed_mps * MPS_TO_KMH

    rows = []
    for i in range(time_base.size):
        rows.append(
            {
                "t_s": float(time_base[i]),
                "wheel_speed_kmh": float(speed_kmh[i]),
                "wheel_speed_ms": float(speed_mps[i]),
                "wheel_speed_mph": float(speed_mph[i]),
                "accel_ms2": None if np.isnan(accel[i]) else float(accel[i]),
                "force_n": None if np.isnan(force_n[i]) else float(force_n[i]),
                "force_lbf": None if np.isnan(force_lbf[i]) else float(force_lbf[i]),
            }
        )

    return {
        "meta": {
            "speed_channels": speed_channels,
            "combine_method": combine_method,
            "vehicle_mass_kg": vehicle_mass_kg,
            "speed_units_mode": speed_units_mode,
            "resample_interval_s": resample_interval_s,
            "smoothing_window": smoothing_window,
            "units_used": sorted(list(units_used_set)),
            "unit_fields": unit_fields,
            "constants": {
                "KMH_TO_MPS": KMH_TO_MPS,
                "MPS_TO_MPH": MPS_TO_MPH,
                "N_TO_LBF": N_TO_LBF,
            },
        },
        "data": rows,
    }


def main() -> None:
    """
    New CLI:
      python coastdown_preprocess.py <mf4_path> <dbc_path> '<json_preprocess_config>'
    """
    if len(sys.argv) < 4:
        print(json.dumps({"error": "Usage: coastdown_preprocess.py <mf4_path> <dbc_path> '<json_preprocess_config>'"}))
        sys.exit(1)

    mf4_path = sys.argv[1]
    dbc_path = sys.argv[2]

    try:
        preprocess_config = json.loads(sys.argv[3])
    except Exception as e:
        print(json.dumps({"error": f"Failed to parse preprocess_config JSON: {e}"}))
        sys.exit(1)

    try:
        result = compute_coastdown_preprocessed_series(
            mf4_path=mf4_path,
            dbc_path=dbc_path,
            preprocess_config=preprocess_config,
        )
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
