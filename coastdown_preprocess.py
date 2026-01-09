import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from asammdf import MDF


KMH_TO_MS = 1.0 / 3.6
MS_TO_MPH = 2.236936
N_TO_LBF = 0.2248089


def compute_coastdown_preprocessed_series(
    mf4_path: str,
    dbc_path: str,
    wheel_speed_channel: str,
    vehicle_mass_kg: float,
) -> dict[str, Any]:
    """
    Loads MF4, applies DBC decoding, extracts the chosen wheel speed channel,
    computes speed conversions, acceleration, and force.
    Returns a JSON-serializable dict.
    """
    mf4_path = str(Path(mf4_path))
    dbc_path = str(Path(dbc_path))

    mdf = MDF(mf4_path)

    # Try multiple CAN bus ids for robustness; stop as soon as selected channel becomes available.
    candidate_bus_ids = [1, 2, 3, 4]
    extracted = None

    last_error = None
    for bus_id in candidate_bus_ids:
        try:
            extracted_try = mdf.extract_bus_logging(
                database_files={"CAN": [(dbc_path, bus_id)]}
            )
            if wheel_speed_channel in extracted_try.channels_db:
                extracted = extracted_try
                break
        except Exception as e:
            last_error = e
            continue

    if extracted is None:
        # Fallback: maybe the signal already exists without decoding, try direct access.
        extracted = mdf

    # Pull signal samples + timestamps
    sig = extracted.get(wheel_speed_channel)  # asammdf Signal object
    speed_kmh = np.asarray(sig.samples, dtype=float)
    time_s = np.asarray(sig.timestamps, dtype=float)

    if speed_kmh.size < 2:
        raise ValueError("Selected wheel speed channel has fewer than 2 samples; cannot compute acceleration.")

    # Convert speed
    speed_ms = speed_kmh * KMH_TO_MS
    speed_mph = speed_ms * MS_TO_MPH

    # Acceleration: (Vn - Vn-1) / (Tn - Tn-1)
    dv = np.diff(speed_ms)
    dt = np.diff(time_s)

    accel = np.empty_like(speed_ms)
    accel[:] = np.nan

    # Safe division (avoid dt<=0)
    valid = dt > 0
    accel[1:] = np.where(valid, dv / dt, np.nan)

    # Force
    force_n = accel * float(vehicle_mass_kg)
    force_lbf = force_n * N_TO_LBF

    # Package as rows (keep it simple for frontend tables/plots later)
    rows = []
    for i in range(speed_kmh.size):
        rows.append(
            {
                "t_s": float(time_s[i]),
                "wheel_speed_kmh": float(speed_kmh[i]),
                "wheel_speed_ms": float(speed_ms[i]),
                "wheel_speed_mph": float(speed_mph[i]),
                "accel_ms2": None if np.isnan(accel[i]) else float(accel[i]),
                "force_n": None if np.isnan(force_n[i]) else float(force_n[i]),
                "force_lbf": None if np.isnan(force_lbf[i]) else float(force_lbf[i]),
            }
        )

    return {
        "meta": {
            "wheel_speed_channel": wheel_speed_channel,
            "vehicle_mass_kg": float(vehicle_mass_kg),
            "constants": {
                "KMH_TO_MS": KMH_TO_MS,
                "MS_TO_MPH": MS_TO_MPH,
                "N_TO_LBF": N_TO_LBF,
            },
        },
        "data": rows,
    }


def main() -> None:
    if len(sys.argv) < 5:
        print(
            json.dumps(
                {
                    "error": (
                        "Usage: coastdown_preprocess.py <mf4_path> <dbc_path> "
                        "<wheel_speed_channel> <vehicle_mass_kg>"
                    )
                }
            )
        )
        sys.exit(1)

    mf4_path = sys.argv[1]
    dbc_path = sys.argv[2]
    wheel_speed_channel = sys.argv[3]
    vehicle_mass_kg = float(sys.argv[4])

    try:
        result = compute_coastdown_preprocessed_series(
            mf4_path=mf4_path,
            dbc_path=dbc_path,
            wheel_speed_channel=wheel_speed_channel,
            vehicle_mass_kg=vehicle_mass_kg,
        )
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
