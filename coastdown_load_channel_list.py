import json
import sys
from pathlib import Path
from asammdf import MDF

def looks_like_wheel_speed(name: str) -> bool:
    n = name.lower()
    tokens = ["wheel", "whl", "wss", "whlspeed", "wheel_speed", "whl_spd"]
    return any(t in n for t in tokens) and "engine" not in n and "gps" not in n

def build_channel_list(mf4_path: str, dbc_path: str):
    mf4_path = str(Path(mf4_path))
    dbc_path = str(Path(dbc_path))

    mdf = MDF(mf4_path)

    decoded = set()
    errors = []

    # Try a handful of common bus IDs
    for bus_id in [1, 2, 3, 4, 5, 6, 7, 8]:
        try:
            extracted = mdf.extract_bus_logging(
                database_files={"CAN": [(dbc_path, bus_id)]}
            )
            for ch in extracted.channels_db.keys():
                if ch.startswith("CAN_DataFrame"):
                    continue
                decoded.add(ch)
        except Exception as e:
            errors.append(f"bus_id={bus_id}: {e}")

    decoded_list = sorted(decoded)

    # Fallback: if decoding produced nothing, list raw MF4 channels
    raw_list = sorted(list(mdf.channels_db.keys()))

    if decoded_list:
        # Nice UX: wheel-speed-like at top but still return everything
        decoded_list.sort(key=lambda x: (not looks_like_wheel_speed(x), len(x)))
        return {"channels": decoded_list, "mode": "dbc_decoded", "errors": []}

    # No decoded channels -> fallback
    raw_list.sort(key=lambda x: (not looks_like_wheel_speed(x), len(x)))
    return {
        "channels": raw_list,
        "mode": "mf4_raw_fallback",
        "errors": errors[:10],  # include some debug context
    }

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: coastdown_load_channel_list.py <mf4_path> <dbc_path>"}))
        sys.exit(1)

    mf4_path = sys.argv[1]
    dbc_path = sys.argv[2]

    try:
        out = build_channel_list(mf4_path, dbc_path)
        print(json.dumps(out))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
