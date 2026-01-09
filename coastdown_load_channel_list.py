import json
import sys
from pathlib import Path

from asammdf import MDF


def build_coastdown_channel_list(mf4_path: str, dbc_path: str) -> list[str]:
    """
    Returns a sorted list of decoded signal channel names available after applying the DBC.
    We try multiple CAN bus ids because different loggers use different bus naming/ids.
    """
    mf4_path = str(Path(mf4_path))
    dbc_path = str(Path(dbc_path))

    mdf = MDF(mf4_path)

    # Try common CAN bus IDs (1..4). If your data always uses one bus, you can simplify.
    candidate_bus_ids = [1, 2, 3, 4]

    decoded_channel_names: set[str] = set()

    for bus_id in candidate_bus_ids:
        try:
            extracted = mdf.extract_bus_logging(
                database_files={"CAN": [(dbc_path, bus_id)]}
            )

            # asammdf keeps channels in a dict-like DB; keys are channel names usable via mdf.get(name)
            for ch_name in extracted.channels_db.keys():
                # Skip low-level/raw channels
                if ch_name.startswith("CAN_DataFrame"):
                    continue
                decoded_channel_names.add(ch_name)

        except Exception:
            # Keep trying other bus IDs
            continue

    return sorted(decoded_channel_names)


def main() -> None:
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: coastdown_load_channel_list.py <mf4_path> <dbc_path>"}))
        sys.exit(1)

    mf4_path = sys.argv[1]
    dbc_path = sys.argv[2]

    try:
        channels = build_coastdown_channel_list(mf4_path, dbc_path)
        print(json.dumps({"channels": channels}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
