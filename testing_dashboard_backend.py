from __future__ import annotations

import csv
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request

try:
    from pymodbus.client import ModbusTcpClient
    MODBUS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import failure depends on environment
    ModbusTcpClient = None
    MODBUS_IMPORT_ERROR = exc


BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "testing_dashboard_config"
AB_DEVICES_CSV = CONFIG_DIR / "ab_devices.csv"
AB_REGISTER_MAP_CSV = CONFIG_DIR / "ab_register_map.csv"

MODBUS_PORT = 502
AB_DEFAULT_UNIT_ID = 1
AB_DEFAULT_TABLE = "holding"
AB_DEFAULT_START_ADDRESS = 0
AB_DEFAULT_REGISTER_COUNT = 20
AB_POLL_INTERVAL_SECONDS = 0.75
AB_MAX_SAMPLES = 180

M700_UNIT_ID = 2
M700_START_ADDRESS = 6999
M700_REGISTER_COUNT = 5
M700_POLL_INTERVAL_SECONDS = 0.5
M700_MAX_SAMPLES = 180

M700_DEVICES = {
    "Thermal Chamber-1": "10.12.4.181",
    "West Noise Room A": "10.10.3.92",
    "West Noise Room B": "10.10.3.90",
    "M1": "10.12.3.192",
    "M2": "10.12.2.49",
    "M3": "10.12.2.117",
    "Thermal Chamber-2": "10.12.1.125",
    "M4": "10.12.5.10",
    "Dyno Room - West": "10.14.72.91",
    "Dyno Room - South": "10.12.5.45",
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def to_signed_16(value: int) -> int:
    return value - 65536 if value > 32767 else value


def combine_32(low_word: int, high_word: int, signed: bool = False) -> int:
    value = (high_word << 16) | low_word
    if signed and value >= 2147483648:
        value -= 4294967296
    return value


def import_error_message() -> str | None:
    if MODBUS_IMPORT_ERROR is None:
        return None
    return f"pymodbus is unavailable: {MODBUS_IMPORT_ERROR}"


def read_modbus_registers(client, table: str, address: int, count: int, unit_id: int):
    if table == "input":
        read_fn = client.read_input_registers
    else:
        read_fn = client.read_holding_registers

    call_shapes = (
        {"address": address, "count": count, "device_id": unit_id},
        {"address": address, "count": count, "slave": unit_id},
    )

    last_error = None
    for kwargs in call_shapes:
        try:
            return read_fn(**kwargs)
        except TypeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    raise RuntimeError("Could not determine pymodbus register read signature.")


class AllenBradleyDashboardService:
    def __init__(self, devices_csv: Path, register_map_csv: Path):
        self.devices_csv = devices_csv
        self.register_map_csv = register_map_csv
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = False
        self.devices = self._load_devices()
        self.register_map = self._load_register_map()
        self.device_states = self._initialize_states()
        self.configuration_error = self._configuration_error()

    def _configuration_error(self) -> str | None:
        if not self.devices_csv.exists():
            return f"AB devices config not found: {self.devices_csv}"
        if not self.register_map_csv.exists():
            return f"AB register map config not found: {self.register_map_csv}"
        return None

    def service_error(self) -> str | None:
        return import_error_message() or self.configuration_error

    def _load_devices(self) -> list[dict[str, Any]]:
        loaded: list[dict[str, Any]] = []
        if not self.devices_csv.exists():
            return loaded

        with self.devices_csv.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = (row.get("name") or "").strip()
                ip = (row.get("ip") or "").strip()
                if not name or not ip:
                    continue

                loaded.append(
                    {
                        "name": name,
                        "ip": ip,
                        "unit_id": int(row.get("unit_id") or AB_DEFAULT_UNIT_ID),
                        "enabled": normalize_bool(row.get("enabled", "true")),
                        "table": (row.get("table") or AB_DEFAULT_TABLE).strip().lower(),
                        "start_address": int(row.get("start_address") or AB_DEFAULT_START_ADDRESS),
                        "register_count": int(row.get("register_count") or AB_DEFAULT_REGISTER_COUNT),
                    }
                )

        return loaded

    def _load_register_map(self) -> list[dict[str, Any]]:
        loaded: list[dict[str, Any]] = []
        if not self.register_map_csv.exists():
            return loaded

        with self.register_map_csv.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not normalize_bool(row.get("enabled", "true")):
                    continue

                loaded.append(
                    {
                        "table": (row.get("table") or AB_DEFAULT_TABLE).strip().lower(),
                        "register": int(row.get("register") or 0),
                        "name": (row.get("name") or "").strip(),
                        "type": (row.get("type") or "uint16").strip().lower(),
                        "scale": float(row.get("scale") or 1),
                        "units": (row.get("units") or "").strip(),
                        "signed": normalize_bool(row.get("signed", "false")),
                        "bit": (row.get("bit") or "").strip(),
                        "bit_name": (row.get("bit_name") or "").strip(),
                        "word_order": (row.get("word_order") or "high_low").strip().lower(),
                        "description": (row.get("description") or "").strip(),
                    }
                )

        return loaded

    def _initialize_states(self) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        for device in self.devices:
            states[device["name"]] = {
                "connected": False,
                "last_error": "",
                "last_update": None,
                "raw_registers": [],
                "decoded": [],
                "samples": deque(maxlen=AB_MAX_SAMPLES),
            }
        return states

    def _register_index(self, register_number: int, start_address: int) -> int:
        if start_address == 0 and register_number > 0:
            return register_number - 1
        return register_number - start_address

    def decode_registers(self, device: dict[str, Any], raw_registers: list[int]) -> list[dict[str, Any]]:
        table = device["table"]
        start_address = device["start_address"]
        decoded: list[dict[str, Any]] = []
        bitfield_groups: dict[int, list[dict[str, Any]]] = {}
        normal_channels: list[dict[str, Any]] = []

        for item in self.register_map:
            if item["table"] != table:
                continue
            if item["type"] == "bitfield":
                bitfield_groups.setdefault(item["register"], []).append(item)
            else:
                normal_channels.append(item)

        for register_number, items in sorted(bitfield_groups.items()):
            index = self._register_index(register_number, start_address)
            if index < 0 or index >= len(raw_registers):
                continue

            raw_value = raw_registers[index]
            bits_out = []
            for item in sorted(items, key=lambda current: int(current["bit"] or 0)):
                bit = int(item["bit"] or 0)
                value = 1 if raw_value & (1 << bit) else 0
                bits_out.append(
                    {
                        "name": item["bit_name"] or f"Bit {bit}",
                        "value": value,
                        "text": "On" if value else "Off",
                        "bit": bit,
                    }
                )

            decoded.append(
                {
                    "name": items[0]["name"] or f"Register {register_number}",
                    "register": register_number,
                    "type": "bitfield",
                    "raw": raw_value,
                    "bits": bits_out,
                    "description": items[0]["description"],
                }
            )

        for item in sorted(normal_channels, key=lambda current: (current["register"], current["name"])):
            register_number = item["register"]
            index = self._register_index(register_number, start_address)
            if index < 0 or index >= len(raw_registers):
                continue

            raw_value = raw_registers[index]
            value = raw_value
            display = str(raw_value)

            if item["type"] in {"int16", "scaled"} and item["signed"]:
                value = to_signed_16(raw_value)

            if item["type"] == "scaled":
                engineering_value = value * item["scale"]
                display = f"{engineering_value:.3f}".rstrip("0").rstrip(".")
            elif item["type"] == "int16":
                display = str(to_signed_16(raw_value) if item["signed"] else raw_value)
            elif item["type"] == "uint16":
                display = str(raw_value)
            elif item["type"] in {"uint32", "int32"}:
                second_index = index + 1
                if second_index >= len(raw_registers):
                    continue

                first_word = raw_registers[index]
                second_word = raw_registers[second_index]
                if item["word_order"] == "low_high":
                    low_word, high_word = first_word, second_word
                else:
                    high_word, low_word = first_word, second_word
                value = combine_32(low_word, high_word, signed=(item["type"] == "int32"))
                display = str(value)

            if item["units"]:
                display = f"{display} {item['units']}"

            decoded.append(
                {
                    "name": item["name"] or f"Register {register_number}",
                    "register": register_number,
                    "type": item["type"],
                    "raw": raw_value,
                    "value": value,
                    "display": display,
                    "units": item["units"],
                    "description": item["description"],
                }
            )

        return decoded

    def start(self) -> None:
        if self._started or self.service_error():
            return

        self._thread = threading.Thread(target=self._poll_forever, daemon=True, name="ab-dashboard-poller")
        self._thread.start()
        self._started = True

    def _poll_forever(self) -> None:
        while True:
            for device in self.devices:
                if not device["enabled"]:
                    continue

                name = device["name"]
                timestamp = _now_iso()
                client = ModbusTcpClient(device["ip"], port=MODBUS_PORT)

                try:
                    if not client.connect():
                        with self._lock:
                            self.device_states[name]["connected"] = False
                            self.device_states[name]["last_error"] = "Connection failed"
                            self.device_states[name]["last_update"] = timestamp
                        continue

                    response = read_modbus_registers(
                        client,
                        table=device["table"],
                        address=device["start_address"],
                        count=device["register_count"],
                        unit_id=device["unit_id"],
                    )

                    if response.isError():
                        with self._lock:
                            self.device_states[name]["connected"] = False
                            self.device_states[name]["last_error"] = f"Read error: {response}"
                            self.device_states[name]["last_update"] = timestamp
                        continue

                    raw_registers = list(response.registers)
                    decoded = self.decode_registers(device, raw_registers)
                    trend_value = raw_registers[0] if raw_registers else None

                    with self._lock:
                        state = self.device_states[name]
                        state["connected"] = True
                        state["last_error"] = ""
                        state["last_update"] = timestamp
                        state["raw_registers"] = raw_registers
                        state["decoded"] = decoded
                        if trend_value is not None:
                            state["samples"].append({"timestamp": timestamp, "value": trend_value})

                except Exception as exc:
                    with self._lock:
                        self.device_states[name]["connected"] = False
                        self.device_states[name]["last_error"] = f"{type(exc).__name__}: {exc}"
                        self.device_states[name]["last_update"] = timestamp
                finally:
                    try:
                        client.close()
                    except Exception:
                        pass

            time.sleep(AB_POLL_INTERVAL_SECONDS)

    def devices_payload(self) -> dict[str, Any]:
        self.start()
        return {
            "devices": self.devices,
            "default_device": self.devices[0]["name"] if self.devices else "",
            "error": self.service_error(),
        }

    def data_payload(self, device_name: str | None) -> tuple[dict[str, Any], int]:
        error = self.service_error()
        if error:
            return {"ok": False, "error": error}, 503

        if not self.devices:
            return {"ok": False, "error": "No AB devices configured."}, 500

        selected_name = device_name or self.devices[0]["name"]
        if selected_name not in self.device_states:
            return {"ok": False, "error": "Invalid device"}, 400

        with self._lock:
            device = next(current for current in self.devices if current["name"] == selected_name)
            state = self.device_states[selected_name]
            return (
                {
                    "ok": state["connected"],
                    "device": device,
                    "last_update": state["last_update"],
                    "last_error": state["last_error"],
                    "raw_registers": list(state["raw_registers"]),
                    "decoded": list(state["decoded"]),
                    "samples": list(state["samples"]),
                },
                200,
            )


class M700DashboardService:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = False
        self.device_states = {
            name: {
                "connected": False,
                "last_error": "",
                "last_update": None,
                "registers": [0] * M700_REGISTER_COUNT,
                "samples_by_register": {
                    str(index): deque(maxlen=M700_MAX_SAMPLES) for index in range(M700_REGISTER_COUNT)
                },
            }
            for name in M700_DEVICES
        }

    def service_error(self) -> str | None:
        return import_error_message()

    def start(self) -> None:
        if self._started or self.service_error():
            return

        self._thread = threading.Thread(target=self._poll_forever, daemon=True, name="m700-dashboard-poller")
        self._thread.start()
        self._started = True

    def _poll_forever(self) -> None:
        while True:
            for device_name, ip_address in M700_DEVICES.items():
                timestamp = _now_iso()
                client = ModbusTcpClient(ip_address, port=MODBUS_PORT)

                try:
                    if not client.connect():
                        with self._lock:
                            self.device_states[device_name]["connected"] = False
                            self.device_states[device_name]["last_error"] = "Connection failed"
                            self.device_states[device_name]["last_update"] = timestamp
                        continue

                    response = read_modbus_registers(
                        client,
                        table="holding",
                        address=M700_START_ADDRESS,
                        count=M700_REGISTER_COUNT,
                        unit_id=M700_UNIT_ID,
                    )

                    if response.isError():
                        with self._lock:
                            self.device_states[device_name]["connected"] = False
                            self.device_states[device_name]["last_error"] = f"Read error: {response}"
                            self.device_states[device_name]["last_update"] = timestamp
                        continue

                    registers = list(response.registers)
                    with self._lock:
                        state = self.device_states[device_name]
                        state["connected"] = True
                        state["last_error"] = ""
                        state["last_update"] = timestamp
                        state["registers"] = registers
                        for index, value in enumerate(registers):
                            state["samples_by_register"][str(index)].append(
                                {"timestamp": timestamp, "value": value}
                            )

                except Exception as exc:
                    with self._lock:
                        self.device_states[device_name]["connected"] = False
                        self.device_states[device_name]["last_error"] = f"{type(exc).__name__}: {exc}"
                        self.device_states[device_name]["last_update"] = timestamp
                finally:
                    try:
                        client.close()
                    except Exception:
                        pass

            time.sleep(M700_POLL_INTERVAL_SECONDS)

    def devices_payload(self) -> dict[str, Any]:
        self.start()
        return {
            "devices": [{"name": name, "ip": ip_address} for name, ip_address in M700_DEVICES.items()],
            "default_device": next(iter(M700_DEVICES), ""),
            "error": self.service_error(),
        }

    def registers_payload(self, device_name: str | None, register_index: str | None) -> tuple[dict[str, Any], int]:
        error = self.service_error()
        if error:
            return {"ok": False, "error": error}, 503

        selected_name = device_name or next(iter(M700_DEVICES), "")
        selected_register = register_index or "0"

        if selected_name not in M700_DEVICES:
            return {"ok": False, "error": "Invalid device"}, 400
        if selected_register not in {str(index) for index in range(M700_REGISTER_COUNT)}:
            return {"ok": False, "error": "Invalid register index"}, 400

        with self._lock:
            state = self.device_states[selected_name]
            return (
                {
                    "ok": state["connected"],
                    "device": {
                        "name": selected_name,
                        "ip": M700_DEVICES[selected_name],
                        "port": MODBUS_PORT,
                        "unit_id": M700_UNIT_ID,
                        "start_address": M700_START_ADDRESS,
                        "register_count": M700_REGISTER_COUNT,
                        "poll_interval_seconds": M700_POLL_INTERVAL_SECONDS,
                    },
                    "last_update": state["last_update"],
                    "registers": list(state["registers"]),
                    "last_error": state["last_error"],
                    "trend_register": int(selected_register),
                    "samples": list(state["samples_by_register"][selected_register]),
                },
                200,
            )


def register_testing_dashboard_routes(app) -> None:
    ab_service = AllenBradleyDashboardService(AB_DEVICES_CSV, AB_REGISTER_MAP_CSV)
    m700_service = M700DashboardService()

    dashboard_bp = Blueprint("testing_dashboard", __name__)

    @dashboard_bp.get("/api/testing-dashboard/ab/devices")
    def testing_dashboard_ab_devices():
        return jsonify(ab_service.devices_payload())

    @dashboard_bp.get("/api/testing-dashboard/ab/data")
    def testing_dashboard_ab_data():
        payload, status = ab_service.data_payload(request.args.get("device"))
        return jsonify(payload), status

    @dashboard_bp.get("/api/testing-dashboard/m700/devices")
    def testing_dashboard_m700_devices():
        return jsonify(m700_service.devices_payload())

    @dashboard_bp.get("/api/testing-dashboard/m700/registers")
    def testing_dashboard_m700_registers():
        payload, status = m700_service.registers_payload(
            request.args.get("device"),
            request.args.get("register"),
        )
        return jsonify(payload), status

    app.register_blueprint(dashboard_bp)
    app.extensions["testing_dashboard_services"] = {
        "ab": ab_service,
        "m700": m700_service,
    }
