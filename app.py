from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import subprocess
import os
import uuid
from werkzeug.utils import secure_filename

from duty_cycle_opt import optimize_duty_cycle
from testing_dashboard_backend import register_testing_dashboard_routes

app = Flask(__name__)
CORS(app)

register_testing_dashboard_routes(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Optional but recommended for large MF4 uploads (adjust as needed)
# app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB


# -------------------------------------------------------------------
# Helpers for uploads
# -------------------------------------------------------------------

ALLOWED_EXTENSIONS = {"dxd", "mf4", "dbc"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(file_storage, upload_dir: str) -> str:
    """
    Saves an uploaded file with a unique name in upload_dir.
    Returns the saved file path.
    """
    if file_storage is None or file_storage.filename is None or file_storage.filename.strip() == "":
        raise ValueError("Missing uploaded file.")

    filename = secure_filename(file_storage.filename)
    if not allowed_file(filename):
        raise ValueError(f"Unsupported file type: {filename}")

    unique_name = f"{uuid.uuid4().hex}_{filename}"
    file_path = os.path.join(upload_dir, unique_name)
    file_storage.save(file_path)
    return file_path


def safe_parse_json_stdout(stdout: str):
    """
    Attempts to parse JSON from a script's stdout.
    Falls back to returning the raw text if it isn't JSON.
    """
    s = (stdout or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return s


def cleanup_file(file_path: str):
    """
    Safely deletes an uploaded file.
    Does nothing if file doesn't exist.
    """
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Warning: Could not delete {file_path}: {e}")


# -------------------------------------------------------------------
# Existing endpoints
# -------------------------------------------------------------------

@app.route("/run_python", methods=["POST"])
def run_python():
    file_path = None
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        uploaded_file = request.files["file"]
        if uploaded_file.filename == "":
            return jsonify({"status": "error", "message": "No file selected"}), 400

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(uploaded_file.filename))
        uploaded_file.save(file_path)

        result = execute_python_script("loadChannelList", file_path)
        return jsonify({"status": "success", "result": result}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        cleanup_file(file_path)


@app.route("/run_calcs", methods=["POST"])
def run_calcs():
    file_path = None
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        uploaded_file = request.files["file"]
        if uploaded_file.filename == "":
            return jsonify({"status": "error", "message": "No file selected"}), 400

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(uploaded_file.filename))
        uploaded_file.save(file_path)

        load_channel = request.form.get("loadChannel")
        rev_channel = request.form.get("revChannel")
        exponents = json.loads(request.form.get("exponents"))

        result = execute_calc_script("runCalcs", file_path, load_channel, rev_channel, exponents)

        if "output" in result:
            return jsonify({"status": "success", "results": result["output"]}), 200
        else:
            return jsonify({"status": "error", "message": result["error"]}), 500

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        cleanup_file(file_path)


@app.route("/findPeaks", methods=["POST"])
def findPeaks():
    file_path = None
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        uploaded_file = request.files["file"]
        if uploaded_file.filename == "":
            return jsonify({"status": "error", "message": "No file selected"}), 400

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(uploaded_file.filename))
        uploaded_file.save(file_path)

        load_channel = request.form.get("loadChannel")
        rev_channel = request.form.get("revChannel")
        prominence = request.form.get("prominence")
        threshold = request.form.get("threshold")

        result = execute_peaks_script("findPeaks", file_path, load_channel, rev_channel, prominence, threshold)

        if "output" in result:
            return jsonify({"status": "success", "results": result["output"]}), 200
        else:
            return jsonify({"status": "error", "message": result["error"]}), 500

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        cleanup_file(file_path)


@app.route("/optimizeDutyCycle", methods=["POST"])
def optimize_duty_cycle_route():
    try:
        data = request.get_json(force=True) or {}

        label = data.get("label", "Custom Scenario")

        wt_exps = data.get("wtExponents", [])
        design_life_values = data.get("designLifeValues", [])

        if not wt_exps or not design_life_values:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "wtExponents and designLifeValues are required.",
                    }
                ),
                400,
            )

        if len(design_life_values) != len(wt_exps):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "designLifeValues must have the same number of rows as wtExponents.",
                    }
                ),
                400,
            )

        iter_min = int(data.get("iterMin", 1))
        iter_max = int(data.get("iterMax", 10))

        rear_bounds = tuple(map(float, data.get("rearBounds", [0, 4000])))
        front_bounds = tuple(map(float, data.get("frontBounds", [0, 5200])))
        cycle_bounds = tuple(map(float, data.get("cycleBounds", [0, 1e8])))
        max_total_cycles = float(data.get("maxTotalCycles", 1e9))

        popsize = int(data.get("popsize", 15))
        maxiter = int(data.get("maxiter", 1000))
        workers = int(data.get("workers", -1))

        result = optimize_duty_cycle(
            wt_exp_list=wt_exps,
            design_life_values=design_life_values,
            label=label,
            iter_min=iter_min,
            iter_max=iter_max,
            rear_bounds=rear_bounds,
            front_bounds=front_bounds,
            cycle_bounds=cycle_bounds,
            max_total_cycles=max_total_cycles,
            popsize=popsize,
            maxiter=maxiter,
            workers=workers,
        )

        if result is None:
            return (jsonify({"status": "error", "message": "No valid solution found."}), 500)

        return jsonify({"status": "success", "result": result}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------------------------------------------------------
# NEW: Coastdown endpoints (MF4 + DBC -> channel list -> preprocessing)
# -------------------------------------------------------------------

@app.route("/api/coastdown/channels", methods=["POST"])
def coastdown_channels():
    """
    Uploads MF4 + DBC and returns decoded channel list for user selection.

    Request: multipart/form-data
      - mf4: file
      - dbc: file

    Response:
      {
        "status": "success",
        "channels": [...],
        "mf4_path": "...",
        "dbc_path": "..."
      }
    """
    try:
        mf4_path = save_uploaded_file(request.files.get("mf4"), app.config["UPLOAD_FOLDER"])
        dbc_path = save_uploaded_file(request.files.get("dbc"), app.config["UPLOAD_FOLDER"])

        result = execute_simple_script("coastdown_load_channel_list", mf4_path, dbc_path)
        parsed = safe_parse_json_stdout(result.get("output", ""))

        if isinstance(parsed, dict) and "channels" in parsed:
            return jsonify(
                {
                    "status": "success",
                    "channels": parsed["channels"],
                    "mf4_path": mf4_path,
                    "dbc_path": dbc_path,
                }
            ), 200

        # If the script returned non-JSON or unexpected JSON, still return something useful
        return jsonify(
            {
                "status": "error",
                "message": "Unexpected response from coastdown_load_channel_list script.",
                "raw": parsed,
            }
        ), 500

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/coastdown/preprocess", methods=["POST"])
def coastdown_preprocess():
    """
    Runs coastdown preprocessing.

    Request: JSON
      {
        "mf4_path": "...",
        "dbc_path": "...",
        "speed_channels": ["WheelSpeed_FL"],   # required (1 or more)
        "combine_method": "mean",              # optional
        "vehicle_mass_kg": 1234.5,             # required
        "speed_units_mode": "auto",            # optional: auto|km/h|m/s|mph
        "resample_interval_s": 0.02,           # optional
        "smoothing_window": 11                 # optional
      }
    """
    try:
        payload = request.get_json(force=True) or {}

        mf4_path = payload.get("mf4_path")
        dbc_path = payload.get("dbc_path")

        speed_channels = payload.get("speed_channels") or []
        combine_method = payload.get("combine_method", "mean")
        vehicle_mass_kg = payload.get("vehicle_mass_kg")

        speed_units_mode = payload.get("speed_units_mode", "auto")
        resample_interval_s = payload.get("resample_interval_s", None)
        smoothing_window = payload.get("smoothing_window", None)

        if not mf4_path or not dbc_path:
            return jsonify({"status": "error", "message": "mf4_path and dbc_path are required."}), 400

        if not isinstance(speed_channels, list) or len(speed_channels) == 0:
            return jsonify({"status": "error", "message": "speed_channels must be a non-empty list."}), 400

        if vehicle_mass_kg is None:
            return jsonify({"status": "error", "message": "vehicle_mass_kg is required."}), 400

        preprocess_config = {
            "speed_channels": speed_channels,
            "combine_method": combine_method,
            "vehicle_mass_kg": float(vehicle_mass_kg),
            "speed_units_mode": speed_units_mode,
            "resample_interval_s": resample_interval_s,
            "smoothing_window": smoothing_window,
        }

        result = execute_simple_script(
            "coastdown_preprocess",
            mf4_path,
            dbc_path,
            json.dumps(preprocess_config),
        )

        parsed = safe_parse_json_stdout(result.get("output", ""))

        if "error" in result and result["error"]:
            return jsonify({"status": "error", "message": result["error"]}), 500

        if isinstance(parsed, dict) and parsed.get("error"):
            return jsonify({"status": "error", "message": parsed["error"]}), 500

        if isinstance(parsed, dict):
            return jsonify({"status": "success", "result": parsed}), 200

        return jsonify({"status": "success", "result": {"raw": parsed}}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/coastdown/segments", methods=["POST"])
def coastdown_segments():
    """
    Detects coastdown segments.

    Request JSON:
    {
      "mf4_path": "...",
      "dbc_path": "...",
      "preprocess_config": {...},
      "segment_config": {...}
    }
    """
    try:
        payload = request.get_json(force=True) or {}

        mf4_path = payload.get("mf4_path")
        dbc_path = payload.get("dbc_path")
        preprocess_config = payload.get("preprocess_config") or {}
        segment_config = payload.get("segment_config") or {}

        if not mf4_path or not dbc_path:
            return jsonify({"status": "error", "message": "mf4_path and dbc_path are required."}), 400

        result = execute_simple_script(
            "coastdown_detect_segments",
            mf4_path,
            dbc_path,
            json.dumps(preprocess_config),
            json.dumps(segment_config),
        )

        parsed = safe_parse_json_stdout(result.get("output", ""))

        if "error" in result and result["error"]:
            return jsonify({"status": "error", "message": result["error"]}), 500

        if isinstance(parsed, dict) and parsed.get("error"):
            return jsonify({"status": "error", "message": parsed["error"]}), 500

        return jsonify({"status": "success", "result": parsed}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/coastdown/fit", methods=["POST"])
def coastdown_fit():
    """
    Fits A/B/C coefficients.

    Request JSON:
    {
      "mf4_path": "...",
      "dbc_path": "...",
      "preprocess_config": {...},
      "segments": [...],
      "segment_ids": [1,2,3]
    }
    """
    mf4_path = None
    dbc_path = None
    try:
        payload = request.get_json(force=True) or {}

        mf4_path = payload.get("mf4_path")
        dbc_path = payload.get("dbc_path")
        preprocess_config = payload.get("preprocess_config") or {}
        segments = payload.get("segments") or []
        if isinstance(segments, dict) and "segments" in segments:
            segments = segments["segments"]

        segment_ids = payload.get("segment_ids") or []
        if not isinstance(segment_ids, list) or len(segment_ids) == 0:
            return jsonify({"status": "error", "message": "segment_ids must be a non-empty list."}), 400

        segment_ids = [int(x) for x in segment_ids]


        if not mf4_path or not dbc_path:
            return jsonify({"status": "error", "message": "mf4_path and dbc_path are required."}), 400

        if not segment_ids:
            return jsonify({"status": "error", "message": "segment_ids must be a non-empty list."}), 400

        fit_config = {
            "segments": segments,
            "segment_ids": segment_ids,
        }

        result = execute_simple_script(
            "coastdown_fit_coefficients",
            mf4_path,
            dbc_path,
            json.dumps(preprocess_config),
            json.dumps(fit_config),
        )

        parsed = safe_parse_json_stdout(result.get("output", ""))

        if "error" in result and result["error"]:
            return jsonify({"status": "error", "message": result["error"]}), 500

        if isinstance(parsed, dict) and parsed.get("error"):
            return jsonify({"status": "error", "message": parsed["error"]}), 500

        return jsonify({"status": "success", "result": parsed}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        cleanup_file(mf4_path)
        cleanup_file(dbc_path)



# -------------------------------------------------------------------
# Script execution helpers
# -------------------------------------------------------------------

def execute_python_script(script_name, file_path):
    try:
        command = f"python {script_name}.py {file_path}"
        result = subprocess.run(command, text=True, capture_output=True)

        if result.returncode == 0:
            return {"output": result.stdout}
        else:
            return {"error": result.stderr}

    except Exception as e:
        return {"error": str(e)}


def execute_calc_script(script_name, file_path, load_channel, rev_channel, exponents):
    try:
        command = f"python {script_name}.py \"{file_path}\" \"{load_channel}\" \"{rev_channel}\" \"{json.dumps(exponents)}\""
        print(command)
        result = subprocess.run(command, text=True, capture_output=True, shell=True)

        if result.returncode == 0:
            return {"output": result.stdout}
        else:
            return {"error": result.stderr}

    except Exception as e:
        return {"error": str(e)}


def execute_peaks_script(script_name, file_path, load_channel, rev_channel, prominence, threshold):
    try:
        command = f"python {script_name}.py \"{file_path}\" \"{load_channel}\" \"{rev_channel}\" \"{prominence}\" \"{threshold}\""
        print(command)
        result = subprocess.run(command, text=True, capture_output=True, shell=True)

        if result.returncode == 0:
            return {"output": result.stdout}
        else:
            return {"error": result.stderr}

    except Exception as e:
        return {"error": str(e)}


def execute_simple_script(script_name: str, *args: str):
    """
    Safe script runner that avoids shell-escaping issues with JSON.
    """
    try:
        cmd = ["python", f"{script_name}.py"]
        cmd.extend(args)

        print("EXEC:", cmd)

        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            shell=False,   # ✅ IMPORTANT
        )

        if result.returncode == 0:
            return {"output": result.stdout}
        else:
            return {"error": result.stderr, "output": result.stdout}

    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
