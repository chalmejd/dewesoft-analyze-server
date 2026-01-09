from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import subprocess
import os
import uuid
from werkzeug.utils import secure_filename

from duty_cycle_opt import optimize_duty_cycle

app = Flask(__name__)
CORS(app)

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


# -------------------------------------------------------------------
# Existing endpoints
# -------------------------------------------------------------------

@app.route("/run_python", methods=["POST"])
def run_python():
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


@app.route("/run_calcs", methods=["POST"])
def run_calcs():
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


@app.route("/findPeaks", methods=["POST"])
def findPeaks():
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
    Runs coastdown preprocessing after the user selects a wheel speed channel.

    Request: JSON
      {
        "mf4_path": "...",
        "dbc_path": "...",
        "selected_speed_channel": "...",
        "vehicle_mass_kg": 1234.5
      }

    Response: JSON returned by coastdown_preprocess.py (meta + data)
    """
    try:
        payload = request.get_json(force=True) or {}

        mf4_path = payload.get("mf4_path")
        dbc_path = payload.get("dbc_path")
        selected_channel = payload.get("selected_speed_channel")
        vehicle_mass_kg = payload.get("vehicle_mass_kg")

        if not mf4_path or not dbc_path:
            return jsonify({"status": "error", "message": "mf4_path and dbc_path are required."}), 400
        if not selected_channel:
            return jsonify({"status": "error", "message": "selected_speed_channel is required."}), 400
        if vehicle_mass_kg is None:
            return jsonify({"status": "error", "message": "vehicle_mass_kg is required."}), 400

        result = execute_simple_script(
            "coastdown_preprocess",
            mf4_path,
            dbc_path,
            selected_channel,
            str(float(vehicle_mass_kg)),
        )

        parsed = safe_parse_json_stdout(result.get("output", ""))

        # If script reported an error via stderr, surface it
        if "error" in result and result["error"]:
            return jsonify({"status": "error", "message": result["error"]}), 500

        # If parsed is dict, return it; otherwise wrap it
        if isinstance(parsed, dict):
            return jsonify({"status": "success", "result": parsed}), 200

        return jsonify({"status": "success", "result": {"raw": parsed}}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


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
    Generic script runner for scripts that just take argv args and output JSON/text to stdout.
    Uses shell=True for consistent quoting behavior with existing code style.
    """
    try:
        quoted_args = " ".join([f"\"{a}\"" for a in args])
        command = f"python {script_name}.py {quoted_args}"
        print(command)
        result = subprocess.run(command, text=True, capture_output=True, shell=True)

        if result.returncode == 0:
            return {"output": result.stdout}
        else:
            # keep stderr for debugging
            return {"error": result.stderr, "output": result.stdout}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
