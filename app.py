from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import subprocess
import os
from duty_cycle_opt import optimize_gear, WT_EXP_DEFAULT, GEARS_DEFAULT

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/run_python", methods=["POST"])
def run_python():
    try:
        # Check if the file is part of the request
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        uploaded_file = request.files["file"]
        if uploaded_file.filename == "":
            return jsonify({"status": "error", "message": "No file selected"}), 400

        # Save the file to the upload directory
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
        uploaded_file.save(file_path)

        # Execute the Python script using the uploaded file
        result = execute_python_script("loadChannelList", file_path)
        return jsonify({"status": "success", "result": result}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    
@app.route("/run_calcs", methods=["POST"])
def run_calcs():
    try:
        # Check if the file is part of the request
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        uploaded_file = request.files["file"]
        if uploaded_file.filename == "":
            return jsonify({"status": "error", "message": "No file selected"}), 400

        # Save the file to the upload directory
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
        uploaded_file.save(file_path)

        # Get the selected channels and exponents from the request
        load_channel = request.form.get("loadChannel")
        rev_channel = request.form.get("revChannel")
        exponents = json.loads(request.form.get("exponents"))

        # Call the runCalcs.py script with the file path, channels, and exponents as arguments
        result = execute_calc_script("runCalcs", file_path, load_channel, rev_channel, exponents)

        # Return the results from the script execution
        if "output" in result:
            return jsonify({"status": "success", "results": result["output"]}), 200
        else:
            return jsonify({"status": "error", "message": result["error"]}), 500

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/findPeaks", methods=["POST"])
def findPeaks():
    try:
        # Check if the file is part of the request
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        uploaded_file = request.files["file"]
        if uploaded_file.filename == "":
            return jsonify({"status": "error", "message": "No file selected"}), 400

        # Save the file to the upload directory
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
        uploaded_file.save(file_path)

        # Get the selected channels and exponents from the request
        load_channel = request.form.get("loadChannel")
        rev_channel = request.form.get("revChannel")
        prominence = request.form.get("prominence")
        threshold = request.form.get("threshold")

        # Call the findPeaks.py script with the file path, channels, prominence, and threshold as arguments
        result = execute_peaks_script("findPeaks", file_path, load_channel, rev_channel, prominence, threshold)

        # Return the results from the script execution
        if "output" in result:
            return jsonify({"status": "success", "results": result["output"]}), 200
        else:
            return jsonify({"status": "error", "message": result["error"]}), 500

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/optimizeDutyCycle", methods=["POST"])
def optimize_duty_cycle():
    try:
        data = request.get_json(force=True) or {}

        gear = data.get("gear", "High")
        if gear not in GEARS_DEFAULT:
            return jsonify({"status": "error", "message": f"Unknown gear '{gear}'"}), 400

        iter_min = int(data.get("iterMin", 1))
        iter_max = int(data.get("iterMax", 10))
        rear_bounds = tuple(map(float, data.get("rearBounds", [0, 4000])))
        front_bounds = tuple(map(float, data.get("frontBounds", [0, 5200])))
        cycle_bounds = tuple(map(float, data.get("cycleBounds", [0, 1e8])))
        max_total_cycles = float(data.get("maxTotalCycles", 1e9))
        wt_exps = data.get("wtExponents", WT_EXP_DEFAULT)

        result = optimize_gear(
            gear_name=gear,
            wt_exp_list=wt_exps,
            design_life_values=GEARS_DEFAULT[gear],
            iter_min=iter_min,
            iter_max=iter_max,
            rear_bounds=rear_bounds,
            front_bounds=front_bounds,
            cycle_bounds=cycle_bounds,
            max_total_cycles=max_total_cycles,
            popsize=int(data.get("popsize", 15)),
            maxiter=int(data.get("maxiter", 40)),
            workers=int(data.get("workers", 1)),  # set >1 later for speed
        )

        if result is None:
            return jsonify({"status": "error", "message": "No valid solution found"}), 500

        return jsonify({"status": "success", "result": result}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def execute_python_script(script_name, file_path):
    try:
        # Construct the command to run the Python script with the file as an argument
        command = f"python {script_name}.py {file_path}"
        result = subprocess.run(command, text=True, capture_output=True)

        if result.returncode == 0:
            return {"output": result.stdout}  # Return the script output
        else:
            return {"error": result.stderr}

    except Exception as e:
        return {"error": str(e)}
    
def execute_calc_script(script_name, file_path, load_channel, rev_channel, exponents):
    try:
        # Construct the command to run the Python script with the necessary arguments
        command = f"python {script_name}.py \"{file_path}\" \"{load_channel}\" \"{rev_channel}\" \"{json.dumps(exponents)}\""
        print(command)
        result = subprocess.run(command, text=True, capture_output=True, shell=True)

        if result.returncode == 0:
            return {"output": result.stdout}  # Return the script output
        else:
            return {"error": result.stderr}

    except Exception as e:
        return {"error": str(e)}

def execute_peaks_script(script_name, file_path, load_channel, rev_channel, prominence, threshold):
    try:
        # Construct the command to run the Python script with the necessary arguments
        command = f"python {script_name}.py \"{file_path}\" \"{load_channel}\" \"{rev_channel}\" \"{prominence}\" \"{threshold}\""
        print(command)
        result = subprocess.run(command, text=True, capture_output=True, shell=True)

        if result.returncode == 0:
            return {"output": result.stdout}  # Return the script output
        else:
            return {"error": result.stderr}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # The server will be accessible locally
