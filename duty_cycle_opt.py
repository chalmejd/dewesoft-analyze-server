# dewesoft-analyze-server-main/duty_cycle_opt.py
import numpy as np
from scipy.optimize import differential_evolution

# You can reuse WT_EXP_DEFAULT and GEARS_DEFAULT from your standalone script
WT_EXP_DEFAULT = [3.0, 6.610, 8.738]

GEARS_DEFAULT = {
    "High": np.array([
        [9.1248195984045E+14, 8.304523587773310E+14, 6.34313109572686E+10],
        [2.1052774438076E+26, 1.888774484389690E+26, 1.79082466323855E+18],
        [3.9609805248929E+33, 3.819191398263470E+33, 8.32426164073660E+22],
    ]),
    "Low": np.array([
        [1.0150775735106E+15, 9.000728882134040E+14, 5.29359208409142E+10],
        [2.3041176558236E+26, 2.067798403993310E+26, 1.41946983751728E+18],
        [4.3349965198631E+33, 4.177665082405250E+33, 6.55723042758595E+22],
    ]),
    "Reverse": np.array([
        [6.9724351117245E+14, 6.176917970829040E+14, 5.99832990593517E+10],
        [1.6778404524204E+26, 1.489468390103020E+26, 1.81440693257572E+18],
        [3.3790518042029E+33, 3.261541113136620E+33, 8.28317750727505E+22],
    ]),
}


def calculate_damage(inputs, wt_exp_list, design_life_values):
    """
    inputs: [rows x 3] -> [rear_torque, front_torque, cycles]
    design_life_values: [num_exp x 3] -> columns = [Front+Rear, Rear, Front]
    """
    inputs = np.asarray(inputs, dtype=float)
    wt_exp_list = np.asarray(wt_exp_list, dtype=float)
    num_exp = len(wt_exp_list)

    total_damage = np.zeros(num_exp * 3, dtype=float)

    for rear_torque, front_torque, cycles in inputs:
        for i, exp in enumerate(wt_exp_list):
            total_damage[i] += (((front_torque + rear_torque) ** exp) * cycles) / design_life_values[i, 0]
            total_damage[i + num_exp] += ((rear_torque ** exp) * cycles) / design_life_values[i, 1]
            total_damage[i + 2 * num_exp] += ((front_torque ** exp) * cycles) / design_life_values[i, 2]

    return total_damage


def objective_function(flat_inputs, wt_exp_list, design_life_values, iter_count,
                       rear_bounds, front_bounds, cycle_bounds, max_total_cycles):
    inputs = np.array(flat_inputs).reshape(iter_count, 3)

    # Constraints
    if np.any(inputs < 0):
        return np.inf
    if np.any((inputs[:, 0] < rear_bounds[0]) | (inputs[:, 0] > rear_bounds[1])):
        return np.inf
    if np.any((inputs[:, 1] < front_bounds[0]) | (inputs[:, 1] > front_bounds[1])):
        return np.inf
    if np.any((inputs[:, 2] < cycle_bounds[0]) | (inputs[:, 2] > cycle_bounds[1])):
        return np.inf
    if np.sum(inputs[:, 2]) > max_total_cycles:
        return np.inf

    total_damage = calculate_damage(inputs, wt_exp_list, design_life_values)
    # Target is damage ~ 1.0 for each component
    return np.sqrt(np.mean((total_damage - 1.0) ** 2))


def optimize_gear(gear_name,
                  wt_exp_list=WT_EXP_DEFAULT,
                  design_life_values=None,
                  iter_min=1,
                  iter_max=10,
                  rear_bounds=(0.0, 4000.0),
                  front_bounds=(0.0, 5200.0),
                  cycle_bounds=(0.0, 1e8),
                  max_total_cycles=1e9,
                  popsize=15,
                  maxiter=40,
                  workers=1):
    if design_life_values is None:
        design_life_values = GEARS_DEFAULT[gear_name]
    design_life_values = np.asarray(design_life_values, dtype=float)

    best_overall = None
    best_iter_count = None
    best_fun = np.inf

    for iter_count in range(iter_min, iter_max + 1):
        bounds = []
        for _ in range(iter_count):
            bounds.append(rear_bounds)
            bounds.append(front_bounds)
            bounds.append(cycle_bounds)

        result = differential_evolution(
            objective_function,
            bounds=bounds,
            args=(wt_exp_list, design_life_values, iter_count,
                  rear_bounds, front_bounds, cycle_bounds, max_total_cycles),
            popsize=popsize,
            maxiter=maxiter,
            tol=0.01,
            updating="deferred",
            workers=workers,   # can be >1 for parallelism, see notes below
        )

        if result.success and result.fun < best_fun:
            best_fun = result.fun
            best_iter_count = iter_count
            best_overall = np.array(result.x).reshape(iter_count, 3)

    if best_overall is None:
        return None

    damage = calculate_damage(best_overall, wt_exp_list, design_life_values)
    return {
        "gear": gear_name,
        "best_iter": int(best_iter_count),
        "rows": best_overall.tolist(),
        "damage": damage.tolist(),
        "wt_exponents": list(wt_exp_list),
    }
