# dewesoft-analyze-server-main/duty_cycle_opt.py
import numpy as np
from scipy.optimize import differential_evolution


def calculate_damage(inputs, wt_exp_list, design_life_values):
    """
    inputs: [rows x 3] -> [rear_torque, front_torque, cycles]
    wt_exp_list: [num_exp]
    design_life_values: [num_exp x 3] -> columns = [Total (F+R), Rear, Front]
    """
    inputs = np.asarray(inputs, dtype=float)
    wt_exp_list = np.asarray(wt_exp_list, dtype=float)
    design_life_values = np.asarray(design_life_values, dtype=float)

    num_exp = len(wt_exp_list)
    if design_life_values.shape != (num_exp, 3):
        raise ValueError(
            f"design_life_values must be shape (num_exp, 3), got {design_life_values.shape}"
        )

    total_damage = np.zeros(num_exp * 3, dtype=float)

    for rear_torque, front_torque, cycles in inputs:
        for i, exp in enumerate(wt_exp_list):
            total_damage[i] += (
                ((front_torque + rear_torque) ** exp) * cycles
                / design_life_values[i, 0]
            )
            total_damage[i + num_exp] += (
                (rear_torque**exp) * cycles / design_life_values[i, 1]
            )
            total_damage[i + 2 * num_exp] += (
                (front_torque**exp) * cycles / design_life_values[i, 2]
            )

    return total_damage


def objective_function(
    flat_inputs,
    wt_exp_list,
    design_life_values,
    iter_count,
    rear_bounds,
    front_bounds,
    cycle_bounds,
    max_total_cycles,
):
    inputs = np.array(flat_inputs).reshape(iter_count, 3)

    # Basic constraints
    if np.any(inputs < 0):
        return np.inf

    # Rear torque bounds
    if np.any((inputs[:, 0] < rear_bounds[0]) | (inputs[:, 0] > rear_bounds[1])):
        return np.inf

    # Front torque bounds
    if np.any((inputs[:, 1] < front_bounds[0]) | (inputs[:, 1] > front_bounds[1])):
        return np.inf

    # Cycle bounds
    if np.any((inputs[:, 2] < cycle_bounds[0]) | (inputs[:, 2] > cycle_bounds[1])):
        return np.inf

    # Total cycles constraint
    if np.sum(inputs[:, 2]) > max_total_cycles:
        return np.inf

    total_damage = calculate_damage(inputs, wt_exp_list, design_life_values)

    # Target is damage ~ 1.0 for each component
    return np.sqrt(np.mean((total_damage - 1.0) ** 2))


def optimize_duty_cycle(
    wt_exp_list,
    design_life_values,
    label="Scenario",
    iter_min=1,
    iter_max=10,
    rear_bounds=(0.0, 4000.0),
    front_bounds=(0.0, 5200.0),
    cycle_bounds=(0.0, 1e8),
    max_total_cycles=1e9,
    popsize=15,
    maxiter=40,
    workers=1,
):
    """
    wt_exp_list: list of exponents [e1, e2, ...]
    design_life_values: list of rows [[Total1, Rear1, Front1], [Total2, Rear2, Front2], ...]
    label: optional string just to tag the scenario
    """
    wt_exp_list = np.asarray(wt_exp_list, dtype=float)
    design_life_values = np.asarray(design_life_values, dtype=float)

    if wt_exp_list.size == 0:
        raise ValueError("At least one exponent is required.")

    if design_life_values.shape != (len(wt_exp_list), 3):
        raise ValueError(
            f"design_life_values must be shape (num_exponents, 3), got {design_life_values.shape}"
        )

    best_overall = None
    best_iter_count = None
    best_fun = np.inf

    for iter_count in range(iter_min, iter_max + 1):
        bounds = []
        for _ in range(iter_count):
            bounds.append(rear_bounds)   # rear torque
            bounds.append(front_bounds)  # front torque
            bounds.append(cycle_bounds)  # cycles

        result = differential_evolution(
            objective_function,
            bounds=bounds,
            args=(
                wt_exp_list,
                design_life_values,
                iter_count,
                rear_bounds,
                front_bounds,
                cycle_bounds,
                max_total_cycles,
            ),
            popsize=popsize,
            maxiter=maxiter,
            tol=0.01,
            updating="deferred",
            workers=workers,  # set to -1 for "all cores" once tested
        )

        if result.success and result.fun < best_fun:
            best_fun = result.fun
            best_iter_count = iter_count
            best_overall = np.array(result.x).reshape(iter_count, 3)

    if best_overall is None:
        return None

    damage = calculate_damage(best_overall, wt_exp_list, design_life_values)

    return {
        "label": label,
        "best_iter": int(best_iter_count),
        "rows": best_overall.tolist(),      # [ [rear, front, cycles], ... ]
        "damage": damage.tolist(),          # flattened list: [Total_exp1.., Rear_exp1.., Front_exp1..]
        "wt_exponents": wt_exp_list.tolist(),
    }
