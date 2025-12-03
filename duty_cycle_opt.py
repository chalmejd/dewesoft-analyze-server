import numpy as np
from math import inf
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
            # Total (front + rear)
            total_damage[i] += (
                ((front_torque + rear_torque) ** exp) * cycles
                / design_life_values[i, 0]
            )
            # Rear
            total_damage[i + num_exp] += (
                (rear_torque**exp) * cycles / design_life_values[i, 1]
            )
            # Front
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
    inputs = np.array(flat_inputs, dtype=float).reshape(iter_count, 3)

    # Constraints
    if np.any(inputs < 0):
        return inf

    if np.any((inputs[:, 0] < rear_bounds[0]) | (inputs[:, 0] > rear_bounds[1])):
        return inf

    if np.any((inputs[:, 1] < front_bounds[0]) | (inputs[:, 1] > front_bounds[1])):
        return inf

    if np.any((inputs[:, 2] < cycle_bounds[0]) | (inputs[:, 2] > cycle_bounds[1])):
        return inf

    if np.sum(inputs[:, 2]) > max_total_cycles:
        return inf

    total_damage = calculate_damage(inputs, wt_exp_list, design_life_values)

    # EXACTLY like your standalone: target ~1.0 (=> ~100% of design life)
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
    maxiter=500,
    workers=-1,
):
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
    best_fun = inf

    for iter_count in range(iter_min, iter_max + 1):
        bounds = []
        for _ in range(iter_count):
            bounds.append(rear_bounds)   # rear torque
            bounds.append(front_bounds)  # front torque
            bounds.append(cycle_bounds)  # cycles

        res = differential_evolution(
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
            strategy="best1bin",
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            updating="deferred",
            workers=workers,  # keep 1 for now; you can go to -1 later
            seed=42,          # matches your standalone DE_PARAMS_DEFAULT
        )

        fun = float(res.fun)
        if not np.isfinite(fun):
            continue

        # IMPORTANT: no res.success check (your original didn't use it)
        if fun < best_fun:
            best_fun = fun
            best_iter_count = iter_count
            best_overall = res.x.reshape(iter_count, 3)

    if best_overall is None:
        return None

    damage = calculate_damage(best_overall, wt_exp_list, design_life_values)

    return {
        "label": label,
        "best_iter": int(best_iter_count),
        "rows": best_overall.tolist(),
        "damage": damage.tolist(),        # fractions of design life (1.0 => 100%)
        "wt_exponents": wt_exp_list.tolist(),
    }
