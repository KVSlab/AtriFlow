import numpy as np
import scipy
from scipy.optimize import minimize

from common import (
    create_main_ref_sr,
    get_pv_flow_rate,
    get_cases,
    load_data,
)


def compute_error(optimal_Q_avg, optimal_n, optimal_bpm):
    # Calculate the values using the optimal parameters
    co_calculated_opt, pv_calculated_opt, mv_calculated_opt = np.array(
        [
            calculate_co_mv_and_pv_peak(
                optimal_Q_avg,
                optimal_n,
                optimal_bpm,
                data,
                model,
                Q_ref,
                area_avg,
                volume_avg,
            )
            for data in all_model_data
        ]
    ).T
    # Compute the errors
    print(f"\n==== Computing errors for model: {model} ====")

    relative_error_co = 100 * np.mean(
        np.abs(co_calculated_opt - reference_co) / reference_co
    )
    relative_error_co_sd = (
            100 * np.abs(np.std(co_calculated_opt) - reference_co_sd) / reference_co_sd
    )

    relative_error_mv = 100 * np.mean(
        np.abs(mv_calculated_opt - reference_mv_velocity) / reference_mv_velocity
    )
    relative_error_mv_sd = (
            100 * np.abs(np.std(mv_calculated_opt) - reference_mv_velocity_sd) / reference_mv_velocity_sd
    )

    relative_error_pv = 100 * np.mean(
        np.abs(pv_calculated_opt - reference_pv_velocity) / reference_pv_velocity
    )
    relative_error_pv_sd = (
            100
            * np.abs(np.std(pv_calculated_opt) - reference_pv_velocity_sd)
            / reference_pv_velocity_sd
    )

    relative_error_bpm = 100 * np.abs(optimal_bpm - reference_bpm) / reference_bpm

    # Print relative errors formatted with two decimal places
    print(
        "Error in CO (mean, SD): ({:.2f}%, {:.2f}%)".format(
            relative_error_co, relative_error_co_sd
        )
    )
    print(
        "Error in MV (mean, SD): ({:.2f}%, {:.2f}%)".format(
            relative_error_mv, relative_error_mv_sd
        )
    )
    print(
        "Error in PV (mean, SD): ({:.2f}%, {:.2f}%)".format(
            relative_error_pv, relative_error_pv_sd
        )
    )
    print("Error in BPM (mean): ({:.2f}%)".format(relative_error_bpm))

    # Calculate the combined total relative error in percentage
    total_relative_error = np.mean(
        [relative_error_co, relative_error_pv, relative_error_mv, relative_error_bpm]
    )
    total_relative_error_sd = np.mean(
        [relative_error_co_sd, relative_error_pv_sd, relative_error_mv_sd]
    )
    print("Average Mean error: {:.2f}%".format(total_relative_error))
    print("Average SD error: {:.2f}%".format(total_relative_error_sd))

    combined_error = np.mean([total_relative_error_sd, total_relative_error])
    print("Total error: {:.2f}%".format(combined_error))


def calculate_Q_A(Q_avg, n, area_i, Q_ref, area_avg):
    Q_i = Q_avg * Q_ref * (area_i / area_avg) ** n
    return Q_i


def calculate_Q_V(Q_avg, n, volume_i, Q_ref, volume_avg):
    Q_i = Q_avg * Q_ref * (volume_i / volume_avg) ** n
    return Q_i


def calculate_Q_umax(area_i, max_u):
    Q_i = max_u * area_i
    return Q_i


def calculate_co_mv_and_pv_peak(
        Q_avg, n, bpm, model_data, model, Q_ref, area_avg, volume_avg, max_u=None, max_pv=None
):
    t_max = 60 / bpm
    scale_ms_to_cms = 100  # to cm/s
    area_i = model_data["outlet_area"]
    volume_i = np.min(model_data["volume"])

    # Calculate Q_mv based on the formula
    Q_mv = None
    if model == "Q-A":
        Q_mv = calculate_Q_A(Q_avg, n, area_i, Q_ref, area_avg)
    elif model == "Q-V":
        Q_mv = calculate_Q_V(Q_avg, n, volume_i, Q_ref, volume_avg)
    elif model == "MaxMV":
        Q_mv = calculate_Q_umax(area_i, max_u)
    elif model == "MaxPV":
        Q_mv = calculate_Q_umax(area_i, max_u)
        # Calculate Q_pv based on volume
        Q_pv, volume_splev = get_pv_flow_rate(t_max, Q_mv, model_data)
        area_pv_total = sum([model_data[f"inlet{i}_area"] for i in range(4)])

        Q_pv = Q_pv / np.max(Q_pv) * max_pv * area_pv_total
        Q_wall = np.gradient(volume_splev, t_max / 1000)
        Q_mv = Q_pv - Q_wall

    if model != "MaxPV":
        # Calculate Q_pv based on volume
        Q_pv, _ = get_pv_flow_rate(t_max, Q_mv, model_data)

    # Step 3: Compute Cardiac Output (CO)
    dt = t_max / len(Q_mv)
    sv = scipy.integrate.trapezoid(Q_mv, dx=dt)  # Average Q
    co = (sv * bpm) / 1000  # Convert from ml to liters

    # Compute MV and PV peak
    area_pv_total = np.sum([model_data[f"inlet{i}_area"] for i in range(4)])  # mm^2
    u_pv_peak = np.max(Q_pv) / area_pv_total * scale_ms_to_cms

    u_mv_peak = np.max(Q_mv) / model_data["outlet_area"] * scale_ms_to_cms

    return co, u_pv_peak, u_mv_peak


def objective_function(params):
    alpha, beta, bpm = params
    w1 = w2 = w3 = w4 = 1.0
    co_calculated, pv_calculated, mv_calculated = np.array(
        [
            calculate_co_mv_and_pv_peak(
                alpha, beta, bpm, data, model, Q_ref, area_avg, volume_avg
            )
            for data in all_model_data
        ]
    ).T

    # Standard deviations
    co_calculated_sd = np.std(co_calculated)
    pv_calculated_sd = np.std(pv_calculated)
    mv_calculated_sd = np.std(mv_calculated)

    # Calculate the components of your objective function element-wise
    component1 = (
            w1 * (np.abs(co_calculated - reference_co) / reference_co)
            + w1 * np.abs(co_calculated_sd - reference_co_sd) / reference_co_sd
    )
    component2 = (
            w2 * (np.abs(pv_calculated - reference_pv_velocity) / reference_pv_velocity)
            + w2 * np.abs(pv_calculated_sd - reference_pv_velocity_sd) / reference_pv_velocity_sd
    )
    component3 = (
            w3 * (np.abs(mv_calculated - reference_mv_velocity) / reference_mv_velocity)
            + w3 * np.abs(mv_calculated_sd - reference_mv_velocity_sd) / reference_mv_velocity_sd
    )
    component4 = w4 * np.abs(bpm - reference_bpm) / reference_bpm

    # Calculate the total objective function as the sum of components
    F = (
            np.sum(component1)
            + np.sum(component2)
            + np.sum(component3)
            + np.sum(component4)
    )

    return F


def perform_optimization(model):
    # Initial guess for alpha, beta, and BPM
    exponent_guess = 0  # n, [-]
    bpm_guess = reference_bpm  # Heart rate, [BPM]
    Q_avg_guess = 74.75  # Ref CO / Ref BPM * 1000
    initial_guess = np.array([Q_avg_guess, exponent_guess, bpm_guess])

    # Model for flow rate
    optimal_values = []

    # Run the optimization
    min_Q_avg = 50
    max_Q_avg = 150

    min_n = 0
    max_n = 3

    # Based on mean +/- SD
    min_bpm = 40
    max_bpm = 100

    method = "L-BFGS-B"
    result = minimize(
        objective_function,
        initial_guess,
        method=method,
        bounds=[(min_Q_avg, max_Q_avg), (min_n, max_n), (min_bpm, max_bpm)],
    )

    results = result.x  # The optimal values
    optimal_values.append(results)
    optimal_Q_avg = optimal_values[0][0]
    optimal_n = optimal_values[0][1]
    optimal_bpm = optimal_values[0][2]

    print(
        f"\nOptimal values for model {model} with method {method}: "
        + f"\nQ_avg={optimal_Q_avg:.2f} "
        + f"\nn={optimal_n:.2f} "
        + f"\nHR_SR={optimal_bpm:.2f}"
    )
    # Calculate errors:
    compute_error(optimal_Q_avg, optimal_n, optimal_bpm)


if __name__ == "__main__":
    # Get references
    reference_dict = create_main_ref_sr()
    reference_co = reference_dict["co"]
    reference_co_sd = reference_dict["co_sd"]
    reference_pv_velocity = reference_dict["u_pv"]
    reference_pv_velocity_sd = reference_dict["u_pv_sd"]
    reference_mv_velocity = reference_dict["u_mv"]
    reference_mv_velocity_sd = reference_dict["u_mv_sd"]
    reference_bpm = reference_dict["hr"]
    reference_bpm_sd = reference_dict["hr_sd"]

    # Set condition and load input data
    condition = "sr"
    cases = get_cases()
    Q_ref, area_avg, volume_avg, all_model_data = load_data(cases, condition)

    for model in ["Q-A", "Q-V"]:
        perform_optimization(model)
