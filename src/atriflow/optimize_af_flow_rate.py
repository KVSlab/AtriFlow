import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import minimize

from atriflow.common import (
    boost_e_wave,
    get_cases,
    get_optimal_values,
    load_data,
    remove_a_wave,
)


def compute_error(optimal_alpha, optimal_bpm, model, counter):
    print(f"\n==== Computing error for model {model} ====")
    # Calculate the values using the optimal parameters
    e_peak_af_opt, e_peak_sr_opt, co_af_opt, co_sr_opt = np.array(
        [
            calculate_co_and_peak_e_wave(
                optimal_alpha, optimal_bpm, data, model, counter
            )
            for data in all_model_data
        ]
    ).T

    # Compute the relative errors in percentage
    relative_error_co = np.mean(
        np.abs((co_sr_opt - co_af_opt) / co_sr_opt) * 100 - reference_co_percent
    )
    relative_error_e_peak = np.mean(
        np.abs((e_peak_af_opt - e_peak_sr_opt) / e_peak_sr_opt) * 100
        - reference_e_peak_percent
    )
    relative_error_bpm = np.abs(
        (optimal_bpm - bpm_sr) / bpm_sr * 100 - reference_bpm_percent
    )

    # Print relative errors formatted with two decimal places
    print("Error in CO: {:.4f}%".format(relative_error_co))
    print("Error in E peak velocity: {:.4f}%".format(relative_error_e_peak))
    print("Error in BPM: {:.4f}%".format(relative_error_bpm))

    # Calculate the combined total relative error in percentage
    total_relative_error = np.mean(
        [relative_error_co, relative_error_e_peak, relative_error_bpm]
    )
    print("Total error: {:.4f}%".format(total_relative_error))


def calculate_Q_A(area_i):
    Q_i = optimal_Q_avg * Q_ref * (area_i / area_avg) ** optimal_n
    return Q_i


def calculate_Q_V(V_i):
    Q_i = optimal_Q_avg * Q_ref * (V_i / volume_avg) ** optimal_n
    return Q_i


def get_Q_af_and_sr(gamma, scale_model, model_data):
    # Calculate Q_i based on the formula
    if scale_model == "Q-A":
        A_i = model_data["outlet_area"]
        Q_i = calculate_Q_A(A_i)
    elif scale_model == "Q-V":
        V_i = model_data["volume"]
        Q_i = calculate_Q_V(V_i)

    # Remove A wave
    Q_removed = remove_a_wave(Q_i)

    # Boost E wave
    Q_boost = boost_e_wave(Q_removed, Q_i, gamma)

    return Q_boost, Q_i


afs = []
timeaf = []
srs = []
timesr = []


def calculate_co_and_peak_e_wave(alpha, bpm, model_data, scale_model, counter):
    t_af = 60 / bpm
    t_sr = 60 / bpm_sr
    time_sr = np.linspace(0, t_sr, 1000)
    time_af = np.linspace(0, t_af, 1000)
    scale = 100  # to cm/s

    # Calculate Q_i based on the formula
    Q_af, Q_sr = get_Q_af_and_sr(alpha, scale_model, model_data)

    if abs(model_data["outlet_area"] - 680) < 1:
        counter["c"] += 1
        afs.append(Q_af)
        srs.append(Q_sr)
        timeaf.append(time_af)
        timesr.append(time_sr)

    # Compute peak E wave
    U_mv_peak_sr = max(Q_sr) / model_data["outlet_area"] * scale
    U_mv_peak_af = max(Q_af) / model_data["outlet_area"] * scale

    # Compute CO (SR)
    dt_sr = t_sr / len(time_sr)
    sv = scipy.integrate.trapezoid(Q_sr, dx=dt_sr)
    hr = bpm_sr
    co_sr = sv * hr / 1000  # Convert from ml to liters

    # Compute CO (AF)
    dt_af = t_af / len(time_af)
    sv = scipy.integrate.trapezoid(Q_af, dx=dt_af)
    hr = bpm
    co_af = sv * hr / 1000  # Convert from ml to liters

    return U_mv_peak_af, U_mv_peak_sr, co_af, co_sr


def objective_function(params):
    alpha, bpm = params
    w1 = w2 = w3 = 1.0

    e_peak_af, e_peak_sr, co_af, co_sr = np.array(
        [
            calculate_co_and_peak_e_wave(alpha, bpm, data, model, counter)
            for data in all_model_data
        ]
    ).T

    component1 = w1 * ((co_sr - co_af) / co_sr * 100 - reference_co_percent) ** 2
    component2 = (
        w2 * ((e_peak_af - e_peak_sr) / e_peak_sr * 100 - reference_e_peak_percent) ** 2
    )
    component3 = w3 * ((bpm - bpm_sr) / bpm_sr * 100 - reference_bpm_percent) ** 2

    F = np.sum(component1) + np.sum(component2) + np.sum(component3)

    return F


def plot_initial_and_converged_solution(af_values, sr_values, time_af, time_sr):
    i_start = 0
    i_mid = 25
    i_end = -1

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    colors = ["#1f77b4", "#ff7f0e"]
    labels = ["SR", "AF"]
    lines = []

    # Plot initial
    text = ["Initial configuration", "Iteration #25", "Converged solution"]
    for j, i in enumerate([i_start, i_mid, i_end]):
        ax = axs[j]
        (line_2,) = ax.plot(
            time_sr[i],
            sr_values[i],
            linestyle="dashed",
            color=colors[0],
            label=labels[0],
            linewidth=4,
        )
        (line_1,) = ax.plot(
            time_af[i],
            af_values[i],
            linestyle="-",
            color=colors[1],
            label=labels[1],
            linewidth=4,
        )

        ax.text(
            0.0108,
            437,
            text[j],
            fontsize=12,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="black",
                alpha=0.5,
            ),
        )

    lines.extend([line_2, line_1])
    for ax in axs:
        ax.grid(which="major", linestyle=":", linewidth="0.5", color="gray")
    # Create a common legend
    fig.legend(
        lines[:2],
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.98),
        fontsize=15,
    )

    for ax in axs:
        ax.set_xlabel("Time [s] ", fontsize=15)
        ax.set_xlim([-0.02, 0.82])
        ax.set_ylim([-10, 460])

    axs[0].set_ylabel("Flow rate  [mL/s]", fontsize=15)
    axs[1].axes.yaxis.set_ticklabels([])
    axs[2].axes.yaxis.set_ticklabels([])

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)

    # Show the plot
    plt.show()


def perform_optimization(model):
    # Create flow rate
    min_alpha = 0
    max_alpha = 5
    min_bpm = 60
    max_bpm = 120

    # Set initial values
    alpha_guess = 0
    bpm_guess = optimal_bpm
    initial_guess = np.array([alpha_guess, bpm_guess])

    optimal_values = []

    method = "L-BFGS-B"
    result = minimize(
        objective_function,
        initial_guess,
        method=method,
        bounds=[(min_alpha, max_alpha), (min_bpm, max_bpm)],
    )
    results = result.x  # The optimal value of N
    optimal_values.append(results)
    optimal_alpha = optimal_values[0][0]
    optimal_bpm_af = optimal_values[0][1]
    print(
        f"\nOptimal values for model {model} with method {method}: "
        + f"\nalpha={optimal_alpha:.2f} "
        + f"\nBPM={optimal_bpm_af:.2f}"
    )

    # Calculate errors:
    compute_error(optimal_alpha, optimal_bpm_af, model, counter)

    return optimal_alpha, optimal_bpm_af


if __name__ == "__main__":
    # From step_1.py
    counter = {"c": 0}
    Q_avgs, n_values, bpms, _, _ = get_optimal_values()
    reference_dict = {
        "bpm_percent": 12.57,  # Increase
        "e_peak_percent": 19.53,  # Increase
        "co_percent": 16.23,  # Decrease
    }
    reference_co_percent = reference_dict["co_percent"]
    reference_e_peak_percent = reference_dict["e_peak_percent"]
    reference_bpm_percent = reference_dict["bpm_percent"]

    condition = "af"
    cases = get_cases()

    Q_ref, area_avg, volume_avg, all_model_data = load_data(cases, condition)

    models = ["Q-A", "Q-V"]
    for model in models:
        optimal_Q_avg = Q_avgs[model]
        optimal_n = n_values[model]
        optimal_bpm = bpm_sr = bpms[model]

        alpha, bpm_af = perform_optimization(model)

    plot_initial_and_converged_solution(afs, srs, timeaf, timesr)
