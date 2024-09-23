import json
import sys
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import splev, splrep
from scipy.signal import argrelextrema

colormap = plt.cm.viridis

ROOT = "data"
INFO_PATH = path.join(ROOT, "geometry/geometry_{}/{}/model_info.json")
FLOW_EXTENSION_LENGTH_PATH = path.join(ROOT, "volumes/flow_extension_lengths_{}.json")
VOLUME_PATH = path.join(ROOT, "volumes/volumes_{}/{}/volume_{}.txt")
FLOW_RATE_PATH = path.join(ROOT, "flow_rate_waveform.csv")
SAVE_PATH = path.join(ROOT, "flow_rates/flow_rates_{}")
DATASET_URL = ""


def create_ref_sr():
    reference_dict = dict(
        ref=[],
        N=[],
        co=[],
        co_sd=[],
        u_mv_peak=[],
        u_mv_peak_sd=[],
        u_pv_peak=[],
        u_pv_peak_sd=[],
    )

    def fill(d):
        N = len(d["ref"])
        for key in d.keys():
            if len(d[key]) != N:
                d[key].append("-")

    # CARDIAC OUTPUT
    reference_dict["ref"].append(r"Chang")
    reference_dict["N"].append(73)
    reference_dict["co"].append(5.57)  # 1.2
    reference_dict["co_sd"].append(0.59)  # 1.2
    fill(reference_dict)
    reference_dict["ref"].append(r"Malayeri")
    reference_dict["N"].append(50)
    reference_dict["co"].append(5.38)  # 1.2
    reference_dict["co_sd"].append(1)  # 1.2
    fill(reference_dict)
    reference_dict["ref"].append(r"Rusinaru (Women)")
    reference_dict["N"].append(1054)
    reference_dict["co"].append(5.04)  # 1.3
    reference_dict["co_sd"].append(1.34)  # 1.3
    fill(reference_dict)
    reference_dict["ref"].append(r"Rusinaru (Men)")
    reference_dict["N"].append(1013)
    reference_dict["co"].append(5.59)  # 1.5
    reference_dict["co_sd"].append(1.5)  # 1.5
    fill(reference_dict)
    reference_dict["ref"].append(r"Yeon (Women)")
    reference_dict["N"].append(512)
    reference_dict["co"].append(4.75)  # 1.51
    reference_dict["co_sd"].append(1.00)
    fill(reference_dict)
    reference_dict["ref"].append(r"Yeon (Men)")
    reference_dict["N"].append(340)
    reference_dict["co"].append(5.87)  # 1.51
    reference_dict["co_sd"].append(1.12)
    fill(reference_dict)

    # PULMONARY VEIN S WAVE PEAK VELOCITY
    reference_dict["ref"].append(r"Gentile")
    reference_dict["N"].append(int(143 / 6 * 2))
    reference_dict["u_pv_peak"].append(52)  # 8.0
    reference_dict["u_pv_peak_sd"].append(8)  # 8.0
    fill(reference_dict)
    reference_dict["ref"].append(r"de Marchi (TTE)")
    reference_dict["N"].append(127)
    reference_dict["u_pv_peak"].append(57.57)  # 12.5
    reference_dict["u_pv_peak_sd"].append(12.20)  # 12.5
    fill(reference_dict)
    reference_dict["ref"].append(r"de Marchi (TEE)")
    reference_dict["N"].append(48)
    reference_dict["u_pv_peak"].append(63.75)  # 14.5
    reference_dict["u_pv_peak_sd"].append(16.04)  # 14.5
    fill(reference_dict)
    reference_dict["ref"].append(r"Klein")
    reference_dict["N"].append(52)
    reference_dict["u_pv_peak"].append(59.38)  # 14.5
    reference_dict["u_pv_peak_sd"].append(9.17)  # 14.5
    fill(reference_dict)

    # MITRAL VALVE E WAVE PEAK VELOCITY
    reference_dict["ref"].append(r"Sutter")
    reference_dict["N"].append(135)
    reference_dict["u_mv_peak"].append(75.74)  # 2.5
    reference_dict["u_mv_peak_sd"].append(19.60)  # 2.5
    fill(reference_dict)
    reference_dict["ref"].append(r"Watanabe")
    reference_dict["N"].append(77)
    reference_dict["u_mv_peak"].append(72.10)  # 10
    reference_dict["u_mv_peak_sd"].append(12.90)  # 10
    fill(reference_dict)
    reference_dict["ref"].append(r"Ma")
    reference_dict["N"].append(83)
    reference_dict["u_mv_peak"].append(73)  # 14
    reference_dict["u_mv_peak_sd"].append(15)  # 14
    fill(reference_dict)
    reference_dict["ref"].append(r"Dalen (Women)")
    reference_dict["N"].append(455)
    reference_dict["u_mv_peak"].append(72.69)  # 15
    reference_dict["u_mv_peak_sd"].append(15.78)  # 15
    reference_dict["u_pv_peak"].append(59.78)  # 12
    reference_dict["u_pv_peak_sd"].append(11.63)  # 12
    fill(reference_dict)
    reference_dict["ref"].append(r"Dalen (Men)")
    reference_dict["N"].append(477)
    reference_dict["u_mv_peak"].append(63.06)  # 15
    reference_dict["u_mv_peak_sd"].append(14.69)  # 15
    reference_dict["u_pv_peak"].append(57.20)  # 11
    reference_dict["u_pv_peak_sd"].append(11.63)  # 11
    fill(reference_dict)

    # Print table
    for i, r in enumerate(reference_dict["ref"]):
        reference_dict["ref"][i] += " et al."

    for key in reference_dict.keys():
        if key == "ref":
            reference_dict[key].append("Reference values")
        elif key == "N":
            reference_dict[key].append("-")
        else:
            values = reference_dict[key]
            s = 0
            N = 0
            for k, val in enumerate(values):
                if val != "-":
                    s += val * reference_dict["N"][k]
                    N += reference_dict["N"][k]
            reference_dict[key].append(s / N)
    return reference_dict


def create_ref_af():
    reference_dict = dict(ref=[], N=[], co=[], co_sd=[], u_mv_peak=[], u_mv_peak_sd=[])

    def fill(d):
        N = len(d["ref"])
        for key in d.keys():
            if len(d[key]) != N:
                d[key].append("-")

    # CARDIAC OUTPUT
    reference_dict["ref"].append(r"Clark")
    reference_dict["N"].append(16)
    reference_dict["co"].append(5.40)  # 1.51
    reference_dict["co_sd"].append(2.4)
    fill(reference_dict)
    reference_dict["ref"].append(r"Pardaens")
    reference_dict["N"].append(15)
    reference_dict["co"].append(3.44)  # 1.51
    reference_dict["co_sd"].append(1.29)
    fill(reference_dict)
    reference_dict["ref"].append(r"Hayashi")
    reference_dict["N"].append(47)
    reference_dict["co"].append(3.84)  # 1.51
    reference_dict["co_sd"].append(0.72)
    fill(reference_dict)
    reference_dict["ref"].append(r"Klavebäck")
    reference_dict["N"].append(44)
    reference_dict["co"].append(3.6)  # 1.51
    reference_dict["co_sd"].append(1.2)
    fill(reference_dict)

    # PEAK E WAVE VELOCITY
    reference_dict["ref"].append(r"Varounis")
    reference_dict["N"].append(24)
    reference_dict["u_mv_peak"].append(104)  # 2.5
    reference_dict["u_mv_peak_sd"].append(28.0)  # 2.5
    fill(reference_dict)
    reference_dict["ref"].append(r"Tsang")
    reference_dict["N"].append(80)
    reference_dict["u_mv_peak"].append(75)  # 2.5
    reference_dict["u_mv_peak_sd"].append(23.0)  # 2.5
    fill(reference_dict)
    reference_dict["ref"].append(r"Chen")
    reference_dict["N"].append(21)
    reference_dict["u_mv_peak"].append(77.81)  # 2.5
    reference_dict["u_mv_peak_sd"].append(19.0)  # 2.5
    fill(reference_dict)
    reference_dict["ref"].append(r"Walek")
    reference_dict["N"].append(70)
    reference_dict["u_mv_peak"].append(90)  # 2.5
    reference_dict["u_mv_peak_sd"].append(20.0)  # 2.5
    fill(reference_dict)
    reference_dict["ref"].append(r"Hayashi")
    reference_dict["N"].append(47)
    reference_dict["u_mv_peak"].append(96)  # 2.5
    reference_dict["u_mv_peak_sd"].append(25.0)  # 2.5
    fill(reference_dict)

    # Print table
    for i, r in enumerate(reference_dict["ref"]):
        reference_dict["ref"][i] += " et al."

    for key in reference_dict.keys():
        if key == "ref":
            reference_dict[key].append("Reference values")
        elif key == "N":
            reference_dict[key].append("-")
        else:
            values = reference_dict[key]
            s = 0
            N = 0
            for k, val in enumerate(values):
                if val != "-":
                    s += val * reference_dict["N"][k]
                    N += reference_dict["N"][k]
            reference_dict[key].append(s / N)
    return reference_dict


def create_main_ref_sr():
    # Mean and SD values computed from reference values
    reference_dict = {
        "u_mv": 69.3,
        "u_mv_sd": 15.5,
        "u_pv": 58.1,
        "u_pv_sd": 11.6,
        "co": 5.3,
        "co_sd": 1.3,
        "hr": 70.9,
        "hr_sd": 12.3,
    }
    return reference_dict


def get_mv_flow_rate(flow_rate_path):
    # Spline input MV flow rate
    t_max = 1
    s_mv = 1e-4
    time = np.linspace(0, t_max, 1000)
    data = pd.read_csv(flow_rate_path)
    q_mv_0 = data["flow_rate"]
    time_mv = data["time"]
    q_mv_0_splrep = splrep(time_mv, q_mv_0, s=s_mv, per=True)
    q_mv = splev(time, q_mv_0_splrep)

    area = np.trapz(q_mv, time)
    q_mv_normalized = q_mv / area

    return q_mv_normalized


def get_volume_without_flow_extensions(
    info, volume_path, flow_extension_values, condition, case, n_cores
):
    _, volume_simulation = np.loadtxt(volume_path.format(condition, case, case)).T
    fli, flo = flow_extension_values

    volume_simulation = volume_simulation[::n_cores]  # To ml / cm^3
    volume = np.array(volume_simulation).copy()

    # Compute flow extension at MV
    area_mv = info["outlet_area"]
    cyl_mv = area_mv * flo * np.sqrt(area_mv / np.pi)
    volume -= cyl_mv

    # Compute flow extension length at PVs
    for i in range(4):
        area_pv = info[f"inlet{i}_area"]
        cyl_pv = area_pv * fli * np.sqrt(area_pv / np.pi)
        volume -= cyl_pv

    return volume, volume_simulation


def get_cases():
    # Define cases
    cases = [
        "0003",
        "0004",
        "0005",
        "0006",
        "0007",
        "0008",
        "0009",
        "0019",
        "0020",
        "0021",
        "0023",
        "0024",
        "0025",
        "0026",
        "0027",
        "0028",
        "0029",
        "0030",
        "0031",
        "0032",
        "0033",
        "0034",
        "0035",
        "0074",
        "0076",
        "0077",
        "0078",
        "0080",
        "0081",
        "1029",
        "1030",
        "1031",
        "1032",
        "1033",
        "1035",
        "1037",
        "1038",
        "1039",
        "2022",
    ]

    print(f"Total number of cases: {len(cases)}")

    return cases


def check_data(path_to_check):
    try:
        with open(path_to_check) as f:
            json.load(f)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Files should be located within 'data' folder")
        print("Required files:")
        print("- 'geometry': Folder with info files (PV area, MV area)")
        print(
            "- 'volumes': Folder with time-dependent volume files and flow extension lengths"
        )
        print("- 'flow_rate_waveform.csv': Generic wave form data points")
        print(f"\n Files/Folders are available online here: {DATASET_URL}")
        sys.exit(1)


def load_data(cases, condition):
    check_data(FLOW_EXTENSION_LENGTH_PATH.format(condition))

    # Load data
    all_model_data = []
    with open(FLOW_EXTENSION_LENGTH_PATH.format(condition)) as f:
        flow_extension_dict = json.load(f)

    time = np.linspace(0, 1, 1000)
    for case in cases:
        n_cores = 20  # Skip every 20th line due to saved in parallel

        with open(INFO_PATH.format(condition, case)) as f:
            data = json.load(f)
        flow_extension_values = flow_extension_dict[case]
        volume, volume_simulation = get_volume_without_flow_extensions(
            data, VOLUME_PATH, flow_extension_values, condition, case, n_cores
        )
        data["volume"] = volume
        data["volume_sim"] = volume_simulation
        volume_scale = 1000  # to mL

        volume = data["volume_sim"] / volume_scale
        volume_splrep = splrep(time, volume, s=1e-2, per=True)
        volume_splev = splev(time, volume_splrep)
        data["volume_splev"] = volume_splev

        all_model_data.append(data)

    # Get averages
    Q_ref = get_mv_flow_rate(FLOW_RATE_PATH)
    area_avg = np.mean([data["outlet_area"] for data in all_model_data])
    volume_avg = np.mean([np.min(data["volume"]) for data in all_model_data])

    return Q_ref, area_avg, volume_avg, all_model_data


def remove_a_wave(Q_scaled):
    Q_mv = Q_scaled

    min_ids = argrelextrema(Q_mv, np.less)[0]
    a_start = min_ids[-1] - 5
    a_end = len(Q_mv)

    def quadratic(i):
        term = (i - a_end) / (a_start - a_end)
        return Q_mv[a_start] * term**2

    # Remove A wave
    ns = len(Q_mv[a_start:])
    i_values = np.linspace(a_start, a_end, ns)
    Q_removed = Q_mv.copy()
    Q_removed[a_start:] = quadratic(i_values)

    return Q_removed


def boost_e_wave(Q_removed, Q_sr, alpha):
    # Boost E wave
    min_ids = argrelextrema(Q_sr, np.less)[0]
    max_ids = argrelextrema(Q_sr, np.greater)[0]
    e_mid = max_ids[-2]
    e_start = min_ids[-2]
    e_end = min_ids[-1]
    e_len_1 = len(Q_sr[e_start:e_mid])
    e_len_2 = len(Q_sr[e_mid:e_end])

    def adjust_curve_up(i, M):
        f = (i - e_start) / (e_mid - e_start)
        return M * np.sin(np.pi * f / 2)

    def adjust_curve_down(i, M):
        f = (e_end - i) / (e_end - e_mid)
        return M * np.sin(np.pi * f / 2)

    i_values_1 = np.linspace(e_start, e_mid, e_len_1)
    i_values_2 = np.linspace(e_mid, e_end, e_len_2)

    main_adjuster_1 = adjust_curve_up(i_values_1, alpha)
    main_adjuster_2 = adjust_curve_down(i_values_2, alpha)

    Q_ewave = Q_removed.copy()
    Q_ewave[e_start:e_mid] += Q_ewave[e_start:e_mid] * main_adjuster_1
    Q_ewave[e_mid:e_end] += Q_ewave[e_mid:e_end] * main_adjuster_2

    return Q_ewave


def get_mv_flow_rate_for_condition(
    time,
    area_mv,
    area_avg,
    volume_la,
    volume_avg,
    Q_bosi,
    optimal_beta,
    model,
    condition,
    optimal_gamma=None,
):
    if model == "Q-A":
        Q_mv = Q_bosi * (area_mv / area_avg) ** optimal_beta
    elif model == "Q-V":
        Q_mv = Q_bosi * (volume_la / volume_avg) ** optimal_beta

    if condition == "af":
        # Remove A wave
        Q_removed = remove_a_wave(Q_mv)

        # Boost E wave
        Q_boost = boost_e_wave(Q_removed, Q_mv, optimal_gamma)
        Q_mv = Q_boost

    # Spline MV flow rate
    s_q = 1e-2
    Q_splrep = splrep(time, Q_mv, s=s_q)
    Q_mv = splev(time, Q_splrep)

    return Q_mv


def plot_flow_rate(cases, Q_mvs, time):
    width = 6
    height = 3
    fig, ax1 = plt.subplots(1, 1, figsize=(width, height))
    colors = colormap(np.linspace(0, 1, len(cases)))
    k = 0
    for i, case in enumerate(cases):
        ax1.plot(1000 * time, Q_mvs[i], label=f"{case}", color=colors[k], linewidth=5)
        k += 1

    ax1.plot(1000 * time, np.mean(Q_mvs, axis=0), "-", color="black", linewidth=5)
    ax1.set_xlabel("Time [ms]", fontsize=15)
    ax1.set_ylabel("Flow rate [mL/s]", fontsize=15)

    ax1.set_xlim(0, max(1000 * time))

    fig.subplots_adjust(
        left=0.126,
        right=0.99,
        bottom=0.165,
        top=0.99,
        wspace=None,
        hspace=0.46,
    )

    plt.show()


def get_pv_flow_rate(T, Q_mv, model_data):
    # Add Q_wall to compute Q_pv
    t_max = int(T * 1000)
    time = np.linspace(0, t_max / 1000, 1000)
    dt = T / len(time)

    volume_splev = model_data["volume_splev"]
    dV_dt = np.gradient(volume_splev, dt)
    Q_wall = dV_dt
    Q_pv = Q_wall + Q_mv

    return Q_pv, volume_splev


def get_optimal_values():
    initial_q_avg = {"Q-A": 97.51, "Q-V": 106.98}
    initial_n = {"Q-A": 0.29, "Q-V": 0.47}
    initial_bpms_sr = {"Q-A": 70.9, "Q-V": 69.78}

    initial_alpha = {"Q-A": 0.21, "Q-V": 0.21}
    initial_bpms_af = {"Q-A": 80.21, "Q-V": 80.22}

    return (
        initial_q_avg,
        initial_n,
        initial_bpms_sr,
        initial_alpha,
        initial_bpms_af,
    )
