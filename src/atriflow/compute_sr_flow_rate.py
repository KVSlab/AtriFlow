from os import makedirs, path

import numpy as np
import scipy
from scipy.interpolate import splev, splrep

from atriflow.common import (
    SAVE_PATH,
    get_cases,
    get_mv_flow_rate_for_condition,
    get_optimal_values,
    get_pv_flow_rate,
    load_data,
    plot_flow_rate,
)


def create_plot_and_save_sr_flow_rate(cases, datas, Q_bosi, model, condition):
    t_end = 60 / optimal_bpm
    t_max = int(t_end * 1000)
    time = np.linspace(0, t_max / 1000, 1000)
    dt = t_max / 1000 / len(time)

    # Scale to cm/s
    velocity_scale = 100

    co_values = []
    u_pvs_peak = []
    u_mvs_peak = []
    q_pvs = []
    q_mvs = []
    u_pvs = []
    volumes = []

    for i, case in enumerate(cases):
        info = datas[i]
        area_mv = info["outlet_area"]
        volume_la = np.min(info["volume"])

        # Get MV flow rate
        Q_mv = get_mv_flow_rate_for_condition(
            time,
            area_mv,
            area_avg,
            volume_la,
            volume_avg,
            Q_bosi,
            optimal_n,
            model,
            condition,
        )

        # Add Q_wall to compute Q_pv
        Q_pv, volume_splev = get_pv_flow_rate(t_end, Q_mv, info)

        # Compute max Q_PV, max Q_MV, and CO
        compute_mean_and_sd(
            dt, co_values, Q_mv, info, velocity_scale, u_mvs_peak, u_pvs_peak, Q_pv
        )

        s_v = 1e-2
        Q_splrep = splrep(time, Q_pv, s=s_v, per=True)
        Q_splev = splev(time, Q_splrep)

        # Plot Velocity
        area_pv_total = sum([info[f"inlet{i}_area"] for i in range(4)])
        u_pv = Q_splev / area_pv_total * velocity_scale
        Q_pv = Q_splev

        save_path = SAVE_PATH.format(condition)
        if not path.isdir(save_path):
            makedirs(save_path, exist_ok=True)
        np.savetxt(
            path.join(save_path, f"flow_rate_{case}_{model}.txt"),
            np.array([time, Q_pv]).T,
        )

        # Plot curves
        volumes.append(volume_splev - volume_splev[0])
        u_pvs.append(u_pv)
        q_pvs.append(Q_pv)
        q_mvs.append(Q_mv)

    plot_flow_rate(cases, u_pvs, time)
    print()
    print(f"Mean and SD values for model: {model}")
    print("CO", np.mean(co_values), np.std(co_values))
    print("U_PV", np.mean(u_pvs_peak), np.std(u_pvs_peak))
    print("U_MV", np.mean(u_mvs_peak), np.std(u_mvs_peak))
    print()


def compute_mean_and_sd(dt, co_values, Q_splev, info, scale, u_mvs, u_pvs, Q_pv):
    sv = scipy.integrate.trapezoid(Q_splev, dx=dt)  # Stroke volume
    co = (optimal_bpm * sv) / 1000
    co_values.append(co)

    area_pv_total = sum([info[f"inlet{i}_area"] for i in range(4)])
    Q_mv_max = np.max(Q_splev)
    Q_pv_max = np.max(Q_pv)
    u_mv_peak = Q_mv_max / info["outlet_area"] * scale
    u_pv_peak = Q_pv_max / area_pv_total * scale
    u_pvs.append(u_pv_peak)
    u_mvs.append(u_mv_peak)


def main():
    global optimal_bpm, area_avg, volume_avg, optimal_n

    cases = get_cases()
    Q_ref, area_avg, volume_avg, all_model_data = load_data(cases, "sr")

    # From step_1.py
    Q_avgs, n_values, bpms, _, _ = get_optimal_values()
    condition = "sr"
    print(f"-- Storing flow rates in {SAVE_PATH.format(condition)}")

    models = ["Q-A", "Q-V"]
    for model in models:
        optimal_Q_avg = Q_avgs[model]
        optimal_n = n_values[model]
        optimal_bpm = bpms[model]

        Q_bosi = optimal_Q_avg * Q_ref

        # Create flow rate
        create_plot_and_save_sr_flow_rate(
            cases, all_model_data, Q_bosi, model, condition
        )


if __name__ == "__main__":
    main()
