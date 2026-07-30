import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import vice
from matplotlib.lines import Line2D


RUN_MODELS = True
FORCE_RERUN = False
USE_SIMPLE_WHEN_NO_MIGRATION = False
MAX_WORKERS = 10
PLOT_SINGLE_GAUSSIAN = True
BASE_DIR = Path("/Users/willmiles/Desktop/Code/VSc/ProjectVICE")
LOCAL_DIR = Path(__file__).resolve().parent
FIGURES_DIR = LOCAL_DIR / "Figures"

vice.yields.ccsne.settings["mg"] = 10.55e-4
vice.yields.ccsne.settings["fe"] = 4.73e-4
vice.yields.sneia.settings["fe"] = 13.25e-4

TGRID = np.linspace(0.0, 3.0, 3001)
TRAPZ = getattr(np, "trapezoid", np.trapz)
TMIN = 0.001
LIGHT_WEIGHT_OFFSET_GYR = 0.1
LIGHT_WEIGHT_BETA = 0.9
SOLAR_METALLICITY = 0.0137
SAMPLE_TIME_GYR = 3.0
INFALL_ONSET_TIME_GYR = 0.5
FIXED_FURTHEST_SATELLITE_OFFSET_GYR = 0.5
ABSOLUTE_QUENCH_TIME_GYR = 3.0
MAJOR_TO_MINOR_MASS_RATIO = 10.0
MERGER_COUNTS = [1,2,3,4,5,6]

OUTPUT_TABLE = FIGURES_DIR / "Changing_N_Events.csv"
OUTPUT_FIGURE = FIGURES_DIR / "Changing_N_Events.png"
OUTPUT_RIGHT_PANEL_FIGURE = FIGURES_DIR / "Changing_N_Events_2.png"

# Edit these first when you want to change the risefall parameter sweep.
# `tau1` is computed as `tau1_over_tau2 * tau2` for each sweep point.
tau1_over_tau2_values = [0.25]
tau2_values = np.linspace(0.05,0.45,12)

eta_values = [0.3]
gas_quench_after_gyr_values = [ABSOLUTE_QUENCH_TIME_GYR for _ in tau2_values]
tau_stars = [0.1]
dtd_a_values = [1.1]
ifr_masses = [10.0]
infall_metallicity_percents = [3.0]

if len(tau2_values) != len(gas_quench_after_gyr_values):
    raise ValueError(
        "tau2_values and gas_quench_after_gyr_values must have the same length "
        "when sweeping them together."
    )

tau2_quench_pairs = list(
    zip(
        sorted(float(value) for value in tau2_values),
        sorted(float(value) for value in gas_quench_after_gyr_values),
    )
)

parameter_grid = [
    (
        tau1_over_tau2,
        tau2,
        eta,
        gas_quench_after_gyr,
        tau_star,
        dtd_a,
        ifr_mass,
        zin_percent,
    )
    for tau1_over_tau2 in tau1_over_tau2_values
    for tau2, gas_quench_after_gyr in tau2_quench_pairs
    for eta in eta_values
    for tau_star in tau_stars
    for dtd_a in dtd_a_values
    for ifr_mass in ifr_masses
    for zin_percent in infall_metallicity_percents
]

if not parameter_grid:
    raise ValueError("At least one value is required in each parameter array")


def dtd(t, a=1.1, tmin=0.15):
    t = np.asarray(t, dtype=float)
    values = np.zeros_like(t, dtype=float)
    valid = t >= float(tmin)
    values[valid] = np.power(t[valid], -a)
    if np.ndim(t) == 0:
        return float(values)
    return values


def light_weight_from_age(age_gyr, beta=LIGHT_WEIGHT_BETA, offset_gyr=LIGHT_WEIGHT_OFFSET_GYR):
    age_gyr = np.asarray(age_gyr, dtype=float)
    weights = 1.5 * np.power(age_gyr + float(offset_gyr), -float(beta))
    if np.ndim(age_gyr) == 0:
        return float(weights)
    return weights


def migration_func(t, t_quench):
    if t_quench == 0:
        return 0.0
    return 6.0 * 0.5 * (np.tanh((t - t_quench) / 0.001) + 1.0)


def _risefall_shape_u(u, ratio):
    u = np.asarray(u, dtype=float)
    ratio = float(ratio)
    return (1.0 - np.exp(-u / ratio)) * np.exp(-u)


def risefall_peak_time(tau1, tau2):
    tau1 = float(tau1)
    tau2 = float(tau2)
    return tau1 * np.log1p(tau2 / tau1)


def risefall_onset_time(mean, tau1, tau2):
    return float(mean) - risefall_peak_time(tau1, tau2)


def build_risefall_ifr(mean, tau1, tau2, ifr_mass, quench_time=0.0):
    mean = float(mean)
    tau1 = float(tau1)
    tau2 = float(tau2)
    if tau1 <= 0 or tau2 <= 0:
        raise ValueError(f"tau1 and tau2 must be > 0, got tau1={tau1}, tau2={tau2}")
    ifr_mass = float(ifr_mass)
    quench_time = float(quench_time)
    peak_time = risefall_peak_time(tau1, tau2)
    time_shift = mean - peak_time
    shifted_time = TGRID - time_shift
    profile = np.zeros_like(TGRID)
    valid = shifted_time >= 0.0
    if quench_time > 0:
        valid &= TGRID < quench_time
    profile[valid] = (1.0 - np.exp(-shifted_time[valid] / tau1)) * np.exp(-shifted_time[valid] / tau2)
    integral = TRAPZ(profile, TGRID)
    if not np.isfinite(integral) or integral <= 0:
        raise ValueError("Rise-fall IFR integral must be finite and > 0")
    grid = (ifr_mass / integral) * profile

    def ifr(t):
        t_arr = np.asarray(t, dtype=float)
        idx = np.clip(
            np.rint((t_arr - TGRID[0]) / (TGRID[1] - TGRID[0])).astype(int),
            0,
            len(grid) - 1,
        )
        values = grid[idx]
        if np.ndim(t) == 0:
            return float(values)
        return values

    return ifr, grid


def many_merger_component_spacing(n_events, furthest_satellite_offset_gyr):
    n_events = int(n_events)
    furthest_satellite_offset_gyr = float(furthest_satellite_offset_gyr)
    if n_events <= 0:
        return 0.0
    return furthest_satellite_offset_gyr / float(n_events)


def many_merger_component_onsets(n_events, furthest_satellite_offset_gyr):
    n_events = int(n_events)
    total_components = n_events + 1
    furthest_satellite_offset_gyr = float(furthest_satellite_offset_gyr)
    if total_components <= 1 or np.isclose(furthest_satellite_offset_gyr, 0.0):
        return np.full(total_components, INFALL_ONSET_TIME_GYR, dtype=float)
    return INFALL_ONSET_TIME_GYR + np.linspace(0.0, furthest_satellite_offset_gyr, total_components)


def progressive_minor_merger_raw_masses(n_events, major_to_minor_ratio):
    n_events = int(n_events)
    ratio = float(major_to_minor_ratio)
    if n_events <= 0:
        return np.array([], dtype=float)
    if ratio <= 0:
        raise ValueError("Major-to-minor mass ratio must be positive.")

    growth_factor = 1.0 + 1.0 / ratio
    # Start with the main progenitor mass M, then add satellites whose masses
    # track 0.1 times the current remnant: 0.1 M, 0.11 M, 0.121 M, ...
    satellite_masses = np.power(growth_factor, np.arange(n_events, dtype=float)) / ratio
    return np.concatenate(([1.0], satellite_masses))


def many_merger_components(tau1, tau2, n_events, total_ifr_mass, furthest_satellite_offset_gyr, major_to_minor_mass_ratio):
    n_events = int(n_events)
    if n_events <= 0:
        return []

    onsets = many_merger_component_onsets(n_events, furthest_satellite_offset_gyr)
    means = onsets + risefall_peak_time(tau1, tau2)
    raw_masses = progressive_minor_merger_raw_masses(n_events, major_to_minor_mass_ratio)
    component_masses = float(total_ifr_mass) * raw_masses / np.sum(raw_masses)

    return [
        {
            "onset": float(onset),
            "mean": float(mean),
            "ifr_mass": float(component_mass),
            "sequence_index": int(i),
            "sequence_role": "main" if i == 0 else "satellite",
        }
        for i, (onset, mean, component_mass) in enumerate(zip(onsets, means, component_masses), start=0)
    ]


def build_model_name(
    tag,
    mean,
    tau1,
    tau2,
    eta,
    quench_time,
    tau_star,
    dtd_a,
    ifr_mass,
    zin_percent,
    furthest_satellite_offset_gyr=None,
    gas_quench_after_gyr=None,
    major_to_minor_mass_ratio=None,
):
    suffix = ""
    if furthest_satellite_offset_gyr is not None:
        suffix = f"_satend{float(furthest_satellite_offset_gyr):.3f}"
    if gas_quench_after_gyr is not None:
        suffix += f"_gasq{float(gas_quench_after_gyr):.3f}"
    if major_to_minor_mass_ratio is not None:
        suffix += f"_mr{float(major_to_minor_mass_ratio):.3f}"
    return (
        f"models/{tag}_mu{mean:.3f}_tau1{tau1:.3f}_tau2{tau2:.3f}_eta{eta:.2f}_qt{quench_time:.3f}"
        f"_taustar{tau_star:.3f}_dtda{dtd_a:.3f}_ifrm{ifr_mass:.3f}_zinpct{zin_percent:.3f}{suffix}"
    )


def format_progress(prefix, completed_runs, total_runs, label):
    return f"{prefix} [{completed_runs}/{total_runs}] {label}"


def tau1_from_tau2(tau2, tau1_over_tau2):
    tau2 = float(tau2)
    tau1_over_tau2 = float(tau1_over_tau2)
    if tau2 <= 0:
        raise ValueError(f"tau2 must be > 0, got tau2={tau2}")
    if tau1_over_tau2 <= 0:
        raise ValueError(f"tau1_over_tau2 must be > 0, got {tau1_over_tau2}")
    return tau1_over_tau2 * tau2


def build_job(
    mean,
    tau1,
    tau2,
    tau1_over_tau2,
    eta,
    quench_time,
    tau_star,
    dtd_a,
    ifr_mass,
    zin_percent,
    tag,
    scenario,
    component_index=0,
    n_components=1,
    n_events=0,
    sequence_index=0,
    sequence_role="",
    furthest_satellite_offset_gyr=None,
    gas_quench_after_gyr=None,
    major_to_minor_mass_ratio=None,
):
    return {
        "name": build_model_name(
            tag,
            mean,
            tau1,
            tau2,
            eta,
            quench_time,
            tau_star,
            dtd_a,
            ifr_mass,
            zin_percent,
            furthest_satellite_offset_gyr=furthest_satellite_offset_gyr,
            gas_quench_after_gyr=gas_quench_after_gyr,
            major_to_minor_mass_ratio=major_to_minor_mass_ratio,
        ),
        "mean": float(mean),
        "tau1": float(tau1),
        "tau2": float(tau2),
        "tau1_over_tau2": float(tau1_over_tau2),
        "eta": float(eta),
        "quench_time": float(quench_time),
        "tau_star": float(tau_star),
        "dtd_a": float(dtd_a),
        "ifr_mass": float(ifr_mass),
        "zin_percent": float(zin_percent),
        "scenario": scenario,
        "component_index": int(component_index),
        "n_components": int(n_components),
        "n_events": int(n_events),
        "sequence_index": int(sequence_index),
        "sequence_role": sequence_role,
        "furthest_satellite_offset_gyr": None if furthest_satellite_offset_gyr is None else float(furthest_satellite_offset_gyr),
        "gas_quench_after_gyr": None if gas_quench_after_gyr is None else float(gas_quench_after_gyr),
        "major_to_minor_mass_ratio": None if major_to_minor_mass_ratio is None else float(major_to_minor_mass_ratio),
    }


def build_single_job(tau1_over_tau2, tau2, eta, gas_quench_after_gyr, tau_star, dtd_a, ifr_mass, zin_percent):
    tau1 = tau1_from_tau2(tau2, tau1_over_tau2)
    onset_time = float(INFALL_ONSET_TIME_GYR)
    mean = onset_time + risefall_peak_time(tau1, tau2)
    quench_time = float(ABSOLUTE_QUENCH_TIME_GYR)
    return build_job(
        mean=mean,
        tau1=tau1,
        tau2=tau2,
        tau1_over_tau2=tau1_over_tau2,
        eta=eta,
        quench_time=quench_time,
        tau_star=tau_star,
        dtd_a=dtd_a,
        ifr_mass=ifr_mass,
        zin_percent=zin_percent,
        tag="risefall_fixedoffset05_manymerger_single",
        scenario="single",
        gas_quench_after_gyr=quench_time,
    )


def build_many_merger_jobs(tau1_over_tau2, tau2, eta, gas_quench_after_gyr, tau_star, dtd_a, ifr_mass, zin_percent, n_events):
    tau1 = tau1_from_tau2(tau2, tau1_over_tau2)
    components = many_merger_components(
        tau1,
        tau2,
        n_events,
        ifr_mass,
        FIXED_FURTHEST_SATELLITE_OFFSET_GYR,
        MAJOR_TO_MINOR_MASS_RATIO,
    )
    jobs = []
    for i, component in enumerate(components, start=1):
        quench_time = float(ABSOLUTE_QUENCH_TIME_GYR)
        jobs.append(
            build_job(
                mean=component["mean"],
                tau1=tau1,
                tau2=tau2,
                tau1_over_tau2=tau1_over_tau2,
                eta=eta,
                quench_time=quench_time,
                tau_star=tau_star,
                dtd_a=dtd_a,
                ifr_mass=component["ifr_mass"],
                zin_percent=zin_percent,
                tag=f"risefall_fixedoffset05_progressive10to1_nevents{n_events}_c{i}",
                scenario=f"merger_{n_events}",
                component_index=i,
                n_components=len(components),
                n_events=n_events,
                sequence_index=component["sequence_index"],
                sequence_role=component["sequence_role"],
                furthest_satellite_offset_gyr=FIXED_FURTHEST_SATELLITE_OFFSET_GYR,
                gas_quench_after_gyr=quench_time,
                major_to_minor_mass_ratio=MAJOR_TO_MINOR_MASS_RATIO,
            )
        )
    return jobs


def get_snapshot_stars(out, sample_time, tmin=0.025, zone_final=0.0):
    s = out.stars
    feh = np.asarray(s["[fe/h]"], dtype=float)
    mgh = np.asarray(s["[mg/h]"], dtype=float)
    mass = np.asarray(s["mass"], dtype=float)
    tform = np.asarray(s["formation_time"], dtype=float)

    keep = (
        np.isfinite(feh)
        & np.isfinite(mgh)
        & np.isfinite(mass)
        & np.isfinite(tform)
        & (mass > 0)
        & (tform >= tmin)
        & (tform <= sample_time)
    )

    if zone_final is not None:
        if "zone_final" in s.keys():
            z = np.asarray(s["zone_final"], dtype=float)
            keep &= np.isfinite(z) & (z == float(zone_final))
        elif "zone_origin" in s.keys():
            z = np.asarray(s["zone_origin"], dtype=float)
            keep &= np.isfinite(z) & (z == float(zone_final))

    return {
        "feh": feh[keep],
        "mgh": mgh[keep],
        "mgfe": mgh[keep] - feh[keep],
        "mass": mass[keep],
        "tform": tform[keep],
    }


def merge_stellar_data(stars_list):
    return {
        "feh": np.concatenate([stars["feh"] for stars in stars_list]),
        "mgh": np.concatenate([stars["mgh"] for stars in stars_list]),
        "mgfe": np.concatenate([stars["mgfe"] for stars in stars_list]),
        "mass": np.concatenate([stars["mass"] for stars in stars_list]),
        "tform": np.concatenate([stars["tform"] for stars in stars_list]),
    }


def stellar_timescale_from_snapshot(stars):
    tform = np.asarray(stars["tform"], dtype=float)
    mass = np.asarray(stars["mass"], dtype=float)

    keep = np.isfinite(tform) & np.isfinite(mass) & (mass > 0)
    if not np.any(keep):
        return np.nan

    tform = tform[keep]
    mass = mass[keep]
    order = np.argsort(tform)
    t_sorted = tform[order]
    mass_cum = np.cumsum(mass[order])
    total_mass = mass_cum[-1]
    if total_mass <= 0:
        return np.nan

    t20 = float(np.interp(0.2 * total_mass, mass_cum, t_sorted))
    t80 = float(np.interp(0.8 * total_mass, mass_cum, t_sorted))
    return t80 - t20


def light_weighted_abundances(stars, observation_time):
    feh = np.asarray(stars["feh"], dtype=float)
    mgh = np.asarray(stars["mgh"], dtype=float)
    mass = np.asarray(stars["mass"], dtype=float)
    tform = np.asarray(stars["tform"], dtype=float)

    age = float(observation_time) - tform
    weights = mass * light_weight_from_age(age)

    keep = (
        np.isfinite(feh)
        & np.isfinite(mgh)
        & np.isfinite(weights)
        & np.isfinite(mass)
        & np.isfinite(tform)
        & (mass > 0)
        & (weights > 0)
    )
    if not np.any(keep):
        return {
            "n_stars": 0,
            "stellar_mass": 0.0,
            "sf_timescale": np.nan,
            "feh": np.nan,
            "mgh": np.nan,
            "mgfe": np.nan,
        }

    total_weight = np.sum(weights[keep])
    feh_mean = np.sum(feh[keep] * weights[keep]) / total_weight
    mgh_mean = np.sum(mgh[keep] * weights[keep]) / total_weight

    return {
        "n_stars": int(np.count_nonzero(keep)),
        "stellar_mass": float(np.sum(mass[keep])),
        "sf_timescale": float(stellar_timescale_from_snapshot(stars)),
        "feh": float(feh_mean),
        "mgh": float(mgh_mean),
        "mgfe": float(mgh_mean - feh_mean),
    }


def run_model(name, ifr_func, eta, tau_star, dtd_a, quench_time, zin_percent):
    zin = float(zin_percent) / 100.0 * SOLAR_METALLICITY
    output_path = BASE_DIR / f"{name}.vice"
    mz = vice.multizone(name=name, n_zones=2)
    mz.n_stars = 1.0
    mz.verbose = False
    no_gas_migration = float(quench_time) == 0.0
    mz.simple = bool(USE_SIMPLE_WHEN_NO_MIGRATION and no_gas_migration)

    z0 = mz.zones[0]
    z0.mode = "ifr"
    z0.func = ifr_func
    z0.elements = ["o", "fe", "mg"]
    z0.Mg0 = 0.0
    z0.tau_star = tau_star
    z0.dt = 0.001
    z0.verbose = False
    z0.eta = float(eta)
    z0.RIa = lambda t, dtd_index=dtd_a: dtd(t, dtd_index)
    z0.recycling = 0.4
    z0.Zin = float(zin)

    z1 = mz.zones[1]
    z1.mode = "ifr"
    z1.func = lambda t: 0.0
    z1.tau_star = 1e12
    z1.elements = ["o", "fe", "mg"]
    z1.Mg0 = 0.0
    z1.dt = 0.001
    z1.verbose = False
    z1.eta = float(eta)
    z1.RIa = z0.RIa

    if not no_gas_migration:
        mz.migration.gas[0][1] = lambda t, q=quench_time: migration_func(t, q)

    if RUN_MODELS and (FORCE_RERUN or not output_path.exists()):
        mz.run(TGRID, overwrite=True)

    return vice.multioutput(str(output_path))


def is_complete_vice_output(output_path):
    output_path = Path(output_path)
    required_paths = [
        output_path / "attributes" / "name.obj",
        output_path / "migration" / "stars.obj",
        output_path / "tracers.out",
        output_path / "zone0.vice" / "history.out",
        output_path / "zone1.vice" / "history.out",
    ]
    return all(path.exists() for path in required_paths)


def remove_incomplete_vice_output(output_path):
    output_path = Path(output_path)
    if output_path.exists() and not is_complete_vice_output(output_path):
        shutil.rmtree(output_path)


def ensure_model_output(job):
    output_path = BASE_DIR / f"{job['name']}.vice"
    existed = output_path.exists()
    valid_output = existed and is_complete_vice_output(output_path)
    if valid_output and not FORCE_RERUN:
        return {"used_cache": True, "job": job}
    if existed and not valid_output:
        remove_incomplete_vice_output(output_path)

    ifr_func, _ = build_risefall_ifr(
        job["mean"],
        job["tau1"],
        job["tau2"],
        job["ifr_mass"],
        quench_time=job["quench_time"],
    )
    run_model(
        job["name"],
        ifr_func,
        job["eta"],
        job["tau_star"],
        job["dtd_a"],
        job["quench_time"],
        job["zin_percent"],
    )
    return {"used_cache": False, "job": job}


def load_output_and_ifr(job):
    _, ifr_grid = build_risefall_ifr(
        job["mean"],
        job["tau1"],
        job["tau2"],
        job["ifr_mass"],
        quench_time=job["quench_time"],
    )
    output_path = BASE_DIR / f"{job['name']}.vice"
    if not is_complete_vice_output(output_path):
        raise FileNotFoundError(f"Incomplete VICE output: {output_path}")
    out = vice.multioutput(str(output_path))
    return out, ifr_grid


def run_single_gaussian(job):
    out, ifr_grid = load_output_and_ifr(job)
    history = out.zones["zone0"].history
    time = np.asarray(history["time"], dtype=float)
    sfr = np.asarray(history["sfr"], dtype=float)
    stars = get_snapshot_stars(out, SAMPLE_TIME_GYR, tmin=TMIN, zone_final=0.0)
    summary = light_weighted_abundances(stars, SAMPLE_TIME_GYR)
    return {"ifr": ifr_grid, "time": time, "sfr": sfr, "summary": summary}


def run_many_merger(component_jobs):
    n_events = component_jobs[0]["n_events"]
    merged_ifr = np.zeros_like(TGRID)
    merged_sfr = None
    merged_time = None
    stars_list = []
    component_histories = []

    for job in component_jobs:
        out, ifr_grid = load_output_and_ifr(job)
        history = out.zones["zone0"].history
        time = np.asarray(history["time"], dtype=float)
        sfr = np.asarray(history["sfr"], dtype=float)
        stars = get_snapshot_stars(out, SAMPLE_TIME_GYR, tmin=TMIN, zone_final=0.0)
        merged_ifr += ifr_grid
        merged_time = time
        merged_sfr = sfr if merged_sfr is None else merged_sfr + sfr
        stars_list.append(stars)
        component_histories.append(
            {
                "component_index": int(job["component_index"]),
                "sequence_index": int(job["sequence_index"]),
                "sequence_role": job["sequence_role"],
                "ifr_time": TGRID.copy(),
                "ifr": ifr_grid.copy(),
                "time": time.copy(),
                "sfr": sfr.copy(),
            }
        )

    merged_stars = merge_stellar_data(stars_list)
    summary = light_weighted_abundances(merged_stars, SAMPLE_TIME_GYR)
    return {
        "means": np.array([job["mean"] for job in component_jobs], dtype=float),
        "component_spacing_gyr": many_merger_component_spacing(n_events, FIXED_FURTHEST_SATELLITE_OFFSET_GYR),
        "base_merger_separation_gyr": float(FIXED_FURTHEST_SATELLITE_OFFSET_GYR),
        "component_histories": component_histories,
        "ifr": merged_ifr,
        "time": merged_time,
        "sfr": merged_sfr,
        "summary": summary,
    }


def run_all_jobs_parallel(jobs):
    if not jobs:
        return

    completed = 0
    total = len(jobs)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_job = {executor.submit(ensure_model_output, job): job for job in jobs}
        for future in as_completed(future_to_job):
            result = future.result()
            job = result["job"]
            completed += 1
            status = "cached" if result["used_cache"] else "ran"
            if job["scenario"] == "single":
                label = (
                    f"{status} single: tau1={job['tau1']:.3f}, tau2={job['tau2']:.3f}, mean={job['mean']:.3f}, "
                    f"ifr_mass={job['ifr_mass']:.3f}"
                )
            else:
                label = (
                    f"{status} {job['scenario']} component {job['component_index']}/{job['n_components']}: "
                    f"tau1={job['tau1']:.3f}, tau2={job['tau2']:.3f}, mean={job['mean']:.3f}, "
                    f"ifr_mass={job['ifr_mass']:.3f}, offset={job['furthest_satellite_offset_gyr']:.3f}"
                )
            print(format_progress("Completed VICE job", completed, total, label))


def build_scenario_styles():
    scenario_colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(MERGER_COUNTS) + int(PLOT_SINGLE_GAUSSIAN)))
    scenario_styles = {}
    color_index = 0
    if PLOT_SINGLE_GAUSSIAN:
        scenario_styles["single"] = {
            "label": "Single Rise-Fall",
            "line_style": "-",
            "line_width": 2.0,
            "marker": "o",
            "color": scenario_colors[color_index],
        }
        color_index += 1

    for i, n_events in enumerate(MERGER_COUNTS):
        scenario_styles[f"merger_{n_events}"] = {
            "label": f"N={n_events}",
            "line_style": "-",
            "line_width": 1.8,
            "marker": "o",
            "color": scenario_colors[color_index],
        }
        color_index += 1

    return scenario_styles


def apply_publication_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 13,
            "axes.labelsize": 13,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.linewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 4,
            "ytick.minor.size": 4,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.minor.width": 0.9,
            "ytick.minor.width": 0.9,
        }
    )


def style_axis(ax):
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=6, width=1.1)
    ax.tick_params(which="minor", length=4, width=0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def format_gas_quench_summary():
    return rf"Gas quench at $t = {ABSOLUTE_QUENCH_TIME_GYR:.2f}$ Gyr"


def plot_timescale_mgfe_panel(ax, rows, scenario_styles):
    multiple_quench_values = len(set(float(row["gas_quench_after_gyr"]) for row in rows)) > 1
    quench_markers = ["o", "s", "^", "D", "P", "X", "v"]
    unique_quench_values = sorted(set(float(row["gas_quench_after_gyr"]) for row in rows))
    quench_to_marker = {
        quench: quench_markers[i % len(quench_markers)]
        for i, quench in enumerate(unique_quench_values)
    }

    for scenario, style in scenario_styles.items():
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        ordered_rows = sorted(
            scenario_rows,
            key=lambda row: (
                float(row["tau2_gyr"]),
                float(row["gas_quench_after_gyr"]),
            ),
        )
        all_timescales = np.array([row["sf_timescale_gyr"] for row in ordered_rows], dtype=float)
        all_mgfe = np.array([row["sampled_mgfe"] for row in ordered_rows], dtype=float)
        for quench_value in unique_quench_values:
            group_rows = [row for row in scenario_rows if np.isclose(float(row["gas_quench_after_gyr"]), quench_value)]
            if not group_rows:
                continue
            timescales = np.array([row["sf_timescale_gyr"] for row in group_rows], dtype=float)
            mgfe = np.array([row["sampled_mgfe"] for row in group_rows], dtype=float)
            label = style["label"]
            if multiple_quench_values:
                label = rf"{label}, $t_{{\rm quench}}={quench_value:.2f}$"
            ax.scatter(
                mgfe,
                timescales,
                color=style["color"],
                s=78,
                edgecolors="black",
                linewidths=0.6,
                alpha=0.95,
                marker=quench_to_marker[quench_value],
                label=label,
            )

        ax.plot(
            all_mgfe,
            all_timescales,
            color=style["color"],
            lw=1.4,
            ls=style["line_style"],
            alpha=0.9,
        )


def plot_results(rows, history_records, output_path):
    apply_publication_style()
    scenario_styles = build_scenario_styles()

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(17.4, 6.1),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]},
    )
    fig.subplots_adjust(wspace=0.26)

    tau_pairs = sorted({(float(record["tau1_gyr"]), float(record["tau2_gyr"])) for record in history_records})
    tau_pair_to_alpha = {
        tau_pair: alpha
        for tau_pair, alpha in zip(
            tau_pairs,
            np.linspace(0.95, 0.35, max(len(tau_pairs), 1)),
        )
    }

    for scenario, style in scenario_styles.items():
        scenario_records = [record for record in history_records if record["scenario"] == scenario]
        for record in scenario_records:
            alpha = tau_pair_to_alpha[(float(record["tau1_gyr"]), float(record["tau2_gyr"]))]
            axs[0].plot(
                record["ifr_time"],
                record["ifr"],
                color=style["color"],
                lw=style["line_width"],
                ls=style["line_style"],
                alpha=alpha,
            )
            axs[1].plot(
                record["time"],
                record["sfr"],
                color=style["color"],
                lw=style["line_width"],
                ls=style["line_style"],
                alpha=alpha,
            )

    plot_timescale_mgfe_panel(axs[2], rows, scenario_styles)

    scenario_handles = [
        Line2D([0], [0], color=style["color"], lw=style["line_width"], ls=style["line_style"], marker=style["marker"], markersize=8, label=style["label"])
        for style in scenario_styles.values()
    ]

    axs[0].set_title("Infall Histories")
    axs[0].set_xlabel("Time [Gyr]")
    axs[0].set_ylabel("IFR")
    axs[0].legend(handles=scenario_handles, frameon=False, loc="upper right", title="Scenario", title_fontsize=10, handlelength=2.6)

    axs[1].set_title("SFR Histories")
    axs[1].set_xlabel("Time [Gyr]")
    axs[1].set_ylabel("SFR")
    axs[1].legend(handles=scenario_handles, frameon=False, loc="upper right", title="Scenario", title_fontsize=10, handlelength=2.6)

    axs[2].set_title(r"[Mg/Fe] vs Star Formation Timescale")
    axs[2].set_xlabel(r"Light-weighted [Mg/Fe] at $t = %.2f$ Gyr" % SAMPLE_TIME_GYR)
    axs[2].set_ylabel(r"Star Formation Timescale, $t_{80} - t_{20}$ [Gyr]")
    axs[2].legend(handles=scenario_handles, frameon=False, loc="lower left", title="Scenario", title_fontsize=10)

    fig.text(
        0.985,
        0.975,
        "\n".join(
            [
                rf"$\Delta t_{{\rm last}} = {FIXED_FURTHEST_SATELLITE_OFFSET_GYR:.2f}$ Gyr",
                rf"MR $= {MAJOR_TO_MINOR_MASS_RATIO:.1f}:1$",
                format_gas_quench_summary(),
                r"Darker lines $\rightarrow$ smaller $\tau_2$",
            ]
        ),
        ha="right",
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.95},
    )

    for ax in axs:
        style_axis(ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_right_panel_only(rows, output_path):
    apply_publication_style()
    scenario_styles = build_scenario_styles()
    fig, ax = plt.subplots(figsize=(8.8, 6.8))

    plot_timescale_mgfe_panel(ax, rows, scenario_styles)

    scenario_handles = [
        Line2D([0], [0], color=style["color"], lw=style["line_width"], ls=style["line_style"], marker=style["marker"], markersize=8, label=style["label"])
        for style in scenario_styles.values()
    ]

    ax.set_title(r"[Mg/Fe] vs Star Formation Timescale")
    ax.set_xlabel(r"Light-weighted [Mg/Fe] at $t = %.2f$ Gyr" % SAMPLE_TIME_GYR)
    ax.set_ylabel(r"Star Formation Timescale, $t_{80} - t_{20}$ [Gyr]")
    ax.legend(handles=scenario_handles, frameon=False, loc="lower left", title="Scenario", title_fontsize=10)
    style_axis(ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_table(rows, output_path):
    if not rows:
        raise ValueError("No results were produced.")
    fieldnames = list(rows[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    start_time = time.perf_counter()
    rows = []
    history_records = []
    job_groups = []
    all_jobs = []

    for tau1_over_tau2, tau2, eta, gas_quench_after_gyr, tau_star, dtd_a, ifr_mass, zin_percent in parameter_grid:
        tau1 = tau1_from_tau2(tau2, tau1_over_tau2)
        single_job = build_single_job(tau1_over_tau2, tau2, eta, gas_quench_after_gyr, tau_star, dtd_a, ifr_mass, zin_percent)
        merger_jobs_by_count = {
            n_components: build_many_merger_jobs(
                tau1_over_tau2,
                tau2,
                eta,
                gas_quench_after_gyr,
                tau_star,
                dtd_a,
                ifr_mass,
                zin_percent,
                n_components,
            )
            for n_components in MERGER_COUNTS
        }
        job_groups.append((tau1_over_tau2, tau1, tau2, gas_quench_after_gyr, ifr_mass, single_job, merger_jobs_by_count))
        all_jobs.append(single_job)
        for jobs in merger_jobs_by_count.values():
            all_jobs.extend(jobs)

    if RUN_MODELS:
        run_all_jobs_parallel(all_jobs)

    for tau1_over_tau2, tau1, tau2, gas_quench_after_gyr, ifr_mass, single_job, merger_jobs_by_count in job_groups:
        single_result = run_single_gaussian(single_job)
        history_records.append(
            {
                "scenario": "single",
                "tau1_over_tau2": float(tau1_over_tau2),
                "tau1_gyr": float(tau1),
                "tau2_gyr": float(tau2),
                "gas_quench_after_gyr": float(gas_quench_after_gyr),
                "ifr_time": TGRID.copy(),
                "time": single_result["time"].copy(),
                "ifr": single_result["ifr"].copy(),
                "sfr": single_result["sfr"].copy(),
            }
        )
        rows.append(
            {
                "scenario": "single",
                "n_merger_events": 0,
                "n_components": 1,
                "tau1_over_tau2": float(single_job["tau1_over_tau2"]),
                "tau1_gyr": float(single_job["tau1"]),
                "tau2_gyr": float(single_job["tau2"]),
                "gas_quench_after_gyr": float(single_job["gas_quench_after_gyr"]),
                "component_spacing_gyr": 0.0,
                "base_merger_separation_gyr": 0.0,
                "ifr_mass": float(ifr_mass),
                "sample_time_gyr": float(SAMPLE_TIME_GYR),
                "ifrm_integral": float(TRAPZ(single_result["ifr"], TGRID)),
                "sf_timescale_gyr": float(single_result["summary"]["sf_timescale"]),
                "sampled_feh": float(single_result["summary"]["feh"]),
                "sampled_mgh": float(single_result["summary"]["mgh"]),
                "sampled_mgfe": float(single_result["summary"]["mgfe"]),
            }
        )

        for n_events in MERGER_COUNTS:
            merger_result = run_many_merger(merger_jobs_by_count[n_events])
            for component_history in merger_result["component_histories"]:
                history_records.append(
                    {
                        "scenario": f"merger_{n_events}",
                        "tau1_over_tau2": float(tau1_over_tau2),
                        "tau1_gyr": float(tau1),
                        "tau2_gyr": float(tau2),
                        "gas_quench_after_gyr": float(gas_quench_after_gyr),
                        "component_index": component_history["component_index"],
                        "ifr_time": component_history["ifr_time"].copy(),
                        "time": component_history["time"].copy(),
                        "ifr": component_history["ifr"].copy(),
                        "sfr": component_history["sfr"].copy(),
                    }
                )
            rows.append(
                {
                    "scenario": f"merger_{n_events}",
                    "n_merger_events": int(n_events),
                    "n_components": int(len(merger_jobs_by_count[n_events])),
                    "tau1_over_tau2": float(merger_jobs_by_count[n_events][0]["tau1_over_tau2"]),
                    "tau1_gyr": float(merger_jobs_by_count[n_events][0]["tau1"]),
                    "tau2_gyr": float(merger_jobs_by_count[n_events][0]["tau2"]),
                    "gas_quench_after_gyr": float(merger_jobs_by_count[n_events][0]["gas_quench_after_gyr"]),
                    "component_spacing_gyr": float(merger_result["component_spacing_gyr"]),
                    "base_merger_separation_gyr": float(merger_result["base_merger_separation_gyr"]),
                    "ifr_mass": float(ifr_mass),
                    "sample_time_gyr": float(SAMPLE_TIME_GYR),
                    "ifrm_integral": float(TRAPZ(merger_result["ifr"], TGRID)),
                    "sf_timescale_gyr": float(merger_result["summary"]["sf_timescale"]),
                    "sampled_feh": float(merger_result["summary"]["feh"]),
                    "sampled_mgh": float(merger_result["summary"]["mgh"]),
                    "sampled_mgfe": float(merger_result["summary"]["mgfe"]),
                }
            )

    write_table(rows, OUTPUT_TABLE)
    plot_results(rows, history_records, OUTPUT_FIGURE)
    plot_right_panel_only(rows, OUTPUT_RIGHT_PANEL_FIGURE)
    print(f"Saved table to {OUTPUT_TABLE}")
    print(f"Saved figure to {OUTPUT_FIGURE}")
    print(f"Saved right-panel figure to {OUTPUT_RIGHT_PANEL_FIGURE}")
    elapsed_seconds = time.perf_counter() - start_time
    print(f"Total runtime: {elapsed_seconds:.1f} s ({elapsed_seconds / 60.0:.2f} min)")


if __name__ == "__main__":
    main()
