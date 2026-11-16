#!/usr/bin/env python3
"""Reproduce every figure and number in the evaluation of the backscatter localization paper.

Usage: python3 reproduce.py [folder holding abs/ and noAbs/; default: data/ next to this script]
Writes figures to figures/, numbers to results/numbers.txt and .json, and all estimates to results/estimates.csv.
"""
import json
import multiprocessing as mp
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # parallelism is per tag position
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------- Configuration, as stated in the paper -----------------------------
FS = 50e6  # receiver sampling rate (Hz)
NUM_ANTENNAS = 4
NUM_TONES = 21  # transmitted carriers, spanning BANDWIDTH with equal spacing
BANDWIDTH = 30e6  # Hz
TAG_SHIFT = -300e3  # backscattered sideband used (Hz)
FILTER_WIDTH = 5e3  # band-pass width around each tone, i.e. +/-2.5 kHz (Hz)
SEGMENT_SAMPLES = 1000  # samples per estimate
NUM_ESTIMATES = 100  # consecutive segments per capture
NUM_SIGNALS = 1  # signal eigenvectors kept; the rest form the noise subspace
AOA_GRID_DEG = np.arange(0.0, 181.0, 1.0)
TOF_STEP = 0.1e-9  # s
TOF_POINTS = 350  # search 0-34.9 ns
ANTENNA_SPACING_WAVELENGTHS = 0.5
CARRIER_SUBSETS = {  # number of waves: tone indices used for estimation
    2: [0, 20],
    3: [0, 10, 20],
    5: [0, 5, 10, 15, 20],
    11: list(range(0, 21, 2)),
    21: list(range(21)),
}
TOF_POINTS_SENSITIVITY = [330, 350, 370, 600]  # also reported, to show the effect of the search range
FIGURE_DPI = 600  # MDPI asks for PNG, JPEG or TIFF, preferably at least 600 dpi
# --------------------------------------------------------------------------------------------------

LIGHT_SPEED = 299792458.0
ENVIRONMENTS = {"abs": "with absorber", "noAbs": "without absorber"}
DISTANCES_CM = [50, 100, 150, 200, 250, 300]
ANGLES_DEG = [30, 60, 90, 120, 150]
HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data"
COLORS = {2: "#2a6fb0", 3: "#c8641a", 5: "#1f9e89", 11: "#8e4bb0", 21: "#7f8c1e"}
LINESTYLES = {2: "-", 3: "--", 5: "-.", 11: ":", 21: "-"}


def load_capture(source, env, position, num_tones=NUM_TONES):
    """Return the capture as a complex array of shape (samples, antennas)."""
    relative = f"{env}/{num_tones}waves/{position}/samples.dat"
    if str(source).endswith(".zip"):
        with zipfile.ZipFile(source) as archive:
            raw = archive.read(f"yud-yamaguchi/data/{relative}")
    else:
        raw = (Path(source) / relative).read_bytes()
    offset = 0
    channels = []
    for _ in range(NUM_ANTENNAS):
        count = int(np.frombuffer(raw, "<i8", 1, offset)[0])
        offset += 8
        values = np.frombuffer(raw, "<f4", 2 * count, offset)
        offset += 8 * count
        channels.append(values[0::2].astype(np.float64) + 1j * values[1::2])
    return np.stack(channels, 1)


def extract_tones(capture, tone_freqs):
    """FFT band-pass each tone and downconvert it; returns (samples, tones * antennas)."""
    n = len(capture)
    spectrum = np.fft.fftshift(np.fft.fft(capture, axis=0), axes=0)
    freqs = FS / n * np.arange(-n / 2, n / 2)
    t = np.arange(n) / FS
    out = []
    for tone in tone_freqs:
        keep = np.abs(freqs - tone) <= FILTER_WIDTH / 2
        band = np.fft.ifft(np.fft.ifftshift(spectrum * keep[:, None], axes=0), axis=0)
        out.append(band * np.exp(-2j * np.pi * tone * t)[:, None])
    return np.concatenate(out, 1).astype(np.complex64).astype(np.complex128)


def tone_frequencies():
    return -BANDWIDTH / 2 + np.arange(NUM_TONES) * BANDWIDTH / (NUM_TONES - 1)


def wired_reference(source, env):
    wired = extract_tones(load_capture(source, env, "wired"), tone_frequencies())
    return np.sum(wired / np.abs(wired), 0)


def music_estimates(calibrated, subset):
    """Joint AoA-ToF MUSIC. Returns {tof_points: array of (aoa_deg, distance_m)} for each search range."""
    columns = np.concatenate([np.arange(NUM_ANTENNAS) + NUM_ANTENNAS * s for s in subset])
    x = calibrated[:, columns].T
    num_tones = len(subset)
    spacing = (subset[1] - subset[0]) * BANDWIDTH / (NUM_TONES - 1)
    max_points = max(TOF_POINTS, *TOF_POINTS_SENSITIVITY)
    tof_grid = np.arange(max_points) * TOF_STEP
    phase = -2j * np.pi * ANTENNA_SPACING_WAVELENGTHS * np.cos(np.deg2rad(AOA_GRID_DEG))
    aoa_steering = np.exp(phase)[None, :] ** np.arange(NUM_ANTENNAS)[:, None]
    tof_steering = np.exp(-2j * np.pi * spacing * tof_grid)[None, :] ** np.arange(num_tones)[:, None]
    results = {points: [] for points in {TOF_POINTS, *TOF_POINTS_SENSITIVITY}}
    for segment in range(NUM_ESTIMATES):
        snapshot = x[:, segment * SEGMENT_SAMPLES : (segment + 1) * SEGMENT_SAMPLES]
        eigenvalues, eigenvectors = np.linalg.eigh(snapshot @ snapshot.conj().T / SEGMENT_SAMPLES)
        order = np.argsort(eigenvalues)[::-1]
        projection = np.zeros((len(AOA_GRID_DEG), max_points))
        for vector in eigenvectors[:, order[:NUM_SIGNALS]].T:
            matrix = vector.reshape(num_tones, NUM_ANTENNAS)
            projection += np.abs(aoa_steering.T @ matrix.conj().T @ tof_steering) ** 2
        spectrum = 1 / np.abs(num_tones * NUM_ANTENNAS - projection)
        for points, found in results.items():
            i, j = np.unravel_index(np.argmax(spectrum[:, :points]), (len(AOA_GRID_DEG), points))
            found.append((AOA_GRID_DEG[i], tof_grid[j] * LIGHT_SPEED / 2))
    return {points: np.array(found) for points, found in results.items()}


def process_position(job):
    source, env, reference, distance_cm, angle = job
    capture = load_capture(source, env, f"{distance_cm}cm{angle}deg")
    calibrated = extract_tones(capture, tone_frequencies() + TAG_SHIFT) / reference
    return (env, distance_cm, angle), {k: music_estimates(calibrated, s) for k, s in CARRIER_SUBSETS.items()}


def errors(estimates, distance_cm, angle):
    """Return AoA error (deg), ToF distance error (m), localization error (m)."""
    aoa, distance = estimates[:, 0], estimates[:, 1]
    true_distance = distance_cm / 100
    a, b = np.deg2rad(aoa), np.deg2rad(angle)
    loc = np.hypot(distance * np.cos(a) - true_distance * np.cos(b), distance * np.sin(a) - true_distance * np.sin(b))
    return np.abs(aoa - angle), np.abs(distance - true_distance), loc


def gather(results, env, waves, points=TOF_POINTS):
    keys = [(env, d, a) for d in DISTANCES_CM for a in ANGLES_DEG]
    parts = [errors(results[k][waves][points], k[1], k[2]) for k in keys]
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e3e3e3", "axes.axisbelow": True,
    })


def plot_cdf(results, env, metric, xlabel, xmax, name):
    fig, ax = plt.subplots(figsize=(4.2, 3.1), layout="constrained")
    for waves in CARRIER_SUBSETS:
        values = np.sort(gather(results, env, waves)[metric])
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), color=COLORS[waves],
                linestyle=LINESTYLES[waves], linewidth=1.6, label=f"{waves} waves")
    ax.set(xlim=(0, xmax), ylim=(0, 1), xlabel=xlabel, ylabel="CDF")
    ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=1, edgecolor="#cccccc")
    fig.savefig(f"figures/{name}.png", dpi=FIGURE_DPI)
    plt.close(fig)


def draw_box(ax, x, values):
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    spread = 1.5 * (q3 - q1)
    low, high = values[values >= q1 - spread].min(), values[values <= q3 + spread].max()
    ax.vlines([x, x], [low, q3], [q1, high], color="#333333", linewidth=1.1)
    ax.hlines([low, high], x - 0.1, x + 0.1, color="#333333", linewidth=1.1)
    ax.add_patch(plt.Rectangle((x - 0.22, q1), 0.44, q3 - q1, facecolor="white", edgecolor=COLORS[2], linewidth=1.3, zorder=3))
    ax.hlines(median, x - 0.22, x + 0.22, color=COLORS[3], linewidth=2, zorder=4)
    return q1, median, q3


def plot_trend(results, env, ymax, name):
    fig, axes = plt.subplots(2, 1, figsize=(5, 4.2), layout="constrained")
    groups = [("Tag-Rx distance (m)", DISTANCES_CM, lambda g: [(g, a) for a in ANGLES_DEG], lambda g: f"{g / 100:g}"),
              ("Actual AoA (°)", ANGLES_DEG, lambda g: [(d, g) for d in DISTANCES_CM], lambda g: f"{g}°")]
    for ax, (xlabel, values, members, label) in zip(axes, groups):
        for i, group in enumerate(values):
            data = np.concatenate([errors(results[(env, d, a)][21][TOF_POINTS], d, a)[2] for d, a in members(group)])
            draw_box(ax, i, data)
        ax.set_xticks(range(len(values)), [label(v) for v in values])
        ax.set(xlabel=xlabel, ylabel="Localization\nerror (m)", ylim=(0, ymax), xlim=(-0.6, len(values) - 0.4))
        ax.grid(axis="x", visible=False)
    fig.savefig(f"figures/{name}.png", dpi=FIGURE_DPI)
    plt.close(fig)


def summarize(results):
    numbers = {"median_errors": {}, "tof_search_sensitivity": {}, "window_edge_fraction": {}}
    for env in ENVIRONMENTS:
        numbers["median_errors"][env] = {}
        numbers["tof_search_sensitivity"][env] = {}
        for waves in CARRIER_SUBSETS:
            aoa, tof, loc = gather(results, env, waves)
            numbers["median_errors"][env][waves] = {
                "aoa_deg": float(np.median(aoa)), "tof_m": float(np.median(tof)), "localization_m": float(np.median(loc))}
            numbers["tof_search_sensitivity"][env][waves] = {
                f"{p * TOF_STEP * 1e9:.1f}ns": float(np.median(gather(results, env, waves, p)[2]))
                for p in TOF_POINTS_SENSITIVITY}
        keys = [(env, d, a) for d in DISTANCES_CM for a in ANGLES_DEG]
        tof = np.concatenate([results[k][21][TOF_POINTS][:, 1] for k in keys])
        top = (TOF_POINTS - 1) * TOF_STEP * LIGHT_SPEED / 2
        numbers["window_edge_fraction"][env] = {
            "at_zero": float(np.mean(tof == 0)), "at_upper_limit": float(np.mean(np.isclose(tof, top)))}
    return numbers


def write_estimates(results):
    """Write every AoA and distance estimate as the processed dataset behind the figures."""
    with open("results/estimates.csv", "w") as out:
        out.write("environment,true_distance_m,true_aoa_deg,carriers,estimate,aoa_deg,distance_m\n")
        for (env, distance_cm, angle), by_waves in sorted(results.items()):
            for waves, by_points in by_waves.items():
                for index, (aoa, distance) in enumerate(by_points[TOF_POINTS]):
                    out.write(f"{env},{distance_cm / 100:g},{angle},{waves},{index},{aoa:g},{distance:.4f}\n")


def write_numbers(numbers):
    Path("results/numbers.json").write_text(json.dumps(numbers, indent=2))
    lines = []
    for env, label in ENVIRONMENTS.items():
        lines.append(f"{label}: median AoA error (deg), ToF error (m), localization error (m)")
        for waves, row in numbers["median_errors"][env].items():
            lines.append(f"  {waves:2d} waves  {row['aoa_deg']:5.1f}  {row['tof_m']:6.4f}  {row['localization_m']:6.4f}")
        lines.append(f"{label}: median localization error (m) by ToF search upper limit")
        for waves, row in numbers["tof_search_sensitivity"][env].items():
            lines.append(f"  {waves:2d} waves  " + "  ".join(f"{k} {v:.4f}" for k, v in row.items()))
        edge = numbers["window_edge_fraction"][env]
        lines.append(f"{label}: 21-wave estimates at ToF 0: {edge['at_zero']:.3f}, at upper limit: {edge['at_upper_limit']:.3f}")
    text = "\n".join(lines) + "\n"
    Path("results/numbers.txt").write_text(text)
    print(text)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_DATA)
    os.chdir(HERE)
    Path("figures").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    references = {env: wired_reference(source, env) for env in ENVIRONMENTS}
    jobs = [(source, env, references[env], d, a) for env in ENVIRONMENTS for d in DISTANCES_CM for a in ANGLES_DEG]
    with mp.Pool() as pool:
        results = dict(pool.map(process_position, jobs))
    style()
    for env, prefix in (("abs", "abs"), ("noAbs", "noAbs")):
        limits = {"abs": (20, 1.5, 1.5), "noAbs": (50, 5, 5)}[env]
        plot_cdf(results, env, 0, "AoA error (°)", limits[0], f"{prefix}AoACDF")
        plot_cdf(results, env, 1, "ToF (distance estimation) error (m)", limits[1], f"{prefix}ToFCDF")
        plot_cdf(results, env, 2, "Localization error (m)", limits[2], f"{prefix}LocalizationCDF")
        plot_trend(results, env, {"abs": 2, "noAbs": 7.5}[env], f"{prefix}LocalizationTrend")
    write_estimates(results)
    write_numbers(summarize(results))


if __name__ == "__main__":
    main()
