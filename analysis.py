"""
Script for evaluating entropy-based change point detection algorithms.

This script generates/loads several time series (Coal Mine Disasters, synthetic ECG, synthetic Finance),
runs selected entropy-based CPD algorithms on each series, and visualizes detected change points
against expected locations.

The plotting includes:
- primary series curve,
- detected change points (scatter),
- expected change-point bands (± tolerance) and centerlines,
- grid with major/minor ticks, and a legend in the upper-right corner.
"""

__author__ = "Kirill Gribanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import matplotlib.pyplot as plt

from approximate_entropy import ApproximateEntropyAlgorithm
from bubble_entropy import BubbleEntropyAlgorithm
from sample_entropy import SampleEntropyAlgorithm
from permutation_entropy import PermutationEntropyAlgorithm


EXPECTED_CPS = {
    "COAL MINE DISASTERS (yearly)": [36],
    "ECG SYNTH (with CPs)": [120],
    "FINANCE SYNTH (with CPs)": [160],
}
EXPECTED_TOL = {
    "COAL MINE DISASTERS (yearly)": 2,
    "ECG SYNTH (with CPs)": 5,
    "FINANCE SYNTH (with CPs)": 5,
}


time_series_list: dict[str, np.ndarray] = {}


coal_series = np.array([
    4,5,4,1,0,4,3,4,0,6,3,3,4,0,2,6,3,3,5,4,
    5,3,1,4,4,1,5,5,3,4,2,5,2,2,3,4,2,1,3,2,
    2,1,1,1,1,3,0,0,1,0,1,1,0,0,3,1,0,3,2,2,
    0,1,1,1,0,1,0,1,0,0,0,2,1,0,0,0,1,1,0,2,
    3,3,1,1,2,1,1,1,1,2,4,2,0,0,0,1,4,0,0,0,
    1,0,0,0,0,0,1,0,0,1,0,1
], dtype=float)

time_series_list["COAL MINE DISASTERS (yearly)"] = coal_series
print(f"[OK] Loaded COAL (manual): {len(coal_series)} points")


np.random.seed(1337)
N_ecg = 240
cp_ecg = 120
t1 = np.linspace(0, 4*np.pi, cp_ecg)
ecg_1 = 1.0 * np.sin(1.2 * t1) + 0.02 * np.random.randn(cp_ecg)
t2 = np.linspace(0, 6*np.pi, N_ecg - cp_ecg)
ecg_2 = 2.5 * np.sin(3.5 * t2) + 2.0 + 0.30 * np.random.randn(N_ecg - cp_ecg)
ecg_2[:3] += 3.0
ecg_synth = np.concatenate([ecg_1, ecg_2])
time_series_list["ECG SYNTH (with CPs)"] = ecg_synth
print(f"[OK] Generated ECG synth: {len(ecg_synth)} points (CP at {cp_ecg})")


rng = np.random.default_rng(2025)
N_fin = 300
cp_fin = 160
r1 = rng.normal(0.0002, 0.006, cp_fin)
r2 = rng.normal(-0.0005, 0.045, N_fin - cp_fin)
rets = np.concatenate([r1, r2])
price = 100.0 * np.exp(np.cumsum(rets))
price[cp_fin:] += 40.0
price[cp_fin:cp_fin+2] += 10.0
time_series_list["FINANCE SYNTH (with CPs)"] = price
print(f"[OK] Generated FINANCE synth: {len(price)} points (CP at {cp_fin})")


def build_algorithms_for(series_name: str):
    """
    Build a list of CPD algorithms configured for a given series name.

    Parameters
    ----------
    series_name : str
        Name of the series as used in `time_series_list`.

    Returns
    -------
    list
        A list of instantiated CPD algorithm objects ready for streaming detection.
    """
    name = series_name.lower()

    if "coal" in name:
        return [
            BubbleEntropyAlgorithm(window_size=34, embedding_dimension=3, time_delay=1, threshold=0.42),
            SampleEntropyAlgorithm(window_size=17, m=2, r=0.99, threshold=0.6),
        ]

    if "ecg" in name:
        ecg_fixed_r = float(0.145 * np.std(ecg_1))
        return [
            ApproximateEntropyAlgorithm(
                window_size=55,
                m=8,
                r=ecg_fixed_r,
                threshold=0.0284
            ),
            SampleEntropyAlgorithm(
                window_size=6,
                m=2,
                r=None,
                r_factor=0.3,
                threshold=0.74
            )
        ]

    if "finance" in name:
        return [
            PermutationEntropyAlgorithm(
                window_size=20,
                embedding_dimension=5,
                time_delay=1,
                threshold=0.9
            ),
            SampleEntropyAlgorithm(
                window_size=8,
                m=3,
                r=None,
                r_factor=0.20,
                threshold=0.5
            ),
        ]

    return [
        SampleEntropyAlgorithm(window_size=80, m=2, r=None, r_factor=0.20, threshold=0.20),
    ]


def plot_series_with_cps(series_name, series, cps, algo_name, expected_cps=None, tol=None):
    """
    Plot a time series with detected and expected change points.

    Parameters
    ----------
    series_name : str
        Display name of the series.
    series : array-like
        Sequence of numeric values representing the time series.
    cps : list[int] or None
        Detected change-point indices to be highlighted.
    algo_name : str
        Label of the algorithm used for detection (shown in the title).
    expected_cps : list[int] or None, optional
        Expected change-point indices for reference visualization.
    tol : int or None, optional
        Tolerance (±) around expected indices for shaded bands.

    Returns
    -------
    None
    """
    plt.figure(figsize=(12, 3))
    plt.plot(series, linewidth=1.3, label=series_name)

    if cps:
        uniq_cps = sorted(set(int(i) for i in cps if 0 <= i < len(series)))
        if uniq_cps:
            plt.scatter(
                uniq_cps,
                np.array(series)[uniq_cps],
                color="red",
                marker="x",
                s=36,
                label="Detected CPs"
            )

    if expected_cps:
        added = False
        for cp in expected_cps:
            if 0 <= cp < len(series):
                if tol is not None and tol > 0:
                    left = max(0, cp - tol)
                    right = min(len(series) - 1, cp + tol)
                    plt.axvspan(left, right, alpha=0.10, color="green")
                if not added:
                    plt.axvline(cp, color="green", linestyle="--", linewidth=1.2, label="Expected CP")
                    added = True
                else:
                    plt.axvline(cp, color="green", linestyle="--", linewidth=1.2)

    plt.title(f"{series_name} — {algo_name}", fontsize=12)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.minorticks_on()
    plt.grid(True, which="major", linestyle="--", alpha=0.45)
    plt.grid(True, which="minor", linestyle=":",  alpha=0.25)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def analyze_series(name, series, algorithms):
    """
    Run streaming detection for a set of algorithms on a given series and visualize results.

    Parameters
    ----------
    name : str
        Series name key used for lookups in EXPECTED_CPS/EXPECTED_TOL.
    series : array-like
        Time series values.
    algorithms : list
        List of instantiated CPD algorithm objects supporting `detect(value)` and `localize(value)`.

    Returns
    -------
    None
    """
    print(f"\n=== Analyzing time series: {name} ===")
    for algo in algorithms:
        if hasattr(algo, "reset"):
            try:
                algo.reset()
            except Exception:
                pass

        cps = []
        for v in series:
            if algo.detect(v):
                cp = algo.localize(v)
                if cp is not None and 0 <= cp < len(series):
                    cps.append(int(cp))

        print(f"{algo.__class__.__name__}: Detected change points: {cps}")
        expected = EXPECTED_CPS.get(name, None)
        tol = EXPECTED_TOL.get(name, None)
        plot_series_with_cps(name, series, cps, algo.__class__.__name__, expected_cps=expected, tol=tol)


for name, series_data in time_series_list.items():
    if not (name.startswith("COAL") or name.startswith("ECG") or name.startswith("FINANCE")):
        continue
    series_arr = np.asarray(series_data, dtype=float)
    series_arr = series_arr[np.isfinite(series_arr)]
    algorithms = build_algorithms_for(name)
    analyze_series(name, series_arr, algorithms)