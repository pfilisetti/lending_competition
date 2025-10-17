"""
Generate histograms and Q-Q plots for lender signals.

Usage:
    .venv/bin/python -m analysis.signal_visualizations

The script saves figures under ``analysis/outputs``:
    - ``signal_histograms.png`` for the marginal distributions of each signal.
    - ``signal_difference_qqplots.png`` for Q-Q plots of pairwise differences.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from utils.utils import SIGNAL_COLUMNS, load_past_loans, replace_zero_with_nan

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def ensure_output_dir(path: Path = OUTPUT_DIR) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_histograms(signals: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> Path:
    fig, axes = plt.subplots(1, len(SIGNAL_COLUMNS), figsize=(4 * len(SIGNAL_COLUMNS), 4), sharey=True)

    for ax, column in zip(axes, SIGNAL_COLUMNS):
        data = signals[column].dropna()
        ax.hist(data, bins=40, density=True, alpha=0.75, color="#1f77b4", edgecolor="black")
        ax.set_title(f"{column} distribution")
        ax.set_xlabel("signal value")
        ax.set_ylabel("density")

    fig.suptitle("Signal distributions (zeros treated as missing)")
    fig.tight_layout()

    output_path = output_dir / "signal_histograms.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_qq_differences(signals: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> Path:
    pairs = [("signal1", "signal2"), ("signal1", "signal3"), ("signal2", "signal3")]
    fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 4))

    for ax, (signal_a, signal_b) in zip(axes, pairs):
        diff = (signals[signal_a] - signals[signal_b]).dropna()
        stats.probplot(diff, dist="norm", plot=ax)
        ax.set_title(f"{signal_a} - {signal_b}")
        ax.set_xlabel("theoretical quantiles")
        ax.set_ylabel("sample quantiles")

    fig.suptitle("Q-Q plots of pairwise signal differences")
    fig.tight_layout()

    output_path = output_dir / "signal_difference_qqplots.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def main() -> None:
    ensure_output_dir()
    df = load_past_loans()
    signals = replace_zero_with_nan(df, SIGNAL_COLUMNS)

    hist_path = plot_histograms(signals)
    qq_path = plot_qq_differences(signals)

    print(f"Saved histogram figure to: {hist_path}")
    print(f"Saved Q-Q plot figure to: {qq_path}")


if __name__ == "__main__":
    main()
