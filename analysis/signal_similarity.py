"""
Quick exploratory analysis of the three lender signals on past loans.

Usage:
    .venv/bin/python -m analysis.signal_similarity

The script prints descriptive statistics, pairwise correlations (Pearson and
Spearman) as well as normality checks on the differences between every pair of
signals. All zeros are treated as missing values since the project encodes
unobserved signals as 0.
"""

from __future__ import annotations

import pandas as pd

from utils.utils import compare_signal_pairs, correlation_matrix, load_past_loans, signal_summary


def _format_header(title: str) -> str:
    line = "-" * len(title)
    return f"{title}\n{line}"


def print_descriptive_stats(df: pd.DataFrame) -> None:
    print(_format_header("Signal Summary (zeros treated as missing)"))
    summary = signal_summary(df)
    print(summary)
    print()


def print_correlations(df: pd.DataFrame) -> None:
    print(_format_header("Pearson Correlation (zeros treated as missing)"))
    pearson = correlation_matrix(df, method="pearson")
    print(pearson)
    print()

    print(_format_header("Spearman Correlation (zeros treated as missing)"))
    spearman = correlation_matrix(df, method="spearman")
    print(spearman)
    print()


def print_pairwise_tests(df: pd.DataFrame) -> None:
    print(_format_header("Pairwise Differences Normality Tests"))
    for result in compare_signal_pairs(df):
        print(
            f"{result.signal_a} - {result.signal_b}: "
            f"mean={result.mean_diff:.4f}, "
            f"std={result.std_diff:.4f}, "
            f"normaltest_stat={result.normaltest_stat:.2f}, "
            f"pvalue={result.normaltest_pvalue:.4f}"
        )
    print()


def main() -> None:
    df = load_past_loans()
    print_descriptive_stats(df)
    print_correlations(df)
    print_pairwise_tests(df)


if __name__ == "__main__":
    main()
