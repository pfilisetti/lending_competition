
"""
Utility helpers for analysing lender signals on past loan data.

These functions encapsulate repeated cleaning steps (handling the zero-coded
missing values), summary statistics and pairwise comparisons that we reuse in
different analysis scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_PAST_LOANS_PATH = Path("data") / "PastLoans.csv"
SIGNAL_COLUMNS = ("signal1", "signal2", "signal3")


def load_past_loans(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the historical loans dataset located at ``data/PastLoans.csv``.

    Parameters
    ----------
    path:
        Optional override to point to a different CSV file.

    Returns
    -------
    pd.DataFrame
        The raw dataframe without any post-processing.
    """

    csv_path = Path(path) if path is not None else DEFAULT_PAST_LOANS_PATH
    return pd.read_csv(csv_path)


def replace_zero_with_nan(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """
    Replace zero entries with NaN for the provided columns.

    Many observations report a value of ``0`` when the private signal is missing.
    Replacing them with ``NaN`` simplifies downstream filtering (``dropna`` and
    masking).
    """

    result = df.copy()
    col_list = list(columns)
    result.loc[:, col_list] = result.loc[:, col_list].replace(0, np.nan)
    return result


def signal_summary(
    df: pd.DataFrame,
    columns: Sequence[str] = SIGNAL_COLUMNS,
    treat_zero_as_missing: bool = True,
) -> pd.DataFrame:
    """
    Compute descriptive statistics for each signal.

    Parameters
    ----------
    df:
        Dataframe containing the signal columns.
    columns:
        Names of the signal columns to summarise.
    treat_zero_as_missing:
        When ``True``, zero values are replaced with ``NaN`` prior to the summary.

    Returns
    -------
    pd.DataFrame
        A dataframe mirroring ``DataFrame.describe()`` with non-missing counts,
        means, standard deviations, min/max and quantiles.
    """

    target = replace_zero_with_nan(df, columns) if treat_zero_as_missing else df
    col_list = list(columns)
    return target.loc[:, col_list].describe()


def correlation_matrix(
    df: pd.DataFrame,
    columns: Sequence[str] = SIGNAL_COLUMNS,
    method: Literal['pearson', 'kendall', 'spearman'] = "pearson",
    treat_zero_as_missing: bool = True,
) -> pd.DataFrame:
    """
    Compute a correlation matrix between signal columns.

    Parameters
    ----------
    method:
        ``pearson`` (default) or any value supported by ``DataFrame.corr``.
    treat_zero_as_missing:
        When ``True``, the correlations only use non-zero pairs.
    """

    target = replace_zero_with_nan(df, columns) if treat_zero_as_missing else df
    col_list = list(columns)
    return target.loc[:, col_list].corr(method=method)


@dataclass
class SignalPairComparison:
    """Container for pairwise signal comparison results."""

    signal_a: str
    signal_b: str
    mean_diff: float
    std_diff: float
    normaltest_stat: float
    normaltest_pvalue: float


def compare_signal_pairs(
    df: pd.DataFrame,
    columns: Sequence[str] = SIGNAL_COLUMNS,
    treat_zero_as_missing: bool = True,
) -> List[SignalPairComparison]:
    """
    Evaluate differences between each pair of signals.

    For each pair we compute:
    - mean/std of the difference ``signal_a - signal_b``;
    - D'Agostino & Pearson normality test on the differences.
    """

    cleaned = replace_zero_with_nan(df, columns) if treat_zero_as_missing else df
    results: List[SignalPairComparison] = []

    for idx, col_a in enumerate(columns):
        for col_b in columns[idx + 1 :]:
            diff = cleaned[col_a] - cleaned[col_b]
            diff = diff.dropna()
            if diff.empty:
                continue

            stat, pvalue = stats.normaltest(diff)
            results.append(
                SignalPairComparison(
                    signal_a=col_a,
                    signal_b=col_b,
                    mean_diff=float(diff.mean()),
                    std_diff=float(diff.std(ddof=1)),
                    normaltest_stat=float(stat),
                    normaltest_pvalue=float(pvalue),
                )
            )

    return results


def signal_pair_iterator(columns: Sequence[str] = SIGNAL_COLUMNS) -> Iterable[tuple[str, str]]:
    """Yield all unique unordered signal pairs."""

    for idx, col_a in enumerate(columns):
        for col_b in columns[idx + 1 :]:
            yield col_a, col_b
