"""Command line tool for estimating trailing stop loss levels from historical prices."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import timedelta
from typing import Sequence

import numpy as np
import pandas as pd
import yfinance as yf


class AnalyzerError(Exception):
    """Raised when the trailing stop analysis cannot be completed."""


@dataclass
class PullbackStatistics:
    """Container for descriptive statistics about pullback magnitudes."""

    count: int
    mean: float
    median: float
    percentile_75: float
    percentile_90: float
    percentile_95: float
    min_pullback: float
    max_pullback: float


def parse_arguments() -> argparse.Namespace:
    """Configure and parse command line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyze historical pullbacks for a stock and suggest a trailing stop loss level."
        )
    )
    parser.add_argument(
        "-s",
        "--symbol",
        help="Ticker symbol to analyze. Prompts interactively when omitted.",
    )
    parser.add_argument(
        "-y",
        "--years",
        type=int,
        default=5,
        help="Number of years of history to download (default: 5).",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help=(
            "Rolling window in days for smoothing prices before swing analysis. Use 1 to disable."
        ),
    )
    parser.add_argument(
        "--crash-threshold",
        type=float,
        default=None,
        help="Maximum pullback percentage considered 'normal'. Larger moves are dropped.",
    )
    parser.add_argument(
        "--min-pullback",
        type=float,
        default=0.5,
        help="Minimum pullback percentage to keep (default: 0.5).",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.9,
        help="Percentile (0-1) used for the trailing stop suggestion (default: 0.9).",
    )
    parser.add_argument(
        "--swing-window",
        type=int,
        default=3,
        help="Days on each side when identifying swing highs and lows (default: 3).",
    )

    return parser.parse_args()


def resolve_symbol(symbol: str | None) -> str:
    """Return a valid ticker symbol, prompting the user if necessary."""

    if symbol:
        return symbol.strip().upper()

    try:
        user_input = input("Enter ticker symbol: ")
    except EOFError as exc:  # pragma: no cover - interactive guard
        raise AnalyzerError("Ticker symbol is required.") from exc

    cleaned = user_input.strip().upper()
    if not cleaned:
        raise AnalyzerError("Ticker symbol is required.")
    return cleaned


def validate_parameters(args: argparse.Namespace) -> None:
    """Validate CLI parameters and raise AnalyzerError for invalid values."""

    if args.years <= 0:
        raise AnalyzerError("Number of years must be positive.")
    if args.smoothing_window <= 0:
        raise AnalyzerError("Smoothing window must be at least 1.")
    if not 0 < args.quantile < 1:
        raise AnalyzerError("Quantile must be between 0 and 1 (exclusive).")
    if args.min_pullback < 0:
        raise AnalyzerError("Minimum pullback must be non-negative.")
    if args.crash_threshold is not None and args.crash_threshold <= 0:
        raise AnalyzerError("Crash threshold must be positive when provided.")
    if args.swing_window <= 0:
        raise AnalyzerError("Swing window must be positive.")


def fetch_adjusted_closes(symbol: str, years: int) -> pd.Series:
    """Download adjusted close prices for the requested ticker and time span."""

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=years)

    try:
        frame = yf.download(
            symbol,
            start=start_date,
            end=end_date + timedelta(days=1),
            auto_adjust=False,
            progress=False,
            actions=False,
        )
    except Exception as exc:  # pragma: no cover - network failure path
        raise AnalyzerError(f"Failed to download data for {symbol}: {exc}") from exc

    if frame.empty or "Adj Close" not in frame:
        raise AnalyzerError(
            f"No adjusted close data found for {symbol}. Verify the ticker and try again."
        )

    prices = frame["Adj Close"].dropna()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()

    if prices.empty:
        raise AnalyzerError("Downloaded data contains no usable prices.")

    return prices


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    """Return a smoothed version of the price series using a centered rolling mean."""

    if window <= 1:
        return series
    smoothed = series.rolling(window=window, center=True, min_periods=1).mean()
    return smoothed


def detect_swings(prices: pd.Series, window: int) -> tuple[list[int], list[int]]:
    """Identify indices of swing highs and lows using a local extrema heuristic."""

    values = prices.to_numpy()
    length = len(values)
    if length == 0:
        return [], []

    highs: list[int] = []
    lows: list[int] = []

    for idx in range(window, length - window):
        value = values[idx]
        prev = values[idx - window : idx]
        nxt = values[idx + 1 : idx + 1 + window]
        if value >= prev.max() and value >= nxt.max():
            highs.append(idx)
        if value <= prev.min() and value <= nxt.min():
            lows.append(idx)

    if not lows or lows[0] != 0:
        lows.insert(0, 0)
    if (length - 1) not in lows:
        lows.append(length - 1)

    return sorted(set(highs)), sorted(set(lows))


def compute_pullbacks(prices: pd.Series, swing_window: int) -> list[float]:
    """Compute pullback percentages following swing highs."""

    highs, lows = detect_swings(prices, swing_window)
    values = prices.to_numpy()
    length = len(values)
    pullbacks: list[float] = []

    if length < 2:
        return pullbacks

    for high_idx in highs:
        if high_idx >= length - 1:
            continue
        previous_lows = [low for low in lows if low < high_idx]
        if not previous_lows:
            continue
        last_low_idx = previous_lows[-1]
        if values[high_idx] <= values[last_low_idx]:
            continue

        peak_price = values[high_idx]
        max_drawdown = 0.0

        for idx in range(high_idx + 1, length):
            price = values[idx]
            if price > peak_price:
                break
            drawdown = (price - peak_price) / peak_price * 100.0
            if drawdown < max_drawdown:
                max_drawdown = drawdown

        if max_drawdown < 0:
            pullbacks.append(abs(max_drawdown))

    return pullbacks


def filter_pullbacks(
    pullbacks: Sequence[float],
    min_pullback: float,
    crash_threshold: float | None,
) -> list[float]:
    """Remove noise and crash-like values from pullback observations."""

    if not pullbacks:
        return []

    arr = np.array(pullbacks, dtype=float)
    arr = arr[arr >= min_pullback]

    if crash_threshold is not None:
        arr = arr[arr <= crash_threshold]

    if arr.size == 0:
        return []

    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    filtered = arr[arr <= upper_bound]

    return filtered.tolist()


def describe_pullbacks(pullbacks: Sequence[float]) -> PullbackStatistics:
    """Generate descriptive statistics for the pullback distribution."""

    series = pd.Series(pullbacks, dtype=float)
    return PullbackStatistics(
        count=int(series.count()),
        mean=float(series.mean()),
        median=float(series.median()),
        percentile_75=float(series.quantile(0.75)),
        percentile_90=float(series.quantile(0.9)),
        percentile_95=float(series.quantile(0.95)),
        min_pullback=float(series.min()),
        max_pullback=float(series.max()),
    )


def round_to_step(value: float, step: float = 0.5) -> float:
    """Round percentage values to the nearest increment (default 0.5%)."""

    if math.isnan(value):
        return value
    return round(value / step) * step


def format_percent(value: float) -> str:
    """Return a string representation of a percentage with one decimal place."""

    return f"{value:.1f}%"


def analyze(symbol: str, args: argparse.Namespace) -> None:
    """Run the full trailing stop analysis and print the final report."""

    prices = fetch_adjusted_closes(symbol, args.years)
    smoothed = smooth_series(prices, args.smoothing_window)
    pullbacks = compute_pullbacks(smoothed, args.swing_window)
    filtered = filter_pullbacks(pullbacks, args.min_pullback, args.crash_threshold)

    if not filtered:
        raise AnalyzerError("No valid pullbacks were detected after filtering.")

    stats = describe_pullbacks(filtered)
    series = pd.Series(filtered, dtype=float)
    suggested_raw = float(series.quantile(args.quantile))

    warning: str | None = None
    if stats.count < 10:
        warning = (
            "Only {count} normal pullbacks were identified. Recommendation may be unreliable."
        ).format(count=stats.count)
        suggested_raw = max(suggested_raw, float(series.max()))

    suggested = round_to_step(suggested_raw, 0.5)

    start_date = prices.index[0].date()
    end_date = prices.index[-1].date()

    print("\nTrailing Stop Analyzer")
    print("=" * 70)
    print(f"Symbol             : {symbol}")
    print(f"History Range      : {start_date} to {end_date} ({len(prices)} trading days)")
    print(f"Smoothing Window   : {args.smoothing_window} day(s)")
    if args.crash_threshold is not None:
        print(f"Crash Threshold    : {args.crash_threshold:.1f}%")
    print("\nPullback Statistics (after filtering)")
    print("-" * 70)
    print(f"Observations       : {stats.count}")
    print(f"Mean               : {format_percent(stats.mean)}")
    print(f"Median             : {format_percent(stats.median)}")
    print(f"75th Percentile    : {format_percent(stats.percentile_75)}")
    print(f"90th Percentile    : {format_percent(stats.percentile_90)}")
    print(f"95th Percentile    : {format_percent(stats.percentile_95)}")
    print(f"Minimum Pullback   : {format_percent(stats.min_pullback)}")
    print(f"Maximum Pullback   : {format_percent(stats.max_pullback)}")

    print("\nSuggested Trailing Stop")
    print("-" * 70)
    print(
        f"Suggested trailing stop: {format_percent(suggested)} "
        f"(based on the {int(args.quantile * 100)}th percentile)"
    )
    if warning:
        print(f"\nWarning: {warning}")


def main() -> None:
    """Entry point for the CLI tool."""

    args = parse_arguments()
    try:
        validate_parameters(args)
        symbol = resolve_symbol(args.symbol)
        analyze(symbol, args)
    except AnalyzerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

    # Example usage:
    #   python trailing_stop_analyzer.py -s AAPL
    #   python trailing_stop_analyzer.py -s MSFT --years 3 --quantile 0.95
    #   python trailing_stop_analyzer.py --years 10
