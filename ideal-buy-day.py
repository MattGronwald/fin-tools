from __future__ import annotations

import argparse
import calendar
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import yfinance as yf
from src.cache_utils import CacheError, get_price_history


def ensure_images_directory() -> str:
    """Create the images directory if it does not exist."""

    images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    return images_dir


IMAGES_DIR = ensure_images_directory()

DaySummary = dict[int, int | str]
FilterDays = Sequence[int] | None


def add_date_columns(df: DataFrame) -> DataFrame:
    """Add month and day columns to the dataframe."""
    df = df.copy()
    df["Month"] = df.index.to_series().dt.month
    df["Day"] = df.index.to_series().dt.day
    return df


def find_best_days_by_month(daily_avg: Series, filter_days: FilterDays) -> DaySummary:
    """Find the day with lowest average price for each month."""
    best_days: DaySummary = {}
    for month in range(1, 13):
        try:
            month_data = daily_avg.xs(month, level="Month")

            if filter_days:
                valid_days = [int(day) for day in filter_days if int(day) in month_data.index]
                if valid_days:
                    month_data = month_data.loc[valid_days]
                else:
                    best_days[month] = "No matching days"
                    continue

            best_day = int(month_data.idxmin())
            best_days[month] = best_day
        except KeyError:
            best_days[month] = "No data"

    return best_days


def print_best_days(best_days: Mapping[int, int | str]) -> None:
    """Format and print the best days for each month."""
    for month in best_days:
        month_name = calendar.month_name[month]
        print(f"Best day to buy in {month_name}: {best_days[month]}")


def get_best_buy_dates(
    ticker_symbol: str,
    years: int = 10,
    filter_days: FilterDays = None,
) -> None:
    """Analyze stock data to find best buying dates."""
    try:
        df = get_price_history(ticker_symbol, years)
    except CacheError as exc:
        print(f"Unable to retrieve historical data: {exc}")
        return

    print(f"Analyzing {len(df)} days of data...")
    df = add_date_columns(df)

    daily_avg = cast(Series, df.groupby(["Month", "Day"])["Low"].mean())

    best_days = find_best_days_by_month(daily_avg, filter_days)
    print_best_days(best_days)

    # Create visualization
    create_day_of_month_visualization(df, ticker_symbol, years, filter_days)


def get_company_name(ticker_symbol: str) -> str:
    """Retrieve company name for the given ticker."""
    try:
        stock = yf.Ticker(ticker_symbol)
        short_name = stock.info.get("shortName")
        return short_name if isinstance(short_name, str) else ticker_symbol
    except Exception:
        return ticker_symbol  # Fallback if company name can't be retrieved


def calculate_day_averages(df: DataFrame, filter_days: FilterDays) -> Series | None:
    """Calculate average price for each day of month across all years."""
    day_avg = cast(Series, df.groupby("Day")["Low"].mean())

    if filter_days:
        filter_set = {int(day) for day in filter_days}
        day_avg = day_avg[day_avg.index.isin(filter_set)]
        if day_avg.empty:
            print("No data available for the specified filter days.")
            return None

    return day_avg


def setup_plot_figure() -> Axes:
    """Create and configure the plot figure."""
    _ = plt.figure(figsize=(12, 9))
    return plt.subplot(111)


def create_bar_chart(ax: Axes, day_avg: Series, filter_days: FilterDays) -> BarContainer:
    """Create the bar chart with appropriate styling."""
    width = 1.1 if filter_days else 0.8
    x_values = [int(index) for index in day_avg.index.tolist()]
    heights = day_avg.to_numpy(dtype=float)
    return ax.bar(x_values, heights, color="skyblue", width=width)


def highlight_min_max_days(bars: BarContainer, day_avg: Series) -> tuple[int, float, int, float]:
    """Highlight the best and worst days on the chart."""
    min_day = int(day_avg.idxmin())
    min_price = float(day_avg.min())
    max_day = int(day_avg.idxmax())
    max_price = float(day_avg.max())

    # Highlight best day in green
    min_idx_raw = day_avg.index.get_loc(min_day)
    if not isinstance(min_idx_raw, (int, np.integer)):
        raise ValueError("Unable to identify a unique minimum day for highlighting.")
    min_idx = int(min_idx_raw)
    bars.patches[min_idx].set_color("green")

    # Highlight worst day in red
    max_idx_raw = day_avg.index.get_loc(max_day)
    if not isinstance(max_idx_raw, (int, np.integer)):
        raise ValueError("Unable to identify a unique maximum day for highlighting.")
    max_idx = int(max_idx_raw)
    bars.patches[max_idx].set_color("red")

    return min_day, min_price, max_day, max_price


def add_chart_styling(filter_days: FilterDays) -> None:
    """Add labels, grid, and configure axes."""
    _ = plt.xlabel("Day of Month", fontsize=12, labelpad=10)
    _ = plt.ylabel("Average Price ($)", fontsize=12)
    _ = plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Set x-axis ticks
    if filter_days:
        _ = plt.xticks(sorted(filter_days), fontsize=11)
    else:
        _ = plt.xticks(range(1, 32), fontsize=11)


def add_average_line(day_avg: Series) -> float:
    """Add horizontal line representing the average price."""
    avg_price = float(day_avg.mean())
    _ = plt.axhline(y=avg_price, color="purple", linestyle="--", alpha=0.7)

    # Add text label for average
    _ = plt.text(
        max(day_avg.index) * 0.9,
        avg_price * 1.015,
        f"Avg: ${avg_price:.2f}",
        va="center",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="purple", boxstyle="round,pad=0.2"),
    )

    return avg_price


def adjust_y_limits() -> tuple[float, float, float]:
    """Adjust y-axis limits to make room for annotations."""
    y_min, y_max = plt.ylim()
    y_range = y_max - y_min
    _ = plt.ylim(y_min - y_range * 0.1, y_max + y_range * 0.15)

    y_min_updated, y_max_updated = plt.ylim()
    return float(y_min_updated), float(y_max_updated), float(y_max_updated - y_min_updated)


def add_annotations(
    min_day: int,
    min_price: float,
    max_day: int,
    max_price: float,
    day_avg: Series,
    y_range: float,
) -> None:
    """Add text annotations for best and worst days."""
    _ = plt.annotate(
        f"Best Day: {min_day}\n${min_price:.2f}",
        xy=(min_day, min_price),
        xytext=(min_day, min_price - y_range * 0.10),
        arrowprops=dict(facecolor="green", shrink=0.05, width=1.5),
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8),
    )

    x_offset = 0
    if len(day_avg) > 3:
        middle_indices = [day_avg.index[i] for i in range(len(day_avg) // 2 - 1, len(day_avg) // 2 + 2)]
        if max_day in middle_indices:
            x_offset = -2

    _ = plt.annotate(
        f"Worst Day: {max_day}\n${max_price:.2f}",
        xy=(max_day, max_price),
        xytext=(max_day + x_offset, max_price + y_range * 0.08),
        arrowprops=dict(facecolor="red", shrink=0.05, width=1.5),
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
    )


def add_summary_statistics(min_day: int, min_price: float, max_day: int, max_price: float) -> None:
    """Add text box with summary statistics."""
    price_diff = max_price - min_price
    percent_diff = (price_diff / max_price) * 100 if max_price else 0.0

    text_info = (
        f"Potential savings: ${price_diff:.2f} ({percent_diff:.1f}%)\n"
        f"Best day: {min_day} (${min_price:.2f})\n"
        f"Worst day: {max_day} (${max_price:.2f})"
    )

    _ = plt.figtext(
        0.78,
        0.15,
        text_info,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8),
        fontsize=10,
    )


def add_titles(company_name: str, ticker_symbol: str, years: int, filter_days: FilterDays) -> None:
    """Add main title and subtitle to the chart."""
    _ = plt.suptitle(
        f"Average Low Price by Day of Month for {company_name} ({ticker_symbol})",
        fontsize=16,
        y=0.98,
    )

    filter_info = (
        f" - Analyzing only days: {', '.join(map(str, sorted(filter_days)))}"
        if filter_days
        else ""
    )
    _ = plt.title(f"{years}-Year Analysis{filter_info}", fontsize=13, pad=20)


def save_chart(ticker_symbol: str, years: int, filter_days: FilterDays) -> None:
    """Save chart to image file."""
    _ = plt.subplots_adjust(top=0.88, bottom=0.12, left=0.1, right=0.9)

    filter_suffix = "_filtered" if filter_days else ""
    filename = f"{ticker_symbol}_{years}yr{filter_suffix}_day_analysis.png"

    image_path = os.path.join(IMAGES_DIR, filename)
    plt.savefig(image_path, bbox_inches="tight", dpi=300)
    print(f"Chart saved as {image_path}")


def create_day_of_month_visualization(
    df: DataFrame,
    ticker_symbol: str,
    years: int,
    filter_days: FilterDays = None,
) -> None:
    """Create visualization for average price by day of month."""
    company_name = get_company_name(ticker_symbol)
    day_avg = calculate_day_averages(df, filter_days)

    if day_avg is None:
        return

    ax = setup_plot_figure()
    bars = create_bar_chart(ax, day_avg, filter_days)
    min_day, min_price, max_day, max_price = highlight_min_max_days(bars, day_avg)

    add_chart_styling(filter_days)
    _ = add_average_line(day_avg)
    _, _, y_range = adjust_y_limits()

    add_annotations(min_day, min_price, max_day, max_price, day_avg, y_range)
    add_summary_statistics(min_day, min_price, max_day, max_price)
    add_titles(company_name, ticker_symbol, years, filter_days)

    save_chart(ticker_symbol, years, filter_days)
    plt.show()


@dataclass(frozen=True)
class CliArgs:
    ticker: str | None
    years: int
    days: list[int] | None
    broker_days: bool


def parse_args(
    default_years: int = 10,
    default_broker_days: Sequence[int] | None = None,
) -> CliArgs:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze the best days to buy a stock.")

    _ = parser.add_argument("ticker", nargs="?", help="Stock ticker symbol")
    _ = parser.add_argument(
        "-y",
        "--years",
        type=int,
        default=default_years,
        help=f"Number of years to analyze (default: {default_years})",
    )
    sample_days = ", ".join(map(str, default_broker_days)) if default_broker_days else "1 15 28"
    _ = parser.add_argument(
        "-d",
        "--days",
        type=int,
        nargs="+",
        help=f"Specific days of month to analyze (e.g., {sample_days})",
    )
    _ = parser.add_argument(
        "--broker-days",
        action="store_true",
        help=(
            "Use default broker days "
            f"({', '.join(map(str, default_broker_days)) if default_broker_days else 'None defined'})"
        ),
    )

    namespace = parser.parse_args()
    ticker_arg = cast(str | None, namespace.ticker)
    years_arg = cast(int, namespace.years)
    days_arg = cast(list[int] | None, namespace.days)
    broker_days_flag = cast(bool, namespace.broker_days)

    return CliArgs(
        ticker=ticker_arg,
        years=years_arg,
        days=list(days_arg) if days_arg else None,
        broker_days=broker_days_flag,
    )


def get_cli_inputs(
    args: CliArgs,
    default_broker_days: Sequence[int],
) -> tuple[str, int, list[int] | None]:
    """Process command line inputs."""
    if args.ticker is None:
        raise ValueError("Ticker argument must be provided when running in CLI mode.")

    symbol = args.ticker.upper()
    years = args.years

    filter_days: list[int] | None = None
    if args.broker_days:
        filter_days = list(default_broker_days)
        print(f"Using default broker days: {filter_days}")
    elif args.days:
        filter_days = [day for day in args.days if 1 <= day <= 31]
        print(f"Analyzing only days: {filter_days}")

    return symbol, years, filter_days


def get_interactive_inputs(
    default_years: int,
    default_broker_days: Sequence[int],
) -> tuple[str, int, list[int] | None]:
    """Get inputs from user in interactive mode."""
    symbol = input("Enter stock ticker symbol: ").upper()

    years = get_years_input(default_years)
    filter_days = get_filter_days_input(default_broker_days)

    return symbol, years, filter_days


def get_years_input(default_years: int) -> int:
    """Get and validate years input from user."""
    while True:
        try:
            years_input = input(f"Enter number of years to analyze (default is {default_years}): ")
            if years_input == "":
                return default_years
            years = int(years_input)
            if years <= 0:
                print("Please enter a positive number of years.")
                continue
            return years
        except ValueError:
            print("Please enter a valid number.")


def get_filter_days_input(default_broker_days: Sequence[int]) -> list[int] | None:
    """Get and validate filter days input from user."""
    filter_option = input("Do you want to analyze specific days only? (y/n, default is n): ").lower()

    if filter_option != "y":
        return None

    print(f"Default broker allowed days: {default_broker_days}")
    custom_days = input("Enter your custom days separated by commas, or press Enter to use defaults: ")

    if not custom_days.strip():
        return list(default_broker_days)

    try:
        filter_days = [int(day.strip()) for day in custom_days.split(",")]
        filter_days = [day for day in filter_days if 1 <= day <= 31]
        if not filter_days:
            print("No valid days provided. Using default days.")
            return list(default_broker_days)
        return filter_days
    except ValueError:
        print("Invalid input. Using default days.")
        return list(default_broker_days)


def main() -> None:
    """Main function to run the analysis."""
    default_years = 10
    default_broker_days = [1, 4, 7, 10, 13, 16, 19, 22, 25]

    args = parse_args(default_years=default_years, default_broker_days=default_broker_days)

    if args.ticker:
        symbol, years, filter_days = get_cli_inputs(args, default_broker_days)
    else:
        symbol, years, filter_days = get_interactive_inputs(default_years, default_broker_days)

    get_best_buy_dates(symbol, years, filter_days)


if __name__ == "__main__":
    main()
