import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import calendar
import matplotlib.pyplot as plt
import os
import pickle
import argparse

def create_required_directories():
    """Create necessary directories for cache and images."""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

    for directory in [cache_dir, images_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    return cache_dir, images_dir

CACHE_DIR, IMAGES_DIR = create_required_directories()

def get_cached_data(ticker_symbol, years):
    """Retrieve data from cache if available and still valid (from today)."""
    today = datetime.now().date().isoformat()
    cache_file = os.path.join(CACHE_DIR, f"{ticker_symbol}_{years}_{today}.pkl")

    if os.path.exists(cache_file):
        print(f"Loading cached data for {ticker_symbol} ({years} years)...")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")

    clean_old_cache_files(ticker_symbol, years, today)
    return None

def clean_old_cache_files(ticker_symbol, years, today):
    """Remove outdated cache files for the ticker."""
    for filename in os.listdir(CACHE_DIR):
        if filename.startswith(f"{ticker_symbol}_{years}_") and not filename.endswith(f"{today}.pkl"):
            try:
                os.remove(os.path.join(CACHE_DIR, filename))
            except:
                pass

def save_to_cache(ticker_symbol, years, data):
    """Save data to cache with today's date in the filename."""
    today = datetime.now().date().isoformat()
    cache_file = os.path.join(CACHE_DIR, f"{ticker_symbol}_{years}_{today}.pkl")

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data cached for {ticker_symbol}")
    except Exception as e:
        print(f"Error caching data: {e}")

def download_stock_data(ticker_symbol, years):
    """Download historical stock data for the given ticker and years."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years)

    print(f"Downloading {years} years of data for {ticker_symbol}...")
    stock = yf.Ticker(ticker_symbol)
    df = None

    try:
        df = stock.history(start=start_date, end=end_date)
    except Exception as exc:
        print(f"Primary Yahoo Finance request failed: {exc}")

    if df is None or df.empty:
        print("Retrying with yf.download fallback...")
        try:
            df = yf.download(
                ticker_symbol,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False
            )
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df = df.xs(ticker_symbol, axis=1, level=-1)
                except (KeyError, ValueError):
                    df.columns = df.columns.get_level_values(0)
        except Exception as fallback_exc:
            print(f"Fallback download failed: {fallback_exc}")
            df = None

    if df is None or df.empty:
        print(f"No data found for ticker {ticker_symbol}")
        return None

    return df

def add_date_columns(df):
    """Add month and day columns to the dataframe."""
    df['Month'] = df.index.to_series().dt.month
    df['Day'] = df.index.to_series().dt.day
    return df

def find_best_days_by_month(daily_avg, filter_days):
    """Find the day with lowest average price for each month."""
    best_days = {}
    for month in range(1, 13):
        try:
            month_data = daily_avg.xs(month, level='Month')

            if filter_days:
                valid_days = [day for day in filter_days if day in month_data.index]
                if valid_days:
                    month_data = month_data.loc[valid_days]
                else:
                    best_days[month] = "No matching days"
                    continue

            best_day = month_data.idxmin()
            best_days[month] = best_day
        except KeyError:
            best_days[month] = "No data"

    return best_days

def print_best_days(best_days):
    """Format and print the best days for each month."""
    for month in best_days:
        month_name = calendar.month_name[month]
        print(f"Best day to buy in {month_name}: {best_days[month]}")

def get_best_buy_dates(ticker_symbol, years=10, filter_days=None):
    """Analyze stock data to find best buying dates."""
    df = get_cached_data(ticker_symbol, years)

    if df is None:
        df = download_stock_data(ticker_symbol, years)
        if df is None:
            return
        save_to_cache(ticker_symbol, years, df)

    print(f"Analyzing {len(df)} days of data...")
    df = add_date_columns(df)

    # Group by month and day to find average price for each calendar day
    daily_avg = df.groupby(['Month', 'Day'])['Low'].mean()

    best_days = find_best_days_by_month(daily_avg, filter_days)
    print_best_days(best_days)

    # Create visualization
    create_day_of_month_visualization(df, ticker_symbol, years, filter_days)

def get_company_name(ticker_symbol):
    """Retrieve company name for the given ticker."""
    try:
        stock = yf.Ticker(ticker_symbol)
        return stock.info.get('shortName', ticker_symbol)
    except:
        return ticker_symbol  # Fallback if company name can't be retrieved

def calculate_day_averages(df, filter_days):
    """Calculate average price for each day of month across all years."""
    day_avg = df.groupby('Day')['Low'].mean()

    if filter_days:
        day_avg = day_avg[day_avg.index.isin(filter_days)]
        if day_avg.empty:
            print("No data available for the specified filter days.")
            return None

    return day_avg

def setup_plot_figure():
    """Create and configure the plot figure."""
    plt.figure(figsize=(12, 9))
    return plt.subplot(111)

def create_bar_chart(ax, day_avg, filter_days):
    """Create the bar chart with appropriate styling."""
    width = 1.1 if filter_days else 0.8
    return ax.bar(day_avg.index, day_avg.values, color='skyblue', width=width)

def highlight_min_max_days(bars, day_avg):
    """Highlight the best and worst days on the chart."""
    min_day = day_avg.idxmin()
    min_price = day_avg.min()
    max_day = day_avg.idxmax()
    max_price = day_avg.max()

    # Highlight best day in green
    min_idx = day_avg.index.get_loc(min_day)
    bars[min_idx].set_color('green')

    # Highlight worst day in red
    max_idx = day_avg.index.get_loc(max_day)
    bars[max_idx].set_color('red')

    return min_day, min_price, max_day, max_price

def add_chart_styling(filter_days):
    """Add labels, grid, and configure axes."""
    plt.xlabel('Day of Month', fontsize=12, labelpad=10)
    plt.ylabel('Average Price ($)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Set x-axis ticks
    if filter_days:
        plt.xticks(sorted(filter_days), fontsize=11)
    else:
        plt.xticks(range(1, 32), fontsize=11)

def add_average_line(day_avg):
    """Add horizontal line representing the average price."""
    avg_price = day_avg.mean()
    plt.axhline(y=avg_price, color='purple', linestyle='--', alpha=0.7)

    # Add text label for average
    plt.text(
        max(day_avg.index) * 0.9,
        avg_price * 1.015,
        f'Avg: ${avg_price:.2f}',
        va='center',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='purple', boxstyle='round,pad=0.2')
    )

    return avg_price

def adjust_y_limits():
    """Adjust y-axis limits to make room for annotations."""
    y_min, y_max = plt.ylim()
    y_range = y_max - y_min
    plt.ylim(y_min - y_range * 0.1, y_max + y_range * 0.15)

    # Return updated limits
    y_min, y_max = plt.ylim()
    return y_min, y_max, y_max - y_min

def add_annotations(min_day, min_price, max_day, max_price, day_avg, y_range):
    """Add text annotations for best and worst days."""
    # Add annotation for best day
    plt.annotate(
        f'Best Day: {min_day}\n${min_price:.2f}',
        xy=(min_day, min_price),
        xytext=(min_day, min_price - y_range*0.10),
        arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
        ha='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8)
    )

    # Add annotation for worst day
    x_offset = 0
    if len(day_avg) > 3:
        if max_day in [day_avg.index[i] for i in range(len(day_avg)//2-1, len(day_avg)//2+2)]:
            x_offset = -2  # Move to the left if in the middle

    plt.annotate(
        f'Worst Day: {max_day}\n${max_price:.2f}',
        xy=(max_day, max_price),
        xytext=(max_day + x_offset, max_price + y_range*0.08),
        arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
        ha='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8)
    )

def add_summary_statistics(min_day, min_price, max_day, max_price):
    """Add text box with summary statistics."""
    price_diff = max_price - min_price
    percent_diff = (price_diff / max_price) * 100

    text_info = (
        f"Potential savings: ${price_diff:.2f} ({percent_diff:.1f}%)\n"
        f"Best day: {min_day} (${min_price:.2f})\n"
        f"Worst day: {max_day} (${max_price:.2f})"
    )

    plt.figtext(
        0.78, 0.15, text_info,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8),
        fontsize=10
    )

def add_titles(company_name, ticker_symbol, years, filter_days):
    """Add main title and subtitle to the chart."""
    plt.suptitle(
        f'Average Low Price by Day of Month for {company_name} ({ticker_symbol})',
        fontsize=16, y=0.98
    )

    filter_info = f" - Analyzing only days: {', '.join(map(str, sorted(filter_days)))}" if filter_days else ""
    plt.title(f'{years}-Year Analysis{filter_info}', fontsize=13, pad=20)

def save_chart(ticker_symbol, years, filter_days):
    """Save chart to image file."""
    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.1, right=0.9)

    filter_suffix = "_filtered" if filter_days else ""
    filename = f'{ticker_symbol}_{years}yr{filter_suffix}_day_analysis.png'

    image_path = os.path.join(IMAGES_DIR, filename)
    plt.savefig(image_path, bbox_inches='tight', dpi=300)
    print(f"Chart saved as {image_path}")

def create_day_of_month_visualization(df, ticker_symbol, years, filter_days=None):
    """Create visualization for average price by day of month."""
    company_name = get_company_name(ticker_symbol)
    day_avg = calculate_day_averages(df, filter_days)

    if day_avg is None:
        return

    ax = setup_plot_figure()
    bars = create_bar_chart(ax, day_avg, filter_days)
    min_day, min_price, max_day, max_price = highlight_min_max_days(bars, day_avg)

    add_chart_styling(filter_days)
    add_average_line(day_avg)
    _, _, y_range = adjust_y_limits()

    add_annotations(min_day, min_price, max_day, max_price, day_avg, y_range)
    add_summary_statistics(min_day, min_price, max_day, max_price)
    add_titles(company_name, ticker_symbol, years, filter_days)

    save_chart(ticker_symbol, years, filter_days)
    plt.show()

def parse_args(default_years=10, default_broker_days=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze the best days to buy a stock.')

    parser.add_argument('ticker', nargs='?', help='Stock ticker symbol')
    parser.add_argument('-y', '--years', type=int, default=default_years,
                        help=f'Number of years to analyze (default: {default_years})')
    parser.add_argument('-d', '--days', type=int, nargs='+',
                        help=f'Specific days of month to analyze (e.g., {", ".join(map(str, default_broker_days)) if default_broker_days else "1 15 28"})')
    parser.add_argument('--broker-days', action='store_true',
                        help=f'Use default broker days ({", ".join(map(str, default_broker_days)) if default_broker_days else "None defined"})')

    return parser.parse_args()

def get_cli_inputs(args, DEFAULT_BROKER_DAYS):
    """Process command line inputs."""
    symbol = args.ticker.upper()
    years = args.years

    # Determine which days to filter by
    filter_days = None
    if args.broker_days:
        filter_days = DEFAULT_BROKER_DAYS
        print(f"Using default broker days: {filter_days}")
    elif args.days:
        filter_days = [day for day in args.days if 1 <= day <= 31]
        print(f"Analyzing only days: {filter_days}")

    return symbol, years, filter_days

def get_interactive_inputs(DEFAULT_YEARS, DEFAULT_BROKER_DAYS):
    """Get inputs from user in interactive mode."""
    symbol = input("Enter stock ticker symbol: ").upper()

    # Get number of years
    years = get_years_input(DEFAULT_YEARS)

    # Get filter days if requested
    filter_days = get_filter_days_input(DEFAULT_BROKER_DAYS)

    return symbol, years, filter_days

def get_years_input(DEFAULT_YEARS):
    """Get and validate years input from user."""
    while True:
        try:
            years_input = input(f"Enter number of years to analyze (default is {DEFAULT_YEARS}): ")
            if years_input == "":
                return DEFAULT_YEARS
            years = int(years_input)
            if years <= 0:
                print("Please enter a positive number of years.")
                continue
            return years
        except ValueError:
            print("Please enter a valid number.")

def get_filter_days_input(DEFAULT_BROKER_DAYS):
    """Get and validate filter days input from user."""
    filter_option = input("Do you want to analyze specific days only? (y/n, default is n): ").lower()

    if filter_option != 'y':
        return None

    print(f"Default broker allowed days: {DEFAULT_BROKER_DAYS}")
    custom_days = input("Enter your custom days separated by commas, or press Enter to use defaults: ")

    if not custom_days.strip():
        return DEFAULT_BROKER_DAYS

    try:
        filter_days = [int(day.strip()) for day in custom_days.split(',')]
        # Validate days are in valid range
        filter_days = [day for day in filter_days if 1 <= day <= 31]
        if not filter_days:
            print("No valid days provided. Using default days.")
            return DEFAULT_BROKER_DAYS
        return filter_days
    except ValueError:
        print("Invalid input. Using default days.")
        return DEFAULT_BROKER_DAYS

def main():
    """Main function to run the analysis."""
    # Define default values
    DEFAULT_YEARS = 10
    DEFAULT_BROKER_DAYS = [1, 4, 7, 10, 13, 16, 19, 22, 25]

    # Parse command line arguments
    args = parse_args(default_years=DEFAULT_YEARS, default_broker_days=DEFAULT_BROKER_DAYS)

    # Check if running with command line arguments
    if args.ticker:
        symbol, years, filter_days = get_cli_inputs(args, DEFAULT_BROKER_DAYS)
    else:
        # Use interactive mode
        symbol, years, filter_days = get_interactive_inputs(DEFAULT_YEARS, DEFAULT_BROKER_DAYS)

    get_best_buy_dates(symbol, years, filter_days)

if __name__ == "__main__":
    main()
