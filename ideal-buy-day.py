import yfinance as yf
from datetime import datetime, timedelta
import calendar
import matplotlib.pyplot as plt
import os
import json
import pickle
import argparse
import sys

# Create directories for cached data and images
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

# Create directories if they don't exist
for directory in [CACHE_DIR, IMAGES_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_cached_data(ticker_symbol, years):
    """Retrieve data from cache if available and still valid (from today)"""
    today = datetime.now().date().isoformat()
    cache_file = os.path.join(CACHE_DIR, f"{ticker_symbol}_{years}_{today}.pkl")

    if os.path.exists(cache_file):
        print(f"Loading cached data for {ticker_symbol} ({years} years)...")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")

    # Clean up old cache files for this ticker
    for filename in os.listdir(CACHE_DIR):
        if filename.startswith(f"{ticker_symbol}_{years}_") and not filename.endswith(f"{today}.pkl"):
            try:
                os.remove(os.path.join(CACHE_DIR, filename))
            except:
                pass

    return None

def save_to_cache(ticker_symbol, years, data):
    """Save data to cache with today's date in the filename"""
    today = datetime.now().date().isoformat()
    cache_file = os.path.join(CACHE_DIR, f"{ticker_symbol}_{years}_{today}.pkl")

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data cached for {ticker_symbol}")
    except Exception as e:
        print(f"Error caching data: {e}")

def get_best_buy_dates(ticker_symbol, years=10, filter_days=None):
    # First check if we have cached data from today
    df = get_cached_data(ticker_symbol, years)

    if df is None:
        # Download specified years of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years)

        print(f"Downloading {years} years of data for {ticker_symbol}...")
        # Get stock data
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            print(f"No data found for ticker {ticker_symbol}")
            return

        # Cache the data
        save_to_cache(ticker_symbol, years, df)

    print(f"Analyzing {len(df)} days of data...")

    # Add month and day columns
    df['Month'] = df.index.to_series().dt.month
    df['Day'] = df.index.to_series().dt.day

    # Group by month and day to find average price for each calendar day
    daily_avg = df.groupby(['Month', 'Day'])['Low'].mean()

    # Find the day with lowest average price for each month
    best_days = {}
    for month in range(1,13):
        try:
            # Get data for this month
            month_data = daily_avg.xs(month, level='Month')

            # If filter days is provided, only consider those days
            if filter_days:
                # First check if we have any filter days in this month's data
                valid_days = [day for day in filter_days if day in month_data.index]
                if valid_days:
                    # Filter the data to only include allowed days
                    month_data = month_data.loc[valid_days]
                else:
                    # If none of the filter days are available for this month
                    best_days[month] = "No matching days"
                    continue

            # Find the day with the lowest average price
            best_day = month_data.idxmin()
            best_days[month] = best_day
        except KeyError:
            # Handle case where there's no data for a particular month
            best_days[month] = "No data"

    # Format and print results
    for month in best_days:
        month_name = calendar.month_name[month]
        print(f"Best day to buy in {month_name}: {best_days[month]}")

    # Create visualization for average price by day of month (across all months)
    create_day_of_month_visualization(df, ticker_symbol, years, filter_days)

def create_day_of_month_visualization(df, ticker_symbol, years, filter_days=None):
    # Get company name
    try:
        stock = yf.Ticker(ticker_symbol)
        company_name = stock.info.get('shortName', ticker_symbol)
    except:
        company_name = ticker_symbol  # Fallback if company name can't be retrieved

    # Calculate average price for each day of month across all years
    day_avg = df.groupby('Day')['Low'].mean()

    # If filter days is provided, only show those days
    if filter_days:
        # Filter the data to only include allowed days
        day_avg = day_avg[day_avg.index.isin(filter_days)]

        if day_avg.empty:
            print("No data available for the specified filter days.")
            return

    # Create figure with explicit space at the top for the title
    plt.figure(figsize=(12, 9))  # Increased height even more for better spacing

    # Create the main plot with adjusted position to leave plenty of space for title
    ax = plt.subplot(111)

    # Calculate optimal bar width based on number of days being shown
    # This makes bars wider when fewer days are displayed
    if filter_days:
        width = 1.1  # Wider bars for filtered days
    else:
        width = 0.8  # Default width

    bars = ax.bar(day_avg.index, day_avg.values, color='skyblue', width=width)

    # Find the day with the lowest average price (best day to buy)
    min_day = day_avg.idxmin()
    min_price = day_avg.min()

    # Find the day with the highest average price (worst day to buy)
    max_day = day_avg.idxmax()
    max_price = day_avg.max()

    # Highlight the best day to buy in green
    min_idx = day_avg.index.get_loc(min_day)
    bars[min_idx].set_color('green')

    # Highlight the worst day to buy in red
    max_idx = day_avg.index.get_loc(max_day)
    bars[max_idx].set_color('red')

    # Add labels
    plt.xlabel('Day of Month', fontsize=12, labelpad=10)
    plt.ylabel('Average Price ($)', fontsize=12)

    # Add a grid to make it easier to read
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Set the x-axis to show only the days we're analyzing
    if filter_days:
        plt.xticks(sorted(filter_days), fontsize=11)
    else:
        plt.xticks(range(1, 32), fontsize=11)

    # Add a horizontal line at the average price
    avg_price = day_avg.mean()
    plt.axhline(y=avg_price, color='purple', linestyle='--', alpha=0.7)

    # Add average price text with a white background for better visibility
    # Position it at the right side but with an offset to avoid x-axis labels
    avg_text = plt.text(
        max(day_avg.index) * 0.9,  # Move it left a bit from the end
        avg_price * 1.015,  # Position it slightly above the line
        f'Avg: ${avg_price:.2f}',
        va='center',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='purple', boxstyle='round,pad=0.2')
    )

    # Position annotations to avoid title - increase vertical spacing
    # Calculate y bounds for better positioning
    y_min, y_max = plt.ylim()
    y_range = y_max - y_min

    # Extend y-axis limits to make room for annotations
    plt.ylim(y_min - y_range * 0.1, y_max + y_range * 0.15)

    # Recalculate y bounds after adjustment
    y_min, y_max = plt.ylim()
    y_range = y_max - y_min

    # Add text annotation for the best day
    plt.annotate(f'Best Day: {min_day}\n${min_price:.2f}',
                 xy=(min_day, min_price),
                 xytext=(min_day, min_price - y_range*0.10),
                 arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
                 ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))

    # Add text annotation for the worst day - with better positioning
    x_offset = 0
    if len(day_avg) > 3:
        # If we have several days, ensure the annotation doesn't overlap
        if max_day in [day_avg.index[i] for i in range(len(day_avg)//2-1, len(day_avg)//2+2)]:
            x_offset = -2  # Move to the left if in the middle

    plt.annotate(f'Worst Day: {max_day}\n${max_price:.2f}',
                 xy=(max_day, max_price),
                 xytext=(max_day + x_offset, max_price + y_range*0.08),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
                 ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

    # Calculate potential savings
    price_diff = max_price - min_price
    percent_diff = (price_diff / max_price) * 100

    # Add a text box with summary statistics
    # Position it in bottom right, but higher up to avoid x-axis labels
    text_info = (f"Potential savings: ${price_diff:.2f} ({percent_diff:.1f}%)\n"
                f"Best day: {min_day} (${min_price:.2f})\n"
                f"Worst day: {max_day} (${max_price:.2f})")

    # Create a separate text box that's positioned better
    plt.figtext(0.78, 0.15, text_info,
               bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8),
               fontsize=10)

    # Now add the titles last to ensure they're on top
    # Add the title at the very top of the figure, outside the axes
    plt.suptitle(f'Average Low Price by Day of Month for {company_name} ({ticker_symbol})',
                fontsize=16, y=0.98)

    # Add a subtitle with years information and filter info if applicable
    filter_info = f" - Analyzing only days: {', '.join(map(str, sorted(filter_days)))}" if filter_days else ""
    plt.title(f'{years}-Year Analysis{filter_info}', fontsize=13, pad=20)

    # Set explicit subplot adjustments to ensure plenty of room for everything
    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.1, right=0.9)

    # Create filename with filter indication if used
    filter_suffix = "_filtered" if filter_days else ""
    filename = f'{ticker_symbol}_{years}yr{filter_suffix}_day_analysis.png'
    
    # Save image to the images directory
    image_path = os.path.join(IMAGES_DIR, filename)
    plt.savefig(image_path, bbox_inches='tight', dpi=300)  # Higher resolution and tight bounding box
    print(f"Chart saved as {image_path}")
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze the best days to buy a stock.')

    parser.add_argument('ticker', nargs='?', help='Stock ticker symbol')
    parser.add_argument('-y', '--years', type=int, default=10,
                        help='Number of years to analyze (default: 10)')
    parser.add_argument('-d', '--days', type=int, nargs='+',
                        help='Specific days of month to analyze (e.g., 1 4 7 10 13 16 19 22 25)')
    parser.add_argument('--broker-days', action='store_true',
                        help='Use default broker days (1, 4, 7, 10, 13, 16, 19, 22, 25)')

    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    # Default broker allowed days
    default_broker_days = [1, 4, 7, 10, 13, 16, 19, 22, 25]

    # Check if we're running with command line arguments
    if args.ticker:
        symbol = args.ticker.upper()
        years = args.years

        # Determine which days to filter by
        filter_days = None
        if args.broker_days:
            filter_days = default_broker_days
            print(f"Using default broker days: {filter_days}")
        elif args.days:
            filter_days = [day for day in args.days if 1 <= day <= 31]
            print(f"Analyzing only days: {filter_days}")

        get_best_buy_dates(symbol, years, filter_days)
        return

    # If no command line arguments, use interactive mode
    symbol = input("Enter stock ticker symbol: ").upper()

    # Get the number of years to analyze with error handling
    while True:
        try:
            years_input = input("Enter number of years to analyze (default is 10): ")
            if years_input == "":
                years = 10
                break
            years = int(years_input)
            if years <= 0:
                print("Please enter a positive number of years.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    # Ask if user wants to filter to specific days
    filter_option = input("Do you want to analyze specific days only? (y/n, default is n): ").lower()

    filter_days = None
    if filter_option == 'y':
        print(f"Default broker allowed days: {default_broker_days}")
        custom_days = input("Enter your custom days separated by commas, or press Enter to use defaults: ")

        if custom_days.strip():
            try:
                filter_days = [int(day.strip()) for day in custom_days.split(',')]
                # Validate days are in valid range
                filter_days = [day for day in filter_days if 1 <= day <= 31]
                if not filter_days:
                    print("No valid days provided. Using default days.")
                    filter_days = default_broker_days
            except ValueError:
                print("Invalid input. Using default days.")
                filter_days = default_broker_days
        else:
            filter_days = default_broker_days

        print(f"Analyzing only days: {filter_days}")

    get_best_buy_dates(symbol, years, filter_days)

if __name__ == "__main__":
    main()
