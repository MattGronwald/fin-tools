import yfinance as yf
from datetime import datetime, timedelta
import calendar
import matplotlib.pyplot as plt
import os
import json
import pickle

# Create a directory for cached data
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

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

def get_best_buy_dates(ticker_symbol, years=10):
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
    create_day_of_month_visualization(df, ticker_symbol, years)

def create_day_of_month_visualization(df, ticker_symbol, years):
    # Get company name
    try:
        stock = yf.Ticker(ticker_symbol)
        company_name = stock.info.get('shortName', ticker_symbol)
    except:
        company_name = ticker_symbol  # Fallback if company name can't be retrieved

    # Calculate average price for each day of month across all years
    day_avg = df.groupby('Day')['Low'].mean()

    # Create figure with explicit space at the top for the title
    plt.figure(figsize=(12, 8))  # Increased height even more

    # Add the title at the very top of the figure, outside the axes
    plt.suptitle(f'Average Low Price by Day of Month for {company_name} ({ticker_symbol})',
                 fontsize=16, y=0.98)

    # Add a subtitle with years information
    plt.title(f'{years}-Year Analysis', fontsize=13, pad=20)

    # Create the main plot
    bars = plt.bar(day_avg.index, day_avg.values, color='skyblue')

    # Find the day with the lowest average price (best day to buy)
    min_day = day_avg.idxmin()
    min_price = day_avg.min()

    # Find the day with the highest average price (worst day to buy)
    max_day = day_avg.idxmax()
    max_price = day_avg.max()

    # Highlight the best day to buy in green
    bars[min_day-1].set_color('green')

    # Highlight the worst day to buy in red
    bars[max_day-1].set_color('red')

    # Add labels
    plt.xlabel('Day of Month')
    plt.ylabel('Average Price ($)')

    # Add a grid to make it easier to read
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Position annotations to avoid title
    # Calculate y bounds for better positioning
    y_min, y_max = plt.ylim()
    y_range = y_max - y_min

    # Add text annotation for the best day
    plt.annotate(f'Best Day: {min_day}\n${min_price:.2f}',
                 xy=(min_day, min_price),
                 xytext=(min_day, min_price - y_range*0.1),  # Position below bar
                 arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
                 ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))

    # Add text annotation for the worst day - with better positioning
    x_offset = 0
    if 10 <= max_day <= 20:
        x_offset = 3  # Move to the side if in the middle

    plt.annotate(f'Worst Day: {max_day}\n${max_price:.2f}',
                 xy=(max_day, max_price),
                 xytext=(max_day + x_offset, max_price + y_range*0.1),  # Position above bar
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
                 ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

    # Set the x-axis to show all days
    plt.xticks(range(1, 32))

    # Add a horizontal line at the average price
    avg_price = day_avg.mean()
    plt.axhline(y=avg_price, color='purple', linestyle='--', alpha=0.7)
    plt.text(31, avg_price, f'Avg: ${avg_price:.2f}', va='center')

    # Calculate potential savings
    price_diff = max_price - min_price
    percent_diff = (price_diff / max_price) * 100

    # Add a text box with summary statistics - moved to bottom right
    text_info = (f"Potential savings: ${price_diff:.2f} ({percent_diff:.1f}%)\n"
                f"Best day: {min_day} (${min_price:.2f})\n"
                f"Worst day: {max_day} (${max_price:.2f})")

    # Position the text box in bottom right
    plt.figtext(0.75, 0.02, text_info, bbox=dict(boxstyle="round,pad=0.5",
                                               fc="white", ec="black", alpha=0.8))

    # Set explicit subplot adjustments to ensure plenty of room for the title
    plt.subplots_adjust(top=0.85)  # Add much more space at the top

    plt.savefig(f'{ticker_symbol}_{years}yr_day_analysis.png')
    print(f"Chart saved as {ticker_symbol}_{years}yr_day_analysis.png")
    plt.show()

def main():
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

    get_best_buy_dates(symbol, years)

if __name__ == "__main__":
    main()
