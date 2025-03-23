import yfinance as yf
from datetime import datetime, timedelta
import calendar
import matplotlib.pyplot as plt

def get_best_buy_dates(ticker_symbol):
    # Download 10 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)

    print(f"Downloading data for {ticker_symbol}...")
    # Get stock data
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(start=start_date, end=end_date)

    if df.empty:
        print(f"No data found for ticker {ticker_symbol}")
        return

    print(f"Analyzing {len(df)} days of data...")

    # Add month and day columns
    df['Month'] = df.index.month
    df['Day'] = df.index.day

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
    create_day_of_month_visualization(df, ticker_symbol)

def create_day_of_month_visualization(df, ticker_symbol):
    # Calculate average price for each day of month across all years
    day_avg = df.groupby('Day')['Low'].mean()

    # Create a bar chart
    plt.figure(figsize=(12, 6))
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

    # Add labels and title
    plt.xlabel('Day of Month')
    plt.ylabel('Average Price ($)')
    plt.title(f'Average Low Price by Day of Month for {ticker_symbol} (10-Year Analysis)')

    # Add a grid to make it easier to read
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add text annotation for the best day
    plt.annotate(f'Best Day: {min_day}\n${min_price:.2f}',
                 xy=(min_day, min_price),
                 xytext=(min_day, min_price*0.95),  # Position below bar
                 arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
                 ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))

    # Add text annotation for the worst day
    plt.annotate(f'Worst Day: {max_day}\n${max_price:.2f}',
                 xy=(max_day, max_price),
                 xytext=(max_day, max_price*1.05),  # Position above bar
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

    # Add a text box with summary statistics
    text_info = (f"Potential savings: ${price_diff:.2f} ({percent_diff:.1f}%)\n"
                f"Best day: {min_day} (${min_price:.2f})\n"
                f"Worst day: {max_day} (${max_price:.2f})")

    plt.figtext(0.15, 0.02, text_info, bbox=dict(boxstyle="round,pad=0.5",
                                                fc="white", ec="black", alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{ticker_symbol}_day_analysis.png')
    print(f"Chart saved as {ticker_symbol}_day_analysis.png")
    plt.show()

def main():
    symbol = input("Enter stock ticker symbol: ").upper()
    get_best_buy_dates(symbol)

if __name__ == "__main__":
    main()
