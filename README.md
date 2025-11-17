# fin-tools

A small collection of Python CLI utilities for retail investors. Each script focuses on
a narrowly scoped workflow so you can mix and match them with your own analysis.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The requirements file pins the runtime dependencies (NumPy, pandas, yfinance, and
matplotlib) as well as the pandas typing stubs for editors.

## Available Tools

### `ideal-buy-day.py`

Find historically attractive days of the month to accumulate a given stock.

```
python ideal-buy-day.py AAPL -y 5
```

Key features:

- Downloads historical OHLC data through Yahoo Finance.
- Caches responses per ticker/year combination so subsequent runs are instant.
- Prints the best historical buying day for each month and generates a chart in
  the `images/` folder.
- Supports interactive prompts as well as optional day-of-month filters.

### `trailing_stop_analyzer.py`

Estimate a reasonable trailing stop percentage that survives routine pullbacks
while still reacting to genuine trend breaks.

```
python trailing_stop_analyzer.py -s MSFT --years 3 --quantile 0.95
```

Highlights:

- Pulls adjusted close prices via `yfinance` for the last _N_ years.
- Optionally smooths the series before locating swing highs/lows.
- Measures pullbacks after every swing high, filters tiny moves and crash-like
  events, and reports descriptive statistics.
- Suggests a trailing stop based on a configurable percentile (defaults to the
  90th) and rounds to sensible 0.5% increments.
- Handles invalid tickers, missing data, and limited histories gracefully.

Use `python trailing_stop_analyzer.py --help` to see the full set of options,
including crash filters, minimum pullbacks, and smoothing windows.

## Development Notes

- Run `python ideal-buy-day.py` or `python trailing_stop_analyzer.py` directly for
  quick smoke-tests.
- Generated artifacts live under `cache/` and `images/`; they are ignored by
  default and can be safely deleted between runs.
