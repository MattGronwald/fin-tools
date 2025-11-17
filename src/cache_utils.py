"""Shared helpers for caching Yahoo Finance price history for CLI tools."""

from __future__ import annotations

import pickle
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


class CacheError(Exception):
    """Raised when cached price data cannot be retrieved."""


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_symbol(symbol: str) -> str:
    """Normalize ticker strings for cache file names."""

    return symbol.upper().replace("/", "-").replace(" ", "-")


def _cache_prefix(symbol: str, years: int) -> str:
    return f"{_safe_symbol(symbol)}_{years}"


def _cache_filename(symbol: str, years: int, stamp: str) -> Path:
    return CACHE_DIR / f"{_cache_prefix(symbol, years)}_{stamp}.pkl"


def _today_stamp() -> str:
    return date.today().isoformat()


def _purge_old_files(symbol: str, years: int, keep_stamp: str) -> None:
    prefix = f"{_cache_prefix(symbol, years)}_"
    suffix = f"_{keep_stamp}.pkl"
    for path in CACHE_DIR.glob(f"{prefix}*.pkl"):
        if not path.name.endswith(suffix):
            try:
                path.unlink()
            except OSError as exc:
                print(f"Unable to delete stale cache file {path.name}: {exc}")


def _load_cache(path: Path) -> pd.DataFrame | None:
    try:
        with path.open("rb") as handle:
            cached_obj = pickle.load(handle)
    except Exception as exc:
        print(f"Error loading cache {path.name}: {exc}")
        return None

    if not isinstance(cached_obj, pd.DataFrame):
        print(f"Cache file {path.name} contained unexpected data. Ignoring.")
        return None

    return cached_obj


def _save_cache(path: Path, data: pd.DataFrame) -> None:
    try:
        with path.open("wb") as handle:
            pickle.dump(data, handle)
    except OSError as exc:
        print(f"Error caching data to {path.name}: {exc}")


def download_price_history(symbol: str, years: int) -> pd.DataFrame:
    """Download OHLCV data (with adjusted closes) for the requested ticker."""

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=years)

    try:
        frame = yf.download(
            symbol,
            start=start_date,
            end=end_date + timedelta(days=1),
            auto_adjust=False,
            actions=False,
            progress=False,
        )
    except Exception as exc:  # pragma: no cover - network errors
        raise CacheError(f"Failed to download data for {symbol}: {exc}") from exc

    if frame.empty:
        raise CacheError(f"No price data returned for {symbol}.")

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.dropna(how="all")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    if frame.empty:
        raise CacheError(f"Downloaded data for {symbol} contains no usable rows.")
    if "Adj Close" not in frame:
        raise CacheError(f"Downloaded data for {symbol} is missing 'Adj Close'.")

    return frame


def get_price_history(symbol: str, years: int, *, allow_cache: bool = True) -> pd.DataFrame:
    """Return cached historical prices for a ticker, downloading when necessary."""

    stamp = _today_stamp()
    cache_path = _cache_filename(symbol, years, stamp)

    if allow_cache and cache_path.exists():
        cached = _load_cache(cache_path)
        if cached is not None:
            print(f"Loading cached data for {symbol} ({years} years)...")
            return cached

    if allow_cache:
        _purge_old_files(symbol, years, stamp)

    print(f"Downloading {years} years of data for {symbol}...")
    frame = download_price_history(symbol, years)
    _save_cache(cache_path, frame)
    _purge_old_files(symbol, years, stamp)
    print(f"Data cached for {symbol}")
    return frame


__all__ = ["CacheError", "CACHE_DIR", "get_price_history", "download_price_history"]
