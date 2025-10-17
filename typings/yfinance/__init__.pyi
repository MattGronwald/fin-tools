from collections.abc import Sequence

import pandas as pd


class Ticker:
    ticker: str
    info: dict[str, object]

    def __init__(self, ticker: str) -> None: ...

    def history(
        self,
        *,
        start: pd.Timestamp | None = ...,
        end: pd.Timestamp | None = ...,
        auto_adjust: bool = ...,
        actions: bool = ...,
        interval: str = ...,
        period: str | None = ...,
        timeout: int = ...
    ) -> pd.DataFrame: ...


def download(
    tickers: Sequence[str] | str,
    *,
    start: pd.Timestamp | None = ...,
    end: pd.Timestamp | None = ...,
    progress: bool = ...,
    auto_adjust: bool = ...
) -> pd.DataFrame: ...


__all__ = ["Ticker", "download"]
