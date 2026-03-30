"""Tool: get_price_history — returns annual price data for a ticker across multiple years."""

from typing import List


def get(ticker: str, years: List[int], data: dict) -> dict:
    ticker = ticker.upper()
    if ticker not in data:
        raise ValueError(f"Unknown ticker: {ticker}")
    result = {}
    for year in years:
        year_str = str(year)
        if year_str not in data[ticker]:
            raise ValueError(f"No data for {ticker} in year {year}")
        if "price" not in data[ticker][year_str]:
            raise ValueError(f"No price data for {ticker} in year {year}")
        result[year_str] = data[ticker][year_str]["price"]
    return result
