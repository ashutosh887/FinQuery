"""Tool: get_cash_flow — returns cash flow data for a ticker and year."""


def get(ticker: str, year: int, data: dict) -> dict:
    ticker = ticker.upper()
    year_str = str(year)
    if ticker not in data:
        raise ValueError(f"Unknown ticker: {ticker}")
    if year_str not in data[ticker]:
        raise ValueError(f"No data for {ticker} in year {year}")
    return data[ticker][year_str]["cash_flow"]
