"""Tool: get_ratios — computes financial ratios from raw data for a ticker and year."""


def get(ticker: str, year: int, data: dict) -> dict:
    ticker = ticker.upper()
    year_str = str(year)
    if ticker not in data:
        raise ValueError(f"Unknown ticker: {ticker}")
    if year_str not in data[ticker]:
        raise ValueError(f"No data for {ticker} in year {year}")

    entry = data[ticker][year_str]
    inc = entry["income_statement"]
    bs = entry["balance_sheet"]
    price = entry["price"]
    shares = entry.get("shares_outstanding", 1)

    eps = inc["eps"]
    avg_price = price["avg_price"]
    market_cap = avg_price * shares
    total_equity = bs["total_equity"]
    total_assets = bs["total_assets"]
    total_debt = bs["total_debt"]
    cash = bs["cash"]
    operating_income = inc["operating_income"]

    pe_ratio = round(avg_price / eps, 2) if eps != 0 else None
    pb_ratio = round(market_cap / total_equity, 2) if total_equity != 0 else None

    ev = market_cap + total_debt - cash
    ev_ebitda = round(ev / operating_income, 2) if operating_income != 0 else None

    roe = round(inc["net_income"] / total_equity, 4) if total_equity != 0 else None
    roa = round(inc["net_income"] / total_assets, 4) if total_assets != 0 else None
    debt_equity = round(total_debt / total_equity, 4) if total_equity != 0 else None

    return {
        "pe_ratio": pe_ratio,
        "pb_ratio": pb_ratio,
        "ev_ebitda": ev_ebitda,
        "roe": roe,
        "roa": roa,
        "debt_equity": debt_equity,
    }
