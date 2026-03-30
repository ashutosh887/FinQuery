"""Tool: compare_to_sector — compares a company's metric to its sector median."""

from . import ratios as ratios_tool


def get(ticker: str, metric: str, year: int, data: dict, sectors: dict) -> dict:
    ticker = ticker.upper()
    year_str = str(year)

    ticker_sectors = sectors.get("_ticker_sectors", {})
    sector = ticker_sectors.get(ticker)
    if sector is None:
        raise ValueError(f"No sector mapping for ticker: {ticker}")

    if sector not in sectors:
        raise ValueError(f"Unknown sector: {sector}")
    if year_str not in sectors[sector]:
        raise ValueError(f"No sector data for {sector} in year {year}")

    sector_data = sectors[sector][year_str]
    if metric not in sector_data:
        raise ValueError(
            f"Unknown metric '{metric}' for sector {sector}. "
            f"Available: {list(sector_data.keys())}"
        )

    company_ratios = ratios_tool.get(ticker, year, data)
    company_value = company_ratios.get(metric)
    if company_value is None:
        raise ValueError(f"Cannot compute {metric} for {ticker} in {year}")

    sector_median = sector_data[metric]

    if sector_median != 0:
        percentile = round(50 + (company_value - sector_median) / abs(sector_median) * 50, 1)
    else:
        percentile = 50.0

    return {
        "ticker": ticker,
        "metric": metric,
        "year": year,
        "value": company_value,
        "sector": sector,
        "sector_median": sector_median,
        "percentile": max(0.0, min(100.0, percentile)),
    }
