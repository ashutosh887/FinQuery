#!/usr/bin/env python3
"""Validate internal consistency of financials.json and sectors.json."""

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "server" / "data"


def main():
    errors = []

    with open(DATA_DIR / "financials.json") as f:
        data = json.load(f)
    with open(DATA_DIR / "sectors.json") as f:
        sectors = json.load(f)

    required_tickers = {"AAPL", "MSFT", "GOOGL", "META", "NVDA", "TSLA", "F", "GM", "JPM", "BAC", "AMZN", "WMT"}
    required_years = {"2019", "2020", "2021", "2022", "2023"}

    # Check all tickers present
    missing_tickers = required_tickers - set(data.keys())
    if missing_tickers:
        errors.append(f"Missing tickers: {missing_tickers}")

    for ticker in data:
        # Check all years present
        missing_years = required_years - set(data[ticker].keys())
        if missing_years:
            errors.append(f"{ticker}: missing years {missing_years}")

        for year in data[ticker]:
            e = data[ticker][year]
            inc = e["income_statement"]
            bs = e["balance_sheet"]
            cf = e["cash_flow"]

            # --- Core invariants ---
            if inc["gross_profit"] != inc["revenue"] - inc["cogs"]:
                errors.append(f"{ticker}/{year}: gross_profit != revenue - cogs")
            if bs["total_equity"] != bs["total_assets"] - bs["total_liabilities"]:
                errors.append(f"{ticker}/{year}: total_equity != total_assets - total_liabilities")
            if cf["fcf"] != cf["operating_cf"] - cf["capex"]:
                errors.append(f"{ticker}/{year}: fcf != operating_cf - capex")
            if inc["operating_income"] >= inc["gross_profit"]:
                errors.append(f"{ticker}/{year}: operating_income >= gross_profit")

            # --- Required fields ---
            for field in ["revenue", "cogs", "gross_profit", "operating_income", "net_income", "eps"]:
                if field not in inc:
                    errors.append(f"{ticker}/{year}: missing income_statement.{field}")
            for field in ["total_assets", "total_liabilities", "total_equity", "cash", "total_debt"]:
                if field not in bs:
                    errors.append(f"{ticker}/{year}: missing balance_sheet.{field}")
            for field in ["operating_cf", "investing_cf", "financing_cf", "fcf", "capex"]:
                if field not in cf:
                    errors.append(f"{ticker}/{year}: missing cash_flow.{field}")

            # Price
            if "price" not in e:
                errors.append(f"{ticker}/{year}: missing price")
            else:
                for field in ["open", "close", "high", "low", "avg_price"]:
                    if field not in e["price"]:
                        errors.append(f"{ticker}/{year}: missing price.{field}")
            if "shares_outstanding" not in e:
                errors.append(f"{ticker}/{year}: missing shares_outstanding")

            # --- Ratios block ---
            if "ratios" not in e:
                errors.append(f"{ticker}/{year}: missing ratios block")
            else:
                r = e["ratios"]
                for field in ["pe_ratio", "pb_ratio", "ev_ebitda", "roe", "roa",
                              "debt_equity", "current_ratio", "gross_margin", "net_margin", "fcf_margin"]:
                    if field not in r:
                        errors.append(f"{ticker}/{year}: missing ratios.{field}")

                # Margin invariants (within 0.001)
                if r.get("gross_margin") is not None and inc["revenue"] > 0:
                    expected = inc["gross_profit"] / inc["revenue"]
                    if abs(r["gross_margin"] - expected) > 0.001:
                        errors.append(f"{ticker}/{year}: gross_margin {r['gross_margin']} != {expected:.4f}")
                if r.get("net_margin") is not None and inc["revenue"] > 0:
                    expected = inc["net_income"] / inc["revenue"]
                    if abs(r["net_margin"] - expected) > 0.001:
                        errors.append(f"{ticker}/{year}: net_margin {r['net_margin']} != {expected:.4f}")

    # --- Sectors ---
    required_sectors = {"technology", "automotive", "banking", "retail"}
    actual_sectors = {k for k in sectors if k != "_ticker_sectors"}
    missing_sectors = required_sectors - actual_sectors
    if missing_sectors:
        errors.append(f"Missing sectors: {missing_sectors}")
    for sector in actual_sectors:
        missing_syears = required_years - set(sectors[sector].keys())
        if missing_syears:
            errors.append(f"Sector {sector}: missing years {missing_syears}")

    # Ticker-sector mapping
    ticker_sectors = sectors.get("_ticker_sectors", {})
    for ticker in required_tickers:
        if ticker not in ticker_sectors:
            errors.append(f"Missing sector mapping for {ticker}")

    # Task-specific: TSLA neg FCF in 2019 and 2020
    for y in ["2019", "2020"]:
        if data["TSLA"][y]["cash_flow"]["fcf"] >= 0:
            errors.append(f"TSLA/{y}: FCF should be negative for task3")

    # META revenue decline 2021->2022
    if data["META"]["2022"]["income_statement"]["revenue"] >= data["META"]["2021"]["income_statement"]["revenue"]:
        errors.append("META: revenue should decline from 2021 to 2022")

    if errors:
        print(f"FAILED -- {len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        tickers = sorted(data.keys())
        print(f"PASSED -- {len(tickers)} tickers, {len(required_years)} years each")
        print(f"  Tickers: {', '.join(tickers)}")
        print(f"  Sectors: {', '.join(sorted(actual_sectors))}")
        print(f"  Ratios: all 10 fields present per ticker/year")
        print(f"  Invariants: gross_profit, equity, fcf, gross_margin, net_margin")
        sys.exit(0)


if __name__ == "__main__":
    main()
