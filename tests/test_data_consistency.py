"""Tests that financials.json data is internally consistent."""

import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent.parent / "server" / "data"


@pytest.fixture(scope="module")
def financials():
    with open(DATA_DIR / "financials.json") as f:
        return json.load(f)


def test_all_tickers_have_required_years(financials):
    for ticker, years in financials.items():
        assert len(years) >= 1, f"{ticker} has no year data"
        for year, data in years.items():
            assert "income_statement" in data, f"{ticker}/{year} missing income_statement"
            assert "balance_sheet" in data, f"{ticker}/{year} missing balance_sheet"
            assert "cash_flow" in data, f"{ticker}/{year} missing cash_flow"
            assert "price" in data, f"{ticker}/{year} missing price"


def test_gross_profit_equals_revenue_minus_cogs(financials):
    for ticker, years in financials.items():
        for year, data in years.items():
            inc = data["income_statement"]
            expected = inc["revenue"] - inc["cogs"]
            assert inc["gross_profit"] == expected, (
                f"{ticker}/{year}: gross_profit {inc['gross_profit']} != "
                f"revenue {inc['revenue']} - cogs {inc['cogs']} = {expected}"
            )


def test_fcf_equals_operating_cf_minus_capex(financials):
    for ticker, years in financials.items():
        for year, data in years.items():
            cf = data["cash_flow"]
            expected = cf["operating_cf"] - cf["capex"]
            assert cf["fcf"] == expected, (
                f"{ticker}/{year}: fcf {cf['fcf']} != "
                f"operating_cf {cf['operating_cf']} - capex {cf['capex']} = {expected}"
            )


def test_total_equity_equals_assets_minus_liabilities(financials):
    for ticker, years in financials.items():
        for year, data in years.items():
            bs = data["balance_sheet"]
            expected = bs["total_assets"] - bs["total_liabilities"]
            assert bs["total_equity"] == expected, (
                f"{ticker}/{year}: total_equity {bs['total_equity']} != "
                f"total_assets {bs['total_assets']} - total_liabilities "
                f"{bs['total_liabilities']} = {expected}"
            )
