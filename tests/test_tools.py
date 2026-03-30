"""Tests for individual tool modules."""

import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent.parent / "server" / "data"


@pytest.fixture(scope="module")
def data():
    with open(DATA_DIR / "financials.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def sectors():
    with open(DATA_DIR / "sectors.json") as f:
        return json.load(f)


def test_income_statement(data):
    from server.tools.income_statement import get

    result = get("AAPL", 2022, data)
    assert result["revenue"] == 394328
    assert result["net_income"] == 99803
    assert result["eps"] == 6.11


def test_income_statement_unknown_ticker(data):
    from server.tools.income_statement import get

    with pytest.raises(ValueError, match="Unknown ticker"):
        get("FAKE", 2022, data)


def test_balance_sheet(data):
    from server.tools.balance_sheet import get

    result = get("MSFT", 2023, data)
    assert result["total_assets"] == 411976
    assert result["cash"] == 34704


def test_cash_flow(data):
    from server.tools.cash_flow import get

    result = get("TSLA", 2020, data)
    assert result["fcf"] == -1503
    assert result["operating_cf"] == 5943


def test_price_history(data):
    from server.tools.price_history import get

    result = get("AAPL", [2020, 2021], data)
    assert "2020" in result
    assert "2021" in result
    assert result["2020"]["close"] == 132.69


def test_ratios(data):
    from server.tools.ratios import get

    result = get("AAPL", 2022, data)
    assert result["pe_ratio"] is not None
    assert result["roe"] is not None
    assert isinstance(result["pe_ratio"], float)


def test_sector_compare(data, sectors):
    from server.tools.sector_compare import get

    result = get("META", "ev_ebitda", 2023, data, sectors)
    assert result["sector"] == "technology"
    assert result["sector_median"] == 18.3
    assert result["value"] is not None
    assert 0.0 <= result["percentile"] <= 100.0
