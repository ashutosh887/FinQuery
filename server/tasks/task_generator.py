"""Procedural task generators for FinQuery.

Each generator produces randomized TaskInstance objects with unique questions
and deterministically computed ground truth from financials.json data.

Combinatorial space:
  Easy:   12 tickers x 6 years x 9 metrics = 648 unique tasks
  Medium: sector/company combos x 5 metrics x 6 years = hundreds
  Hard:   C(12,3) x year windows x 4 anomaly patterns = thousands
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


COMPANY_NAMES: Dict[str, str] = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google (Alphabet)",
    "META": "Meta Platforms",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "F": "Ford",
    "GM": "General Motors",
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "AMZN": "Amazon",
    "WMT": "Walmart",
}

SECTOR_COMPANIES: Dict[str, List[str]] = {
    "technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
    "automotive": ["TSLA", "F", "GM"],
    "banking": ["JPM", "BAC"],
    "retail": ["AMZN", "WMT"],
}

ALL_YEARS: List[int] = [2019, 2020, 2021, 2022, 2023, 2024]


@dataclass
class TaskInstance:
    """A single generated task with its ground truth and metadata."""

    task_id: str
    task_description: str
    ground_truth: Dict[str, Any]
    relevant_tickers: List[str]
    relevant_years: List[int]
    difficulty: str
    max_steps: int
    required_fetches: List[str]
    min_fetches: int
    expected_intermediates: List[float]
    grader_id: str  # "task1", "task2", "task3"


# ---------------------------------------------------------------------------
# Easy Task Generator
# ---------------------------------------------------------------------------

def _income(d: dict, t: str, y: int) -> dict:
    return d[t][str(y)]["income_statement"]


def _balance(d: dict, t: str, y: int) -> dict:
    return d[t][str(y)]["balance_sheet"]


def _cashflow(d: dict, t: str, y: int) -> dict:
    return d[t][str(y)]["cash_flow"]


def _price(d: dict, t: str, y: int) -> dict:
    return d[t][str(y)]["price"]


EASY_METRICS = [
    {
        "id": "net_profit_margin",
        "display": "net profit margin",
        "unit": "percentage",
        "fetch_types": ["income_statement"],
        "compute": lambda d, t, y: _income(d, t, y)["net_income"] / _income(d, t, y)["revenue"] * 100,
    },
    {
        "id": "gross_margin",
        "display": "gross margin",
        "unit": "percentage",
        "fetch_types": ["income_statement"],
        "compute": lambda d, t, y: _income(d, t, y)["gross_profit"] / _income(d, t, y)["revenue"] * 100,
    },
    {
        "id": "operating_margin",
        "display": "operating margin",
        "unit": "percentage",
        "fetch_types": ["income_statement"],
        "compute": lambda d, t, y: _income(d, t, y)["operating_income"] / _income(d, t, y)["revenue"] * 100,
    },
    {
        "id": "debt_to_equity",
        "display": "debt-to-equity ratio",
        "unit": "ratio",
        "fetch_types": ["balance_sheet"],
        "compute": lambda d, t, y: _balance(d, t, y)["total_debt"] / _balance(d, t, y)["total_equity"],
    },
    {
        "id": "return_on_assets",
        "display": "return on assets (ROA)",
        "unit": "percentage",
        "fetch_types": ["income_statement", "balance_sheet"],
        "compute": lambda d, t, y: _income(d, t, y)["net_income"] / _balance(d, t, y)["total_assets"] * 100,
    },
    {
        "id": "return_on_equity",
        "display": "return on equity (ROE)",
        "unit": "percentage",
        "fetch_types": ["income_statement", "balance_sheet"],
        "compute": lambda d, t, y: _income(d, t, y)["net_income"] / _balance(d, t, y)["total_equity"] * 100,
    },
    {
        "id": "fcf_margin",
        "display": "free cash flow margin",
        "unit": "percentage",
        "fetch_types": ["cash_flow", "income_statement"],
        "compute": lambda d, t, y: _cashflow(d, t, y)["fcf"] / _income(d, t, y)["revenue"] * 100,
    },
    {
        "id": "price_change_pct",
        "display": "annual stock price change",
        "unit": "percentage",
        "fetch_types": ["price_history"],
        "compute": lambda d, t, y: (_price(d, t, y)["close"] - _price(d, t, y)["open"]) / _price(d, t, y)["open"] * 100,
    },
    {
        "id": "capex_to_revenue",
        "display": "capital expenditure as a percentage of revenue",
        "unit": "percentage",
        "fetch_types": ["cash_flow", "income_statement"],
        "compute": lambda d, t, y: _cashflow(d, t, y)["capex"] / _income(d, t, y)["revenue"] * 100,
    },
]


class EasyTaskGenerator:
    """Generates single-metric computation questions."""

    def __init__(self, data: dict, sectors: dict) -> None:
        self.data = data
        self.tickers = sorted(data.keys())

    def generate(self, seed: Optional[int] = None) -> TaskInstance:
        rng = random.Random(seed)

        for _ in range(100):
            ticker = rng.choice(self.tickers)
            year = rng.choice(ALL_YEARS)
            metric = rng.choice(EASY_METRICS)

            try:
                answer = round(metric["compute"](self.data, ticker, year), 2)
            except (KeyError, ZeroDivisionError, TypeError):
                continue

            company = COMPANY_NAMES.get(ticker, ticker)
            if metric["unit"] == "percentage":
                desc = (
                    f"What was {company}'s {metric['display']} for fiscal year {year}? "
                    f"Express as a percentage rounded to 2 decimal places."
                )
            else:
                desc = (
                    f"What was {company}'s {metric['display']} for fiscal year {year}? "
                    f"Round to 2 decimal places."
                )

            required = [f"{ft}:{ticker}:{year}" for ft in metric["fetch_types"]]

            return TaskInstance(
                task_id="task1_easy",
                task_description=desc,
                ground_truth={
                    "answer": answer,
                    "tolerance_exact": 0.05,
                    "tolerance_partial": 0.5,
                    "required_fetches": required,
                    "min_fetches": len(required),
                },
                relevant_tickers=[ticker],
                relevant_years=[year],
                difficulty="easy",
                max_steps=10,
                required_fetches=required,
                min_fetches=len(required),
                expected_intermediates=[answer],
                grader_id="task1",
            )

        return self._fallback()

    def _fallback(self) -> TaskInstance:
        inc = self.data["AAPL"]["2022"]["income_statement"]
        answer = round(inc["net_income"] / inc["revenue"] * 100, 2)
        req = ["income_statement:AAPL:2022"]
        return TaskInstance(
            task_id="task1_easy",
            task_description=(
                "What was Apple's net profit margin for fiscal year 2022? "
                "Express as a percentage rounded to 2 decimal places."
            ),
            ground_truth={
                "answer": answer,
                "tolerance_exact": 0.05,
                "tolerance_partial": 0.5,
                "required_fetches": req,
                "min_fetches": 1,
            },
            relevant_tickers=["AAPL"],
            relevant_years=[2022],
            difficulty="easy",
            max_steps=10,
            required_fetches=req,
            min_fetches=1,
            expected_intermediates=[answer],
            grader_id="task1",
        )


# ---------------------------------------------------------------------------
# Medium Task Generator
# ---------------------------------------------------------------------------

MEDIUM_METRICS: Dict[str, Dict[str, Any]] = {
    "pe_ratio": {"display": "P/E ratio", "lower_is_better": True},
    "pb_ratio": {"display": "P/B ratio", "lower_is_better": True},
    "ev_ebitda": {"display": "EV/EBITDA", "lower_is_better": True},
    "roe": {"display": "return on equity (ROE)", "lower_is_better": False},
    "debt_equity": {"display": "debt-to-equity ratio", "lower_is_better": True},
}


class MediumTaskGenerator:
    """Generates multi-company ratio comparison questions."""

    def __init__(self, data: dict, sectors: dict) -> None:
        self.data = data
        self.sectors = sectors

    def generate(self, seed: Optional[int] = None) -> TaskInstance:
        rng = random.Random(seed)

        for _ in range(100):
            sector = rng.choice(list(SECTOR_COMPANIES.keys()))
            companies = SECTOR_COMPANIES[sector]
            metric_id = rng.choice(list(MEDIUM_METRICS.keys()))
            metric_info = MEDIUM_METRICS[metric_id]
            year = rng.choice(ALL_YEARS)

            if len(companies) >= 3:
                selected = sorted(rng.sample(companies, 3))
            else:
                selected = sorted(companies)

            try:
                values: Dict[str, float] = {}
                for t in selected:
                    val = self.data[t][str(year)]["ratios"].get(metric_id)
                    if val is None:
                        raise ValueError
                    values[t] = val

                sector_median = self.sectors[sector][str(year)][metric_id]
            except (KeyError, ValueError):
                continue

            deltas = {t: round(v - sector_median, 2) for t, v in values.items()}

            if metric_info["lower_is_better"]:
                winner = min(deltas, key=lambda k: deltas[k])
            else:
                winner = max(deltas, key=lambda k: deltas[k])

            winning_delta = deltas[winner]

            names = [COMPANY_NAMES.get(t, t) for t in selected]
            if len(selected) >= 3:
                companies_str = f"{names[0]}, {names[1]}, and {names[2]}"
                prefix = "Among"
            else:
                companies_str = f"{names[0]} and {names[1]}"
                prefix = "Between"

            sector_display = sector.capitalize()

            desc = (
                f"{prefix} {companies_str}, which company had the most favorable "
                f"{metric_info['display']} relative to the {sector_display} sector median in {year}? "
                f"By how many points did it differ from the sector median? Submit your answer as "
                f'{{"company": "TICKER", "delta": NUMBER}}.'
            )

            required: List[str] = []
            for t in selected:
                required.append(f"ratios:{t}:{year}")
                required.append(f"sector_compare:{t}:{metric_id}:{year}")

            intermediates = [round(v, 2) for v in values.values()] + [winning_delta]

            return TaskInstance(
                task_id="task2_medium",
                task_description=desc,
                ground_truth={
                    "correct_company": winner,
                    "correct_delta": winning_delta,
                    "delta_tolerance": 0.5,
                    "required_fetches": required,
                    "min_fetches": len(selected),
                },
                relevant_tickers=selected,
                relevant_years=[year],
                difficulty="medium",
                max_steps=20,
                required_fetches=required,
                min_fetches=len(selected),
                expected_intermediates=intermediates,
                grader_id="task2",
            )

        return self._fallback()

    def _fallback(self) -> TaskInstance:
        tickers = ["GOOGL", "META", "MSFT"]
        year = 2023
        values = {t: self.data[t]["2023"]["ratios"]["ev_ebitda"] for t in tickers}
        median = self.sectors["technology"]["2023"]["ev_ebitda"]
        deltas = {t: round(v - median, 2) for t, v in values.items()}
        winner = min(deltas, key=lambda k: deltas[k])
        req = []
        for t in tickers:
            req.append(f"ratios:{t}:{year}")
            req.append(f"sector_compare:{t}:ev_ebitda:{year}")
        intermediates = [round(v, 2) for v in values.values()] + [deltas[winner]]
        return TaskInstance(
            task_id="task2_medium",
            task_description=(
                "Among Google (Alphabet), Meta Platforms, and Microsoft, which company had the most "
                "favorable EV/EBITDA relative to the Technology sector median in 2023? By how many "
                'points did it differ from the sector median? Submit your answer as '
                '{"company": "TICKER", "delta": NUMBER}.'
            ),
            ground_truth={
                "correct_company": winner,
                "correct_delta": deltas[winner],
                "delta_tolerance": 0.5,
                "required_fetches": req,
                "min_fetches": 3,
            },
            relevant_tickers=tickers,
            relevant_years=[year],
            difficulty="medium",
            max_steps=20,
            required_fetches=req,
            min_fetches=3,
            expected_intermediates=intermediates,
            grader_id="task2",
        )


# ---------------------------------------------------------------------------
# Hard Task Generator
# ---------------------------------------------------------------------------

def _safe_check(fn, data: dict, ticker: str, year: int) -> bool:
    try:
        return bool(fn(data, ticker, year))
    except (KeyError, TypeError, ZeroDivisionError):
        return False


ANOMALY_PATTERNS = [
    {
        "id": "negative_fcf_and_high_pe",
        "condition_a": {
            "key": "negative_fcf_years",
            "display": "negative free cash flow",
            "check": lambda d, t, y: d[t][str(y)]["cash_flow"]["fcf"] < 0,
            "min_count": 2,
            "fetch_types": ["cash_flow"],
        },
        "condition_b": {
            "key": "pe_above_30_years",
            "display": "a P/E ratio above 30",
            "check": lambda d, t, y: (
                d[t][str(y)]["ratios"]["pe_ratio"] is not None
                and d[t][str(y)]["ratios"]["pe_ratio"] > 30
            ),
            "min_count": 1,
            "fetch_types": ["ratios"],
        },
    },
    {
        "id": "high_debt_and_low_returns",
        "condition_a": {
            "key": "high_debt_years",
            "display": "a debt-to-equity ratio above 2.0",
            "check": lambda d, t, y: (
                d[t][str(y)]["ratios"]["debt_equity"] is not None
                and d[t][str(y)]["ratios"]["debt_equity"] > 2.0
            ),
            "min_count": 2,
            "fetch_types": ["balance_sheet"],
        },
        "condition_b": {
            "key": "low_roa_years",
            "display": "return on assets below 5%",
            "check": lambda d, t, y: (
                d[t][str(y)]["ratios"]["roa"] is not None
                and d[t][str(y)]["ratios"]["roa"] < 0.05
            ),
            "min_count": 1,
            "fetch_types": ["ratios"],
        },
    },
    {
        "id": "negative_income_and_high_valuation",
        "condition_a": {
            "key": "negative_income_years",
            "display": "negative net income",
            "check": lambda d, t, y: d[t][str(y)]["income_statement"]["net_income"] < 0,
            "min_count": 2,
            "fetch_types": ["income_statement"],
        },
        "condition_b": {
            "key": "high_pb_years",
            "display": "a P/B ratio above 10",
            "check": lambda d, t, y: (
                d[t][str(y)]["ratios"]["pb_ratio"] is not None
                and d[t][str(y)]["ratios"]["pb_ratio"] > 10
            ),
            "min_count": 1,
            "fetch_types": ["ratios"],
        },
    },
    {
        "id": "negative_ocf_and_low_margins",
        "condition_a": {
            "key": "negative_ocf_years",
            "display": "negative operating cash flow",
            "check": lambda d, t, y: d[t][str(y)]["cash_flow"]["operating_cf"] < 0,
            "min_count": 2,
            "fetch_types": ["cash_flow"],
        },
        "condition_b": {
            "key": "negative_margin_years",
            "display": "a negative net profit margin",
            "check": lambda d, t, y: (
                d[t][str(y)]["ratios"]["net_margin"] is not None
                and d[t][str(y)]["ratios"]["net_margin"] < 0
            ),
            "min_count": 1,
            "fetch_types": ["income_statement"],
        },
    },
]


class HardTaskGenerator:
    """Generates multi-year anomaly detection questions."""

    def __init__(self, data: dict, sectors: dict) -> None:
        self.data = data
        self.tickers = sorted(data.keys())

    def generate(self, seed: Optional[int] = None) -> TaskInstance:
        rng = random.Random(seed)

        for _ in range(200):
            companies = sorted(rng.sample(self.tickers, 3))
            window_size = rng.choice([3, 4, 5])
            max_start = 2024 - window_size + 1
            start_year = rng.randint(2019, max_start)
            years = list(range(start_year, start_year + window_size))
            pattern = rng.choice(ANOMALY_PATTERNS)

            cond_a = pattern["condition_a"]
            cond_b = pattern["condition_b"]

            a_years: Dict[str, List[int]] = {}
            b_years: Dict[str, List[int]] = {}
            qualifying: List[str] = []

            for t in companies:
                a_matched = [y for y in years if _safe_check(cond_a["check"], self.data, t, y)]
                b_matched = [y for y in years if _safe_check(cond_b["check"], self.data, t, y)]

                if len(a_matched) >= cond_a["min_count"] and len(b_matched) >= cond_b["min_count"]:
                    qualifying.append(t)
                    a_years[t] = a_matched
                    b_years[t] = b_matched

            if not qualifying:
                continue

            names = [COMPANY_NAMES.get(t, t) for t in companies]
            companies_str = f"{names[0]}, {names[1]}, and {names[2]}"
            year_range = f"{years[0]}-{years[-1]}"

            desc = (
                f"Among {companies_str} \u2014 which company had {cond_a['display']} in "
                f"at least {cond_a['min_count']} of the {len(years)} fiscal years from "
                f"{year_range}, AND had {cond_b['display']} in any of those same years? "
                f"For each qualifying company, state which specific years had "
                f"{cond_a['display']} and which years had {cond_b['display']}. Submit as "
                f'{{"qualifying_companies": ["TICKER"], "details": {{"TICKER": '
                f'{{"{cond_a["key"]}": [YEAR, ...], "{cond_b["key"]}": [YEAR, ...]}}}}}}.'
            )

            fetch_types = sorted(set(cond_a["fetch_types"]) | set(cond_b["fetch_types"]))
            required: List[str] = []
            for t in companies:
                for y in years:
                    for ft in fetch_types:
                        required.append(f"{ft}:{t}:{y}")

            min_fetches = len(companies) * 2

            return TaskInstance(
                task_id="task3_hard",
                task_description=desc,
                ground_truth={
                    "qualifying_companies": qualifying,
                    "condition_a_key": cond_a["key"],
                    "condition_b_key": cond_b["key"],
                    "condition_a_years": a_years,
                    "condition_b_years": b_years,
                    "required_fetches": required,
                    "min_fetches": min_fetches,
                },
                relevant_tickers=companies,
                relevant_years=years,
                difficulty="hard",
                max_steps=40,
                required_fetches=required,
                min_fetches=min_fetches,
                expected_intermediates=[],
                grader_id="task3",
            )

        return self._fallback()

    def _fallback(self) -> TaskInstance:
        companies = ["F", "GM", "TSLA"]
        years = [2020, 2021, 2022, 2023]
        pattern = ANOMALY_PATTERNS[0]  # negative_fcf_and_high_pe
        cond_a = pattern["condition_a"]
        cond_b = pattern["condition_b"]

        a_years: Dict[str, List[int]] = {}
        b_years: Dict[str, List[int]] = {}
        qualifying: List[str] = []

        for t in companies:
            a_matched = [y for y in years if _safe_check(cond_a["check"], self.data, t, y)]
            b_matched = [y for y in years if _safe_check(cond_b["check"], self.data, t, y)]
            if len(a_matched) >= cond_a["min_count"] and len(b_matched) >= cond_b["min_count"]:
                qualifying.append(t)
                a_years[t] = a_matched
                b_years[t] = b_matched

        fetch_types = sorted(set(cond_a["fetch_types"]) | set(cond_b["fetch_types"]))
        required = [f"{ft}:{t}:{y}" for t in companies for y in years for ft in fetch_types]

        return TaskInstance(
            task_id="task3_hard",
            task_description=(
                "Among Ford, General Motors, and Tesla \u2014 which company had negative free "
                "cash flow in at least 2 of the 4 fiscal years from 2020-2023, AND had a P/E "
                "ratio above 30 in any of those same years? For each qualifying company, state "
                "which specific years had negative free cash flow and which years had a P/E "
                'ratio above 30. Submit as {"qualifying_companies": ["TICKER"], "details": '
                '{"TICKER": {"negative_fcf_years": [YEAR, ...], "pe_above_30_years": '
                "[YEAR, ...]}}}."
            ),
            ground_truth={
                "qualifying_companies": qualifying,
                "condition_a_key": "negative_fcf_years",
                "condition_b_key": "pe_above_30_years",
                "condition_a_years": a_years,
                "condition_b_years": b_years,
                "required_fetches": required,
                "min_fetches": 6,
            },
            relevant_tickers=companies,
            relevant_years=years,
            difficulty="hard",
            max_steps=40,
            required_fetches=required,
            min_fetches=6,
            expected_intermediates=[],
            grader_id="task3",
        )
