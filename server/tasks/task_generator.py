"""Procedural task generators for FinQuery.

Combinatorial space:
  Easy:   25 tickers x 9 years x 11 metrics = ~2475 unique tasks
  Medium: 7 sectors x company combos x 5 metrics x 9 years = hundreds
  Hard:   C(25,3) x year windows x 6 anomaly patterns = thousands
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


COMPANY_NAMES: Dict[str, str] = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google (Alphabet)",
    "META": "Meta Platforms", "NVDA": "NVIDIA", "ORCL": "Oracle", "CRM": "Salesforce",
    "TSLA": "Tesla", "F": "Ford", "GM": "General Motors", "TM": "Toyota",
    "JPM": "JPMorgan Chase", "BAC": "Bank of America", "WFC": "Wells Fargo", "GS": "Goldman Sachs",
    "AMZN": "Amazon", "WMT": "Walmart", "COST": "Costco",
    "JNJ": "Johnson & Johnson", "UNH": "UnitedHealth", "PFE": "Pfizer",
    "XOM": "ExxonMobil", "CVX": "Chevron",
    "CAT": "Caterpillar", "BA": "Boeing",
}

SECTOR_COMPANIES: Dict[str, List[str]] = {
    "technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "ORCL", "CRM"],
    "automotive": ["TSLA", "F", "GM", "TM"],
    "banking": ["JPM", "BAC", "WFC", "GS"],
    "retail": ["AMZN", "WMT", "COST"],
    "healthcare": ["JNJ", "UNH", "PFE"],
    "energy": ["XOM", "CVX"],
    "industrials": ["CAT", "BA"],
}

ALL_YEARS: List[int] = list(range(2017, 2026))


@dataclass
class TaskInstance:
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
    grader_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── helpers ──────────────────────────────────────────────────────

def _inc(d, t, y):
    return d[t][str(y)]["income_statement"]

def _bs(d, t, y):
    return d[t][str(y)]["balance_sheet"]

def _cf(d, t, y):
    return d[t][str(y)]["cash_flow"]

def _pr(d, t, y):
    return d[t][str(y)]["price"]

def _rat(d, t, y):
    return d[t][str(y)]["ratios"]


# ── Easy metrics ─────────────────────────────────────────────────

EASY_METRICS = [
    dict(id="net_profit_margin", display="net profit margin", unit="percentage",
         fetch_types=["income_statement"],
         compute=lambda d, t, y: _inc(d, t, y)["net_income"] / _inc(d, t, y)["revenue"] * 100),
    dict(id="gross_margin", display="gross margin", unit="percentage",
         fetch_types=["income_statement"],
         compute=lambda d, t, y: _inc(d, t, y)["gross_profit"] / _inc(d, t, y)["revenue"] * 100),
    dict(id="operating_margin", display="operating margin", unit="percentage",
         fetch_types=["income_statement"],
         compute=lambda d, t, y: _inc(d, t, y)["operating_income"] / _inc(d, t, y)["revenue"] * 100),
    dict(id="debt_to_equity", display="debt-to-equity ratio", unit="ratio",
         fetch_types=["balance_sheet"],
         compute=lambda d, t, y: _bs(d, t, y)["total_debt"] / _bs(d, t, y)["total_equity"]),
    dict(id="return_on_assets", display="return on assets (ROA)", unit="percentage",
         fetch_types=["income_statement", "balance_sheet"],
         compute=lambda d, t, y: _inc(d, t, y)["net_income"] / _bs(d, t, y)["total_assets"] * 100),
    dict(id="return_on_equity", display="return on equity (ROE)", unit="percentage",
         fetch_types=["income_statement", "balance_sheet"],
         compute=lambda d, t, y: _inc(d, t, y)["net_income"] / _bs(d, t, y)["total_equity"] * 100),
    dict(id="fcf_margin", display="free cash flow margin", unit="percentage",
         fetch_types=["cash_flow", "income_statement"],
         compute=lambda d, t, y: _cf(d, t, y)["fcf"] / _inc(d, t, y)["revenue"] * 100),
    dict(id="price_change_pct", display="annual stock price change", unit="percentage",
         fetch_types=["price_history"],
         compute=lambda d, t, y: (_pr(d, t, y)["close"] - _pr(d, t, y)["open"]) / _pr(d, t, y)["open"] * 100),
    dict(id="capex_to_revenue", display="capital expenditure as a percentage of revenue", unit="percentage",
         fetch_types=["cash_flow", "income_statement"],
         compute=lambda d, t, y: _cf(d, t, y)["capex"] / _inc(d, t, y)["revenue"] * 100),
    dict(id="eps_growth", display="year-over-year EPS growth rate", unit="percentage",
         fetch_types=["income_statement"],
         compute=lambda d, t, y: (_inc(d, t, y)["eps"] - _inc(d, t, y - 1)["eps"]) / abs(_inc(d, t, y - 1)["eps"]) * 100),
    dict(id="revenue_growth", display="year-over-year revenue growth rate", unit="percentage",
         fetch_types=["income_statement"],
         compute=lambda d, t, y: (_inc(d, t, y)["revenue"] - _inc(d, t, y - 1)["revenue"]) / _inc(d, t, y - 1)["revenue"] * 100),
]


class EasyTaskGenerator:
    def __init__(self, data: dict, sectors: dict) -> None:
        self.data = data
        self.tickers = sorted(data.keys())

    def generate(self, seed: Optional[int] = None) -> TaskInstance:
        rng = random.Random(seed)
        for _ in range(100):
            ticker = rng.choice(self.tickers)
            metric = rng.choice(EASY_METRICS)
            needs_prev = metric["id"] in ("eps_growth", "revenue_growth")
            year = rng.choice(ALL_YEARS[1:] if needs_prev else ALL_YEARS)
            try:
                answer = round(metric["compute"](self.data, ticker, year), 2)
            except (KeyError, ZeroDivisionError, TypeError):
                continue

            company = COMPANY_NAMES.get(ticker, ticker)
            if metric["unit"] == "percentage":
                desc = (f"What was {company}'s {metric['display']} for fiscal year {year}? "
                        f"Express as a percentage rounded to 2 decimal places.")
            else:
                desc = (f"What was {company}'s {metric['display']} for fiscal year {year}? "
                        f"Round to 2 decimal places.")

            req_years = [year - 1, year] if needs_prev else [year]
            required = [f"{ft}:{ticker}:{y}" for y in req_years for ft in metric["fetch_types"]]

            return TaskInstance(
                task_id="task1_easy", task_description=desc,
                ground_truth=dict(answer=answer, tolerance_exact=0.05, tolerance_partial=0.5,
                                  required_fetches=required, min_fetches=len(set(required))),
                relevant_tickers=[ticker], relevant_years=req_years,
                difficulty="easy", max_steps=10,
                required_fetches=required, min_fetches=len(set(required)),
                expected_intermediates=[answer], grader_id="task1",
                metadata=dict(metric=metric["id"], ticker=ticker, year=year),
            )
        return self._fallback()

    def _fallback(self) -> TaskInstance:
        inc = self.data["AAPL"]["2022"]["income_statement"]
        answer = round(inc["net_income"] / inc["revenue"] * 100, 2)
        return TaskInstance(
            task_id="task1_easy",
            task_description="What was Apple's net profit margin for fiscal year 2022? Express as a percentage rounded to 2 decimal places.",
            ground_truth=dict(answer=answer, tolerance_exact=0.05, tolerance_partial=0.5,
                              required_fetches=["income_statement:AAPL:2022"], min_fetches=1),
            relevant_tickers=["AAPL"], relevant_years=[2022],
            difficulty="easy", max_steps=10,
            required_fetches=["income_statement:AAPL:2022"], min_fetches=1,
            expected_intermediates=[answer], grader_id="task1",
            metadata=dict(metric="net_profit_margin", ticker="AAPL", year=2022),
        )


# ── Medium ───────────────────────────────────────────────────────

MEDIUM_METRICS: Dict[str, Dict[str, Any]] = {
    "pe_ratio":    dict(display="P/E ratio", lower_is_better=True),
    "pb_ratio":    dict(display="P/B ratio", lower_is_better=True),
    "ev_ebitda":   dict(display="EV/EBITDA", lower_is_better=True),
    "roe":         dict(display="return on equity (ROE)", lower_is_better=False),
    "debt_equity": dict(display="debt-to-equity ratio", lower_is_better=True),
}


class MediumTaskGenerator:
    def __init__(self, data: dict, sectors: dict) -> None:
        self.data = data
        self.sectors = sectors

    def generate(self, seed: Optional[int] = None) -> TaskInstance:
        rng = random.Random(seed)
        for _ in range(100):
            sector = rng.choice(list(SECTOR_COMPANIES.keys()))
            companies = SECTOR_COMPANIES[sector]
            metric_id = rng.choice(list(MEDIUM_METRICS.keys()))
            info = MEDIUM_METRICS[metric_id]
            year = rng.choice(ALL_YEARS)
            selected = sorted(rng.sample(companies, min(3, len(companies))))

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
            winner = min(deltas, key=lambda k: deltas[k]) if info["lower_is_better"] else max(deltas, key=lambda k: deltas[k])
            winning_delta = deltas[winner]

            names = [COMPANY_NAMES.get(t, t) for t in selected]
            if len(selected) >= 3:
                co_str = f"{names[0]}, {names[1]}, and {names[2]}"
                prefix = "Among"
            else:
                co_str = f"{names[0]} and {names[1]}"
                prefix = "Between"

            desc = (f"{prefix} {co_str}, which company had the most favorable "
                    f"{info['display']} relative to the {sector.capitalize()} sector median in {year}? "
                    f"By how many points did it differ from the sector median? "
                    f'Submit your answer as {{"company": "TICKER", "delta": NUMBER}}.')

            required: List[str] = []
            for t in selected:
                required.append(f"ratios:{t}:{year}")
                required.append(f"sector_compare:{t}:{metric_id}:{year}")

            return TaskInstance(
                task_id="task2_medium", task_description=desc,
                ground_truth=dict(correct_company=winner, correct_delta=winning_delta,
                                  delta_tolerance=0.5, required_fetches=required, min_fetches=len(selected)),
                relevant_tickers=selected, relevant_years=[year],
                difficulty="medium", max_steps=20,
                required_fetches=required, min_fetches=len(selected),
                expected_intermediates=[round(v, 2) for v in values.values()] + [winning_delta],
                grader_id="task2",
                metadata=dict(metric=metric_id, sector=sector, year=year,
                              companies=selected, direction="lower" if info["lower_is_better"] else "higher"),
            )
        return self._fallback()

    def _fallback(self) -> TaskInstance:
        tickers = ["GOOGL", "META", "MSFT"]
        values = {t: self.data[t]["2023"]["ratios"]["ev_ebitda"] for t in tickers}
        median = self.sectors["technology"]["2023"]["ev_ebitda"]
        deltas = {t: round(v - median, 2) for t, v in values.items()}
        winner = min(deltas, key=lambda k: deltas[k])
        req = [f for t in tickers for f in [f"ratios:{t}:2023", f"sector_compare:{t}:ev_ebitda:2023"]]
        return TaskInstance(
            task_id="task2_medium",
            task_description=("Among Google (Alphabet), Meta Platforms, and Microsoft, which company had the most "
                              "favorable EV/EBITDA relative to the Technology sector median in 2023? By how many "
                              'points did it differ from the sector median? Submit your answer as '
                              '{"company": "TICKER", "delta": NUMBER}.'),
            ground_truth=dict(correct_company=winner, correct_delta=deltas[winner],
                              delta_tolerance=0.5, required_fetches=req, min_fetches=3),
            relevant_tickers=tickers, relevant_years=[2023],
            difficulty="medium", max_steps=20, required_fetches=req, min_fetches=3,
            expected_intermediates=[round(v, 2) for v in values.values()] + [deltas[winner]],
            grader_id="task2",
            metadata=dict(metric="ev_ebitda", sector="technology", year=2023),
        )


# ── Hard ─────────────────────────────────────────────────────────

def _safe_check(fn, data, ticker, year):
    try:
        return bool(fn(data, ticker, year))
    except (KeyError, TypeError, ZeroDivisionError):
        return False


ANOMALY_PATTERNS = [
    dict(id="negative_fcf_and_high_pe",
         condition_a=dict(key="negative_fcf_years", display="negative free cash flow",
                          check=lambda d, t, y: _cf(d, t, y)["fcf"] < 0,
                          min_count=2, fetch_types=["cash_flow"]),
         condition_b=dict(key="pe_above_30_years", display="a P/E ratio above 30",
                          check=lambda d, t, y: _rat(d, t, y)["pe_ratio"] is not None and _rat(d, t, y)["pe_ratio"] > 30,
                          min_count=1, fetch_types=["ratios"])),
    dict(id="high_debt_and_low_returns",
         condition_a=dict(key="high_debt_years", display="a debt-to-equity ratio above 2.0",
                          check=lambda d, t, y: _rat(d, t, y)["debt_equity"] is not None and _rat(d, t, y)["debt_equity"] > 2.0,
                          min_count=2, fetch_types=["balance_sheet"]),
         condition_b=dict(key="low_roa_years", display="return on assets below 5%",
                          check=lambda d, t, y: _rat(d, t, y)["roa"] is not None and _rat(d, t, y)["roa"] < 0.05,
                          min_count=1, fetch_types=["ratios"])),
    dict(id="negative_income_and_high_valuation",
         condition_a=dict(key="negative_income_years", display="negative net income",
                          check=lambda d, t, y: _inc(d, t, y)["net_income"] < 0,
                          min_count=2, fetch_types=["income_statement"]),
         condition_b=dict(key="high_pb_years", display="a P/B ratio above 10",
                          check=lambda d, t, y: _rat(d, t, y)["pb_ratio"] is not None and _rat(d, t, y)["pb_ratio"] > 10,
                          min_count=1, fetch_types=["ratios"])),
    dict(id="negative_ocf_and_low_margins",
         condition_a=dict(key="negative_ocf_years", display="negative operating cash flow",
                          check=lambda d, t, y: _cf(d, t, y)["operating_cf"] < 0,
                          min_count=2, fetch_types=["cash_flow"]),
         condition_b=dict(key="negative_margin_years", display="a negative net profit margin",
                          check=lambda d, t, y: _rat(d, t, y)["net_margin"] is not None and _rat(d, t, y)["net_margin"] < 0,
                          min_count=1, fetch_types=["income_statement"])),
    dict(id="cash_burn_and_price_decline",
         condition_a=dict(key="cash_burn_years", display="negative operating cash flow",
                          check=lambda d, t, y: _cf(d, t, y)["operating_cf"] < 0,
                          min_count=2, fetch_types=["cash_flow"]),
         condition_b=dict(key="price_decline_years", display="an annual stock price decline",
                          check=lambda d, t, y: _pr(d, t, y)["close"] < _pr(d, t, y)["open"],
                          min_count=1, fetch_types=["price_history"])),
    dict(id="high_pe_and_low_roa",
         condition_a=dict(key="high_pe_years", display="a P/E ratio above 25",
                          check=lambda d, t, y: _rat(d, t, y)["pe_ratio"] is not None and _rat(d, t, y)["pe_ratio"] > 25,
                          min_count=2, fetch_types=["ratios"]),
         condition_b=dict(key="low_roa_years", display="return on assets below 5%",
                          check=lambda d, t, y: _rat(d, t, y)["roa"] is not None and _rat(d, t, y)["roa"] < 0.05,
                          min_count=1, fetch_types=["ratios"])),
]


class HardTaskGenerator:
    def __init__(self, data: dict, sectors: dict) -> None:
        self.data = data
        self.tickers = sorted(data.keys())

    def generate(self, seed: Optional[int] = None) -> TaskInstance:
        rng = random.Random(seed)
        for _ in range(200):
            companies = sorted(rng.sample(self.tickers, 3))
            window_size = rng.choice([3, 4, 5])
            max_start = 2025 - window_size + 1
            start_year = rng.randint(2017, max_start)
            years = list(range(start_year, start_year + window_size))
            pattern = rng.choice(ANOMALY_PATTERNS)
            ca, cb = pattern["condition_a"], pattern["condition_b"]

            a_years: Dict[str, List[int]] = {}
            b_years: Dict[str, List[int]] = {}
            qualifying: List[str] = []
            for t in companies:
                am = [y for y in years if _safe_check(ca["check"], self.data, t, y)]
                bm = [y for y in years if _safe_check(cb["check"], self.data, t, y)]
                if len(am) >= ca["min_count"] and len(bm) >= cb["min_count"]:
                    qualifying.append(t)
                    a_years[t] = am
                    b_years[t] = bm

            if not qualifying:
                continue

            names = [COMPANY_NAMES.get(t, t) for t in companies]
            desc = (f"Among {names[0]}, {names[1]}, and {names[2]} \u2014 which company had "
                    f"{ca['display']} in at least {ca['min_count']} of the {len(years)} fiscal years from "
                    f"{years[0]}-{years[-1]}, AND had {cb['display']} in any of those same years? "
                    f"For each qualifying company, state which specific years had {ca['display']} "
                    f"and which years had {cb['display']}. Submit as "
                    f'{{"qualifying_companies": ["TICKER"], "details": {{"TICKER": '
                    f'{{"{ca["key"]}": [YEAR, ...], "{cb["key"]}": [YEAR, ...]}}}}}}.')

            fetch_types = sorted(set(ca["fetch_types"]) | set(cb["fetch_types"]))
            required = [f"{ft}:{t}:{y}" for t in companies for y in years for ft in fetch_types]

            return TaskInstance(
                task_id="task3_hard", task_description=desc,
                ground_truth=dict(qualifying_companies=qualifying,
                                  condition_a_key=ca["key"], condition_b_key=cb["key"],
                                  condition_a_years=a_years, condition_b_years=b_years,
                                  required_fetches=required, min_fetches=len(companies) * 2),
                relevant_tickers=companies, relevant_years=years,
                difficulty="hard", max_steps=40,
                required_fetches=required, min_fetches=len(companies) * 2,
                expected_intermediates=[], grader_id="task3",
                metadata=dict(pattern=pattern["id"], companies=companies, years=years),
            )
        return self._fallback()

    def _fallback(self) -> TaskInstance:
        companies = ["F", "GM", "TSLA"]
        years = [2020, 2021, 2022, 2023]
        ca, cb = ANOMALY_PATTERNS[0]["condition_a"], ANOMALY_PATTERNS[0]["condition_b"]
        a_years: Dict[str, List[int]] = {}
        b_years: Dict[str, List[int]] = {}
        qualifying: List[str] = []
        for t in companies:
            am = [y for y in years if _safe_check(ca["check"], self.data, t, y)]
            bm = [y for y in years if _safe_check(cb["check"], self.data, t, y)]
            if len(am) >= ca["min_count"] and len(bm) >= cb["min_count"]:
                qualifying.append(t)
                a_years[t] = am
                b_years[t] = bm
        fetch_types = sorted(set(ca["fetch_types"]) | set(cb["fetch_types"]))
        required = [f"{ft}:{t}:{y}" for t in companies for y in years for ft in fetch_types]
        return TaskInstance(
            task_id="task3_hard",
            task_description=("Among Ford, General Motors, and Tesla \u2014 which company had negative free "
                              "cash flow in at least 2 of the 4 fiscal years from 2020-2023, AND had a P/E "
                              "ratio above 30 in any of those same years? For each qualifying company, state "
                              "which specific years had negative free cash flow and which years had a P/E "
                              'ratio above 30. Submit as {"qualifying_companies": ["TICKER"], "details": '
                              '{"TICKER": {"negative_fcf_years": [YEAR, ...], "pe_above_30_years": [YEAR, ...]}}).'),
            ground_truth=dict(qualifying_companies=qualifying,
                              condition_a_key="negative_fcf_years", condition_b_key="pe_above_30_years",
                              condition_a_years=a_years, condition_b_years=b_years,
                              required_fetches=required, min_fetches=6),
            relevant_tickers=companies, relevant_years=years,
            difficulty="hard", max_steps=40, required_fetches=required, min_fetches=6,
            expected_intermediates=[], grader_id="task3",
            metadata=dict(pattern="negative_fcf_and_high_pe"),
        )


# ── Batch / Composite ───────────────────────────────────────────

def generate_batch(generators: Dict[str, Any], task_id: str, size: int, seed: int) -> List[TaskInstance]:
    gen = generators[task_id]
    return [gen.generate(seed=seed + i) for i in range(size)]


def generate_composite_batch(generators: Dict[str, Any], task_specs: List[Dict], size: int, seed: int) -> List[TaskInstance]:
    total_weight = sum(s.get("weight", 1) for s in task_specs)
    tasks: List[TaskInstance] = []
    rng = random.Random(seed)
    difficulty_map = {"easy": "task1_easy", "medium": "task2_medium", "hard": "task3_hard"}

    for spec in task_specs:
        w = spec.get("weight", 1)
        tid = spec.get("task_id") or difficulty_map.get(spec.get("difficulty", "easy"), "task1_easy")
        count = max(1, round(size * w / total_weight))
        gen = generators.get(tid)
        if gen:
            for i in range(count):
                tasks.append(gen.generate(seed=rng.randint(0, 2**31)))

    rng.shuffle(tasks)
    return tasks[:size]
