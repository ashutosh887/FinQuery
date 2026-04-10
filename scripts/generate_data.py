#!/usr/bin/env python3
"""Generate expanded synthetic financial data for FinQuery.

Output: 25 companies, 7 sectors, 9 years (2017-2025).
All accounting identities enforced. Replaces server/data/financials.json and sectors.json.
"""

import json
import math
import random
import statistics
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "server" / "data"
YEARS = list(range(2017, 2026))

TICKER_SECTORS = {
    "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology",
    "META": "technology", "NVDA": "technology", "ORCL": "technology", "CRM": "technology",
    "TSLA": "automotive", "F": "automotive", "GM": "automotive", "TM": "automotive",
    "JPM": "banking", "BAC": "banking", "WFC": "banking", "GS": "banking",
    "AMZN": "retail", "WMT": "retail", "COST": "retail",
    "JNJ": "healthcare", "UNH": "healthcare", "PFE": "healthcare",
    "XOM": "energy", "CVX": "energy",
    "CAT": "industrials", "BA": "industrials",
}

SECTOR_METRICS = ["pe_ratio", "pb_ratio", "ev_ebitda", "roe", "debt_equity"]

# ── Company profiles ─────────────────────────────────────────────
# Base values approximate 2020 actuals. All monetary in $M.
# gm=gross_margin, om=operating_margin, nm=net_margin
# am=asset/revenue multiplier, eq=equity fraction of assets
# dp=debt fraction of assets, cp=cash fraction of assets
# cx=capex/revenue, ocf=operating_cf/revenue
# sh=shares_outstanding (millions), bp=base_stock_price (2020)
# rg=revenue_growth, pg=price_growth (CAGR)
# revs=revenue_shocks {year: multiplier}, mo=margin_overrides {year: {field: val}}

PROFILES = {
    "ORCL": dict(sector="technology", rev=39068, gm=0.80, om=0.35, nm=0.26,
                 am=2.5, eq=0.12, dp=0.55, cp=0.15, cx=0.03, ocf=0.35,
                 sh=3100, bp=57, rg=0.04, pg=0.15, revs={}, mo={}),
    "CRM":  dict(sector="technology", rev=17098, gm=0.73, om=0.04, nm=0.02,
                 am=2.8, eq=0.45, dp=0.15, cp=0.12, cx=0.04, ocf=0.30,
                 sh=900, bp=200, rg=0.22, pg=0.20, revs={}, mo={
                     2022: dict(om=0.06, nm=0.04), 2023: dict(om=0.10, nm=0.08),
                     2024: dict(om=0.14, nm=0.11), 2025: dict(om=0.18, nm=0.14)}),
    "TM":   dict(sector="automotive", rev=275000, gm=0.20, om=0.07, nm=0.05,
                 am=1.8, eq=0.40, dp=0.30, cp=0.10, cx=0.05, ocf=0.12,
                 sh=14000, bp=140, rg=0.03, pg=0.08,
                 revs={2020: 0.82, 2021: 1.12}, mo={}),
    "WFC":  dict(sector="banking", rev=58000, gm=0.58, om=0.18, nm=0.10,
                 am=25.0, eq=0.08, dp=0.30, cp=0.15, cx=0.02, ocf=0.20,
                 sh=4100, bp=28, rg=0.01, pg=0.05,
                 revs={2020: 0.85}, mo={2020: dict(nm=0.03)}),
    "GS":   dict(sector="banking", rev=44560, gm=0.65, om=0.30, nm=0.24,
                 am=25.0, eq=0.06, dp=0.25, cp=0.15, cx=0.03, ocf=0.25,
                 sh=360, bp=210, rg=0.05, pg=0.12,
                 revs={2020: 0.90, 2021: 1.35}, mo={}),
    "COST": dict(sector="retail", rev=163220, gm=0.13, om=0.035, nm=0.025,
                 am=0.35, eq=0.30, dp=0.15, cp=0.15, cx=0.02, ocf=0.05,
                 sh=443, bp=360, rg=0.09, pg=0.18, revs={}, mo={}),
    "JNJ":  dict(sector="healthcare", rev=82584, gm=0.66, om=0.25, nm=0.20,
                 am=2.0, eq=0.40, dp=0.20, cp=0.10, cx=0.04, ocf=0.25,
                 sh=2630, bp=150, rg=0.04, pg=0.05, revs={}, mo={}),
    "UNH":  dict(sector="healthcare", rev=257141, gm=0.24, om=0.08, nm=0.06,
                 am=0.8, eq=0.35, dp=0.25, cp=0.08, cx=0.01, ocf=0.08,
                 sh=950, bp=350, rg=0.10, pg=0.15, revs={}, mo={}),
    "PFE":  dict(sector="healthcare", rev=41908, gm=0.79, om=0.25, nm=0.22,
                 am=2.0, eq=0.45, dp=0.25, cp=0.08, cx=0.03, ocf=0.30,
                 sh=5560, bp=37, rg=0.03, pg=0.05,
                 revs={2021: 1.95, 2022: 2.40, 2023: 0.70, 2024: 0.45, 2025: 0.50},
                 mo={2021: dict(gm=0.83, om=0.35, nm=0.30),
                     2022: dict(gm=0.85, om=0.40, nm=0.35),
                     2023: dict(om=0.15, nm=0.10),
                     2024: dict(om=0.08, nm=0.02),
                     2025: dict(om=0.10, nm=0.05)}),
    "XOM":  dict(sector="energy", rev=178574, gm=0.30, om=0.04, nm=0.01,
                 am=1.8, eq=0.50, dp=0.20, cp=0.03, cx=0.10, ocf=0.12,
                 sh=4270, bp=42, rg=0.0, pg=0.08,
                 revs={2020: 0.65, 2021: 1.30, 2022: 2.10, 2023: 0.90, 2024: 0.80},
                 mo={2020: dict(om=-0.12, nm=-0.13, ocf=0.04),
                     2022: dict(om=0.18, nm=0.14, ocf=0.22),
                     2023: dict(om=0.12, nm=0.09)}),
    "CVX":  dict(sector="energy", rev=94692, gm=0.40, om=0.06, nm=0.02,
                 am=1.5, eq=0.55, dp=0.15, cp=0.02, cx=0.10, ocf=0.15,
                 sh=1920, bp=84, rg=0.02, pg=0.10,
                 revs={2020: 0.60, 2021: 1.40, 2022: 2.00, 2023: 0.85, 2024: 0.78},
                 mo={2020: dict(om=-0.10, nm=-0.10, ocf=0.05),
                     2022: dict(om=0.16, nm=0.12, ocf=0.20)}),
    "CAT":  dict(sector="industrials", rev=41748, gm=0.35, om=0.14, nm=0.10,
                 am=1.8, eq=0.20, dp=0.45, cp=0.10, cx=0.04, ocf=0.18,
                 sh=540, bp=175, rg=0.06, pg=0.15, revs={2020: 0.88}, mo={}),
    "BA":   dict(sector="industrials", rev=58158, gm=0.14, om=0.03, nm=0.01,
                 am=2.0, eq=0.05, dp=0.50, cp=0.10, cx=0.02, ocf=0.05,
                 sh=580, bp=160, rg=0.0, pg=-0.02,
                 revs={2019: 0.88, 2020: 0.55, 2021: 0.70, 2022: 1.10, 2023: 1.15, 2024: 1.20},
                 mo={2019: dict(om=-0.05, nm=-0.08, ocf=-0.04),
                     2020: dict(om=-0.20, nm=-0.20, ocf=-0.15),
                     2021: dict(om=-0.10, nm=-0.08, ocf=-0.05),
                     2022: dict(om=-0.02, nm=-0.04, ocf=0.02),
                     2023: dict(om=0.04, nm=0.02, ocf=0.06)}),
}

# Growth adjustments for existing companies' new years (2017, 2018, 2025)
EXISTING_GROWTH = {
    "AAPL": 0.06, "MSFT": 0.14, "GOOGL": 0.18, "META": 0.20, "NVDA": 0.25,
    "TSLA": 0.30, "F": 0.02, "GM": 0.03, "JPM": 0.05, "BAC": 0.04,
    "AMZN": 0.22, "WMT": 0.03,
}


def make_record(rev, gm, om, nm, am, eq, dp, cp, cx, ocf_pct, sh, p_open, p_close, p_hi, p_lo):
    """Build a complete financial record from key parameters."""
    revenue = round(rev)
    cogs = round(revenue * (1 - gm))
    gross_profit = revenue - cogs
    operating_income = round(revenue * om)
    if operating_income >= gross_profit:
        operating_income = gross_profit - 1
    net_income = round(revenue * nm)
    eps = round(net_income / sh, 2) if sh > 0 else 0.0

    total_assets = max(round(revenue * am), 1)
    total_equity = max(round(total_assets * eq), 1)
    total_liabilities = total_assets - total_equity
    total_debt = round(total_assets * dp)
    cash = round(total_assets * cp)

    operating_cf = round(revenue * ocf_pct)
    capex = abs(round(revenue * cx))
    fcf = operating_cf - capex
    investing_cf = round(-capex * 1.3)
    financing_cf = round(-(operating_cf + investing_cf) * 0.6)

    avg_price = round((p_hi + p_lo) / 2, 2)
    market_cap = avg_price * sh

    pe = round(avg_price / eps, 2) if eps > 0 else None
    pb = round(market_cap / total_equity, 2) if total_equity > 0 else None
    ev = market_cap + total_debt - cash
    ev_eb = round(ev / operating_income, 2) if operating_income > 0 else None
    roe = round(net_income / total_equity, 4) if total_equity > 0 else None
    roa = round(net_income / total_assets, 4) if total_assets > 0 else None
    de = round(total_debt / total_equity, 4) if total_equity > 0 else None
    cr = round(cash / total_debt, 4) if total_debt > 0 else round(cash / max(total_liabilities, 1), 4)
    gm_r = round(gross_profit / revenue, 4) if revenue > 0 else None
    nm_r = round(net_income / revenue, 4) if revenue > 0 else None
    fm = round(fcf / revenue, 4) if revenue > 0 else None

    return {
        "income_statement": {
            "revenue": revenue, "cogs": cogs, "gross_profit": gross_profit,
            "operating_income": operating_income, "net_income": net_income, "eps": eps,
        },
        "balance_sheet": {
            "total_assets": total_assets, "total_liabilities": total_liabilities,
            "total_equity": total_equity, "cash": cash, "total_debt": total_debt,
        },
        "cash_flow": {
            "operating_cf": operating_cf, "investing_cf": investing_cf,
            "financing_cf": financing_cf, "fcf": fcf, "capex": capex,
        },
        "price": {
            "open": round(p_open, 2), "close": round(p_close, 2),
            "high": round(p_hi, 2), "low": round(p_lo, 2), "avg_price": avg_price,
        },
        "shares_outstanding": sh,
        "ratios": {
            "pe_ratio": pe, "pb_ratio": pb, "ev_ebitda": ev_eb,
            "roe": roe, "roa": roa, "debt_equity": de,
            "current_ratio": round(cr, 4), "gross_margin": gm_r,
            "net_margin": nm_r, "fcf_margin": fm,
        },
    }


def generate_new_company(profile, rng):
    """Generate 9 years of data for a new company from its profile."""
    p = profile
    base_year = 2020
    result = {}

    prev_close = p["bp"]
    for year in YEARS:
        offset = year - base_year
        growth_factor = (1 + p["rg"]) ** offset
        rev_shock = p.get("revs", {}).get(year, 1.0)
        noise = 1 + rng.uniform(-0.02, 0.02)
        rev = p["rev"] * growth_factor * rev_shock * noise

        mo = p.get("mo", {}).get(year, {})
        gm = mo.get("gm", p["gm"]) + rng.uniform(-0.005, 0.005)
        om = mo.get("om", p["om"]) + rng.uniform(-0.005, 0.005)
        nm = mo.get("nm", p["nm"]) + rng.uniform(-0.003, 0.003)
        ocf_pct = mo.get("ocf", p["ocf"]) + rng.uniform(-0.01, 0.01)

        if om >= gm:
            om = gm - 0.02

        am = p["am"] * (1 + rng.uniform(-0.05, 0.05))
        price_factor = (1 + p["pg"]) ** offset
        p_open = prev_close
        annual_return = p["pg"] + rng.uniform(-0.15, 0.15)
        if year in p.get("revs", {}):
            annual_return += (rev_shock - 1) * 0.5
        p_close = p_open * (1 + annual_return)
        p_close = max(p_close, 1.0)
        p_hi = max(p_open, p_close) * (1 + rng.uniform(0.05, 0.25))
        p_lo = min(p_open, p_close) * (1 - rng.uniform(0.05, 0.25))
        p_lo = max(p_lo, 0.50)
        prev_close = p_close

        sh = round(p["sh"] * (1 - 0.01 * offset + rng.uniform(-0.01, 0.01)))
        sh = max(sh, 10)

        record = make_record(
            rev, gm, om, nm, am, p["eq"], p["dp"], p["cp"], p["cx"],
            ocf_pct, sh, p_open, p_close, p_hi, p_lo,
        )
        result[str(year)] = record

    return result


def extrapolate_existing(data, ticker, target_year, rng):
    """Generate a new year for an existing company by extrapolating from nearest year."""
    existing_years = sorted(int(y) for y in data[ticker].keys())
    growth = EXISTING_GROWTH.get(ticker, 0.05)

    if target_year < min(existing_years):
        anchor_year = min(existing_years)
        years_back = anchor_year - target_year
        factor = (1 + growth) ** (-years_back)
    else:
        anchor_year = max(existing_years)
        years_fwd = target_year - anchor_year
        factor = (1 + growth) ** years_fwd

    anchor = data[ticker][str(anchor_year)]
    inc = anchor["income_statement"]
    bs = anchor["balance_sheet"]
    cf = anchor["cash_flow"]
    pr = anchor["price"]
    sh = anchor["shares_outstanding"]

    noise = 1 + rng.uniform(-0.02, 0.02)
    rev = inc["revenue"] * factor * noise
    gm = inc["gross_profit"] / inc["revenue"] if inc["revenue"] > 0 else 0.3
    om = inc["operating_income"] / inc["revenue"] if inc["revenue"] > 0 else 0.1
    nm = inc["net_income"] / inc["revenue"] if inc["revenue"] > 0 else 0.05
    am = bs["total_assets"] / inc["revenue"] if inc["revenue"] > 0 else 2.0
    eq_pct = bs["total_equity"] / bs["total_assets"] if bs["total_assets"] > 0 else 0.3
    dp = bs["total_debt"] / bs["total_assets"] if bs["total_assets"] > 0 else 0.3
    cp = bs["cash"] / bs["total_assets"] if bs["total_assets"] > 0 else 0.1
    cx = cf["capex"] / inc["revenue"] if inc["revenue"] > 0 else 0.03
    ocf_pct = cf["operating_cf"] / inc["revenue"] if inc["revenue"] > 0 else 0.15

    gm += rng.uniform(-0.005, 0.005)
    om += rng.uniform(-0.005, 0.005)
    nm += rng.uniform(-0.003, 0.003)

    if om >= gm:
        om = gm - 0.02

    price_growth = growth + rng.uniform(-0.10, 0.10)
    if target_year < anchor_year:
        p_close = pr["open"] / ((1 + price_growth) ** (anchor_year - target_year))
    else:
        p_close = pr["close"] * ((1 + price_growth) ** (target_year - anchor_year))

    p_close = max(p_close, 1.0)
    if target_year < anchor_year:
        p_open = p_close / (1 + price_growth + rng.uniform(-0.05, 0.05))
    else:
        p_open = pr["close"]

    p_open = max(p_open, 0.50)
    p_hi = max(p_open, p_close) * (1 + rng.uniform(0.05, 0.20))
    p_lo = min(p_open, p_close) * (1 - rng.uniform(0.05, 0.20))
    p_lo = max(p_lo, 0.50)

    new_sh = round(sh * (1 + rng.uniform(-0.02, 0.02)))

    return make_record(rev, gm, om, nm, am, eq_pct, dp, cp, cx, ocf_pct,
                       new_sh, p_open, p_close, p_hi, p_lo)


def compute_sectors(data):
    """Compute sector medians from company data."""
    sectors_out = {}
    for sector in sorted(set(TICKER_SECTORS.values())):
        sectors_out[sector] = {}
        tickers_in_sector = [t for t, s in TICKER_SECTORS.items() if s == sector]

        for year in YEARS:
            year_str = str(year)
            metric_values = {m: [] for m in SECTOR_METRICS}

            for ticker in tickers_in_sector:
                if ticker not in data or year_str not in data[ticker]:
                    continue
                ratios = data[ticker][year_str].get("ratios", {})
                for m in SECTOR_METRICS:
                    val = ratios.get(m)
                    if val is not None:
                        metric_values[m].append(val)

            year_data = {}
            for m in SECTOR_METRICS:
                vals = metric_values[m]
                if vals:
                    year_data[m] = round(statistics.median(vals), 2)
                else:
                    year_data[m] = 0.0
            sectors_out[sector][year_str] = year_data

    sectors_out["_ticker_sectors"] = dict(TICKER_SECTORS)
    return sectors_out


def validate(data):
    """Quick invariant check."""
    errors = 0
    for ticker in data:
        for year_str in data[ticker]:
            e = data[ticker][year_str]
            inc, bs, cf = e["income_statement"], e["balance_sheet"], e["cash_flow"]
            if inc["gross_profit"] != inc["revenue"] - inc["cogs"]:
                print(f"  {ticker}/{year_str}: gross_profit mismatch")
                errors += 1
            if bs["total_equity"] != bs["total_assets"] - bs["total_liabilities"]:
                print(f"  {ticker}/{year_str}: equity mismatch")
                errors += 1
            if cf["fcf"] != cf["operating_cf"] - cf["capex"]:
                print(f"  {ticker}/{year_str}: fcf mismatch")
                errors += 1
            if inc["operating_income"] >= inc["gross_profit"]:
                print(f"  {ticker}/{year_str}: operating_income >= gross_profit")
                errors += 1
    return errors


def main():
    rng = random.Random(42)

    with open(OUTPUT_DIR / "financials.json") as f:
        data = json.load(f)

    print(f"Existing data: {len(data)} tickers, years per ticker: {len(list(data.values())[0])}")

    for ticker in list(data.keys()):
        for year in YEARS:
            if str(year) not in data[ticker]:
                data[ticker][str(year)] = extrapolate_existing(data, ticker, year, rng)

    for ticker, profile in PROFILES.items():
        if ticker not in data:
            data[ticker] = generate_new_company(profile, rng)
            print(f"  Generated {ticker} ({profile['sector']})")

    errs = validate(data)
    if errs:
        print(f"VALIDATION FAILED: {errs} errors")
        return

    sectors = compute_sectors(data)

    with open(OUTPUT_DIR / "financials.json", "w") as f:
        json.dump(data, f, indent=2)
    with open(OUTPUT_DIR / "sectors.json", "w") as f:
        json.dump(sectors, f, indent=2)

    tickers = sorted(data.keys())
    sectors_list = sorted(s for s in sectors if s != "_ticker_sectors")
    print(f"\nGenerated: {len(tickers)} tickers, {len(YEARS)} years, {len(sectors_list)} sectors")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Sectors: {', '.join(sectors_list)}")
    print(f"  Years: {YEARS[0]}-{YEARS[-1]}")
    total_records = sum(len(data[t]) for t in data)
    print(f"  Total records: {total_records}")


if __name__ == "__main__":
    main()
