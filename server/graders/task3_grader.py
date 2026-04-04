"""Task 3 (Hard) — Multi-Year Anomaly Detection grader.

Question: Among TSLA, F, GM — which had negative FCF in >= 2 of 4 years (2020-2023)
AND P/E > 30 in any of those years? State which years for each.

Ground truth: Only TSLA qualifies.
  - Negative FCF years: 2020, 2021
  - P/E > 30 years: 2020, 2021, 2022, 2023
"""

GROUND_TRUTH = {
    "qualifying_companies": ["TSLA"],
    "fcf_negative_years": {
        "TSLA": [2020, 2021],
    },
    "pe_above_30_years": {
        "TSLA": [2020, 2021, 2022, 2023],
    },
    "required_fetches": [
        "cash_flow:TSLA:2020", "cash_flow:TSLA:2021", "cash_flow:TSLA:2022", "cash_flow:TSLA:2023",
        "cash_flow:F:2020", "cash_flow:F:2021", "cash_flow:F:2022", "cash_flow:F:2023",
        "cash_flow:GM:2020", "cash_flow:GM:2021", "cash_flow:GM:2022", "cash_flow:GM:2023",
        "ratios:TSLA:2020", "ratios:TSLA:2021", "ratios:TSLA:2022", "ratios:TSLA:2023",
        "ratios:F:2020", "ratios:F:2021", "ratios:F:2022", "ratios:F:2023",
        "ratios:GM:2020", "ratios:GM:2021", "ratios:GM:2022", "ratios:GM:2023",
    ],
    "min_fetches": 6,
}


def _score_years(details: dict, ground_truth_years: dict, key: str) -> float:
    total_correct = 0
    total_expected = 0
    for company, expected_years in ground_truth_years.items():
        total_expected += len(expected_years)
        submitted = details.get(company, {}).get(key, [])
        submitted_set = set(submitted)
        for y in expected_years:
            if y in submitted_set:
                total_correct += 1
    return total_correct / total_expected if total_expected > 0 else 0.0


def grade(answer, ground_truth=None, **kwargs):
    gt = ground_truth or GROUND_TRUTH
    score = 0.0
    breakdown = {}

    if not isinstance(answer, dict):
        return {"score": 0.0, "breakdown": {"error": "expected dict answer"}}

    raw_companies = answer.get("qualifying_companies", [])
    if isinstance(raw_companies, str):
        raw_companies = [raw_companies]
    submitted_companies = {c.upper().strip() for c in raw_companies}
    expected_companies = set(gt["qualifying_companies"])

    overlap = len(submitted_companies & expected_companies)
    total = len(expected_companies)
    company_score = (overlap / total) * 0.30 if total > 0 else 0.0
    false_positives = len(submitted_companies - expected_companies)
    company_score = max(0.0, company_score - false_positives * 0.10)
    breakdown["correct_companies"] = round(company_score, 4)
    score += company_score

    details = answer.get("details", {})
    # Normalize keys to uppercase
    details = {k.upper(): v for k, v in details.items()} if isinstance(details, dict) else {}
    fcf_score = _score_years(details, gt["fcf_negative_years"], "negative_fcf_years")
    breakdown["correct_fcf_years"] = round(fcf_score * 0.30, 4)
    score += fcf_score * 0.30

    pe_score = _score_years(details, gt["pe_above_30_years"], "pe_above_30_years")
    breakdown["correct_pe_years"] = round(pe_score * 0.30, 4)
    score += pe_score * 0.30

    # Efficiency bonus (0.10) handled by reward engine, not grader.

    return {"score": round(score, 4), "breakdown": breakdown}
