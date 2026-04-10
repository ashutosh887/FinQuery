"""Task 3 (Hard) — Multi-Year Anomaly Detection grader.

Supports multiple anomaly patterns via configurable condition keys.
Scoring: companies=0.30, condition_a=0.30, condition_b=0.30, efficiency=0.10.
"""

GROUND_TRUTH = {
    "qualifying_companies": ["TSLA"],
    "condition_a_key": "negative_fcf_years",
    "condition_b_key": "pe_above_30_years",
    "condition_a_years": {
        "TSLA": [2020, 2021],
    },
    "condition_b_years": {
        "TSLA": [2020, 2021, 2022, 2023],
    },
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
        return {"score": 0.01, "breakdown": {"error": "expected dict answer"}}

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
    details = {k.upper(): v for k, v in details.items()} if isinstance(details, dict) else {}

    cond_a_key = gt.get("condition_a_key", "negative_fcf_years")
    cond_b_key = gt.get("condition_b_key", "pe_above_30_years")
    cond_a_data = gt.get("condition_a_years", gt.get("fcf_negative_years", {}))
    cond_b_data = gt.get("condition_b_years", gt.get("pe_above_30_years", {}))

    a_score = _score_years(details, cond_a_data, cond_a_key)
    breakdown[f"correct_{cond_a_key}"] = round(a_score * 0.30, 4)
    score += a_score * 0.30

    b_score = _score_years(details, cond_b_data, cond_b_key)
    breakdown[f"correct_{cond_b_key}"] = round(b_score * 0.30, 4)
    score += b_score * 0.30

    return {"score": max(0.01, min(0.99, round(score, 4))), "breakdown": breakdown}
