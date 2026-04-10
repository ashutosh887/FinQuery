"""Task 2 (Medium) — Multi-Company Ratio Comparison grader.

Scoring: company=0.40, delta=0.40, efficiency=0.20.
"""

GROUND_TRUTH = {
    "correct_company": "META",
    "correct_delta": -5.47,
    "delta_tolerance": 0.5,
    "required_fetches": [
        "ratios:MSFT:2023",
        "ratios:GOOGL:2023",
        "ratios:META:2023",
        "sector_compare:MSFT:ev_ebitda:2023",
        "sector_compare:GOOGL:ev_ebitda:2023",
        "sector_compare:META:ev_ebitda:2023",
    ],
    "min_fetches": 3,
}


def grade(answer, ground_truth=None, step_count=None, max_steps=20):
    gt = ground_truth or GROUND_TRUTH
    score = 0.0
    breakdown = {}

    if isinstance(answer, dict):
        company = str(answer.get("company", "")).upper().strip()
        delta = answer.get("delta")
    elif isinstance(answer, str):
        company = answer.strip().upper()
        delta = None
    else:
        return {"score": 0.01, "breakdown": {"error": "invalid answer format"}}

    if company == gt["correct_company"]:
        breakdown["correct_company"] = 0.40
        score += 0.40
    else:
        breakdown["correct_company"] = 0.0

    if delta is not None:
        try:
            diff = abs(float(delta) - gt["correct_delta"])
            if diff <= gt["delta_tolerance"]:
                breakdown["correct_delta"] = 0.40
                score += 0.40
            else:
                breakdown["correct_delta"] = 0.0
        except (TypeError, ValueError):
            breakdown["correct_delta"] = 0.0
    else:
        breakdown["correct_delta"] = 0.0

    if step_count is not None and step_count <= max_steps * 0.5:
        breakdown["efficiency"] = 0.20
        score += 0.20
    else:
        breakdown["efficiency"] = 0.0

    return {"score": max(0.01, min(0.99, round(score, 4))), "breakdown": breakdown}
