"""Task 1 (Easy) — Single-Metric Computation grader.

Question: What was Apple's net profit margin for fiscal year 2022?
Ground truth: 25.31% (net_income 99803 / revenue 394328 * 100)
"""

GROUND_TRUTH = {
    "answer": 25.31,
    "tolerance_exact": 0.05,
    "tolerance_partial": 0.5,
    "required_fetches": ["income_statement:AAPL:2022"],
    "min_fetches": 1,
}


def grade(answer, ground_truth=None, **kwargs):
    gt = ground_truth or GROUND_TRUTH
    try:
        val = float(answer)
    except (TypeError, ValueError):
        return {"score": 0.0, "breakdown": {"accuracy": 0.0, "reason": "non-numeric answer"}}

    diff = abs(val - gt["answer"])
    if diff <= gt["tolerance_exact"]:
        accuracy = 1.0
    elif diff <= gt["tolerance_partial"]:
        accuracy = 0.5
    else:
        accuracy = 0.0

    return {
        "score": accuracy,
        "breakdown": {
            "accuracy": accuracy,
            "expected": gt["answer"],
            "submitted": val,
            "difference": round(diff, 4),
        },
    }
