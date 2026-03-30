"""Tests for grader modules."""

from server.graders import task1_grader, task2_grader, task3_grader


def test_task1_exact():
    result = task1_grader.grade(25.31)
    assert result["score"] == 1.0


def test_task1_close():
    result = task1_grader.grade(25.5)
    assert result["score"] == 0.5


def test_task1_wrong():
    result = task1_grader.grade(50.0)
    assert result["score"] == 0.0


def test_task1_non_numeric():
    result = task1_grader.grade("not a number")
    assert result["score"] == 0.0


def test_task2_correct():
    result = task2_grader.grade({"company": "META", "delta": -5.47})
    assert result["score"] == 1.0


def test_task2_right_company_wrong_delta():
    result = task2_grader.grade({"company": "META", "delta": 10.0})
    assert result["score"] == 0.5


def test_task2_wrong_company():
    result = task2_grader.grade({"company": "MSFT", "delta": -5.47})
    assert result["score"] == 0.5


def test_task2_string_answer():
    result = task2_grader.grade("META")
    assert result["score"] == 0.5


def test_task3_correct():
    answer = {
        "qualifying_companies": ["TSLA"],
        "details": {
            "TSLA": {
                "negative_fcf_years": [2020, 2021],
                "pe_above_30_years": [2020, 2021, 2022, 2023],
            }
        },
    }
    result = task3_grader.grade(answer)
    assert result["score"] == 0.9


def test_task3_partial():
    answer = {
        "qualifying_companies": ["TSLA"],
        "details": {
            "TSLA": {
                "negative_fcf_years": [2020],
                "pe_above_30_years": [2020, 2021],
            }
        },
    }
    result = task3_grader.grade(answer)
    assert 0.0 < result["score"] < 0.9


def test_task3_wrong():
    answer = {
        "qualifying_companies": ["GM"],
        "details": {},
    }
    result = task3_grader.grade(answer)
    assert result["score"] < 0.3


def test_task3_non_dict():
    result = task3_grader.grade("TSLA")
    assert result["score"] == 0.0
