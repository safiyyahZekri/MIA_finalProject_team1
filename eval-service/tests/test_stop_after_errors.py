from scripts.run_answer_eval import trailing_errors


def test_counts_only_the_unbroken_run_of_errors_at_the_end():
    """Run decomposition-on-100 hit a workspace usage limit at question 45 and
    recorded the remaining 56 questions as errors; the runner now stops after
    a few in a row instead."""
    rows = [{"error": None}, {"error": "503"}, {"error": None}, {"error": "503"}, {"error": "usage limit"}]

    assert trailing_errors(rows) == 2


def test_no_trailing_errors():
    assert trailing_errors([{"error": "503"}, {"error": None}]) == 0
    assert trailing_errors([]) == 0
