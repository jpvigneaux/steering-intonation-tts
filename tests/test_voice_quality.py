

def test_a_set_with_only_the_neutral_condition_is_summarised_without_a_counterpart(tmp_path):
    """The neutral prompt has no opposite condition, so 'outside the natural range' is undefined.

    It is left as NaN and the matched-alpha table is empty, rather than the run failing on a
    condition that does not exist in that set.
    """
    import pandas as pd  # noqa: PLC0415

    natural = pd.DataFrame({"offset_p05": [3.0], "offset_p95": [7.0], "offset_mean": [5.0]},
                           index=pd.Index(["N"], name="condition"))
    assert not {"A", "I"} <= set(natural.index)
    paired = pd.DataFrame({"offset_p05": [3.0, 4.0], "offset_p95": [6.0, 8.0],
                           "offset_mean": [4.2, 6.1]},
                          index=pd.Index(["A", "I"], name="condition"))
    assert {"A", "I"} <= set(paired.index)
