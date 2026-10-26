import numpy as np
import pandas as pd
import pytest

from app.services.evaluator import DataQualityEvaluator
from tests.data_utils import make_df

ev = DataQualityEvaluator.evaluate


def base():
    return make_df(400, seed=1, missing=False).drop(columns="student_id")


def test_identical_data_scores_perfectly():
    real = base()
    m = ev(real, real.copy())
    assert m["average_ks_statistic"] == 0.0
    assert all(v["tvd"] == 0.0 for v in m["categorical_metrics"].values())
    assert m["correlation_frobenius_delta"] == 0.0 and m["domain_validity_percent"] == 100.0
    assert m["overall_fidelity_score"] == 100.0


def test_resampled_data_from_the_same_distribution_scores_high():
    m = ev(base(), make_df(400, seed=2, missing=False).drop(columns="student_id"))
    assert m["average_ks_statistic"] < 0.15 and m["overall_fidelity_score"] > 85


def test_shifted_numeric_and_skewed_categories_are_penalised():
    real = base()
    bad = real.copy()
    bad["hours"] = bad["hours"] + 12
    bad["score"] = bad["score"].sample(frac=1, random_state=0).to_numpy()  # breaks the hours-score correlation
    bad["gender"] = "Male"
    m = ev(real, bad)
    assert m["ks_metrics"]["hours"]["status"] == "Failed"
    assert m["categorical_metrics"]["gender"]["tvd"] > 0.4
    assert m["correlation_mean_abs_delta"] > 0.3  # only the hours-score pair is broken
    assert m["domain_validity_percent"] < 100  # shifted hours leave the real [min, max]
    assert m["overall_fidelity_score"] < ev(real, real.copy())["overall_fidelity_score"] - 30


def test_invalid_categories_and_missing_values_reduce_validity():
    real = base()
    syn = real.copy()
    syn.loc[syn.index[:40], "grade"] = "Alien"
    syn.loc[syn.index[40:80], "age"] = np.nan
    m = ev(real, syn)
    assert m["categorical_metrics"]["grade"]["invalid_categories"] == ["Alien"]
    assert m["domain_validity_percent"] == pytest.approx(100 * (1 - 80 / (len(real) * 5)), abs=0.1)


def test_metrics_are_computed_not_constant():
    real = base()
    a = ev(real, real.sample(frac=1, random_state=1))["overall_fidelity_score"]
    b = ev(real, real.assign(hours=real["hours"] * 0.5))["overall_fidelity_score"]
    assert a != b


def test_constant_and_degenerate_columns_do_not_crash():
    real = pd.DataFrame({"k": [3.0] * 50, "n": np.arange(50.0), "c": ["a"] * 50})
    m = ev(real, real.copy())
    assert m["ks_metrics"]["k"]["ks_statistic"] == 0.0
    assert any("Constant" in w for w in m["warnings"])
    assert ev(real, real.assign(k=4.0))["ks_metrics"]["k"]["ks_statistic"] == 1.0


def test_unavailable_metrics_are_none_not_invented():
    only_cat = pd.DataFrame({"c": list("ab") * 20})
    m = ev(only_cat, only_cat.copy())
    assert m["average_ks_statistic"] is None and m["correlation_frobenius_delta"] is None
    assert m["fidelity_components"]["numeric"] is None and m["overall_fidelity_score"] == 100.0
    one_num = pd.DataFrame({"x": np.arange(40.0)})
    assert ev(one_num, one_num.copy())["correlation_frobenius_delta"] is None


def test_small_samples_and_extra_columns_produce_warnings():
    real = base()
    m = ev(real, real.head(10).assign(extra=1))
    assert any("Small sample" in w for w in m["warnings"]) and any("extra" in w for w in m["warnings"])


def test_empty_or_disjoint_inputs_raise():
    real = base()
    with pytest.raises(ValueError):
        ev(real, real.iloc[0:0])
    with pytest.raises(ValueError):
        ev(real, pd.DataFrame({"other": [1, 2, 3]}))


def test_limitations_are_stated_and_weights_sum_to_one():
    m = ev(base(), base())
    assert "privacy" in m["limitations"]
    assert sum(m["fidelity_weights"].values()) == pytest.approx(1.0)
