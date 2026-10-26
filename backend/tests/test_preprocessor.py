import numpy as np
import pandas as pd
import pytest

from app.services.preprocessor import DataPreprocessor, PreprocessingError
from tests.data_utils import make_df


def schema_for(df, tmp_path):
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    return DataPreprocessor.inspect_csv(str(p))


def cfg(name, typ, strat="median", included=True):
    return {"name": name, "selectedType": typ, "imputationStrategy": strat, "included": included}


def test_inspect_infers_types_and_excludes_ids(tmp_path):
    rows, cols, schema, preview = schema_for(make_df(), tmp_path)
    by = {c["name"]: c for c in schema}
    assert (rows, cols) == (300, 6) and len(preview) == 50
    assert by["student_id"]["selectedType"] == "drop" and not by["student_id"]["included"]
    assert by["gender"]["selectedType"] == "categorical"
    assert by["hours"]["selectedType"] == "numeric" and by["hours"]["nullCount"] == 6
    assert by["age"]["selectedType"] == "numeric"


def test_id_heuristic_does_not_swallow_ordinary_columns(tmp_path):
    df = pd.DataFrame({"paid": [1.5, 2.5, 3.5], "midterm": [1, 2, 3], "userId": ["a", "b", "c"], "x_id": [1, 2, 3]})
    by = {c["name"]: c for c in schema_for(df, tmp_path)[2]}
    assert by["paid"]["included"] and by["midterm"]["included"]
    assert not by["userId"]["included"] and not by["x_id"]["included"]


def test_continuous_unique_float_column_is_not_an_identifier(tmp_path):
    df = pd.DataFrame({"x": np.linspace(0.1, 9.9, 40), "c": ["a", "b"] * 20})
    assert {c["name"]: c for c in schema_for(df, tmp_path)[2]}["x"]["included"]


def test_all_null_column_is_excluded_not_crashing(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "empty": [None, None, None]})
    by = {c["name"]: c for c in schema_for(df, tmp_path)[2]}
    assert not by["empty"]["included"]


@pytest.mark.parametrize("content,fragment", [(b"", "empty"), (b"a,b\n", "no data rows")])
def test_read_csv_rejects_unusable_files(tmp_path, content, fragment):
    p = tmp_path / "bad.csv"
    p.write_bytes(content)
    with pytest.raises(PreprocessingError, match=fragment):
        DataPreprocessor.inspect_csv(str(p))


def test_missing_values_are_imputed_and_output_is_normalised(tmp_path):
    df = make_df()
    schema = schema_for(df, tmp_path)[2]
    pre = DataPreprocessor()
    X = pre.fit_transform(df, schema)
    assert X.shape[0] == 300 and not np.isnan(X).any()
    assert X.min() >= 0.0 and X.max() <= 1.0
    # 'hours' median fill, 'grade' mode fill
    assert pre.fill_values["hours"] == pytest.approx(df["hours"].median())
    assert pre.fill_values["grade"] == df["grade"].mode().iloc[0]
    assert pre.processed_feature_names[:2] == ["gender_Female", "gender_Male"]
    assert [d["name"] for d in pre.dropped_columns] == ["student_id"]


def test_mean_strategy_and_drop_rows_strategy(tmp_path):
    df = make_df(60)
    schema = schema_for(df, tmp_path)[2]
    for c in schema:
        if c["name"] == "hours":
            c["imputationStrategy"] = "mean"
    pre = DataPreprocessor()
    pre.fit(df, schema)
    assert pre.fill_values["hours"] == pytest.approx(df["hours"].mean())

    for c in schema:
        if c["name"] == "hours":
            c["imputationStrategy"] = "drop"
    pre = DataPreprocessor()
    X = pre.fit_transform(df, schema)
    assert X.shape[0] == 54 and any("removed" in w for w in pre.warnings)


def test_invalid_numeric_values_become_missing_with_warning():
    df = pd.DataFrame({"v": ["1", "2", "oops", "4", "5", "6"], "c": list("aabbab")})
    pre = DataPreprocessor()
    X = pre.fit_transform(df, [cfg("v", "numeric"), cfg("c", "categorical", "mode")])
    assert not np.isnan(X).any() and any("non-numeric" in w for w in pre.warnings)


def test_mostly_non_numeric_column_marked_numeric_is_an_error():
    df = pd.DataFrame({"v": ["a", "b", "c", "1"]})
    with pytest.raises(PreprocessingError, match="not numbers"):
        DataPreprocessor().fit(df, [cfg("v", "numeric")])


def test_constant_columns_are_kept_and_reproduced():
    df = pd.DataFrame({"k": [7.0] * 20, "n": np.arange(20.0), "c": ["z"] * 20})
    pre = DataPreprocessor()
    X = pre.fit_transform(df, [cfg("k", "numeric"), cfg("n", "numeric"), cfg("c", "categorical", "mode")])
    assert not np.isnan(X).any()
    assert sum("constant" in w for w in pre.warnings) == 1 and sum("single category" in w for w in pre.warnings) == 1
    out = pre.inverse_transform(np.full((5, X.shape[1]), 0.5, dtype=np.float32))
    assert (out["k"] == 7.0).all() and (out["c"] == "z").all()


def test_config_errors_are_reported_not_ignored():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(PreprocessingError, match="not found"):
        DataPreprocessor().fit(df, [cfg("a", "numeric"), cfg("ghost", "numeric")])
    with pytest.raises(PreprocessingError, match="selectedType"):
        DataPreprocessor().fit(df, [cfg("a", "weird")])
    with pytest.raises(PreprocessingError, match="imputationStrategy"):
        DataPreprocessor().fit(df, [cfg("a", "numeric", "magic")])
    with pytest.raises(PreprocessingError, match="No usable columns"):
        DataPreprocessor().fit(df, [cfg("a", "numeric", included=False)])


def test_too_many_categories_is_an_error():
    df = pd.DataFrame({"c": [f"v{i}" for i in range(80)]})
    with pytest.raises(PreprocessingError, match="distinct values"):
        DataPreprocessor().fit(df, [cfg("c", "categorical", "mode")])


def test_inverse_transform_restores_types_and_valid_categories(tmp_path):
    df = make_df()
    pre = DataPreprocessor()
    X = pre.fit_transform(df, schema_for(df, tmp_path)[2])
    out = pre.inverse_transform(X, rng=np.random.default_rng(1))
    assert list(out.columns) == ["gender", "grade", "age", "hours", "score"]
    assert pd.api.types.is_integer_dtype(out["age"]) and pd.api.types.is_float_dtype(out["hours"])
    assert set(out["gender"]) <= set(df["gender"]) and set(out["grade"]) <= set(df["grade"].dropna())
    assert out["age"].between(df["age"].min(), df["age"].max()).all()
    assert (out["hours"].round(1) == out["hours"]).all()  # 1 decimal preserved


def test_save_load_roundtrip_gives_identical_transformations(tmp_path):
    df = make_df()
    pre = DataPreprocessor()
    X = pre.fit_transform(df, schema_for(df, tmp_path)[2])
    pre.save(str(tmp_path / "p.json"))
    loaded = DataPreprocessor()
    loaded.load(str(tmp_path / "p.json"))
    np.testing.assert_array_equal(loaded.transform(df), X)
    pd.testing.assert_frame_equal(loaded.inverse_transform(X), pre.inverse_transform(X))
    assert loaded.warnings == pre.warnings and loaded.dropped_columns == pre.dropped_columns


def test_transform_never_refits_and_handles_unseen_categories(tmp_path):
    df = make_df()
    pre = DataPreprocessor()
    pre.fit(df, schema_for(df, tmp_path)[2])
    scalers_before = dict(pre.numeric_scalers["hours"])
    new = df.head(5).copy()
    new["gender"] = "Other"                 # unseen category
    new.loc[new.index[0], "hours"] = 999.0  # outside the fitted range
    X = pre.transform(new)
    assert pre.unseen_category_count == 5
    assert X[:, :2].sum() == 0              # all-zero one-hot block
    assert X.max() <= 1.0 and pre.numeric_scalers["hours"] == scalers_before


def test_corrupt_or_incompatible_pipeline_file_raises(tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    (tmp_path / "old.json").write_text('{"numeric_scalers": {}}')
    with pytest.raises(PreprocessingError, match="corrupt"):
        DataPreprocessor().load(str(tmp_path / "bad.json"))
    with pytest.raises(PreprocessingError, match="incompatible"):
        DataPreprocessor().load(str(tmp_path / "old.json"))
    with pytest.raises(FileNotFoundError):
        DataPreprocessor().load(str(tmp_path / "missing.json"))
