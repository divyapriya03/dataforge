"""End-to-end tests of the dataset -> preprocess -> train -> generate -> evaluate -> export pipeline.

Each test runs against its own temporary data directory and SQLite file (see conftest.py).
"""
import io
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from app.core import config
from app.core import database as db
from app.models.vae import VariationalAutoencoder
from tests.conftest import train, upload
from tests.data_utils import csv_bytes, make_df

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def count(table):
    con = db.get_db_connection()
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


def generate(client, model, **overrides):
    body = {"model_id": model["model_id"], "dataset_id": model["dataset_id"], "num_records": 200, "random_seed": 7}
    body.update(overrides)
    return client.post("/api/generate", json=body)


def files_in(kind):
    return sorted(os.listdir(config.get_dir(kind)))


# ------------------------------------------------------------------ 1. valid upload
def test_valid_upload_is_stored_and_described(client, env):
    ds = upload(client)
    assert ds["rowCount"] == 300 and ds["columnCount"] == 6 and ds["id"].startswith("ds_")
    by = {c["name"]: c for c in ds["columns"]}
    assert by["student_id"]["included"] is False and by["gender"]["selectedType"] == "categorical"
    assert files_in("uploads") == [f"{ds['id']}_students.csv"]
    # the path persisted in SQLite is relative to the data dir, so the project can be moved
    assert db.get_dataset(ds["id"])["file_path"] == f"uploads/{ds['id']}_students.csv"
    assert client.get(f"/api/datasets/{ds['id']}").json()["preview"][0]["gender"] in ("Female", "Male")
    assert [d["id"] for d in client.get("/api/datasets").json()] == [ds["id"]]
    # the upload response carries the same stored timestamp as the list and detail endpoints
    assert ds["uploadDate"] == client.get("/api/datasets").json()[0]["uploadDate"] == client.get(f"/api/datasets/{ds['id']}").json()["uploadDate"]
    assert ds["uploadDate"] != "Just now" and len(ds["uploadDate"]) == 19


def test_upload_strips_directory_parts_from_the_filename(client):
    r = client.post("/api/datasets/upload", files={"file": ("..\\..\\evil.csv", csv_bytes(make_df(20)), "text/csv")})
    assert r.status_code == 200 and r.json()["filename"] == "evil.csv"
    assert all(os.path.dirname(f) == "" for f in files_in("uploads"))


# ------------------------------------------------------------------ 2. invalid uploads
@pytest.mark.parametrize("name,content,fragment", [
    ("empty.csv", b"", "empty"),
    ("header_only.csv", b"a,b,c\n", "no data rows"),
    ("notes.txt", b"a,b\n1,2\n", "Only CSV"),
])
def test_invalid_uploads_are_rejected_and_leave_nothing_behind(client, name, content, fragment):
    r = client.post("/api/datasets/upload", files={"file": (name, content, "text/csv")})
    assert r.status_code == 400 and fragment in r.json()["detail"]
    assert files_in("uploads") == [] and count("datasets") == 0


def test_oversized_upload_is_rejected_with_413(client, monkeypatch):
    monkeypatch.setattr(config, "max_upload_bytes", lambda: 200)
    r = client.post("/api/datasets/upload", files={"file": ("big.csv", csv_bytes(make_df(200)), "text/csv")})
    assert r.status_code == 413
    assert files_in("uploads") == [] and count("datasets") == 0


# ------------------------------------------------------------------ 3/4. preprocessing endpoint
def test_preprocess_reports_layout_dropped_columns_and_warnings(client, dataset):
    r = client.post("/api/preprocess", json={"dataset_id": dataset["id"], "columns": dataset["columns"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["processed_dimensions"] == 2 + 4 + 3 and body["rows_used"] == 300
    assert body["feature_names"][:2] == ["gender_Female", "gender_Male"]
    assert body["dropped_columns"] == [{"name": "student_id", "reason": "excluded by configuration"}]
    assert body["ready_for_training"] is True


def test_preprocess_rejects_bad_configuration(client, dataset):
    cols = json.loads(json.dumps(dataset["columns"]))
    cols[1]["selectedType"] = "weird"
    r = client.post("/api/preprocess", json={"dataset_id": dataset["id"], "columns": cols})
    assert r.status_code == 400 and "selectedType" in r.json()["detail"]
    r = client.post("/api/preprocess", json={"dataset_id": "ds_nope", "columns": dataset["columns"]})
    assert r.status_code == 404


def test_dataset_with_only_unusable_columns_cannot_be_trained(client):
    df = pd.DataFrame({"user_id": range(30), "blank": [None] * 30})
    ds = upload(client, df)
    r = train(client, ds["id"])
    assert r.status_code == 400 and "No usable columns" in r.json()["detail"]
    assert count("models") == 0


# ------------------------------------------------------------------ 5. successful training
def test_training_fits_and_persists_a_model(client, dataset, model):
    assert model["status"] == "completed" and model["model_type"] == "numpy-vae"
    assert model["current_epoch"] == model["total_epochs"] == 30 == len(model["loss_history"])
    losses = [p["totalLoss"] for p in model["loss_history"]]
    assert all(np.isfinite(losses)) and losses[-1] < losses[0]
    assert model["completed_at"] and model["error_message"] is None
    assert model["params"]["epochs"] == 30 and model["params"]["latent_dim"] == 4 and model["params"]["random_seed"] == 3
    assert "columns" not in model["params"]

    row = db.get_model(model["model_id"])
    assert row["weights_path"] == f"models/{model['model_id']}_weights.npz"
    assert row["preprocessor_path"] == f"models/{model['model_id']}_preprocessor.json"
    assert {os.path.basename(row["weights_path"]), os.path.basename(row["preprocessor_path"])} <= set(files_in("models"))
    # status and list endpoints return the stored (not recomputed) state
    assert client.get(f"/api/models/status/{model['model_id']}").json() == model
    assert [m["model_id"] for m in client.get("/api/models").json()] == [model["model_id"]]


def test_training_with_same_seed_is_reproducible(client, dataset):
    a = train(client, dataset["id"], epochs=5).json()
    b = train(client, dataset["id"], epochs=5).json()
    assert a["loss_history"] == b["loss_history"] and a["model_id"] != b["model_id"]


def test_custom_column_configuration_is_used_for_training(client, dataset):
    cols = json.loads(json.dumps(dataset["columns"]))
    for c in cols:
        if c["name"] == "score":
            c["included"], c["selectedType"] = False, "drop"
    m = train(client, dataset["id"], columns=cols, epochs=3).json()
    pipeline = json.load(open(os.path.join(config.get_dir("models"), f"{m['model_id']}_preprocessor.json")))
    assert "score" not in pipeline["column_order"]
    gen = generate(client, m).json()
    assert "score" not in gen["preview_records"][0]


# ------------------------------------------------------------------ 6. failed training configuration
@pytest.mark.parametrize("body", [
    {"hidden_layers": [64]},
    {"hidden_layers": [64, 32, 16]},
    {"epochs": 0},
    {"latent_dim": 1},
    {"learning_rate": 0},
    {"batch_size": 2},
    {"model_name": " "},
])
def test_invalid_training_configuration_is_rejected_without_records(client, dataset, body):
    r = train(client, dataset["id"], **body)
    assert r.status_code == 422
    assert count("models") == 0 and files_in("models") == []


def test_training_unknown_dataset_and_tiny_dataset(client):
    assert train(client, "ds_missing").status_code == 404
    small = upload(client, pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": list("xyx")}), name="tiny.csv")
    r = train(client, small["id"])
    assert r.status_code == 400 and "At least" in r.json()["detail"]
    assert count("models") == 0


def test_dataset_file_missing_on_disk_is_reported(client, dataset, env):
    os.remove(os.path.join(config.get_dir("uploads"), os.path.basename(db.get_dataset(dataset["id"])["file_path"])))
    r = train(client, dataset["id"])
    assert r.status_code == 404 and "missing" in r.json()["detail"]
    assert count("models") == 0


def test_divergent_training_is_recorded_as_failed_not_completed(client, dataset, monkeypatch):
    def nan_epoch(self, X, batch_size):
        return float("nan"), float("nan"), float("nan")
    monkeypatch.setattr(VariationalAutoencoder, "train_epoch", nan_epoch)
    r = train(client, dataset["id"])
    assert r.status_code == 422 and "diverged" in r.json()["detail"]
    rows = client.get("/api/models").json()
    assert len(rows) == 1 and rows[0]["status"] == "failed" and "diverged" in rows[0]["error_message"]
    assert files_in("models") == []
    # a failed model can never be used for generation
    g = generate(client, {"model_id": rows[0]["model_id"], "dataset_id": dataset["id"]})
    assert g.status_code == 409 and "not ready" in g.json()["detail"]
    assert count("experiments") == 0


def test_failure_while_saving_is_recorded_without_leaking_details(client, dataset, monkeypatch):
    def boom(self, path):
        raise OSError("disk exploded at /secret/location")
    monkeypatch.setattr(VariationalAutoencoder, "save", boom)
    r = train(client, dataset["id"])
    assert r.status_code == 500 and "secret" not in r.text
    m = client.get("/api/models").json()[0]
    assert m["status"] == "failed" and "secret" not in (m["error_message"] or "")


# ------------------------------------------------------------------ 7. persistence & reload
def test_saved_weights_reload_into_an_identical_model(client, model):
    row = db.get_model(model["model_id"])
    params = row["params"]
    pre = json.load(open(os.path.join(config.get_data_dir(), row["preprocessor_path"])))
    path = os.path.join(config.get_data_dir(), row["weights_path"])
    a = VariationalAutoencoder(len(pre["processed_feature_names"]), params["latent_dim"], params["hidden_layers"])
    b = VariationalAutoencoder(len(pre["processed_feature_names"]), params["latent_dim"], params["hidden_layers"], seed=1)
    a.load(path)
    b.load(path)
    np.testing.assert_array_equal(a.generate(25, seed=3), b.generate(25, seed=3))


# ------------------------------------------------------------------ 8/9. generation
def test_generation_returns_requested_rows_with_the_original_schema(client, dataset, model, env):
    r = generate(client, model, num_records=250)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["num_records"] == 250 and len(body["preview_records"]) == 50
    assert body["experiment_id"].startswith("gen_") and body["output_filename"] in files_in("synthetic")

    out = pd.read_csv(os.path.join(config.get_dir("synthetic"), body["output_filename"]))
    original = make_df()
    assert len(out) == 250
    assert list(out.columns) == ["gender", "grade", "age", "hours", "score"]  # original order, id column excluded
    assert not out.isna().any().any()
    assert set(out["gender"]) <= set(original["gender"]) and set(out["grade"]) <= set(original["grade"].dropna())
    assert pd.api.types.is_integer_dtype(out["age"]) and out["age"].between(18, 29).all()
    assert out["hours"].between(original["hours"].min(), original["hours"].max()).all()
    # synthetic rows are generated, not copied from the original
    assert not out.merge(original, on=list(out.columns)).shape[0] > 25


def test_same_seed_reproduces_and_different_seed_differs(client, model):
    a = generate(client, model, random_seed=11).json()["preview_records"]
    b = generate(client, model, random_seed=11).json()["preview_records"]
    c = generate(client, model, random_seed=12).json()["preview_records"]
    assert a == b and a != c


def test_temperature_changes_the_output(client, model):
    low = generate(client, model, temperature=0.1, random_seed=1).json()["preview_records"]
    high = generate(client, model, temperature=2.0, random_seed=1).json()["preview_records"]
    assert low != high
    assert np.std([r["hours"] for r in low]) < np.std([r["hours"] for r in high])


def test_generation_uses_the_models_own_dataset_by_default(client, model):
    r = client.post("/api/generate", json={"model_id": model["model_id"], "num_records": 20})
    assert r.status_code == 200 and r.json()["dataset_id"] == model["dataset_id"]


def test_trained_model_produces_reasonable_statistics(client, dataset):
    """Sanity check on this small deterministic dataset only; not a general accuracy claim.

    The shared 30-epoch fixture is too short for good marginals (mean KS about 0.29 there), so this
    test trains with a realistic budget.
    """
    m = train(client, dataset["id"], epochs=200, learning_rate=0.003, latent_dim=8).json()
    r = generate(client, m, num_records=500).json()
    ev = client.get(f"/api/evaluate/{r['experiment_id']}").json()
    assert ev["domain_validity_percent"] == 100.0
    assert ev["average_ks_statistic"] < 0.2
    assert all(v["tvd"] < 0.1 for v in ev["categorical_metrics"].values())


# ------------------------------------------------------------------ 10. restart
def test_generation_works_in_a_new_process(client, model, env):
    first = generate(client, model, random_seed=5, num_records=60).json()
    script = (
        "import json, sys\n"
        "from fastapi.testclient import TestClient\n"
        "from app.main import app\n"
        "with TestClient(app) as c:\n"
        "    r = c.post('/api/generate', json={'model_id': sys.argv[1], 'num_records': 60, 'random_seed': 5})\n"
        "    e = c.get('/api/evaluate/' + r.json()['experiment_id'])\n"
        "    print(json.dumps({'status': r.status_code, 'preview': r.json()['preview_records'], 'eval': e.status_code}))\n"
    )
    env_vars = {**os.environ, "DATAFORGE_DATA_DIR": config.get_data_dir(), "DATAFORGE_DB_PATH": config.get_db_path()}
    done = subprocess.run([sys.executable, "-c", script, model["model_id"]], cwd=BACKEND_DIR, env=env_vars,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    second = json.loads(done.stdout.strip().splitlines()[-1])
    assert second["status"] == 200 and second["eval"] == 200
    assert second["preview"] == first["preview_records"]  # same artifacts + same seed => same rows


# ------------------------------------------------------------------ 11/12. evaluation
def test_evaluation_is_computed_from_real_data_and_persisted(client, dataset, model):
    g = generate(client, model, num_records=300).json()
    r = client.get(f"/api/evaluate/{g['experiment_id']}")
    assert r.status_code == 200
    ev = r.json()
    assert ev["overall_fidelity_score"] == g["fidelity_score"]
    assert ev["domain_validity_percent"] == g["domain_validity_percent"]
    assert set(ev["ks_metrics"]) == {"age", "hours", "score"} and set(ev["categorical_metrics"]) == {"gender", "grade"}
    assert ev["correlation_frobenius_delta"] is not None and ev["limitations"] and ev["warnings"] is not None
    assert [s["feature"] for s in ev["summary_comparisons"]] == ["age", "hours", "score"]

    # recompute independently from the files on disk and compare with what was stored
    from app.services.evaluator import DataQualityEvaluator
    real = pd.read_csv(os.path.join(config.get_dir("uploads"), os.listdir(config.get_dir("uploads"))[0]))
    syn = pd.read_csv(os.path.join(config.get_dir("synthetic"), g["output_filename"]))
    again = DataQualityEvaluator.evaluate(real, syn)
    assert again["overall_fidelity_score"] == pytest.approx(ev["overall_fidelity_score"], abs=0.2)  # CSV rounding only

    stored = db.get_evaluation(g["experiment_id"])
    assert stored["overall_score"] == ev["overall_fidelity_score"] and stored["details"]["fidelity_weights"]
    assert client.get(f"/api/evaluate/{g['experiment_id']}").json() == ev  # retrieval is stable


def test_evaluation_for_a_legacy_experiment_without_a_stored_row_is_computed_once(client, model):
    g = generate(client, model).json()
    con = db.get_db_connection()
    con.execute("DELETE FROM evaluations WHERE experiment_id = ?", (g["experiment_id"],))
    con.commit()
    con.close()
    ev = client.get(f"/api/evaluate/{g['experiment_id']}").json()
    assert ev["overall_fidelity_score"] > 0 and db.get_evaluation(g["experiment_id"]) is not None


# ------------------------------------------------------------------ 13. export & history
def test_export_downloads_the_generated_csv_and_history_lists_it(client, dataset, model):
    g = generate(client, model, num_records=120).json()
    r = client.get(f"/api/export/{g['experiment_id']}")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert g["output_filename"] in r.headers["content-disposition"]
    df = pd.read_csv(io.StringIO(r.text))
    assert len(df) == 120 and list(df.columns) == ["gender", "grade", "age", "hours", "score"]

    hist = client.get("/api/experiments").json()
    assert len(hist) == 1
    h = hist[0]
    assert (h["id"], h["dataset_id"], h["model_id"]) == (g["experiment_id"], dataset["id"], model["model_id"])
    assert h["dataset_name"] == "students.csv" and h["model_name"] == model["name"]
    assert h["output_filename"] == g["output_filename"] and h["fidelity_score"] == g["fidelity_score"]
    assert h["status"] == "completed" and h["output_available"] is True and h["model_type"] == "numpy-vae"
    assert "output_file_path" not in h and config.get_data_dir() not in json.dumps(hist)  # no internal paths leak


# ------------------------------------------------------------------ 14. invalid ids / missing artifacts
def test_unknown_ids_return_404(client, dataset, model):
    assert generate(client, {"model_id": "vae_nope", "dataset_id": dataset["id"]}).status_code == 404
    assert client.get("/api/models/status/vae_nope").status_code == 404
    assert client.get("/api/evaluate/gen_nope").status_code == 404
    assert client.get("/api/export/gen_nope").status_code == 404
    assert client.get("/api/datasets/ds_nope").status_code == 404
    assert client.post("/api/generate", json={"model_id": model["model_id"], "dataset_id": "ds_nope"}).status_code == 400


def test_generation_request_validation(client, model, monkeypatch):
    assert generate(client, model, num_records=5).status_code == 422
    assert generate(client, model, temperature=9).status_code == 422
    monkeypatch.setenv("DATAFORGE_MAX_GENERATE_ROWS", "100")
    r = generate(client, model, num_records=101)
    assert r.status_code == 422 and "100" in r.json()["detail"]
    assert generate(client, model, num_records=100).status_code == 200
    assert count("experiments") == 1


def test_model_trained_on_another_dataset_is_rejected(client, model):
    other = upload(client, make_df(50, seed=9), name="other.csv")
    r = generate(client, {"model_id": model["model_id"], "dataset_id": other["id"]})
    assert r.status_code == 400 and "trained on dataset" in r.json()["detail"]


@pytest.mark.parametrize("victim", ["weights", "preprocessor"])
def test_missing_artifacts_give_a_clear_409(client, model, victim):
    row = db.get_model(model["model_id"])
    os.remove(os.path.join(config.get_data_dir(), row["weights_path" if victim == "weights" else "preprocessor_path"]))
    r = generate(client, model)
    assert r.status_code == 409 and {"weights": "weights file", "preprocessor": "preprocessing file"}[victim] in r.json()["detail"]
    assert count("experiments") == 0 and files_in("synthetic") == []


def test_corrupt_and_incompatible_artifacts_give_409_not_500(client, model):
    row = db.get_model(model["model_id"])
    weights = os.path.join(config.get_data_dir(), row["weights_path"])
    pipeline = os.path.join(config.get_data_dir(), row["preprocessor_path"])
    good_weights = open(weights, "rb").read()

    open(weights, "wb").write(b"garbage")
    assert generate(client, model).status_code == 409

    np.savez(weights, **{k: v[:1] for k, v in np.load(io.BytesIO(good_weights)).items()})  # wrong shapes
    r = generate(client, model)
    assert r.status_code == 409 and "incompatible" in r.json()["detail"]

    open(weights, "wb").write(good_weights)
    open(pipeline, "w").write("{broken")
    r = generate(client, model)
    assert r.status_code == 409 and "preprocessing file" in r.json()["detail"]

    open(pipeline, "w").write(json.dumps({"numeric_scalers": {}}))
    assert generate(client, model).status_code == 409
    assert count("experiments") == 0


def test_export_when_the_csv_was_deleted(client, model):
    g = generate(client, model).json()
    os.remove(os.path.join(config.get_dir("synthetic"), g["output_filename"]))
    assert client.get(f"/api/export/{g['experiment_id']}").status_code == 404


# ------------------------------------------------------------------ 15. database consistency on failure
def test_failed_generation_leaves_no_rows_and_no_files(client, model, monkeypatch):
    from app.services.evaluator import DataQualityEvaluator

    def boom(real, syn):
        raise RuntimeError("evaluation blew up: /private/path")
    monkeypatch.setattr(DataQualityEvaluator, "evaluate", staticmethod(boom))
    r = generate(client, model)
    assert r.status_code == 500 and "private" not in r.text
    assert count("experiments") == 0 and count("evaluations") == 0 and files_in("synthetic") == []


def test_database_failure_during_generation_removes_the_csv(client, model, monkeypatch):
    def boom(exp, evaluation):
        raise RuntimeError("db down")
    monkeypatch.setattr(db, "insert_experiment_with_evaluation", boom)
    assert generate(client, model).status_code == 500
    assert count("experiments") == 0 and files_in("synthetic") == []


# ------------------------------------------------------------------ legacy data (Step 1 audit, P0)
def test_records_with_absolute_paths_from_another_machine_still_work(client, env):
    """Rows written on another machine store absolute paths that do not exist here; files are found by name."""
    ds = upload(client)
    other_machine = "C:\\Users\\other\\project\\backend\\data\\uploads\\" + os.path.basename(db.get_dataset(ds["id"])["file_path"])
    con = db.get_db_connection()
    con.execute("UPDATE datasets SET file_path = ? WHERE id = ?", (other_machine, ds["id"]))
    con.commit()
    con.close()
    assert client.get(f"/api/datasets/{ds['id']}").json()["preview"]
    m = train(client, ds["id"], epochs=3).json()
    assert m["status"] == "completed"

    # make the model look like a legacy one: absolute foreign weights path, no preprocessor_path,
    # and the pipeline stored under the old per-dataset name
    row = db.get_model(m["model_id"])
    models_dir = config.get_dir("models")
    os.replace(os.path.join(config.get_data_dir(), row["preprocessor_path"]), os.path.join(models_dir, f"preprocessor_{ds['id']}.json"))
    con = db.get_db_connection()
    con.execute("UPDATE models SET weights_path = ?, preprocessor_path = NULL WHERE id = ?",
                ("C:\\Users\\other\\backend\\data\\models\\" + os.path.basename(row["weights_path"]), m["model_id"]))
    con.commit()
    con.close()
    g = generate(client, m)
    assert g.status_code == 200, g.text
    assert client.get(f"/api/export/{g.json()['experiment_id']}").status_code == 200


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "online"


# ------------------------------------------------------------------ export history integrity (Step 4)
def test_each_experiment_exports_its_own_dataset_only(client, model):
    a = generate(client, model, random_seed=1, num_records=40).json()
    b = generate(client, model, random_seed=2, num_records=80).json()
    csv_a = client.get(f"/api/export/{a['experiment_id']}")
    csv_b = client.get(f"/api/export/{b['experiment_id']}")
    assert a["output_filename"] in csv_a.headers["content-disposition"]
    assert b["output_filename"] in csv_b.headers["content-disposition"]
    assert len(pd.read_csv(io.StringIO(csv_a.text))) == 40 and len(pd.read_csv(io.StringIO(csv_b.text))) == 80
    assert csv_a.content == open(os.path.join(config.get_dir("synthetic"), a["output_filename"]), "rb").read()
    assert csv_a.content != csv_b.content


def test_history_reports_a_removed_output_file_instead_of_a_downloadable_experiment(client, model):
    g = generate(client, model).json()
    os.remove(os.path.join(config.get_dir("synthetic"), g["output_filename"]))
    item = client.get("/api/experiments").json()[0]
    assert item["status"] == "output_missing" and item["output_available"] is False
    assert client.get(f"/api/export/{g['experiment_id']}").status_code == 404
    assert client.get(f"/api/evaluate/{g['experiment_id']}").status_code == 200  # stored evaluation survives


def test_failed_generation_never_appears_in_history(client, model, monkeypatch):
    from app.services.evaluator import DataQualityEvaluator

    def boom(real, syn):
        raise RuntimeError("nope")
    monkeypatch.setattr(DataQualityEvaluator, "evaluate", staticmethod(boom))
    assert generate(client, model).status_code == 500
    assert client.get("/api/experiments").json() == []


def test_history_survives_a_deleted_dataset_record(client, dataset, model):
    g = generate(client, model).json()
    con = db.get_db_connection()
    con.execute("DELETE FROM datasets WHERE id = ?", (dataset["id"],))
    con.commit()
    con.close()
    item = client.get("/api/experiments").json()[0]
    assert item["id"] == g["experiment_id"] and item["dataset_name"] is None and item["dataset_id"] == dataset["id"]
    assert client.get(f"/api/export/{g['experiment_id']}").status_code == 200


def test_history_is_newest_first_even_within_the_same_second(client, model):
    ids = [generate(client, model, random_seed=i, num_records=20).json()["experiment_id"] for i in range(5)]
    listed = [e["id"] for e in client.get("/api/experiments").json()]
    assert listed == list(reversed(ids))


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5])
def test_invalid_generation_seeds_are_client_errors_not_server_errors(client, model, seed):
    r = generate(client, model, random_seed=seed)
    assert r.status_code == 422
    assert count("experiments") == 0 and files_in("synthetic") == []


def test_boundary_generation_seeds_work(client, model):
    assert generate(client, model, random_seed=0, num_records=20).status_code == 200
    assert generate(client, model, random_seed=2**32 - 1, num_records=20).status_code == 200
