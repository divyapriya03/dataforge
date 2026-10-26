"""Negative / recovery tests (Step 6): the application must report failures clearly, keep existing data intact
and never leave records or files that suggest a success that did not happen."""
import os
import sqlite3

import pandas as pd
import pytest

from app.core import config
from app.core import database as db
from app.models.vae import VariationalAutoencoder
from tests.conftest import train, upload
from tests.test_pipeline_api import count, files_in, generate


def snapshot(env):
    """Everything persistent: all table rows plus the names and sizes of all stored files."""
    con = sqlite3.connect(env / "test.db")
    rows = {t: con.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall() for t in ("datasets", "models", "experiments", "evaluations")}
    con.close()
    files = {}
    for kind in config.KINDS:
        for f in sorted(os.listdir(config.get_dir(kind))):
            files[f"{kind}/{f}"] = os.path.getsize(os.path.join(config.get_dir(kind), f))
    return rows, files


# ------------------------------------------------------------------ database cannot be accessed
@pytest.mark.parametrize("broken", ["directory", "garbage"])
def test_an_inaccessible_database_gives_a_generic_500_and_nothing_else_breaks(client, env, monkeypatch, broken):
    ds = upload(client)
    good_db = os.environ["DATAFORGE_DB_PATH"]
    bad = env / "bad.db"
    if broken == "directory":
        bad.mkdir()
    else:
        bad.write_bytes(b"this is not a sqlite database" * 20)
    monkeypatch.setenv("DATAFORGE_DB_PATH", str(bad))

    for method, path in [("get", "/api/datasets"), ("get", "/api/models"), ("get", "/api/experiments"), ("get", f"/api/datasets/{ds['id']}")]:
        r = getattr(client, method)(path)
        assert r.status_code == 500 and r.json() == {"detail": "Internal server error."}
        assert str(env) not in r.text and "sqlite" not in r.text.lower()
    assert client.get("/api/health").status_code == 200  # the process keeps serving

    monkeypatch.setenv("DATAFORGE_DB_PATH", good_db)  # the database comes back: nothing was lost
    assert [d["id"] for d in client.get("/api/datasets").json()] == [ds["id"]]


def test_an_upload_that_cannot_be_recorded_leaves_no_file(client, env, monkeypatch):
    def boom(*a, **k):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(db, "insert_dataset", boom)
    r = client.post("/api/datasets/upload", files={"file": ("a.csv", b"x,y\n1,2\n3,4\n", "text/csv")})
    assert r.status_code == 500 and "locked" not in r.text
    assert files_in("uploads") == [] and count("datasets") == 0


# ------------------------------------------------------------------ failure midway
def test_training_that_crashes_midway_is_recorded_as_failed_without_artifacts(client, dataset, env, monkeypatch):
    calls = {"n": 0}
    original = VariationalAutoencoder.train_epoch

    def crash_on_third(self, X, batch_size):
        calls["n"] += 1
        if calls["n"] == 3:
            raise MemoryError("out of memory at /secret/place")
        return original(self, X, batch_size)

    monkeypatch.setattr(VariationalAutoencoder, "train_epoch", crash_on_third)
    r = train(client, dataset["id"], epochs=10)
    assert r.status_code == 500 and "secret" not in r.text and "MemoryError" not in r.text
    [m] = client.get("/api/models").json()
    assert m["status"] == "failed" and m["loss_history"] == [] and m["completed_at"]
    assert "secret" not in (m["error_message"] or "")
    assert files_in("models") == []
    assert client.post("/api/generate", json={"model_id": m["model_id"], "num_records": 20}).status_code == 409


def test_a_failure_while_writing_the_csv_leaves_no_partial_file_or_record(client, model, monkeypatch):
    def disk_full(self, *a, **k):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(pd.DataFrame, "to_csv", disk_full)
    r = generate(client, model)
    assert r.status_code == 500 and "space" not in r.text.lower()
    assert files_in("synthetic") == [] and count("experiments") == 0 and count("evaluations") == 0


def test_failed_operations_do_not_change_stored_data(client, dataset, model, env, monkeypatch):
    ok = generate(client, model)
    assert ok.status_code == 200
    before = snapshot(env)
    # a series of failing requests
    assert train(client, "ds_nope").status_code == 404
    assert train(client, dataset["id"], hidden_layers=[64]).status_code == 422
    assert generate(client, model, num_records=3).status_code == 422
    assert generate(client, model, random_seed=-5).status_code == 422
    assert client.post("/api/generate", json={"model_id": "vae_nope", "num_records": 20}).status_code == 404
    assert client.get("/api/export/gen_nope").status_code == 404
    assert client.post("/api/datasets/upload", files={"file": ("e.csv", b"", "text/csv")}).status_code == 400
    monkeypatch.setattr(VariationalAutoencoder, "train_epoch", lambda self, X, batch_size: (float("nan"),) * 3)
    assert train(client, dataset["id"]).status_code == 422
    after = snapshot(env)
    assert after[1] == before[1]                                    # no file added, removed or changed
    assert after[0]["datasets"] == before[0]["datasets"] and after[0]["experiments"] == before[0]["experiments"]
    assert after[0]["evaluations"] == before[0]["evaluations"]
    assert [r[0] for r in after[0]["models"]][-len(before[0]["models"]):] == [r[0] for r in before[0]["models"]]  # only a 'failed' row was added
    assert len(after[0]["models"]) == len(before[0]["models"]) + 1


# ------------------------------------------------------------------ evaluation that cannot be computed
def test_metrics_that_cannot_be_computed_are_null_not_invented(client):
    ds = upload(client, pd.DataFrame({"cat": list("abcab") * 8}), name="onlycat.csv")
    m = train(client, ds["id"], epochs=3).json()
    exp = client.post("/api/generate", json={"model_id": m["model_id"], "num_records": 30}).json()["experiment_id"]
    ev = client.get(f"/api/evaluate/{exp}").json()
    assert ev["average_ks_statistic"] is None and ev["correlation_frobenius_delta"] is None
    assert ev["ks_metrics"] == {} and set(ev["categorical_metrics"]) == {"cat"}
    assert any("Correlation" in w for w in ev["warnings"])


def test_an_experiment_without_evaluation_and_without_files_is_a_404(client, model):
    g = generate(client, model).json()
    con = db.get_db_connection()
    con.execute("DELETE FROM evaluations WHERE experiment_id = ?", (g["experiment_id"],))
    con.commit()
    con.close()
    os.remove(os.path.join(config.get_dir("synthetic"), g["output_filename"]))
    r = client.get(f"/api/evaluate/{g['experiment_id']}")
    assert r.status_code == 404 and str(config.get_data_dir()) not in r.text


# ------------------------------------------------------------------ resource limits
def test_the_generation_limit_can_be_lowered_by_configuration(client, model, monkeypatch):
    monkeypatch.setenv("DATAFORGE_MAX_GENERATE_ROWS", "50")
    assert generate(client, model, num_records=51).status_code == 422
    assert generate(client, model, num_records=50).status_code == 200


def test_the_upload_limit_is_enforced_while_streaming(client, env, monkeypatch):
    monkeypatch.setenv("DATAFORGE_MAX_UPLOAD_MB", "1")
    big = ("a,b\n" + "1,2\n" * 400000).encode()  # about 1.6 MB
    r = client.post("/api/datasets/upload", files={"file": ("big.csv", big, "text/csv")})
    assert r.status_code == 413 and "1 MB" in r.json()["detail"]
    assert files_in("uploads") == [] and count("datasets") == 0
