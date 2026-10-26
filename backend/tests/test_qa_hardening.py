"""Regression tests for defects found during the Step 6 QA pass (each failed before its fix)."""
import io
import os
import re
import uuid

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import endpoints
from app.core import config
from app.main import app
from tests.conftest import train
from tests.data_utils import csv_bytes, make_df

GOOD = csv_bytes(make_df(30, missing=False))


def upload_named(client, name, content=GOOD):
    return client.post("/api/datasets/upload", files={"file": (name, content, "text/csv")})


# ------------------------------------------------------------------ F1: file names
@pytest.mark.parametrize("raw,expected", [
    ("students.csv", "students.csv"),
    ("my data (final).csv", "my_data_final_.csv"),
    ("we<ir>d|na?me*.csv", "we_ir_d_na_me_.csv"),
    ("dön nées.csv", "d_n_n_es.csv"),
    ("..hidden.csv", "hidden.csv"),
    ("", "dataset.csv"),
    ("???", "dataset.csv"),
])
def test_safe_filename(raw, expected):
    assert config.safe_filename(raw) == expected


def test_safe_filename_limits_length_and_keeps_the_extension():
    name = config.safe_filename("x" * 400 + ".csv")
    assert len(name) <= 80 and name.endswith(".csv")
    assert len(config.safe_stem("y" * 400 + ".csv")) <= 60


@pytest.mark.parametrize("name", [
    "x" * 300 + ".csv",                 # used to return 500: the OS rejects the file name
    "we<ir>d|na?me*.csv",               # characters Windows forbids in file names
    "CON.csv", "NUL.csv", "a:b.csv", "dön nées.csv", "  .csv", "../../evil.csv", "..\\..\\evil.csv",
])
def test_any_file_name_is_accepted_safely_and_stays_inside_the_upload_folder(client, env, name):
    r = upload_named(client, name)
    assert r.status_code == 200, r.text
    files = os.listdir(config.get_dir("uploads"))
    assert len(files) == 1 and re.fullmatch(r"[A-Za-z0-9._-]+", files[0]) and len(files[0]) < 120
    assert os.listdir(env) == sorted(os.listdir(env)) and set(os.listdir(env)) <= {"data", "test.db"}  # nothing written elsewhere
    assert "/" not in r.json()["filename"] and "\\" not in r.json()["filename"]


def test_generated_file_names_are_safe_for_awkward_dataset_names(client):
    ds = upload_named(client, "we<ir>d|na?me*.csv", csv_bytes(make_df(60))).json()
    m = train(client, ds["id"], epochs=3).json()
    g = client.post("/api/generate", json={"model_id": m["model_id"], "num_records": 20})
    assert g.status_code == 200, g.text
    name = g.json()["output_filename"]
    assert re.fullmatch(r"[A-Za-z0-9._-]+", name)
    assert client.get(f"/api/export/{g.json()['experiment_id']}").status_code == 200
    assert name in client.get("/api/experiments").json()[0]["output_filename"]


def test_export_with_non_ascii_dataset_name_has_a_valid_header(client):
    ds = upload_named(client, "données été.csv", csv_bytes(make_df(60))).json()
    m = train(client, ds["id"], epochs=3).json()
    exp = client.post("/api/generate", json={"model_id": m["model_id"], "num_records": 20}).json()["experiment_id"]
    r = client.get(f"/api/export/{exp}")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    r.headers["content-disposition"].encode("latin-1")  # header must be latin-1 safe


# ------------------------------------------------------------------ never overwrite / collisions
def test_a_leftover_file_is_never_overwritten_or_deleted_by_a_new_upload(client, env, monkeypatch):
    uploads = config.get_dir("uploads")
    precious = os.path.join(uploads, "ds_aaaaaaaa_students.csv")
    open(precious, "wb").write(b"precious,data\n1,2\n")
    hexes = iter(["aaaaaaaa" + "0" * 24, "bbbbbbbb" + "0" * 24])
    monkeypatch.setattr(endpoints.uuid, "uuid4", lambda: uuid.UUID(next(hexes)))
    r = upload_named(client, "students.csv")
    assert r.status_code == 200 and r.json()["id"] == "ds_bbbbbbbb"
    assert open(precious, "rb").read() == b"precious,data\n1,2\n"


def test_new_id_skips_ids_that_are_already_used(monkeypatch):
    hexes = iter(["1" * 32, "1" * 32, "2" * 32])
    monkeypatch.setattr(endpoints.uuid, "uuid4", lambda: uuid.UUID(next(hexes)))
    assert endpoints._new_id("vae", lambda i: i == "vae_11111111") == "vae_22222222"


def test_a_colliding_model_id_does_not_overwrite_an_existing_model(client, dataset, monkeypatch):
    first = train(client, dataset["id"], epochs=2).json()
    taken = first["model_id"].split("_", 1)[1]
    hexes = iter([taken + "0" * 24, "cafe0000" + "0" * 24])
    monkeypatch.setattr(endpoints.uuid, "uuid4", lambda: uuid.UUID(next(hexes)))
    second = train(client, dataset["id"], epochs=2).json()
    assert second["model_id"] == "vae_cafe0000" != first["model_id"]
    assert client.get(f"/api/models/status/{first['model_id']}").json() == first  # untouched


# ------------------------------------------------------------------ F2: CORS
def test_only_configured_origins_are_allowed(env):
    with TestClient(app) as c:
        ok = c.get("/api/health", headers={"Origin": "http://localhost:5173"})
        assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
        bad = c.get("/api/health", headers={"Origin": "http://evil.example"})
        assert bad.status_code == 200 and "access-control-allow-origin" not in bad.headers
        pre = c.options("/api/generate", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
        assert "access-control-allow-origin" not in pre.headers
        pre_ok = c.options("/api/generate", headers={"Origin": "http://127.0.0.1:5173", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
        assert pre_ok.status_code == 200 and pre_ok.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
        assert "access-control-allow-credentials" not in ok.headers  # no credentialed cross-origin requests


def test_cors_origins_can_be_configured(monkeypatch):
    assert "http://localhost:5173" in config.cors_origins()
    monkeypatch.setenv("DATAFORGE_CORS_ORIGINS", "https://app.example.org/, http://10.0.0.5:8080")
    assert config.cors_origins() == ["https://app.example.org", "http://10.0.0.5:8080"]


# ------------------------------------------------------------------ F3: wrong delimiter
@pytest.mark.parametrize("sep", [";", "\t", "|"])
def test_files_with_another_delimiter_are_rejected_with_a_hint(client, sep):
    df = make_df(30, missing=False)
    content = df.to_csv(index=False, sep=sep).encode()
    r = upload_named(client, "other.csv", content)
    assert r.status_code == 400 and "comma-separated" in r.json()["detail"]
    assert client.get("/api/datasets").json() == [] and os.listdir(config.get_dir("uploads")) == []


def test_a_genuine_single_column_file_is_still_accepted(client):
    r = upload_named(client, "one.csv", pd.DataFrame({"value": range(15)}).to_csv(index=False).encode())
    assert r.status_code == 200 and r.json()["columnCount"] == 1


# ------------------------------------------------------------------ path handling / information exposure
@pytest.mark.parametrize("path", [
    "/api/export/..%2f..%2fdataforge.db", "/api/export/%2e%2e%2f%2e%2e%2fetc", "/api/evaluate/..%5c..%5cx",
    "/api/datasets/%00", "/api/models/status/..%2f..", "/api/export/gen_x%00.csv",
])
def test_identifier_parameters_cannot_reach_the_filesystem(client, env, path):
    r = client.get(path)
    assert r.status_code == 404
    assert str(env) not in r.text


def test_error_responses_never_contain_server_paths(client, dataset, model, env):
    from app.core import database as db
    row = db.get_model(model["model_id"])
    for victim in (row["weights_path"], row["preprocessor_path"]):
        path = os.path.join(config.get_data_dir(), victim)
        os.rename(path, path + ".bak")
        r = client.post("/api/generate", json={"model_id": model["model_id"], "num_records": 20})
        os.rename(path + ".bak", path)
        assert r.status_code == 409
        assert config.get_data_dir() not in r.text and str(env) not in r.text and ":\\" not in r.text
    open(os.path.join(config.get_data_dir(), row["weights_path"]), "wb").write(b"junk")
    r = client.post("/api/generate", json={"model_id": model["model_id"], "num_records": 20})
    assert r.status_code == 409 and str(env) not in r.text


def test_model_files_are_loaded_without_pickle(tmp_path):
    """An .npz containing a pickled object must be refused instead of executing anything."""
    import numpy as np
    from app.models.vae import VariationalAutoencoder
    v = VariationalAutoencoder(3, 2, [8, 4], seed=0)
    evil = tmp_path / "evil.npz"
    np.savez(evil, **{k: np.array([object()], dtype=object) for k in v.params})
    with pytest.raises(ValueError):
        v.load(str(evil))
