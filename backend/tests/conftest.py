"""Test fixtures.

The app creates its SQLite tables when ``app.main`` is imported, so the storage environment
variables are pointed at a throw-away directory *before* that import. Every test then gets its own
data directory and database; the real ``backend/data`` and ``backend/dataforge.db`` are never touched.
"""
import os
import tempfile

_BOOT = tempfile.mkdtemp(prefix="dataforge_boot_")
os.environ["DATAFORGE_DATA_DIR"] = os.path.join(_BOOT, "data")
os.environ["DATAFORGE_DB_PATH"] = os.path.join(_BOOT, "boot.db")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core import database as db  # noqa: E402
from app.main import app  # noqa: E402
from tests.data_utils import csv_bytes, make_df  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fresh data dir + database for one test."""
    monkeypatch.setenv("DATAFORGE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DATAFORGE_DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    return tmp_path


@pytest.fixture
def client(env):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def upload(client, df=None, name="students.csv"):
    df = make_df() if df is None else df
    r = client.post("/api/datasets/upload", files={"file": (name, csv_bytes(df), "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()


def train(client, dataset_id, **overrides):
    body = {"dataset_id": dataset_id, "epochs": 30, "latent_dim": 4, "random_seed": 3, "batch_size": 32}
    body.update(overrides)
    return client.post("/api/models/train", json=body)


@pytest.fixture
def dataset(client):
    return upload(client)


@pytest.fixture
def model(client, dataset):
    r = train(client, dataset["id"])
    assert r.status_code == 200, r.text
    return r.json()
