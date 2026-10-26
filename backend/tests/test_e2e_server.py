"""End-to-end test against a REAL uvicorn process over HTTP (not the in-process TestClient).

Each test starts the server with a temporary data directory and database, walks through the whole
workflow, restarts the server and checks that records and artifacts persist. Real project data is
never touched. Requires the `requests` package (see requirements-dev.txt).
"""
import hashlib
import io
import os
import socket
import sqlite3
import subprocess
import sys
import time

import pandas as pd
import pytest
import requests

from tests.data_utils import csv_bytes, make_df

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, data_dir, db_path):
        self.env = {**os.environ, "DATAFORGE_DATA_DIR": str(data_dir), "DATAFORGE_DB_PATH": str(db_path)}
        self.proc = None
        self.base = ""

    def start(self):
        port = free_port()
        self.base = f"http://127.0.0.1:{port}/api"
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=BACKEND_DIR, env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 40
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited during start-up")
            try:
                if requests.get(f"{self.base}/health", timeout=1).status_code == 200:
                    return self
            except requests.RequestException:
                time.sleep(0.2)
        raise RuntimeError("server did not become healthy")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def get(self, path, **kw):
        return requests.get(f"{self.base}{path}", timeout=60, **kw)

    def post(self, path, **kw):
        return requests.post(f"{self.base}{path}", timeout=120, **kw)


@pytest.fixture
def server(tmp_path):
    srv = Server(tmp_path / "data", tmp_path / "e2e.db")
    srv.tmp = tmp_path
    srv.start()
    yield srv
    srv.stop()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_full_workflow_persists_across_a_server_restart(server):
    original = make_df()

    # 1-3. health, empty state, upload -> dataset id
    assert server.get("/health").json()["status"] == "online"
    assert server.get("/datasets").json() == [] and server.get("/experiments").json() == []
    up = server.post("/datasets/upload", files={"file": ("students.csv", csv_bytes(original), "text/csv")})
    assert up.status_code == 200
    ds = up.json()
    assert ds["id"].startswith("ds_") and ds["rowCount"] == 300 and ds["columnCount"] == 6
    assert server.get(f"/datasets/{ds['id']}").json()["uploadDate"] == ds["uploadDate"]

    # 4. preprocessing (dry run) on the intended dataset
    prep = server.post("/preprocess", json={"dataset_id": ds["id"], "columns": ds["columns"]}).json()
    assert prep["dataset_id"] == ds["id"] and prep["ready_for_training"] and prep["rows_used"] == 300
    assert any(d["name"] == "student_id" for d in prep["dropped_columns"])

    # 5-8. train, model available, generate, row count and schema
    model = server.post("/models/train", json={"dataset_id": ds["id"], "epochs": 40, "latent_dim": 4, "random_seed": 3}).json()
    assert model["status"] == "completed" and model["dataset_id"] == ds["id"] and model["model_id"].startswith("vae_")
    assert [m["model_id"] for m in server.get("/models").json()] == [model["model_id"]]
    assert server.get(f"/models/status/{model['model_id']}").json()["status"] == "completed"

    gen = server.post("/generate", json={"model_id": model["model_id"], "num_records": 120, "random_seed": 9}).json()
    exp_id = gen["experiment_id"]
    assert gen["dataset_id"] == ds["id"] and gen["model_id"] == model["model_id"] and gen["num_records"] == 120
    csv_response = server.get(f"/export/{exp_id}")
    assert csv_response.status_code == 200 and csv_response.headers["content-type"].startswith("text/csv")
    synthetic = pd.read_csv(io.BytesIO(csv_response.content))
    assert len(synthetic) == 120 and list(synthetic.columns) == ["gender", "grade", "age", "hours", "score"]
    assert not synthetic.isna().any().any()
    assert set(synthetic["gender"]) <= set(original["gender"]) and set(synthetic["grade"]) <= set(original["grade"].dropna())

    # 9-10. evaluation computed from real data
    ev = server.get(f"/evaluate/{exp_id}").json()
    assert ev["experiment_id"] == exp_id and ev["overall_fidelity_score"] == gen["fidelity_score"]
    assert set(ev["ks_metrics"]) == {"age", "hours", "score"} and set(ev["categorical_metrics"]) == {"gender", "grade"}
    assert ev["domain_validity_percent"] == 100.0 and ev["limitations"]

    # 11. history matches
    [item] = server.get("/experiments").json()
    assert (item["id"], item["dataset_id"], item["model_id"]) == (exp_id, ds["id"], model["model_id"])
    assert item["fidelity_score"] == ev["overall_fidelity_score"] and item["num_records"] == 120
    assert item["status"] == "completed" and item["output_available"] is True
    assert str(server.tmp) not in str(server.get("/experiments").json())  # no filesystem paths in responses

    before = {
        "datasets": server.get("/datasets").json(),
        "models": server.get("/models").json(),
        "experiments": server.get("/experiments").json(),
        "evaluation": ev,
        "csv": sha(csv_response.content),
        "preview": gen["preview_records"],
    }

    # 12. restart the backend process
    server.stop()
    server.start()

    # 13. everything is still there and identical
    assert server.get("/datasets").json() == before["datasets"]
    assert server.get("/models").json() == before["models"]
    assert server.get("/experiments").json() == before["experiments"]
    assert server.get(f"/evaluate/{exp_id}").json() == before["evaluation"]
    assert sha(server.get(f"/export/{exp_id}").content) == before["csv"]

    # 14. the persisted model artifacts still generate (same seed -> same rows as before the restart)
    again = server.post("/generate", json={"model_id": model["model_id"], "num_records": 120, "random_seed": 9}).json()
    assert again["preview_records"] == before["preview"]
    assert again["experiment_id"] != exp_id
    assert [e["id"] for e in server.get("/experiments").json()][:2] == [again["experiment_id"], exp_id]  # newest first
    # the first experiment's file is untouched by the second generation
    assert sha(server.get(f"/export/{exp_id}").content) == before["csv"]


def test_error_paths_over_http(server):
    ds = server.post("/datasets/upload", files={"file": ("students.csv", csv_bytes(make_df()), "text/csv")}).json()

    # uploads
    assert server.post("/datasets/upload", files={"file": ("empty.csv", b"", "text/csv")}).status_code == 400
    assert server.post("/datasets/upload", files={"file": ("notes.txt", b"a,b\n1,2\n", "text/plain")}).status_code == 400
    assert len(server.get("/datasets").json()) == 1  # failed uploads left no records

    # invalid ids and parameters
    assert server.get("/datasets/ds_nope").status_code == 404
    assert server.post("/preprocess", json={"dataset_id": "ds_nope", "columns": ds["columns"]}).status_code == 404
    assert server.post("/models/train", json={"dataset_id": "ds_nope"}).status_code == 404
    for bad in ({"epochs": 0}, {"hidden_layers": [64]}, {"learning_rate": 0}, {"latent_dim": 1}):
        assert server.post("/models/train", json={"dataset_id": ds["id"], **bad}).status_code == 422
    assert server.get("/models").json() == []  # nothing recorded for rejected requests

    model = server.post("/models/train", json={"dataset_id": ds["id"], "epochs": 5, "random_seed": 1}).json()
    mid = model["model_id"]
    assert server.post("/generate", json={"model_id": "vae_nope", "num_records": 20}).status_code == 404
    for bad in ({"num_records": 5}, {"num_records": 20, "temperature": 9}, {"num_records": 20, "random_seed": -1}):
        assert server.post("/generate", json={"model_id": mid, **bad}).status_code == 422
    assert server.post("/generate", json={"model_id": mid, "dataset_id": "ds_other", "num_records": 20}).status_code == 400
    assert server.get("/experiments").json() == []  # empty history, nothing from failed requests

    # a good experiment, then break its artifacts one by one
    exp = server.post("/generate", json={"model_id": mid, "num_records": 30, "random_seed": 2}).json()["experiment_id"]
    data_dir = server.tmp / "data"

    # missing evaluation row -> recomputed from the real files, then stored
    con = sqlite3.connect(server.tmp / "e2e.db")
    con.execute("DELETE FROM evaluations WHERE experiment_id = ?", (exp,))
    con.commit()
    con.close()
    recomputed = server.get(f"/evaluate/{exp}")
    assert recomputed.status_code == 200 and recomputed.json()["experiment_id"] == exp

    # missing model weights -> clear 409, no new experiment
    weights = next((data_dir / "models").glob(f"{mid}_weights.npz"))
    weights.rename(weights.with_suffix(".bak"))
    r = server.post("/generate", json={"model_id": mid, "num_records": 20})
    assert r.status_code == 409 and "weights" in r.json()["detail"] and str(data_dir) not in r.text
    assert len(server.get("/experiments").json()) == 1

    # missing generated CSV -> export 404, history says so, stored evaluation survives, no silent regeneration
    csv_file = next((data_dir / "synthetic").glob(f"*{exp}.csv"))
    csv_file.unlink()
    assert server.get(f"/export/{exp}").status_code == 404
    assert not csv_file.exists()
    [item] = server.get("/experiments").json()
    assert item["status"] == "output_missing" and item["output_available"] is False
    assert server.get(f"/evaluate/{exp}").status_code == 200
    assert server.get("/export/gen_nope").status_code == 404 and server.get("/evaluate/gen_nope").status_code == 404
