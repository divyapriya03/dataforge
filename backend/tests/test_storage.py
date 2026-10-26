import os
import sqlite3

import pytest

from app.core import config
from app.core import database as db


def test_paths_are_relative_to_the_data_dir_and_cwd_independent(env, tmp_path, monkeypatch):
    p = tmp_path / "data" / "uploads" / "a.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    stored = config.to_stored_path(str(p))
    assert stored == "uploads/a.csv"
    monkeypatch.chdir(tmp_path.parent)
    assert config.resolve_stored_path(stored, "uploads") == str(p)


def test_legacy_absolute_paths_from_another_machine_resolve_by_file_name(env):
    legacy = "C:\\Users\\someone\\OneDrive\\project\\backend\\data\\uploads\\ds_1_x.csv"
    assert config.resolve_stored_path(legacy, "uploads") == os.path.join(config.get_dir("uploads"), "ds_1_x.csv")
    assert config.resolve_stored_path(None, "uploads") is None


def test_init_db_migrates_an_old_database_without_losing_rows(tmp_path, monkeypatch):
    old = tmp_path / "old.db"
    con = sqlite3.connect(old)
    con.executescript("""
        CREATE TABLE datasets (id TEXT PRIMARY KEY, filename TEXT NOT NULL, file_path TEXT NOT NULL, row_count INTEGER NOT NULL,
                               column_count INTEGER NOT NULL, schema_json TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE models (id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, name TEXT NOT NULL, params_json TEXT NOT NULL,
                             final_loss REAL, recon_loss REAL, kl_loss REAL, loss_history_json TEXT,
                             status TEXT NOT NULL DEFAULT 'initialized', weights_path TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE evaluations (id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT NOT NULL UNIQUE, overall_score REAL NOT NULL,
                             ks_metrics_json TEXT NOT NULL, correlation_delta REAL NOT NULL, validity_percentage REAL NOT NULL,
                             summary_stats_diff_json TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        INSERT INTO datasets VALUES ('ds_old','f.csv','/x/f.csv',3,2,'[]','2026-01-01');
        INSERT INTO models (id,dataset_id,name,params_json,status,weights_path) VALUES ('vae_old','ds_old','m','{"epochs": 5}','completed','/x/w.npz');
        INSERT INTO evaluations (experiment_id,overall_score,ks_metrics_json,correlation_delta,validity_percentage,summary_stats_diff_json)
               VALUES ('gen_old', 70.0, '{}', 0.04, 100.0, '[]');
    """)
    con.commit()
    con.close()
    monkeypatch.setenv("DATAFORGE_DB_PATH", str(old))
    db.init_db()
    db.init_db()  # idempotent
    m = db.get_model("vae_old")
    assert m["weights_path"] == "/x/w.npz" and m["model_type"] is None and m["preprocessor_path"] is None
    assert db.get_dataset("ds_old")["row_count"] == 3
    assert db.get_evaluation("gen_old")["details"] == {}


def test_experiment_and_evaluation_are_written_atomically(env):
    exp = dict(exp_id="gen_1", dataset_id="d", model_id="m", num_records=10, temp=1.0, seed=1,
               output_path="synthetic/x.csv", fidelity=80.0, validity=100.0)
    bad_eval = dict(score=None, ks_dict={}, corr_delta=0.0, validity=100.0, summary_diff=[], details={})  # NOT NULL violation
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_experiment_with_evaluation(exp, bad_eval)
    assert db.get_experiment("gen_1") is None
    db.insert_experiment_with_evaluation(exp, {**bad_eval, "score": 80.0})
    assert db.get_experiment("gen_1") and db.get_evaluation("gen_1")["overall_score"] == 80.0
