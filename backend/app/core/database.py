import sqlite3
import json
import os
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataforge.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Datasets Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        row_count INTEGER NOT NULL,
        column_count INTEGER NOT NULL,
        schema_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 2. Models Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS models (
        id TEXT PRIMARY KEY,
        dataset_id TEXT NOT NULL,
        name TEXT NOT NULL,
        params_json TEXT NOT NULL,
        final_loss REAL,
        recon_loss REAL,
        kl_loss REAL,
        loss_history_json TEXT,
        status TEXT NOT NULL DEFAULT 'initialized',
        weights_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
    );
    """)

    # 3. Experiments / Generations Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY,
        dataset_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        num_records INTEGER NOT NULL,
        temperature REAL NOT NULL DEFAULT 1.0,
        random_seed INTEGER DEFAULT 42,
        output_file_path TEXT NOT NULL,
        fidelity_score REAL,
        validity_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
        FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
    );
    """)

    # 4. Evaluations Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id TEXT NOT NULL UNIQUE,
        overall_score REAL NOT NULL,
        ks_metrics_json TEXT NOT NULL,
        correlation_delta REAL NOT NULL,
        validity_percentage REAL NOT NULL,
        summary_stats_diff_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
    );
    """)

    conn.commit()
    conn.close()

# Database Helper Functions

def insert_dataset(dataset_id: str, filename: str, file_path: str, row_count: int, col_count: int, schema_dict: List[Dict]):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO datasets (id, filename, file_path, row_count, column_count, schema_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (dataset_id, filename, file_path, row_count, col_count, json.dumps(schema_dict)))
    conn.commit()
    conn.close()

def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        d = dict(row)
        d['columns'] = json.loads(d['schema_json'])
        return d
    return None

def list_datasets() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d['columns'] = json.loads(d['schema_json'])
        result.append(d)
    return result

def insert_model(model_id: str, dataset_id: str, name: str, params: Dict, weights_path: str = ""):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO models (id, dataset_id, name, params_json, status, weights_path)
        VALUES (?, ?, ?, ?, 'initialized', ?)
    """, (model_id, dataset_id, name, json.dumps(params), weights_path))
    conn.commit()
    conn.close()

def update_model_progress(model_id: str, status: str, final_loss: float = None, recon_loss: float = None, kl_loss: float = None, loss_history: List = None, weights_path: str = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    updates = ["status = ?"]
    params = [status]

    if final_loss is not None:
        updates.append("final_loss = ?")
        params.append(final_loss)
    if recon_loss is not None:
        updates.append("recon_loss = ?")
        params.append(recon_loss)
    if kl_loss is not None:
        updates.append("kl_loss = ?")
        params.append(kl_loss)
    if loss_history is not None:
        updates.append("loss_history_json = ?")
        params.append(json.dumps(loss_history))
    if weights_path is not None:
        updates.append("weights_path = ?")
        params.append(weights_path)

    params.append(model_id)
    cursor.execute(f"UPDATE models SET {', '.join(updates)} WHERE id = ?", params)
    conn.commit()
    conn.close()

def get_model(model_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models WHERE id = ?", (model_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        d = dict(row)
        d['params'] = json.loads(d['params_json'])
        d['loss_history'] = json.loads(d['loss_history_json']) if d['loss_history_json'] else []
        return d
    return None

def list_models() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d['params'] = json.loads(d['params_json'])
        d['loss_history'] = json.loads(d['loss_history_json']) if d['loss_history_json'] else []
        result.append(d)
    return result

def insert_experiment(exp_id: str, dataset_id: str, model_id: str, num_records: int, temp: float, seed: int, output_path: str, fidelity: float, validity: float):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO experiments (id, dataset_id, model_id, num_records, temperature, random_seed, output_file_path, fidelity_score, validity_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (exp_id, dataset_id, model_id, num_records, temp, seed, output_path, fidelity, validity))
    conn.commit()
    conn.close()

def get_experiment(exp_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def list_experiments() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT e.*, d.filename as dataset_name, m.name as model_name
        FROM experiments e
        LEFT JOIN datasets d ON e.dataset_id = d.id
        LEFT JOIN models m ON e.model_id = m.id
        ORDER BY e.created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def insert_evaluation(exp_id: str, score: float, ks_dict: Dict, corr_delta: float, validity: float, summary_diff: Dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO evaluations (experiment_id, overall_score, ks_metrics_json, correlation_delta, validity_percentage, summary_stats_diff_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (exp_id, score, json.dumps(ks_dict), corr_delta, validity, json.dumps(summary_diff)))
    conn.commit()
    conn.close()

def get_evaluation(exp_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM evaluations WHERE experiment_id = ?", (exp_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        d = dict(row)
        d['ks_metrics'] = json.loads(d['ks_metrics_json'])
        d['summary_stats_diff'] = json.loads(d['summary_stats_diff_json'])
        return d
    return None
