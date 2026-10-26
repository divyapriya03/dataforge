import requests
import json
import os
import sys

# Manual smoke test against a RUNNING server. It writes data (a model, a generation, an upload), so point
# it at a scratch server: DATAFORGE_API_URL=http://127.0.0.1:8001/api python test_backend_api.py
# The automated, isolated tests live in tests/ (run `python -m pytest`).
BASE_URL = os.environ.get("DATAFORGE_API_URL", "http://127.0.0.1:8000/api")

def run_tests():
    print("==================================================")
    print("DATAFORGE BACKEND API AUTOMATED VERIFICATION SUITE")
    print("==================================================")

    # 1. Healthcheck
    print("\n[TEST 1] GET /api/health")
    res = requests.get(f"{BASE_URL}/health")
    assert res.status_code == 200, f"Healthcheck failed: {res.text}"
    print("✓ Success:", res.json()["status"], "-", res.json()["deep_learning_engine"])

    # 2. List datasets
    print("\n[TEST 2] GET /api/datasets")
    res = requests.get(f"{BASE_URL}/datasets")
    assert res.status_code == 200, f"List datasets failed: {res.text}"
    datasets = res.json()
    assert len(datasets) > 0, "No datasets found in database"
    # Prefer the seeded sample dataset; otherwise any dataset large enough to train on (>= 10 rows).
    target = next((d for d in datasets if d["id"] == "ds_student_01"), None) or next(
        (d for d in datasets if d["rowCount"] >= 10), None
    )
    assert target is not None, "No dataset with at least 10 rows found"
    dataset_id = target["id"]
    print(f"✓ Success: Found {len(datasets)} dataset(s). Target dataset: {dataset_id} ({target['filename']})")

    # 3. Get single dataset info
    print(f"\n[TEST 3] GET /api/datasets/{dataset_id}")
    res = requests.get(f"{BASE_URL}/datasets/{dataset_id}")
    assert res.status_code == 200, f"Get dataset failed: {res.text}"
    ds_data = res.json()
    print(f"✓ Success: Rows = {ds_data['rowCount']}, Columns = {ds_data['columnCount']}, Schema columns = {len(ds_data['columns'])}")

    # 4. Preprocess dataset
    print("\n[TEST 4] POST /api/preprocess")
    payload = {
        "dataset_id": dataset_id,
        "columns": ds_data["columns"]
    }
    res = requests.post(f"{BASE_URL}/preprocess", json=payload)
    assert res.status_code == 200, f"Preprocessing failed: {res.text}"
    prep_data = res.json()
    print(f"✓ Success: Processed dimensions = {prep_data['processed_dimensions']}, Features = {prep_data['feature_names']}")

    # 5. Train VAE model
    print("\n[TEST 5] POST /api/models/train (Training 20 epochs)")
    train_payload = {
        "dataset_id": dataset_id,
        "latent_dim": 8,
        "hidden_layers": [128, 64],
        "learning_rate": 0.002,
        "batch_size": 32,
        "epochs": 20,
        "beta_weight": 1.0
    }
    res = requests.post(f"{BASE_URL}/models/train", json=train_payload)
    assert res.status_code == 200, f"Model training failed: {res.text}"
    train_data = res.json()
    model_id = train_data["model_id"]
    print(f"✓ Success: Trained model {model_id} across {train_data['total_epochs']} epochs.")
    print(f"  Final Loss: {train_data['final_loss']} (Recon MSE: {train_data['recon_loss']}, KL Div: {train_data['kl_loss']})")

    # 6. Check model status
    print(f"\n[TEST 6] GET /api/models/status/{model_id}")
    res = requests.get(f"{BASE_URL}/models/status/{model_id}")
    assert res.status_code == 200, f"Model status check failed: {res.text}"
    status_data = res.json()
    print(f"✓ Success: Status = {status_data['status']}, Recorded loss points = {len(status_data['loss_history'])}")

    # 7. Generate synthetic records
    print("\n[TEST 7] POST /api/generate (Generating 200 synthetic records)")
    gen_payload = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "num_records": 200,
        "temperature": 1.0,
        "random_seed": 42
    }
    res = requests.post(f"{BASE_URL}/generate", json=gen_payload)
    assert res.status_code == 200, f"Synthetic generation failed: {res.text}"
    gen_data = res.json()
    exp_id = gen_data["experiment_id"]
    print(f"✓ Success: Generated {gen_data['num_records']} records. Output file: {gen_data['output_filename']}")
    print(f"  Initial Fidelity Score: {gen_data['fidelity_score']}%")
    print(f"  Preview record sample: {gen_data['preview_records'][0]}")

    # 8. Evaluate quality
    print(f"\n[TEST 8] GET /api/evaluate/{exp_id}")
    res = requests.get(f"{BASE_URL}/evaluate/{exp_id}")
    assert res.status_code == 200, f"Evaluation retrieval failed: {res.text}"
    eval_data = res.json()
    print(f"✓ Success: Overall Fidelity Score = {eval_data['overall_fidelity_score']}%")
    print(f"  Domain Validity = {eval_data['domain_validity_percent']}%")
    print(f"  Correlation Delta = {eval_data['correlation_frobenius_delta']}")
    print(f"  Avg KS Statistic = {eval_data['average_ks_statistic']}")

    # 9. Download synthetic CSV
    print(f"\n[TEST 9] GET /api/export/{exp_id}")
    res = requests.get(f"{BASE_URL}/export/{exp_id}")
    assert res.status_code == 200, f"Export CSV failed: {res.text}"
    assert "text/csv" in res.headers.get("content-type", ""), "Response is not CSV"
    lines = res.text.strip().split("\n")
    print(f"✓ Success: Downloaded CSV with {len(lines)} lines (Header + {len(lines)-1} data rows)")
    print(f"  Header: {lines[0]}")

    # 10. List experiment history
    print("\n[TEST 10] GET /api/experiments")
    res = requests.get(f"{BASE_URL}/experiments")
    assert res.status_code == 200, f"List experiments failed: {res.text}"
    exp_list = res.json()
    print(f"✓ Success: Found {len(exp_list)} recorded experiment(s) in SQLite database.")

    # 11. Upload new CSV dataset
    print("\n[TEST 11] POST /api/datasets/upload (Testing CSV file upload)")
    test_csv_path = "test_upload.csv"
    with open(test_csv_path, "w") as f:
        f.write("user_id,age,salary,department\nU1,25,50000,Engineering\nU2,32,75000,Marketing\nU3,45,110000,Management\n")
    
    with open(test_csv_path, "rb") as f:
        files = {"file": ("test_upload.csv", f, "text/csv")}
        res = requests.post(f"{BASE_URL}/datasets/upload", files=files)
    os.remove(test_csv_path)

    assert res.status_code == 200, f"Upload failed: {res.text}"
    up_data = res.json()
    print(f"✓ Success: Uploaded {up_data['filename']} with ID {up_data['id']} ({up_data['rowCount']} rows, {up_data['columnCount']} columns)")

    print("\n==================================================")
    print("ALL 11 BACKEND API ENDPOINTS VERIFIED AND PASSING!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
