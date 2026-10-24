import os
import shutil
import uuid
import json
import pandas as pd
import numpy as np
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from app.core import database as db
from app.schemas.api_models import (
    DatasetResponse,
    PreprocessRequest,
    PreprocessResponse,
    TrainModelRequest,
    ModelStatusResponse,
    GenerateRequest,
    GenerateResponse,
    EvaluateResponse,
    ExperimentHistoryItem,
    ColumnMetaSchema,
    LossPoint,
)
from app.services.preprocessor import DataPreprocessor
from app.models.vae import VariationalAutoencoder
from app.services.evaluator import DataQualityEvaluator

router = APIRouter(prefix="/api", tags=["DataForge API"])

# Resolve base directory three levels up from this file
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
SYNTHETIC_DIR = os.path.join(BASE_DIR, "data", "synthetic")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_DIR, exist_ok=True)

# 1. POST /api/datasets/upload — Upload CSV file
@router.post("/datasets/upload", response_model=DatasetResponse)
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail='Only CSV files (.csv) are supported.')

    dataset_id = f'ds_{uuid.uuid4().hex[:8]}'
    clean_filename = f'{dataset_id}_{file.filename}'
    file_path = os.path.join(UPLOAD_DIR, clean_filename)

    # Save uploaded file
    with open(file_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Validate and inspect CSV
    try:
        row_count, col_count, schema, preview = DataPreprocessor.inspect_csv(file_path)
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f'Failed to parse CSV: {str(e)}')

    # Record in DB
    db.insert_dataset(dataset_id, file.filename, file_path, row_count, col_count, schema)

    return {
        "id": dataset_id,
        "filename": file.filename,
        "rowCount": row_count,
        "columnCount": col_count,
        "uploadDate": "Just now",
        "columns": schema,
        "preview": preview,
    }

# 2. GET /api/datasets — List all uploaded datasets
@router.get("/datasets", response_model=List[DatasetResponse])
def get_datasets():
    datasets = db.list_datasets()
    result = []
    for d in datasets:
        preview = []
        if os.path.exists(d["file_path"]):
            try:
                df = pd.read_csv(d["file_path"], nrows=15)
                preview = df.fillna("").to_dict(orient="records")
            except Exception:
                pass
        result.append({
            "id": d["id"],
            "filename": d["filename"],
            "rowCount": d["row_count"],
            "columnCount": d["column_count"],
            "uploadDate": d["created_at"],
            "columns": d["columns"],
            "preview": preview,
        })
    return result

# 3. GET /api/datasets/{dataset_id} — Retrieve dataset information
@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
def get_dataset_details(dataset_id: str):
    dataset = db.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    preview = []
    if os.path.exists(dataset["file_path"]):
        try:
            df = pd.read_csv(dataset["file_path"], nrows=50)
            preview = df.fillna("").to_dict(orient="records")
        except Exception:
            pass

    return {
        "id": dataset["id"],
        "filename": dataset["filename"],
        "rowCount": dataset["row_count"],
        "columnCount": dataset["column_count"],
        "uploadDate": dataset["created_at"],
        "columns": dataset["columns"],
        "preview": preview,
    }

# 4. POST /api/preprocess — Preprocess dataset features
@router.post("/preprocess", response_model=PreprocessResponse)
def preprocess_dataset(req: PreprocessRequest):
    dataset = db.get_dataset(req.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = pd.read_csv(dataset["file_path"])
    col_configs = [c.dict() for c in req.columns]

    preprocessor = DataPreprocessor()
    try:
        X_norm = preprocessor.fit_transform(df, col_configs)
        pipe_path = os.path.join(MODEL_DIR, f"preprocessor_{req.dataset_id}.json")
        preprocessor.save(pipe_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")

    return {
        "dataset_id": req.dataset_id,
        "processed_dimensions": int(X_norm.shape[1]),
        "feature_names": preprocessor.processed_feature_names,
        "ready_for_training": True,
    }

# 5. POST /api/models/train — Start VAE model training
@router.post("/models/train", response_model=ModelStatusResponse)
def train_model(req: TrainModelRequest):
    dataset = db.get_dataset(req.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = pd.read_csv(dataset["file_path"])
    preprocessor = DataPreprocessor()
    col_configs = dataset["columns"]

    try:
        X_norm = preprocessor.fit_transform(df, col_configs)
        pipe_path = os.path.join(MODEL_DIR, f"preprocessor_{req.dataset_id}.json")
        preprocessor.save(pipe_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data pipeline setup error: {str(e)}")

    model_id = f"vae_{uuid.uuid4().hex[:8]}"
    model_name = req.model_name or f"VAE-{dataset['filename'].split('.')[0]}-Z{req.latent_dim}"
    weights_path = os.path.join(MODEL_DIR, f"{model_id}_weights.npz")

    input_dim = X_norm.shape[1]
    vae = VariationalAutoencoder(
        input_dim=input_dim,
        latent_dim=req.latent_dim,
        hidden_dims=req.hidden_layers,
        learning_rate=req.learning_rate,
        beta=req.beta_weight,
    )

    db.insert_model(model_id, req.dataset_id, model_name, req.dict(), weights_path)

    loss_history: List[LossPoint] = []
    final_tot, final_rec, final_kl = 0.0, 0.0, 0.0
    for ep in range(1, req.epochs + 1):
        tot_l, rec_l, kl_l = vae.train_epoch(X_norm, batch_size=req.batch_size)
        loss_history.append({
            "epoch": ep,
            "totalLoss": float(round(tot_l, 4)),
            "reconLoss": float(round(rec_l, 4)),
            "klLoss": float(round(kl_l, 4)),
        })
        final_tot, final_rec, final_kl = tot_l, rec_l, kl_l

    # Save trained weights
    vae.save(weights_path)
    db.update_model_progress(
        model_id=model_id,
        status="completed",
        final_loss=float(round(final_tot, 4)),
        recon_loss=float(round(final_rec, 4)),
        kl_loss=float(round(final_kl, 4)),
        loss_history=loss_history,
        weights_path=weights_path,
    )

    return {
        "model_id": model_id,
        "dataset_id": req.dataset_id,
        "name": model_name,
        "status": "completed",
        "current_epoch": req.epochs,
        "total_epochs": req.epochs,
        "final_loss": float(round(final_tot, 4)),
        "recon_loss": float(round(final_rec, 4)),
        "kl_loss": float(round(final_kl, 4)),
        "loss_history": loss_history,
    }

# 6. GET /api/models/status/{model_id} — Check training status
@router.get("/models/status/{model_id}", response_model=ModelStatusResponse)
def get_model_status(model_id: str):
    m = db.get_model(model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")

    total_epochs = m["params"].get("epochs", 50)
    current_epoch = len(m["loss_history"])
    return {
        "model_id": m["id"],
        "dataset_id": m["dataset_id"],
        "name": m["name"],
        "status": m["status"],
        "current_epoch": current_epoch,
        "total_epochs": total_epochs,
        "final_loss": m["final_loss"],
        "recon_loss": m["recon_loss"],
        "kl_loss": m["kl_loss"],
        "loss_history": m["loss_history"],
    }

# 7. GET /api/models — List all trained models
@router.get("/models", response_model=List[ModelStatusResponse])
def get_models():
    models = db.list_models()
    result = []
    for m in models:
        total_epochs = m["params"].get("epochs", 50)
        current_epoch = len(m["loss_history"])
        result.append({
            "model_id": m["id"],
            "dataset_id": m["dataset_id"],
            "name": m["name"],
            "status": m["status"],
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "final_loss": m["final_loss"],
            "recon_loss": m["recon_loss"],
            "kl_loss": m["kl_loss"],
            "loss_history": m["loss_history"],
        })
    return result

# 8. POST /api/generate — Generate synthetic records
@router.post("/generate", response_model=GenerateResponse)
def generate_records(req: GenerateRequest):
    model_record = db.get_model(req.model_id)
    if not model_record:
        raise HTTPException(status_code=404, detail="Model not found")

    dataset = db.get_dataset(req.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    pipe_path = os.path.join(MODEL_DIR, f"preprocessor_{req.dataset_id}.json")
    if not os.path.exists(pipe_path):
        raise HTTPException(status_code=400, detail="Preprocessor pipeline not found for this dataset.")

    preprocessor = DataPreprocessor()
    preprocessor.load(pipe_path)

    # Initialize VAE with trained architecture and load weights
    input_dim = len(preprocessor.processed_feature_names)
    params = model_record["params"]
    vae = VariationalAutoencoder(
        input_dim=input_dim,
        latent_dim=params["latent_dim"],
        hidden_dims=params.get("hidden_layers", [128, 64]),
    )
    vae.load(model_record["weights_path"])

    # Sample latent vectors & decode
    synth_norm = vae.generate(n_samples=req.num_records, temperature=req.temperature, seed=req.random_seed)
    synthetic_df = preprocessor.inverse_transform(synth_norm)

    experiment_id = f"gen_{uuid.uuid4().hex[:8]}"
    output_filename = f"synthetic_{dataset['filename'].replace('.csv', '')}_{experiment_id}.csv"
    output_path = os.path.join(SYNTHETIC_DIR, output_filename)
    synthetic_df.to_csv(output_path, index=False)

    # Evaluate fidelity against real dataset
    real_df = pd.read_csv(dataset["file_path"])
    eval_metrics = DataQualityEvaluator.evaluate(real_df, synthetic_df)

    db.insert_experiment(
        exp_id=experiment_id,
        dataset_id=req.dataset_id,
        model_id=req.model_id,
        num_records=req.num_records,
        temp=req.temperature,
        seed=req.random_seed,
        output_path=output_path,
        fidelity=eval_metrics["overall_fidelity_score"],
        validity=eval_metrics["domain_validity_percent"],
    )

    db.insert_evaluation(
        exp_id=experiment_id,
        score=eval_metrics["overall_fidelity_score"],
        ks_dict=eval_metrics["ks_metrics"],
        corr_delta=eval_metrics["correlation_frobenius_delta"],
        validity=eval_metrics["domain_validity_percent"],
        summary_diff=eval_metrics["summary_comparisons"],
    )

    preview = synthetic_df.head(50).to_dict(orient="records")

    return {
        "experiment_id": experiment_id,
        "dataset_id": req.dataset_id,
        "model_id": req.model_id,
        "num_records": req.num_records,
        "temperature": req.temperature,
        "output_filename": output_filename,
        "fidelity_score": eval_metrics["overall_fidelity_score"],
        "preview_records": preview,
    }

# 9. GET /api/evaluate/{experiment_id} — Retrieve evaluation results
@router.get("/evaluate/{experiment_id}", response_model=EvaluateResponse)
def get_evaluation_metrics(experiment_id: str):
    eval_rec = db.get_evaluation(experiment_id)
    if not eval_rec:
        # Check if experiment exists
        exp = db.get_experiment(experiment_id)
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")
        dataset = db.get_dataset(exp["dataset_id"])
        real_df = pd.read_csv(dataset["file_path"])
        synth_df = pd.read_csv(exp["output_file_path"])
        metrics = DataQualityEvaluator.evaluate(real_df, synth_df)
        db.insert_evaluation(
            exp_id=experiment_id,
            score=metrics["overall_fidelity_score"],
            ks_dict=metrics["ks_metrics"],
            corr_delta=metrics["correlation_frobenius_delta"],
            validity=metrics["domain_validity_percent"],
            summary_diff=metrics["summary_comparisons"],
        )
        eval_rec = db.get_evaluation(experiment_id)

    avg_ks = (
        float(round(np.mean([v["ks_statistic"] for v in eval_rec["ks_metrics"].values()]), 4))
        if eval_rec["ks_metrics"]
        else 0.04
    )
    return {
        "experiment_id": experiment_id,
        "overall_fidelity_score": eval_rec["overall_score"],
        "domain_validity_percent": eval_rec["validity_percentage"],
        "average_ks_statistic": avg_ks,
        "correlation_frobenius_delta": eval_rec["correlation_delta"],
        "ks_metrics": eval_rec["ks_metrics"],
        "summary_comparisons": eval_rec["summary_stats_diff"],
    }

# 10. GET /api/export/{experiment_id} — Download the generated CSV
@router.get("/export/{experiment_id}")
def download_synthetic_csv(experiment_id: str):
    exp = db.get_experiment(experiment_id)
    if not exp or not os.path.exists(exp["output_file_path"]):
        raise HTTPException(status_code=404, detail="Synthetic CSV file not found")

    filename = os.path.basename(exp["output_file_path"])
    return FileResponse(
        path=exp["output_file_path"],
        media_type="text/csv",
        filename=filename,
    )

# 11. GET /api/experiments — Retrieve experiment history
@router.get("/experiments", response_model=List[ExperimentHistoryItem])
def get_experiment_history():
    history = db.list_experiments()
    return history
