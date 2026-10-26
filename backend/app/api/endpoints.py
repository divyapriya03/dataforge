import glob
import logging
import ntpath
import os
import uuid
from pathlib import PureWindowsPath
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from app.core import config
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
    LossPoint,
)
from app.services.preprocessor import DataPreprocessor, PreprocessingError
from app.models.vae import VariationalAutoencoder
from app.services.evaluator import DataQualityEvaluator

log = logging.getLogger("dataforge")

router = APIRouter(prefix="/api", tags=["DataForge API"])

MODEL_TYPE = "numpy-vae"  # hand-written NumPy variational autoencoder (app/models/vae.py)


class TrainingError(Exception):
    """Training did not produce a usable model; the message is safe to return to the client."""


# ----------------------------------------------------------------------------- helpers
def _internal_error(action: str, exc: Exception) -> HTTPException:
    """Log the real exception server-side; give the client a generic message."""
    log.exception("%s failed: %s", action, type(exc).__name__)
    return HTTPException(status_code=500, detail=f"{action} failed because of an internal error. See the server log.")


def _stored_file(stored: Optional[str], kind: str) -> Optional[str]:
    path = config.resolve_stored_path(stored, kind)
    return path if path and os.path.exists(path) else None


def _read_dataset_df(dataset: Dict[str, Any]) -> pd.DataFrame:
    path = _stored_file(dataset["file_path"], "uploads")
    if path is None:
        raise HTTPException(status_code=404, detail=f"The file for dataset '{dataset['id']}' is missing on the server.")
    try:
        return DataPreprocessor.read_csv(path)
    except PreprocessingError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _preview(dataset: Dict[str, Any], rows: int) -> List[Dict[str, Any]]:
    path = _stored_file(dataset["file_path"], "uploads")
    if path is None:
        return []
    try:
        return pd.read_csv(path, nrows=rows).fillna("").to_dict(orient="records")
    except Exception:
        return []


def _model_response(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_id": m["id"],
        "dataset_id": m["dataset_id"],
        "name": m["name"],
        "status": m["status"],
        "current_epoch": len(m["loss_history"]),
        "total_epochs": m["params"].get("epochs", 50),
        "final_loss": m["final_loss"],
        "recon_loss": m["recon_loss"],
        "kl_loss": m["kl_loss"],
        "loss_history": m["loss_history"],
        "model_type": m.get("model_type") or MODEL_TYPE,
        "params": m["params"],
        "error_message": m.get("error_message"),
        "created_at": m.get("created_at"),
        "completed_at": m.get("completed_at"),
    }


def _new_id(prefix: str, exists) -> str:
    """A random id that is not already used (INSERT OR REPLACE would otherwise overwrite a record on a collision)."""
    for _ in range(20):
        candidate = f"{prefix}_{uuid.uuid4().hex[:8]}"
        if not exists(candidate):
            return candidate
    raise RuntimeError("could not allocate a unique id")


def _remove_quietly(*paths: Optional[str]):
    for p in paths:
        if p:
            try:
                os.remove(p)
            except OSError:
                pass


# ----------------------------------------------------------------------------- 1. upload
@router.post("/datasets/upload", response_model=DatasetResponse)
async def upload_dataset(file: UploadFile = File(...)):
    # Display name: no directory part, no control characters, bounded length. The name used on disk is sanitized separately.
    filename = "".join(ch for ch in PureWindowsPath(file.filename or "").name if ch.isprintable())
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are supported.")
    if len(filename) > 200:
        filename = filename[:196] + filename[-4:]  # shorten the name, keep the extension

    upload_dir = config.get_dir("uploads")
    # an id counts as used when it has a record OR a file (a leftover file must never be overwritten)
    dataset_id = _new_id("ds", lambda i: db.get_dataset(i) is not None or bool(glob.glob(os.path.join(upload_dir, f"{i}_*"))))
    file_path = os.path.join(upload_dir, f"{dataset_id}_{config.safe_filename(filename)}")

    limit = config.max_upload_bytes()
    written = 0
    created = False
    try:
        with open(file_path, "xb") as buffer:  # exclusive: never overwrite an existing file
            created = True
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise HTTPException(status_code=413, detail=f"File is larger than the {limit // (1024 * 1024)} MB limit.")
                buffer.write(chunk)

        try:
            row_count, col_count, schema, preview = DataPreprocessor.inspect_csv(file_path)
        except PreprocessingError as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")
        except Exception as e:
            log.warning("CSV inspection failed: %s", type(e).__name__)
            raise HTTPException(status_code=400, detail="Failed to parse CSV: the file is not a readable tabular CSV.")
    except Exception:
        if created:
            _remove_quietly(file_path)  # never keep a file that has no database record
        raise

    try:
        db.insert_dataset(dataset_id, filename, config.to_stored_path(file_path), row_count, col_count, schema)
    except Exception as e:
        _remove_quietly(file_path)
        raise _internal_error("Saving the dataset", e)

    stored = db.get_dataset(dataset_id)  # same timestamp the list/detail endpoints will return
    return {
        "id": dataset_id,
        "filename": filename,
        "rowCount": row_count,
        "columnCount": col_count,
        "uploadDate": stored["created_at"] if stored else "",
        "columns": schema,
        "preview": preview,
    }


# ----------------------------------------------------------------------------- 2. list
@router.get("/datasets", response_model=List[DatasetResponse])
def get_datasets():
    result = []
    for d in db.list_datasets():
        result.append({
            "id": d["id"],
            "filename": d["filename"],
            "rowCount": d["row_count"],
            "columnCount": d["column_count"],
            "uploadDate": d["created_at"],
            "columns": d["columns"],
            "preview": _preview(d, 15),
        })
    return result


# ----------------------------------------------------------------------------- 3. detail
@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
def get_dataset_details(dataset_id: str):
    dataset = db.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {
        "id": dataset["id"],
        "filename": dataset["filename"],
        "rowCount": dataset["row_count"],
        "columnCount": dataset["column_count"],
        "uploadDate": dataset["created_at"],
        "columns": dataset["columns"],
        "preview": _preview(dataset, 50),
    }


# ----------------------------------------------------------------------------- 4. preprocess
@router.post("/preprocess", response_model=PreprocessResponse)
def preprocess_dataset(req: PreprocessRequest):
    """Dry run of the preprocessing pipeline for a column configuration.

    Validates the configuration against the real data and reports the resulting feature layout,
    dropped columns and warnings. The fitted result is saved as ``preprocess_<dataset_id>.json`` for
    inspection; training always re-fits its own pipeline and stores it with the model.
    """
    dataset = db.get_dataset(req.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = _read_dataset_df(dataset)
    preprocessor = DataPreprocessor()
    try:
        X_norm = preprocessor.fit_transform(df, [c.model_dump() for c in req.columns])
        preprocessor.save(os.path.join(config.get_dir("models"), f"preprocess_{req.dataset_id}.json"))
    except PreprocessingError as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")
    except Exception as e:
        raise _internal_error("Preprocessing", e)

    return {
        "dataset_id": req.dataset_id,
        "processed_dimensions": int(X_norm.shape[1]),
        "feature_names": preprocessor.processed_feature_names,
        "ready_for_training": X_norm.shape[0] >= config.MIN_TRAIN_ROWS,
        "rows_used": int(X_norm.shape[0]),
        "dropped_columns": preprocessor.dropped_columns,
        "warnings": preprocessor.warnings,
    }


# ----------------------------------------------------------------------------- 5. train
@router.post("/models/train", response_model=ModelStatusResponse)
def train_model(req: TrainModelRequest):
    """Fit a VAE on the dataset and save it. Runs synchronously; the response is the final state.

    Nothing is written to the database until the request, dataset and preprocessing are validated.
    From then on the model row is created as ``training`` and ends as ``completed`` (weights saved
    and re-loaded successfully) or ``failed`` (with ``error_message``); it is never left ``completed``
    unless the artifacts exist.
    """
    dataset = db.get_dataset(req.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = _read_dataset_df(dataset)
    col_configs = [c.model_dump() for c in req.columns] if req.columns else dataset["columns"]
    preprocessor = DataPreprocessor()
    try:
        X_norm = preprocessor.fit_transform(df, col_configs)
    except PreprocessingError as e:
        raise HTTPException(status_code=400, detail=f"Data pipeline setup error: {e}")
    except Exception as e:
        raise _internal_error("Data pipeline setup", e)
    if X_norm.shape[0] < config.MIN_TRAIN_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"At least {config.MIN_TRAIN_ROWS} usable rows are required to train; this dataset has {X_norm.shape[0]}.",
        )

    model_id = _new_id("vae", db.get_model)
    model_name = req.model_name or f"VAE-{dataset['filename'].rsplit('.', 1)[0]}-Z{req.latent_dim}"
    models_dir = config.get_dir("models")
    weights_path = os.path.join(models_dir, f"{model_id}_weights.npz")
    pipeline_path = os.path.join(models_dir, f"{model_id}_preprocessor.json")

    params = req.model_dump(exclude={"columns"})
    db.insert_model(
        model_id, req.dataset_id, model_name, params,
        weights_path=config.to_stored_path(weights_path), status="training", model_type=MODEL_TYPE,
        preprocessor_path=config.to_stored_path(pipeline_path),
    )

    loss_history: List[LossPoint] = []
    try:
        vae = VariationalAutoencoder(
            input_dim=X_norm.shape[1],
            latent_dim=req.latent_dim,
            hidden_dims=req.hidden_layers,
            learning_rate=req.learning_rate,
            beta=req.beta_weight,
            seed=req.random_seed,
        )
        for ep in range(1, req.epochs + 1):
            tot_l, rec_l, kl_l = vae.train_epoch(X_norm, batch_size=req.batch_size)
            if not all(np.isfinite([tot_l, rec_l, kl_l])):
                raise TrainingError(
                    f"Training diverged at epoch {ep} (loss is not finite). Try a smaller learning rate."
                )
            loss_history.append({
                "epoch": ep,
                "totalLoss": float(round(tot_l, 4)),
                "reconLoss": float(round(rec_l, 4)),
                "klLoss": float(round(kl_l, 4)),
            })

        # Persist artifacts, then prove they can be loaded before declaring success.
        vae.save(weights_path)
        preprocessor.save(pipeline_path)
        check = VariationalAutoencoder(
            input_dim=X_norm.shape[1], latent_dim=req.latent_dim, hidden_dims=req.hidden_layers
        )
        check.load(weights_path)
        DataPreprocessor().load(pipeline_path)
    except Exception as e:
        _remove_quietly(weights_path, pipeline_path)
        message = str(e) if isinstance(e, TrainingError) else "Training failed because of an internal error."
        if not isinstance(e, TrainingError):
            log.exception("Training of %s failed", model_id)
        db.update_model_progress(model_id=model_id, status="failed", error_message=message)
        raise HTTPException(status_code=422 if isinstance(e, TrainingError) else 500, detail=message)

    last = loss_history[-1]
    db.update_model_progress(
        model_id=model_id,
        status="completed",
        final_loss=last["totalLoss"],
        recon_loss=last["reconLoss"],
        kl_loss=last["klLoss"],
        loss_history=loss_history,
    )
    return _model_response(db.get_model(model_id))


# ----------------------------------------------------------------------------- 6. status
@router.get("/models/status/{model_id}", response_model=ModelStatusResponse)
def get_model_status(model_id: str):
    m = db.get_model(model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    return _model_response(m)


# ----------------------------------------------------------------------------- 7. list models
@router.get("/models", response_model=List[ModelStatusResponse])
def get_models():
    return [_model_response(m) for m in db.list_models()]


# ----------------------------------------------------------------------------- 8. generate
def _load_model_for_generation(model_record: Dict[str, Any], dataset_id: str):
    """Rebuild (preprocessor, VAE) from the stored artifacts, or raise a clear 409."""
    weights = _stored_file(model_record.get("weights_path"), "models")
    pipeline_stored = model_record.get("preprocessor_path") or f"models/preprocessor_{dataset_id}.json"  # legacy models
    pipeline = _stored_file(pipeline_stored, "models")
    if weights is None:
        raise HTTPException(status_code=409, detail="The model's weights file is missing on the server.")
    if pipeline is None:
        raise HTTPException(status_code=409, detail="The model's preprocessing file is missing on the server.")

    preprocessor = DataPreprocessor()
    try:
        preprocessor.load(pipeline)
    except (PreprocessingError, OSError) as e:
        raise HTTPException(status_code=409, detail=f"The model's preprocessing file cannot be used: {e}")

    params = model_record["params"]
    try:
        vae = VariationalAutoencoder(
            input_dim=len(preprocessor.processed_feature_names),
            latent_dim=params["latent_dim"],
            hidden_dims=params.get("hidden_layers", [128, 64]),
        )
        vae.load(weights)
    except (KeyError, ValueError, OSError) as e:
        raise HTTPException(status_code=409, detail=f"The model's weights file cannot be used: {e}")
    return preprocessor, vae


@router.post("/generate", response_model=GenerateResponse)
def generate_records(req: GenerateRequest):
    model_record = db.get_model(req.model_id)
    if not model_record:
        raise HTTPException(status_code=404, detail="Model not found")
    if model_record["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Model is not ready for generation (status: {model_record['status']}).")

    dataset_id = req.dataset_id or model_record["dataset_id"]
    if dataset_id != model_record["dataset_id"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model_id}' was trained on dataset '{model_record['dataset_id']}', not '{dataset_id}'.",
        )
    dataset = db.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    max_rows = config.max_generate_rows()
    if req.num_records > max_rows:
        raise HTTPException(status_code=422, detail=f"num_records must not exceed {max_rows}.")

    preprocessor, vae = _load_model_for_generation(model_record, dataset_id)
    real_df = _read_dataset_df(dataset)
    absent = [c for c in preprocessor.column_order if c not in real_df.columns]
    if absent:
        raise HTTPException(status_code=409, detail=f"The dataset file no longer contains the model's columns: {absent}")

    experiment_id = _new_id("gen", db.get_experiment)
    output_filename = f"synthetic_{config.safe_stem(dataset['filename'])}_{experiment_id}.csv"
    output_path = os.path.join(config.get_dir("synthetic"), output_filename)
    tmp_path = output_path + ".tmp"

    try:
        rng = np.random.default_rng(req.random_seed)
        synth_norm = vae.generate(n_samples=req.num_records, temperature=req.temperature, rng=rng)
        synthetic_df = preprocessor.inverse_transform(synth_norm, rng=rng)

        if list(synthetic_df.columns) != preprocessor.column_order or len(synthetic_df) != req.num_records:
            raise RuntimeError("generated data does not match the expected schema")
        if synthetic_df.isna().any().any():
            raise RuntimeError("generated data contains missing values")

        eval_metrics = DataQualityEvaluator.evaluate(real_df, synthetic_df)
        synthetic_df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, output_path)

        db.insert_experiment_with_evaluation(
            {
                "exp_id": experiment_id, "dataset_id": dataset_id, "model_id": req.model_id,
                "num_records": req.num_records, "temp": req.temperature, "seed": req.random_seed,
                "output_path": config.to_stored_path(output_path),
                "fidelity": eval_metrics["overall_fidelity_score"],
                "validity": eval_metrics["domain_validity_percent"],
            },
            _evaluation_row(eval_metrics),
        )
    except Exception as e:
        _remove_quietly(tmp_path, output_path)  # no CSV without a database record
        raise _internal_error("Generation", e)

    return {
        "experiment_id": experiment_id,
        "dataset_id": dataset_id,
        "model_id": req.model_id,
        "num_records": req.num_records,
        "temperature": req.temperature,
        "output_filename": output_filename,
        "fidelity_score": eval_metrics["overall_fidelity_score"],
        "domain_validity_percent": eval_metrics["domain_validity_percent"],
        "preview_records": synthetic_df.head(50).to_dict(orient="records"),
    }


def _evaluation_row(m: Dict[str, Any]) -> Dict[str, Any]:
    """Map evaluator output to the columns of the evaluations table."""
    corr = m["correlation_frobenius_delta"]
    return {
        "score": m["overall_fidelity_score"],
        "ks_dict": m["ks_metrics"],
        "corr_delta": 0.0 if corr is None else corr,  # column is NOT NULL; availability is kept in details
        "validity": m["domain_validity_percent"],
        "summary_diff": m["summary_comparisons"],
        "details": {
            "correlation_available": corr is not None,
            "correlation_mean_abs_delta": m["correlation_mean_abs_delta"],
            "categorical_metrics": m["categorical_metrics"],
            "fidelity_components": m["fidelity_components"],
            "fidelity_weights": m["fidelity_weights"],
            "warnings": m["warnings"],
            "limitations": m["limitations"],
        },
    }


# ----------------------------------------------------------------------------- 9. evaluate
@router.get("/evaluate/{experiment_id}", response_model=EvaluateResponse)
def get_evaluation_metrics(experiment_id: str):
    exp = db.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    eval_rec = db.get_evaluation(experiment_id)
    if not eval_rec:
        # Evaluation was never stored (older experiment): compute it now from the real files.
        dataset = db.get_dataset(exp["dataset_id"])
        synth_path = _stored_file(exp["output_file_path"], "synthetic")
        if not dataset or synth_path is None:
            raise HTTPException(status_code=404, detail="The data needed to evaluate this experiment is missing.")
        try:
            metrics = DataQualityEvaluator.evaluate(_read_dataset_df(dataset), pd.read_csv(synth_path))
            row = _evaluation_row(metrics)
            db.insert_evaluation(experiment_id, row["score"], row["ks_dict"], row["corr_delta"],
                                 row["validity"], row["summary_diff"], row["details"])
        except HTTPException:
            raise
        except Exception as e:
            raise _internal_error("Evaluation", e)
        eval_rec = db.get_evaluation(experiment_id)

    details = eval_rec["details"]
    ks_values = [v["ks_statistic"] for v in eval_rec["ks_metrics"].values()]
    corr_available = details.get("correlation_available", True)  # rows stored before `details` always had a value
    return {
        "experiment_id": experiment_id,
        "overall_fidelity_score": eval_rec["overall_score"],
        "domain_validity_percent": eval_rec["validity_percentage"],
        "average_ks_statistic": float(round(np.mean(ks_values), 4)) if ks_values else None,
        "correlation_frobenius_delta": eval_rec["correlation_delta"] if corr_available else None,
        "ks_metrics": eval_rec["ks_metrics"],
        "summary_comparisons": eval_rec["summary_stats_diff"],
        "correlation_mean_abs_delta": details.get("correlation_mean_abs_delta"),
        "categorical_metrics": details.get("categorical_metrics", {}),
        "fidelity_components": details.get("fidelity_components", {}),
        "fidelity_weights": details.get("fidelity_weights", {}),
        "warnings": details.get("warnings", []),
        "limitations": details.get("limitations"),
    }


# ----------------------------------------------------------------------------- 10. export
@router.get("/export/{experiment_id}")
def download_synthetic_csv(experiment_id: str):
    exp = db.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    path = _stored_file(exp["output_file_path"], "synthetic")
    if path is None:
        raise HTTPException(status_code=404, detail="Synthetic CSV file not found")
    return FileResponse(path=path, media_type="text/csv", filename=os.path.basename(path))


# ----------------------------------------------------------------------------- 11. history
@router.get("/experiments", response_model=List[ExperimentHistoryItem])
def get_experiment_history():
    history = db.list_experiments()
    for item in history:
        item["output_filename"] = ntpath.basename(item["output_file_path"]) if item.get("output_file_path") else None
        item["output_available"] = _stored_file(item.get("output_file_path"), "synthetic") is not None
        item["status"] = "completed" if item["output_available"] else "output_missing"
    return history
