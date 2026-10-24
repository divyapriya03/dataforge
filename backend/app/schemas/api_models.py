from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ColumnMetaSchema(BaseModel):
    name: str
    inferredType: str
    selectedType: str
    nullCount: int
    nullPercentage: float
    imputationStrategy: str
    included: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    categories: Optional[List[str]] = None

class DatasetResponse(BaseModel):
    id: str
    filename: str
    rowCount: int
    columnCount: int
    uploadDate: str
    columns: List[ColumnMetaSchema]
    preview: Optional[List[Dict[str, Any]]] = None

class PreprocessRequest(BaseModel):
    dataset_id: str
    columns: List[ColumnMetaSchema]

class PreprocessResponse(BaseModel):
    dataset_id: str
    processed_dimensions: int
    feature_names: List[str]
    ready_for_training: bool

class TrainModelRequest(BaseModel):
    dataset_id: str
    model_name: Optional[str] = None
    latent_dim: int = Field(default=8, ge=2, le=64)
    hidden_layers: List[int] = Field(default=[128, 64])
    learning_rate: float = Field(default=0.001, gt=0.0)
    batch_size: int = Field(default=32, ge=8)
    epochs: int = Field(default=50, ge=1, le=500)
    beta_weight: float = Field(default=1.0, ge=0.0)

class LossPoint(BaseModel):
    epoch: int
    totalLoss: float
    reconLoss: float
    klLoss: float

class ModelStatusResponse(BaseModel):
    model_id: str
    dataset_id: str
    name: str
    status: str
    current_epoch: int
    total_epochs: int
    final_loss: Optional[float] = None
    recon_loss: Optional[float] = None
    kl_loss: Optional[float] = None
    loss_history: List[LossPoint] = []

class GenerateRequest(BaseModel):
    model_id: str
    dataset_id: str
    num_records: int = Field(default=1000, ge=10, le=50000)
    temperature: float = Field(default=1.0, ge=0.1, le=2.5)
    random_seed: int = Field(default=42)

class GenerateResponse(BaseModel):
    experiment_id: str
    dataset_id: str
    model_id: str
    num_records: int
    temperature: float
    output_filename: str
    fidelity_score: float
    preview_records: List[Dict[str, Any]]

class EvaluateResponse(BaseModel):
    experiment_id: str
    overall_fidelity_score: float
    domain_validity_percent: float
    average_ks_statistic: float
    correlation_frobenius_delta: float
    ks_metrics: Dict[str, Any]
    summary_comparisons: List[Dict[str, Any]]

class ExperimentHistoryItem(BaseModel):
    id: str
    dataset_id: str
    model_id: str
    dataset_name: Optional[str] = None
    model_name: Optional[str] = None
    num_records: int
    temperature: float
    fidelity_score: Optional[float] = None
    validity_score: Optional[float] = None
    created_at: str
