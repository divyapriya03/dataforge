from pydantic import BaseModel, Field, field_validator
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
    rows_used: Optional[int] = None
    dropped_columns: List[Dict[str, str]] = []
    warnings: List[str] = []

class TrainModelRequest(BaseModel):
    dataset_id: str
    model_name: Optional[str] = None
    latent_dim: int = Field(default=8, ge=2, le=64)
    hidden_layers: List[int] = Field(default=[128, 64])
    learning_rate: float = Field(default=0.001, gt=0.0, le=1.0)
    batch_size: int = Field(default=32, ge=8, le=4096)
    epochs: int = Field(default=50, ge=1, le=500)
    beta_weight: float = Field(default=1.0, ge=0.0, le=100.0)
    random_seed: Optional[int] = Field(default=None, ge=0, le=2**32 - 1)
    # Optional per-column preprocessing configuration; defaults to the dataset's stored schema.
    columns: Optional[List[ColumnMetaSchema]] = None

    @field_validator("model_name")
    @classmethod
    def _name_ok(cls, v):
        if v is not None:
            v = v.strip()
            if not v or len(v) > 100:
                raise ValueError("model_name must be 1-100 characters")
        return v

    @field_validator("hidden_layers")
    @classmethod
    def _layers_ok(cls, v):
        # The NumPy VAE has exactly two hidden layers in the encoder and in the decoder.
        if len(v) != 2 or any(n < 2 or n > 1024 for n in v):
            raise ValueError("hidden_layers must list exactly two layer sizes between 2 and 1024, e.g. [128, 64]")
        return v

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
    model_type: Optional[str] = None
    params: Optional[Dict[str, Any]] = None  # training parameters as submitted (hyperparameters, seed)
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

class GenerateRequest(BaseModel):
    model_id: str
    dataset_id: Optional[str] = None  # defaults to the model's own dataset; if given it must match
    # Upper limit is configurable (DATAFORGE_MAX_GENERATE_ROWS) and enforced by the endpoint.
    num_records: int = Field(default=1000, ge=10)
    temperature: float = Field(default=1.0, ge=0.1, le=2.5)
    random_seed: int = Field(default=42, ge=0, le=2**32 - 1)  # NumPy seeds must be non-negative

class GenerateResponse(BaseModel):
    experiment_id: str
    dataset_id: str
    model_id: str
    num_records: int
    temperature: float
    output_filename: str
    fidelity_score: float
    domain_validity_percent: Optional[float] = None
    preview_records: List[Dict[str, Any]]

class EvaluateResponse(BaseModel):
    experiment_id: str
    overall_fidelity_score: float
    domain_validity_percent: float
    average_ks_statistic: Optional[float] = None          # None when there are no numeric columns
    correlation_frobenius_delta: Optional[float] = None   # None when fewer than 2 usable numeric columns
    ks_metrics: Dict[str, Any]
    summary_comparisons: List[Dict[str, Any]]
    # Additive fields (None / empty for evaluations stored before they existed)
    correlation_mean_abs_delta: Optional[float] = None
    categorical_metrics: Dict[str, Any] = {}
    fidelity_components: Dict[str, Optional[float]] = {}
    fidelity_weights: Dict[str, float] = {}
    warnings: List[str] = []
    limitations: Optional[str] = None

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
    random_seed: Optional[int] = None
    output_filename: Optional[str] = None
    # An experiment row only exists after generation fully succeeded, so it is 'completed' while its CSV is
    # on disk and 'output_missing' when the file has been removed since. Failed generations are never stored.
    status: str = "completed"
    output_available: bool = False
    model_type: Optional[str] = None
    created_at: str
