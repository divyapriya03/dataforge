// Types mirroring the FastAPI schemas in backend/app/schemas/api_models.py (field names are the real ones).

export interface ApiColumn {
  name: string;
  inferredType: string;
  selectedType: string;
  nullCount: number;
  nullPercentage: number;
  imputationStrategy: string;
  included: boolean;
  min?: number | null;
  max?: number | null;
  mean?: number | null;
  std?: number | null;
  categories?: string[] | null;
}

/** GET /api/datasets, GET /api/datasets/{id}, POST /api/datasets/upload */
export interface ApiDataset {
  id: string;
  filename: string;
  rowCount: number;
  columnCount: number;
  uploadDate: string;
  columns: ApiColumn[];
  preview?: Record<string, unknown>[] | null;
}

/** GET /api/health */
export interface ApiHealth {
  status: string;
  message: string;
  version: string;
  deep_learning_engine: string;
}

/** POST /api/preprocess (a dry run; training fits its own pipeline) */
export interface ApiPreprocessResponse {
  dataset_id: string;
  processed_dimensions: number;
  feature_names: string[];
  ready_for_training: boolean;
  rows_used?: number | null;
  dropped_columns: { name: string; reason: string }[];
  warnings: string[];
}

/** POST /api/models/train body. Everything except dataset_id is optional on the server (defaults apply). */
export interface TrainRequest {
  dataset_id: string;
  /** Per-column preprocessing configuration (as edited in Dataset Studio); the server uses the dataset's stored schema when omitted. */
  columns?: ApiColumn[];
  model_name?: string;
  latent_dim?: number;
  hidden_layers?: number[];
  learning_rate?: number;
  batch_size?: number;
  epochs?: number;
  beta_weight?: number;
  random_seed?: number;
}

export interface ApiLossPoint {
  epoch: number;
  totalLoss: number;
  reconLoss: number;
  klLoss: number;
}

/** POST /api/models/train, GET /api/models, GET /api/models/status/{id} */
export interface ApiModel {
  model_id: string;
  dataset_id: string;
  name: string;
  /** 'initialized' | 'training' | 'completed' | 'failed' */
  status: string;
  current_epoch: number;
  total_epochs: number;
  final_loss: number | null;
  recon_loss: number | null;
  kl_loss: number | null;
  loss_history: ApiLossPoint[];
  model_type?: string | null;
  params?: Record<string, unknown> | null;
  error_message?: string | null;
  created_at?: string | null;
  completed_at?: string | null;
}

/** GET /api/experiments (one item per successful generation, newest first) */
export interface ApiExperiment {
  id: string;
  dataset_id: string;
  model_id: string;
  dataset_name?: string | null;
  model_name?: string | null;
  model_type?: string | null;
  num_records: number;
  temperature: number;
  fidelity_score?: number | null;
  validity_score?: number | null;
  random_seed?: number | null;
  output_filename?: string | null;
  /** 'completed' (CSV on disk) | 'output_missing' (CSV removed since generation) */
  status: string;
  output_available: boolean;
  /** SQLite CURRENT_TIMESTAMP, UTC, "YYYY-MM-DD HH:MM:SS" */
  created_at: string;
}

export interface ApiKsMetric {
  ks_statistic: number | null;
  p_value: number | null;
  wasserstein_distance: number | null;
  status: string;
}

export interface ApiCategoricalMetric {
  tvd: number | null;
  status: string;
  invalid_categories?: string[];
  real_proportions?: Record<string, number>;
  synthetic_proportions?: Record<string, number>;
}

export interface ApiSummaryRow {
  feature: string;
  realMean: number | null;
  syntheticMean: number | null;
  realStd: number | null;
  syntheticStd: number | null;
  ksStatistic: number | null;
  pValue: number | null;
  status: string;
}

/** GET /api/evaluate/{experiment_id}; fields documented in BACKEND_DEVELOPMENT.md section 6 */
export interface ApiEvaluation {
  experiment_id: string;
  overall_fidelity_score: number;
  domain_validity_percent: number;
  average_ks_statistic: number | null;
  correlation_frobenius_delta: number | null;
  correlation_mean_abs_delta?: number | null;
  ks_metrics: Record<string, ApiKsMetric>;
  categorical_metrics?: Record<string, ApiCategoricalMetric>;
  summary_comparisons?: ApiSummaryRow[];
  fidelity_components?: Record<string, number | null>;
  fidelity_weights?: Record<string, number>;
  warnings?: string[];
  limitations?: string | null;
}

/** POST /api/generate body. dataset_id defaults to the model's own dataset on the server. */
export interface GenerateRequest {
  model_id: string;
  dataset_id?: string;
  num_records: number;
  temperature?: number;
  random_seed?: number;
}

/** POST /api/generate response */
export interface ApiGenerateResponse {
  experiment_id: string;
  dataset_id: string;
  model_id: string;
  num_records: number;
  temperature: number;
  output_filename: string;
  fidelity_score: number;
  domain_validity_percent?: number | null;
  preview_records: Record<string, unknown>[];
}
