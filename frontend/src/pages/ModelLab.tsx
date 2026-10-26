import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  Database,
  Info,
  Loader2,
  PlayCircle,
  RefreshCw,
  XCircle,
} from 'lucide-react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { inflight, useInflight } from '../services/inflight';
import { API_BASE_URL, TRAIN_TIMEOUT_MS, getDatasets, previewPreprocessing, startTraining, type ApiErrorKind } from '../services/api';
import {
  DEFAULT_FORM,
  LIMITS,
  MIN_TRAIN_ROWS,
  validateTrainingForm,
  type TrainingFormErrors,
  type TrainingFormValues,
} from '../services/trainingConfig';
import type { ApiColumn, ApiDataset, ApiModel, ApiPreprocessResponse, TrainRequest } from '../types/api';
import type { ModelRecord } from '../types';

interface ModelLabProps {
  /** Models loaded from the backend (GET /api/models), owned by App. */
  models: ModelRecord[];
  activeModel: ModelRecord | null;
  backendOnline: boolean;
  backendChecking: boolean;
  /** Called with the backend's response after a training run completed successfully. */
  onModelTrained: (model: ApiModel) => void;
  /** Re-reads GET /api/models; resolves to an error message or null. */
  onRefreshModels: () => Promise<string | null>;
  onSelectModel: (modelId: string) => void;
  /** With a model ID, the Data Generator opens with that model selected. */
  onNavigateToGenerator: (modelId?: string) => void;
  /** Dataset to select first (from the URL or the app's current selection). */
  initialDatasetId?: string;
  /** Column configurations edited in Dataset Studio, by dataset ID; sent with the training request. */
  columnConfigs?: Record<string, ApiColumn[]>;
  onSelectDataset?: (datasetId: string) => void;
}

type TrainState =
  | { phase: 'idle' }
  | { phase: 'running'; startedAt: number }
  | { phase: 'success'; result: ApiModel; seconds: number }
  | { phase: 'error'; message: string; kind: ApiErrorKind; seconds: number };

type PreviewState =
  | { phase: 'idle' }
  | { phase: 'loading' }
  | { phase: 'ok'; data: ApiPreprocessResponse }
  | { phase: 'error'; message: string };

const inputBase =
  'w-full bg-slate-950 border rounded-lg px-3 py-2 text-sm text-slate-200 outline-none focus-visible:ring-2 focus-visible:ring-indigo-500';
const card = 'bg-slate-900/80 border border-slate-800 rounded-2xl p-6';

const FIELD_HELP: Record<keyof TrainingFormValues, { label: string; help: string }> = {
  epochs: {
    label: 'Epochs',
    help: `Passes over the data (${LIMITS.epochs.min}–${LIMITS.epochs.max}). Backend default is 50; datasets of a few hundred rows did noticeably better with 200 or more in backend tests.`,
  },
  learningRate: {
    label: 'Learning rate',
    help: 'Adam step size (greater than 0, at most 1). Backend default 0.001; very large values can make training diverge.',
  },
  batchSize: {
    label: 'Batch size',
    help: `Rows per update (${LIMITS.batchSize.min}–${LIMITS.batchSize.max}).`,
  },
  latentDim: {
    label: 'Latent dimension',
    help: `Size of the bottleneck the model samples from when generating (${LIMITS.latentDim.min}–${LIMITS.latentDim.max}).`,
  },
  hiddenLayer1: {
    label: 'Hidden layer 1',
    help: `Units in the outer hidden layer (${LIMITS.hiddenLayer.min}–${LIMITS.hiddenLayer.max}).`,
  },
  hiddenLayer2: {
    label: 'Hidden layer 2',
    help: `Units in the inner hidden layer (${LIMITS.hiddenLayer.min}–${LIMITS.hiddenLayer.max}).`,
  },
  betaWeight: {
    label: 'KL weight (β)',
    help: `Weight of the KL term in the loss (${LIMITS.betaWeight.min}–${LIMITS.betaWeight.max}). Backend default 1.`,
  },
  randomSeed: {
    label: 'Random seed (optional)',
    help: 'Whole number for reproducible training. Leave empty for a random seed.',
  },
  modelName: {
    label: 'Model name (optional)',
    help: `Up to ${LIMITS.modelNameMaxLength} characters. Leave empty to use the generated name.`,
  },
};

const formatLoss = (v: number | null | undefined) => (v === null || v === undefined ? 'unavailable' : v.toFixed(4));

/** SQLite CURRENT_TIMESTAMP values look like "2026-10-24 05:10:34" and are UTC. */
const parseServerTime = (s?: string | null): number | null => {
  if (!s) return null;
  const t = Date.parse(s.includes('T') ? s : `${s.replace(' ', 'T')}Z`);
  return Number.isNaN(t) ? null : t;
};

const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
  const style =
    status === 'completed'
      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
      : status === 'failed'
      ? 'bg-rose-500/10 text-rose-400 border-rose-500/30'
      : status === 'training'
      ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
      : 'bg-slate-700/30 text-slate-300 border-slate-600/40';
  return <span className={`inline-block px-2 py-0.5 rounded-full text-[11px] font-semibold border ${style}`}>{status}</span>;
};

interface FieldProps {
  name: keyof TrainingFormValues;
  values: TrainingFormValues;
  errors: TrainingFormErrors;
  disabled: boolean;
  onChange: (name: keyof TrainingFormValues, value: string) => void;
  inputMode?: 'numeric' | 'decimal' | 'text';
  placeholder?: string;
}

const Field: React.FC<FieldProps> = ({ name, values, errors, disabled, onChange, inputMode = 'numeric', placeholder }) => {
  const id = `train-${name}`;
  const error = errors[name];
  return (
    <div>
      <label htmlFor={id} className="block text-slate-300 text-xs font-medium mb-1">
        {FIELD_HELP[name].label}
      </label>
      <input
        id={id}
        type="text"
        inputMode={inputMode}
        value={values[name]}
        placeholder={placeholder}
        disabled={disabled}
        aria-invalid={error ? true : undefined}
        aria-describedby={`${id}-help${error ? ` ${id}-error` : ''}`}
        onChange={(e) => onChange(name, e.target.value)}
        className={`${inputBase} ${error ? 'border-rose-500' : 'border-slate-800'} disabled:opacity-60`}
      />
      <p id={`${id}-help`} className="text-[11px] text-slate-500 mt-1">
        {FIELD_HELP[name].help}
      </p>
      {error && (
        <p id={`${id}-error`} role="alert" className="text-[11px] text-rose-400 mt-1">
          {error}
        </p>
      )}
    </div>
  );
};

export const ModelLab: React.FC<ModelLabProps> = ({
  models,
  activeModel,
  backendOnline,
  backendChecking,
  onModelTrained,
  onRefreshModels,
  onSelectModel,
  onNavigateToGenerator,
  initialDatasetId,
  columnConfigs,
  onSelectDataset,
}) => {
  // ---- datasets (always from the backend)
  const [datasets, setDatasets] = useState<ApiDataset[]>([]);
  const [datasetsPhase, setDatasetsPhase] = useState<'loading' | 'ready' | 'error'>('loading');
  const [datasetsError, setDatasetsError] = useState<string>('');
  const [selectedId, setSelectedId] = useState<string>('');

  const loadDatasets = useCallback(async () => {
    setDatasetsPhase('loading');
    const res = await getDatasets();
    if (res.success && Array.isArray(res.data)) {
      setDatasets(res.data);
      setSelectedId((cur) => {
        const has = (id?: string) => !!id && res.data!.some((d) => d.id === id);
        return has(cur) ? cur : has(initialDatasetId) ? initialDatasetId! : res.data![0]?.id ?? '';
      });
      setDatasetsPhase('ready');
    } else {
      setDatasets([]);
      setDatasetsError(res.error ?? 'The server returned an unexpected dataset list.');
      setDatasetsPhase('error');
    }
  }, []);

  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  // a dataset chosen elsewhere (URL parameter, Dataset Studio) takes over the selection
  useEffect(() => {
    if (initialDatasetId && datasets.some((d) => d.id === initialDatasetId)) setSelectedId(initialDatasetId);
  }, [initialDatasetId, datasets]);

  const selected = datasets.find((d) => d.id === selectedId) ?? null;
  const customColumns = selected ? columnConfigs?.[selected.id] : undefined;
  const effectiveColumns = customColumns ?? selected?.columns ?? [];
  const usableColumns = effectiveColumns.filter((c) => c.included && c.selectedType !== 'drop').length;
  const blockReason = !selected
    ? 'Select a dataset to train on.'
    : selected.rowCount < MIN_TRAIN_ROWS
    ? `This dataset has ${selected.rowCount} row(s); training needs at least ${MIN_TRAIN_ROWS}.`
    : usableColumns === 0
    ? 'This dataset has no usable columns (every column is excluded).'
    : '';

  // ---- preprocessing check (dry run)
  const [preview, setPreview] = useState<PreviewState>({ phase: 'idle' });
  useEffect(() => {
    setPreview({ phase: 'idle' });
  }, [selectedId]);

  const runPreview = async () => {
    if (!selected) return;
    setPreview({ phase: 'loading' });
    const res = await previewPreprocessing(selected.id, effectiveColumns);
    setPreview(res.success && res.data ? { phase: 'ok', data: res.data } : { phase: 'error', message: res.error ?? 'Preprocessing check failed.' });
  };

  // ---- form
  const [values, setValues] = useState<TrainingFormValues>(DEFAULT_FORM);
  const [errors, setErrors] = useState<TrainingFormErrors>({});
  const setField = (name: keyof TrainingFormValues, value: string) => {
    setValues((v) => ({ ...v, [name]: value }));
    setErrors((e) => (e[name] ? { ...e, [name]: undefined } : e));
  };

  // ---- training
  const [train, setTrain] = useState<TrainState>({ phase: 'idle' });
  const [submitted, setSubmitted] = useState<TrainRequest | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const trainingPending = useInflight('train'); // true while ANY training request of this app is running
  const [refreshNote, setRefreshNote] = useState<string>('');

  useEffect(() => {
    if (train.phase !== 'running') return;
    setElapsed(0);
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - train.startedAt) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [train]);

  const refreshModels = async () => {
    const err = await onRefreshModels();
    setRefreshNote(err ? `Could not refresh the model list: ${err}` : '');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (blockReason || inflight.has('train')) return; // one training request at a time, even after leaving the page
    const { errors: found, request: validated } = validateTrainingForm(values, selectedId);
    setErrors(found);
    if (!validated) return;
    // columns edited in Dataset Studio are part of the request; otherwise the server uses the dataset's stored schema
    const request: TrainRequest = customColumns ? { ...validated, columns: customColumns } : validated;

    if (!inflight.begin('train')) return;
    const startedAt = Date.now();
    setSubmitted(request);
    setTrain({ phase: 'running', startedAt });
    try {
      const res = await startTraining(request);
      const seconds = Math.round((Date.now() - startedAt) / 100) / 10;
      if (res.success && res.data && typeof res.data.model_id === 'string') {
        if (res.data.status !== 'completed') {
          setTrain({ phase: 'error', kind: 'format', seconds, message: `The server reported status "${res.data.status}" instead of "completed".` });
        } else {
          setTrain({ phase: 'success', result: res.data, seconds });
          onModelTrained(res.data);
        }
      } else if (res.success) {
        setTrain({ phase: 'error', kind: 'format', seconds, message: 'The server returned a response without a model_id.' });
      } else {
        setTrain({ phase: 'error', kind: res.kind ?? 'http', seconds, message: res.error ?? 'Training failed.' });
      }
      await refreshModels(); // a failed run is also stored on the server; show the real list either way
    } finally {
      inflight.end('train');
    }
  };

  const running = train.phase === 'running' || trainingPending;
  const trainingElsewhere = trainingPending && train.phase !== 'running';

  // a training started from an earlier visit to this page has ended: show what the server now has
  const wasPending = useRef(false);
  useEffect(() => {
    if (wasPending.current && !trainingPending && train.phase !== 'running') refreshModels();
    wasPending.current = trainingPending;
  }, [trainingPending]);
  const result = train.phase === 'success' ? train.result : null;
  const chartData = result?.loss_history ?? [];

  const serverSeconds = (() => {
    const a = parseServerTime(result?.created_at);
    const b = parseServerTime(result?.completed_at);
    return a !== null && b !== null && b >= a ? Math.round((b - a) / 1000) : null;
  })();

  const paramRows: [string, string][] = result
    ? (() => {
        const p = (result.params ?? {}) as Record<string, unknown>;
        const show = (v: unknown) => (v === undefined ? 'unavailable' : v === null ? 'not set (random)' : Array.isArray(v) ? v.join(', ') : String(v));
        return [
          ['Epochs', show(p.epochs)],
          ['Learning rate', show(p.learning_rate)],
          ['Batch size', show(p.batch_size)],
          ['Latent dimension', show(p.latent_dim)],
          ['Hidden layers', show(p.hidden_layers)],
          ['KL weight (β)', show(p.beta_weight)],
          ['Random seed', show(p.random_seed)],
        ];
      })()
    : [];

  return (
    <div className="space-y-6">
      {/* A. Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-900/80 border border-slate-800 p-5 rounded-2xl">
        <div className="min-w-0">
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            <BrainCircuit className="w-5 h-5 text-indigo-400" aria-hidden="true" />
            ModelLab
          </h2>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl">
            Train a model on one of your uploaded datasets. The trained model is saved by the backend and can then be used to generate synthetic tabular data.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span
            role="status"
            data-testid="backend-status"
            className={`inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-full border ${
              backendChecking
                ? 'text-amber-400 border-amber-500/30'
                : backendOnline
                ? 'text-emerald-400 border-emerald-500/30'
                : 'text-rose-400 border-rose-500/30'
            }`}
          >
            {backendChecking ? 'Checking backend…' : backendOnline ? 'Backend connected' : `Backend unreachable (${API_BASE_URL})`}
          </span>
          {activeModel && (
            <button
              type="button"
              onClick={() => onNavigateToGenerator()}
              className="inline-flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-4 py-2 rounded-lg focus-visible:ring-2 focus-visible:ring-indigo-300 outline-none"
            >
              Go to Data Generator <ArrowRight className="w-3.5 h-3.5" aria-hidden="true" />
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <form onSubmit={handleSubmit} noValidate className="lg:col-span-2 space-y-6" aria-label="Training setup">
          {/* B. Dataset selection */}
          <section className={card} aria-labelledby="dataset-heading">
            <h3 id="dataset-heading" className="flex items-center gap-2 text-indigo-400 font-semibold text-sm border-b border-slate-800 pb-3 mb-4">
              <Database className="w-4 h-4" aria-hidden="true" /> 1. Dataset
            </h3>

            {datasetsPhase === 'loading' && (
              <p className="text-sm text-slate-400 flex items-center gap-2" role="status">
                <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" /> Loading datasets…
              </p>
            )}

            {datasetsPhase === 'error' && (
              <div role="alert" className="text-sm text-rose-300 bg-rose-500/10 border border-rose-500/30 rounded-lg p-3 flex items-start justify-between gap-3">
                <span>Could not load datasets: {datasetsError}</span>
                <button type="button" onClick={loadDatasets} className="text-xs underline whitespace-nowrap focus-visible:ring-2 focus-visible:ring-indigo-400 outline-none rounded">
                  Retry
                </button>
              </div>
            )}

            {datasetsPhase === 'ready' && datasets.length === 0 && (
              <div className="text-sm text-slate-300 bg-slate-950/60 border border-slate-800 rounded-lg p-4">
                <p className="font-medium">No datasets found.</p>
                <p className="text-xs text-slate-400 mt-1">Upload a CSV in Dataset Studio first; it will appear here.</p>
              </div>
            )}

            {datasetsPhase === 'ready' && datasets.length > 0 && (
              <div className="space-y-3">
                <div>
                  <label htmlFor="dataset-select" className="block text-slate-300 text-xs font-medium mb-1">
                    Dataset to train on
                  </label>
                  <select
                    id="dataset-select"
                    value={selectedId}
                    disabled={running}
                    onChange={(e) => {
                      setSelectedId(e.target.value);
                      onSelectDataset?.(e.target.value);
                    }}
                    className={`${inputBase} border-slate-800`}
                  >
                    {datasets.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.filename} ({d.id})
                      </option>
                    ))}
                  </select>
                </div>

                {selected && (
                  <div className="text-xs bg-slate-950/60 border border-slate-800 rounded-lg p-3 space-y-1" data-testid="dataset-info">
                    <p className="text-slate-300">
                      <span className="font-semibold">{selected.filename}</span> · <span className="font-mono">{selected.id}</span>
                    </p>
                    <p className="text-slate-400">
                      {selected.rowCount.toLocaleString('en-US')} rows · {selected.columnCount} columns · {usableColumns} usable for training
                      {selected.columnCount - usableColumns > 0 && ` (${selected.columnCount - usableColumns} excluded, e.g. identifiers)`}
                    </p>
                    {customColumns && (
                      <p className="text-indigo-300" data-testid="custom-columns-note">Training will use the column configuration you edited in Dataset Studio.</p>
                    )}
                    <p className="text-slate-500">Validated when uploaded. Training prepares the data itself (missing-value handling, scaling, one-hot encoding), so no separate preprocessing step is required.</p>
                    <button
                      type="button"
                      onClick={runPreview}
                      disabled={preview.phase === 'loading' || running}
                      className="mt-1 text-indigo-300 underline disabled:opacity-60 focus-visible:ring-2 focus-visible:ring-indigo-400 outline-none rounded"
                    >
                      {preview.phase === 'loading' ? 'Checking…' : 'Check preprocessing (dry run)'}
                    </button>
                    {preview.phase === 'ok' && (
                      <div className="text-slate-300 mt-1" data-testid="preprocess-result">
                        <p>
                          {preview.data.processed_dimensions} model input features from {preview.data.rows_used ?? 'unknown'} rows ·{' '}
                          {preview.data.ready_for_training ? 'ready for training' : 'not ready for training'}
                        </p>
                        {preview.data.dropped_columns.length > 0 && (
                          <p className="text-slate-400">Not used: {preview.data.dropped_columns.map((c) => `${c.name} (${c.reason})`).join('; ')}</p>
                        )}
                        {preview.data.warnings.map((w) => (
                          <p key={w} className="text-amber-300">⚠ {w}</p>
                        ))}
                      </div>
                    )}
                    {preview.phase === 'error' && (
                      <p role="alert" className="text-rose-300 mt-1">{preview.message}</p>
                    )}
                  </div>
                )}
              </div>
            )}
          </section>

          {/* C. Configuration */}
          <section className={card} aria-labelledby="config-heading">
            <h3 id="config-heading" className="text-indigo-400 font-semibold text-sm border-b border-slate-800 pb-3 mb-4">
              2. Training configuration
            </h3>
            <p className="flex items-start gap-2 text-xs text-slate-400 bg-slate-950/60 border border-slate-800 rounded-lg p-3 mb-4">
              <Info className="w-4 h-4 text-indigo-400 flex-shrink-0 mt-px" aria-hidden="true" />
              <span>
                The backend trains a fixed model type: a variational autoencoder (VAE) with exactly <strong>two</strong> hidden layers in the encoder and in the
                decoder and LeakyReLU activations. You can change the layer sizes but not the number of layers or the activation.
              </span>
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Field name="epochs" values={values} errors={errors} disabled={running} onChange={setField} />
              <Field name="learningRate" values={values} errors={errors} disabled={running} onChange={setField} inputMode="decimal" />
              <Field name="batchSize" values={values} errors={errors} disabled={running} onChange={setField} />
              <Field name="randomSeed" values={values} errors={errors} disabled={running} onChange={setField} placeholder="random" />
              <Field name="latentDim" values={values} errors={errors} disabled={running} onChange={setField} />
              <Field name="betaWeight" values={values} errors={errors} disabled={running} onChange={setField} inputMode="decimal" />
              <Field name="hiddenLayer1" values={values} errors={errors} disabled={running} onChange={setField} />
              <Field name="hiddenLayer2" values={values} errors={errors} disabled={running} onChange={setField} />
              <div className="sm:col-span-2">
                <Field name="modelName" values={values} errors={errors} disabled={running} onChange={setField} inputMode="text" />
              </div>
            </div>
          </section>

          {/* D. Action */}
          <div className="space-y-2">
            {Object.values(errors).some(Boolean) && (
              <p role="alert" className="text-xs text-rose-300">Fix the highlighted fields before starting training.</p>
            )}
            {blockReason && datasetsPhase === 'ready' && <p className="text-xs text-amber-300">{blockReason}</p>}
            <button
              type="submit"
              disabled={running || !!blockReason || datasetsPhase !== 'ready'}
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:text-slate-400 disabled:cursor-not-allowed text-white text-sm font-semibold px-6 py-3 rounded-xl shadow-md shadow-indigo-600/30 focus-visible:ring-2 focus-visible:ring-indigo-300 outline-none"
            >
              {running ? <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" /> : <PlayCircle className="w-4 h-4" aria-hidden="true" />}
              {running ? 'Training…' : 'Start Training'}
            </button>
          </div>
        </form>

        {/* E + F. Status and results */}
        <div className="space-y-6">
          <section className={card} aria-labelledby="status-heading" aria-live="polite">
            <h3 id="status-heading" className="text-indigo-400 font-semibold text-sm border-b border-slate-800 pb-3 mb-4">
              Training status
            </h3>
            {train.phase === 'idle' && !trainingElsewhere && (
              <p className="text-sm text-slate-300" data-testid="status-ready">
                <StatusBadge status="ready" /> <span className="ml-2">Waiting for you to start a run.</span>
              </p>
            )}
            {trainingElsewhere && (
              <p data-testid="status-training-elsewhere" className="text-sm text-amber-300 flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" /> A training request started earlier is still running on the server. Wait for it to finish; the result will appear in the model list below.
              </p>
            )}
            {train.phase === 'running' && (
              <div data-testid="status-training" className="space-y-2">
                <p className="text-sm text-slate-200 flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin text-amber-400" aria-hidden="true" />
                  <StatusBadge status="training" /> <span>Waiting for the server… {elapsed}s</span>
                </p>
                <p className="text-xs text-slate-400">
                  The backend trains inside this request and sends nothing until it is finished, so there is no progress bar or epoch counter. Keep this page open; the request gives up after{' '}
                  {Math.round(TRAIN_TIMEOUT_MS / 60000)} minutes.
                </p>
              </div>
            )}
            {train.phase === 'success' && (
              <p className="text-sm text-emerald-300 flex items-center gap-2" data-testid="status-completed">
                <CheckCircle2 className="w-4 h-4" aria-hidden="true" /> <StatusBadge status="completed" />
                <span>Server confirmed in {train.seconds}s.</span>
              </p>
            )}
            {train.phase === 'error' && (
              <div data-testid="status-failed" role="alert" className="space-y-2">
                <p className="text-sm text-rose-300 flex items-start gap-2">
                  <XCircle className="w-4 h-4 flex-shrink-0 mt-0.5" aria-hidden="true" />
                  <span>
                    <StatusBadge status="failed" /> <span className="ml-1">{train.message}</span>
                  </span>
                </p>
                {(train.kind === 'timeout' || train.kind === 'network') && (
                  <p className="text-xs text-amber-300 flex items-start gap-2">
                    <AlertTriangle className="w-4 h-4 flex-shrink-0" aria-hidden="true" />
                    <span>
                      The result is unknown. The server may still be training, or may have finished. Use “Refresh models” below; a finished run appears there with its real status.
                    </span>
                  </p>
                )}
                {train.kind === 'http' && <p className="text-xs text-slate-400">No model is reported as completed for this run.</p>}
              </div>
            )}
          </section>

          {result && (
            <section className={card} aria-labelledby="result-heading" data-testid="training-result">
              <h3 id="result-heading" className="text-indigo-400 font-semibold text-sm border-b border-slate-800 pb-3 mb-4">
                Training result
              </h3>
              <dl className="grid grid-cols-[auto,1fr] gap-x-4 gap-y-1.5 text-xs">
                <dt className="text-slate-500">Model ID</dt>
                <dd className="text-slate-200 font-mono break-all" data-testid="result-model-id">{result.model_id}</dd>
                <dt className="text-slate-500">Name</dt>
                <dd className="text-slate-200 break-words">{result.name}</dd>
                <dt className="text-slate-500">Dataset ID</dt>
                <dd className="text-slate-200 font-mono break-all" data-testid="result-dataset-id">{result.dataset_id}</dd>
                <dt className="text-slate-500">Model type</dt>
                <dd className="text-slate-200">{result.model_type ?? 'unavailable'}</dd>
                <dt className="text-slate-500">Status</dt>
                <dd><StatusBadge status={result.status} /></dd>
                <dt className="text-slate-500">Epochs run</dt>
                <dd className="text-slate-200">{result.current_epoch} of {result.total_epochs}</dd>
                <dt className="text-slate-500">Started (server)</dt>
                <dd className="text-slate-200">{result.created_at ? `${result.created_at} UTC` : 'unavailable'}</dd>
                <dt className="text-slate-500">Completed (server)</dt>
                <dd className="text-slate-200">{result.completed_at ? `${result.completed_at} UTC` : 'unavailable'}</dd>
                <dt className="text-slate-500">Duration</dt>
                <dd className="text-slate-200">
                  {serverSeconds !== null ? `≈ ${serverSeconds}s (server timestamps)` : 'unavailable'} · {train.phase === 'success' ? `${train.seconds}s request time in this browser` : ''}
                </dd>
              </dl>

              <h4 className="text-slate-300 text-xs font-semibold mt-4 mb-1">Configuration used</h4>
              <dl className="grid grid-cols-[auto,1fr] gap-x-4 gap-y-1 text-xs">
                {paramRows.map(([k, v]) => (
                  <React.Fragment key={k}>
                    <dt className="text-slate-500">{k}</dt>
                    <dd className="text-slate-200">{v}</dd>
                  </React.Fragment>
                ))}
              </dl>

              <h4 className="text-slate-300 text-xs font-semibold mt-4 mb-1">Final training loss (last epoch)</h4>
              <dl className="grid grid-cols-[auto,1fr] gap-x-4 gap-y-1 text-xs">
                <dt className="text-slate-500">Total</dt>
                <dd className="text-slate-200 font-mono">{formatLoss(result.final_loss)}</dd>
                <dt className="text-slate-500">Reconstruction</dt>
                <dd className="text-slate-200 font-mono">{formatLoss(result.recon_loss)}</dd>
                <dt className="text-slate-500">KL divergence</dt>
                <dd className="text-slate-200 font-mono">{formatLoss(result.kl_loss)}</dd>
              </dl>
              <p className="text-[11px] text-slate-500 mt-2">
                These are training losses, not a measure of synthetic-data quality. Quality is evaluated after generation. Training runs are identified by the model ID; experiment IDs are created when you generate data.
              </p>

              {chartData.length > 0 && (
                <div className="mt-4 h-52" aria-label="Training loss per epoch">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="epoch" stroke="#64748b" fontSize={10} />
                      <YAxis stroke="#64748b" fontSize={10} />
                      <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', fontSize: 11 }} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line type="monotone" dataKey="totalLoss" name="Total" stroke="#818cf8" dot={false} />
                      <Line type="monotone" dataKey="reconLoss" name="Reconstruction" stroke="#34d399" dot={false} />
                      <Line type="monotone" dataKey="klLoss" name="KL" stroke="#f472b6" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              <button
                type="button"
                onClick={() => {
                  onSelectModel(result.model_id);
                  onNavigateToGenerator(result.model_id);
                }}
                className="mt-4 w-full inline-flex items-center justify-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold px-4 py-2.5 rounded-lg focus-visible:ring-2 focus-visible:ring-emerald-300 outline-none"
              >
                Use this model in Data Generator <ArrowRight className="w-3.5 h-3.5" aria-hidden="true" />
              </button>
            </section>
          )}
        </div>
      </div>

      {/* G. Models stored on the backend */}
      <section className={card} aria-labelledby="models-heading">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3 mb-4">
          <h3 id="models-heading" className="text-indigo-400 font-semibold text-sm">Models on the backend</h3>
          <button
            type="button"
            onClick={refreshModels}
            className="inline-flex items-center gap-1.5 text-xs text-slate-300 border border-slate-700 hover:border-indigo-500 rounded-lg px-3 py-1.5 focus-visible:ring-2 focus-visible:ring-indigo-400 outline-none"
          >
            <RefreshCw className="w-3.5 h-3.5" aria-hidden="true" /> Refresh models
          </button>
        </div>
        {refreshNote && <p role="alert" className="text-xs text-rose-300 mb-3">{refreshNote}</p>}
        {models.length === 0 ? (
          <p className="text-sm text-slate-400">No models yet. Train one above.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <caption className="sr-only">Trained models stored by the backend</caption>
              <thead className="text-slate-500">
                <tr>
                  <th scope="col" className="py-2 pr-4 font-medium">Model</th>
                  <th scope="col" className="py-2 pr-4 font-medium">Dataset</th>
                  <th scope="col" className="py-2 pr-4 font-medium">Status</th>
                  <th scope="col" className="py-2 pr-4 font-medium">Epochs</th>
                  <th scope="col" className="py-2 pr-4 font-medium">Created (UTC)</th>
                  <th scope="col" className="py-2 font-medium"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.id} className="border-t border-slate-800 text-slate-300" data-testid={`model-row-${m.id}`}>
                    <td className="py-2 pr-4">
                      <div className="font-medium">{m.name}</div>
                      <div className="font-mono text-[11px] text-slate-500">{m.id}</div>
                    </td>
                    <td className="py-2 pr-4 font-mono">{m.datasetId}</td>
                    <td className="py-2 pr-4"><StatusBadge status={m.status === 'idle' ? 'initialized' : m.status} /></td>
                    <td className="py-2 pr-4">{m.lossHistory.length}{m.params.epochs ? ` / ${m.params.epochs}` : ''}</td>
                    <td className="py-2 pr-4">{m.createdAt || 'unavailable'}</td>
                    <td className="py-2 text-right">
                      {m.status === 'completed' && (
                        <button
                          type="button"
                          onClick={() => onSelectModel(m.id)}
                          aria-pressed={activeModel?.id === m.id}
                          className={`px-2.5 py-1 rounded-md border focus-visible:ring-2 focus-visible:ring-indigo-400 outline-none ${
                            activeModel?.id === m.id ? 'border-indigo-500 text-indigo-300' : 'border-slate-700 text-slate-300 hover:border-indigo-500'
                          }`}
                        >
                          {activeModel?.id === m.id ? 'Selected' : 'Select'}
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
};
