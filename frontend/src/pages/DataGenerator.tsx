import React, { useState } from 'react';
import { ArrowRight, BarChart3, BrainCircuit, CheckCircle2, Download, History, Loader2, Sparkles, XCircle } from 'lucide-react';
import { inflight, useInflight } from '../services/inflight';
import { GENERATE_TIMEOUT_MS, generateSynthetic, type ApiErrorKind } from '../services/api';
import { useCsvDownload } from '../hooks/useCsvDownload';
import { fmtPct } from '../services/format';
import type { ModelRecord } from '../types';
import type { ApiDataset, ApiGenerateResponse, GenerateRequest } from '../types/api';

interface DataGeneratorProps {
  /** Models from the backend; only completed ones can generate. */
  models: ModelRecord[];
  datasets: ApiDataset[];
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
  /** Called after the server confirmed the generation (the experiment is stored and exportable). */
  onGenerated: (result: ApiGenerateResponse) => void;
  onNavigateToModelLab: () => void;
  onNavigateToQuality: (experimentId: string) => void;
  onNavigateToHistory: () => void;
}

// Limits of GenerateRequest (backend/app/schemas/api_models.py). The maximum number of rows is
// configurable on the server (DATAFORGE_MAX_GENERATE_ROWS, default 50000); the server's message is shown if it is lower.
export const GEN_LIMITS = {
  records: { min: 10, max: 50000 },
  temperature: { min: 0.1, max: 2.5 },
  seed: { min: 0, max: 2 ** 32 - 1 },
} as const;

type Fields = { numRecords: string; temperature: string; randomSeed: string };
type Errors = Partial<Record<keyof Fields, string>>;
type RunState =
  | { phase: 'idle' }
  | { phase: 'running' }
  | { phase: 'success'; result: ApiGenerateResponse }
  | { phase: 'error'; message: string; kind: ApiErrorKind };

export function validateGeneration(values: Fields, modelId: string, datasetId: string): { errors: Errors; request?: GenerateRequest } {
  const errors: Errors = {};
  const n = values.numRecords.trim() === '' ? NaN : Number(values.numRecords);
  if (!Number.isInteger(n)) errors.numRecords = 'Number of records must be a whole number.';
  else if (n < GEN_LIMITS.records.min || n > GEN_LIMITS.records.max)
    errors.numRecords = `Number of records must be between ${GEN_LIMITS.records.min} and ${GEN_LIMITS.records.max.toLocaleString('en-US')}.`;

  const t = values.temperature.trim() === '' ? NaN : Number(values.temperature);
  if (!Number.isFinite(t)) errors.temperature = 'Temperature must be a number.';
  else if (t < GEN_LIMITS.temperature.min || t > GEN_LIMITS.temperature.max)
    errors.temperature = `Temperature must be between ${GEN_LIMITS.temperature.min} and ${GEN_LIMITS.temperature.max}.`;

  let seed: number | undefined;
  if (values.randomSeed.trim() !== '') {
    const s = Number(values.randomSeed);
    if (!Number.isInteger(s) || s < GEN_LIMITS.seed.min || s > GEN_LIMITS.seed.max) errors.randomSeed = `Random seed must be a whole number between 0 and ${GEN_LIMITS.seed.max.toLocaleString('en-US')}.`;
    else seed = s;
  }
  if (!modelId) return { errors };
  if (Object.keys(errors).length) return { errors };
  return { errors, request: { model_id: modelId, dataset_id: datasetId, num_records: n, temperature: t, ...(seed !== undefined ? { random_seed: seed } : {}) } };
}

const card = 'bg-slate-900/80 border border-slate-800 rounded-2xl p-6';
const input = 'w-full bg-slate-950 border rounded-lg px-3 py-2 text-sm text-slate-200 outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:opacity-60';
const btn = 'inline-flex items-center justify-center gap-1.5 text-xs font-semibold rounded-lg px-3 py-2 outline-none focus-visible:ring-2 focus-visible:ring-indigo-300 disabled:opacity-50 disabled:cursor-not-allowed';

export const DataGenerator: React.FC<DataGeneratorProps> = ({
  models,
  datasets,
  selectedModelId,
  onSelectModel,
  onGenerated,
  onNavigateToModelLab,
  onNavigateToQuality,
  onNavigateToHistory,
}) => {
  const usableModels = models.filter((m) => m.status === 'completed');
  const model = usableModels.find((m) => m.id === selectedModelId) ?? usableModels[0] ?? null;
  const dataset = model ? datasets.find((d) => d.id === model.datasetId) ?? null : null;

  const [values, setValues] = useState<Fields>({ numRecords: '1000', temperature: '1', randomSeed: '' });
  const [errors, setErrors] = useState<Errors>({});
  const [run, setRun] = useState<RunState>({ phase: 'idle' });
  const generatingPending = useInflight('generate'); // survives leaving and re-entering the page
  const csv = useCsvDownload();

  const setField = (name: keyof Fields, value: string) => {
    setValues((v) => ({ ...v, [name]: value }));
    setErrors((e) => (e[name] ? { ...e, [name]: undefined } : e));
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!model || inflight.has('generate')) return;
    const { errors: found, request } = validateGeneration(values, model.id, model.datasetId);
    setErrors(found);
    if (!request) return;
    if (!inflight.begin('generate')) return;
    setRun({ phase: 'running' });
    try {
      const res = await generateSynthetic(request);
      if (res.success && res.data && typeof res.data.experiment_id === 'string') {
        setRun({ phase: 'success', result: res.data });
        onGenerated(res.data);
      } else if (res.success) {
        setRun({ phase: 'error', kind: 'format', message: 'The server returned a response without an experiment ID.' });
      } else {
        setRun({ phase: 'error', kind: res.kind ?? 'http', message: res.error ?? 'Generation failed.' });
      }
    } finally {
      inflight.end('generate');
    }
  };

  const running = run.phase === 'running' || generatingPending;
  const result = run.phase === 'success' ? run.result : null;
  const columns = result && result.preview_records.length > 0 ? Object.keys(result.preview_records[0]) : [];
  const downloadingResult = result ? csv.downloading.includes(result.experiment_id) : false;

  const field = (name: keyof Fields, label: string, help: string, placeholder?: string) => (
    <div>
      <label htmlFor={`gen-${name}`} className="block text-slate-300 text-xs font-medium mb-1">{label}</label>
      <input
        id={`gen-${name}`}
        type="text"
        inputMode="decimal"
        value={values[name]}
        placeholder={placeholder}
        disabled={running}
        aria-invalid={errors[name] ? true : undefined}
        aria-describedby={`gen-${name}-help${errors[name] ? ` gen-${name}-error` : ''}`}
        onChange={(e) => setField(name, e.target.value)}
        className={`${input} ${errors[name] ? 'border-rose-500' : 'border-slate-800'}`}
      />
      <p id={`gen-${name}-help`} className="text-[11px] text-slate-500 mt-1">{help}</p>
      {errors[name] && <p id={`gen-${name}-error`} role="alert" className="text-[11px] text-rose-400 mt-1">{errors[name]}</p>}
    </div>
  );

  return (
    <div className="space-y-6">
      <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-2xl">
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-indigo-400" aria-hidden="true" /> Data Generator
        </h2>
        <p className="text-xs text-slate-400 mt-1 max-w-2xl">
          Sample new rows from a trained model. Each generation is saved on the server as an experiment with its own CSV file and saved evaluation.
        </p>
      </div>

      {usableModels.length === 0 ? (
        <div className={card} data-testid="no-models">
          <p className="text-sm text-slate-200 font-medium">No trained models yet.</p>
          <p className="text-xs text-slate-400 mt-1">Train a model first; completed models appear here.</p>
          <button type="button" onClick={onNavigateToModelLab} className={`${btn} mt-3 bg-indigo-600 hover:bg-indigo-500 text-white`}>
            <BrainCircuit className="w-3.5 h-3.5" aria-hidden="true" /> Go to Model Lab
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <form onSubmit={submit} noValidate aria-label="Generation setup" className={`${card} lg:col-span-1 space-y-4`}>
            <h3 className="text-indigo-400 font-semibold text-sm border-b border-slate-800 pb-3">Generation settings</h3>
            <div>
              <label htmlFor="gen-model" className="block text-slate-300 text-xs font-medium mb-1">Trained model</label>
              <select id="gen-model" value={model?.id ?? ''} disabled={running} onChange={(e) => onSelectModel(e.target.value)} className={`${input} border-slate-800`}>
                {usableModels.map((m) => (
                  <option key={m.id} value={m.id}>{m.name} ({m.id})</option>
                ))}
              </select>
              <p className="text-[11px] text-slate-500 mt-1" data-testid="model-dataset">
                Trained on {dataset ? `${dataset.filename} (${dataset.id})` : `dataset ${model?.datasetId} (record not found)`}. Rows are generated for this dataset's columns.
              </p>
            </div>
            {field('numRecords', 'Number of records', `${GEN_LIMITS.records.min} to ${GEN_LIMITS.records.max.toLocaleString('en-US')} (the server may set a lower limit).`)}
            {field('temperature', 'Temperature', `${GEN_LIMITS.temperature.min} to ${GEN_LIMITS.temperature.max}. Scales the random latent sample: lower stays near typical rows, higher is more varied.`)}
            {field('randomSeed', 'Random seed (optional)', 'Whole number for reproducible output. Leave empty to use the server default (42).', '42')}
            <button
              type="submit"
              disabled={running}
              className={`${btn} w-full py-3 text-sm bg-indigo-600 hover:bg-indigo-500 text-white`}
            >
              {running ? <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" /> : <Sparkles className="w-4 h-4" aria-hidden="true" />}
              {running ? 'Generating…' : 'Generate synthetic data'}
            </button>
          </form>

          <section className={`${card} lg:col-span-2 min-w-0`} aria-labelledby="gen-result-heading" aria-live="polite">
            <h3 id="gen-result-heading" className="text-indigo-400 font-semibold text-sm border-b border-slate-800 pb-3 mb-4">Result</h3>
            {run.phase === 'idle' && <p className="text-sm text-slate-400" data-testid="gen-idle">Nothing generated yet in this session. Choose settings and generate.</p>}
            {running && (
              <p className="text-sm text-slate-200 flex items-center gap-2" data-testid="gen-running">
                <Loader2 className="w-4 h-4 animate-spin text-amber-400" aria-hidden="true" /> Waiting for the server… (gives up after {Math.round(GENERATE_TIMEOUT_MS / 60000)} minutes)
              </p>
            )}
            {run.phase === 'error' && (
              <div role="alert" data-testid="gen-error" className="text-sm text-rose-300 bg-rose-500/10 border border-rose-500/30 rounded-lg p-3 flex gap-2">
                <XCircle className="w-4 h-4 flex-shrink-0 mt-0.5" aria-hidden="true" />
                <span>
                  Generation failed: {run.message}
                  {(run.kind === 'network' || run.kind === 'timeout') && ' The outcome is unknown; check Export History to see whether the experiment was saved.'}
                </span>
              </div>
            )}

            {result && (
              <div className="space-y-4" data-testid="gen-result">
                <p className="text-sm text-emerald-300 flex items-center gap-2"><CheckCircle2 className="w-4 h-4" aria-hidden="true" /> Generated and saved by the server.</p>
                <dl className="grid grid-cols-[auto,1fr] gap-x-4 gap-y-1.5 text-xs">
                  <dt className="text-slate-500">Experiment ID</dt><dd className="font-mono text-slate-200" data-testid="gen-experiment-id">{result.experiment_id}</dd>
                  <dt className="text-slate-500">Model ID</dt><dd className="font-mono text-slate-200">{result.model_id}</dd>
                  <dt className="text-slate-500">Dataset ID</dt><dd className="font-mono text-slate-200">{result.dataset_id}</dd>
                  <dt className="text-slate-500">Records</dt><dd className="text-slate-200">{result.num_records.toLocaleString('en-US')}</dd>
                  <dt className="text-slate-500">Temperature</dt><dd className="text-slate-200">{result.temperature}</dd>
                  <dt className="text-slate-500">Output file</dt><dd className="text-slate-200">{result.output_filename}</dd>
                  <dt className="text-slate-500">Fidelity score (heuristic)</dt><dd className="text-slate-200" data-testid="gen-fidelity">{fmtPct(result.fidelity_score)}</dd>
                  <dt className="text-slate-500">Validity</dt><dd className="text-slate-200" data-testid="gen-validity">{fmtPct(result.domain_validity_percent)}</dd>
                </dl>
                <p className="text-[11px] text-slate-500">The fidelity score is a heuristic summary of statistical similarity, not a guarantee of data quality or privacy. Open the quality comparison for the full saved evaluation.</p>

                <div className="flex flex-wrap gap-2">
                  <button type="button" disabled={downloadingResult} onClick={() => csv.download(result.experiment_id)} className={`${btn} bg-indigo-600 hover:bg-indigo-500 text-white`}>
                    {downloadingResult ? <Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden="true" /> : <Download className="w-3.5 h-3.5" aria-hidden="true" />}
                    {downloadingResult ? 'Downloading…' : 'Download CSV'}
                  </button>
                  <button type="button" onClick={() => onNavigateToQuality(result.experiment_id)} className={`${btn} border border-slate-700 hover:border-indigo-500 text-slate-200`}>
                    <BarChart3 className="w-3.5 h-3.5" aria-hidden="true" /> Compare quality <ArrowRight className="w-3.5 h-3.5" aria-hidden="true" />
                  </button>
                  <button type="button" onClick={onNavigateToHistory} className={`${btn} border border-slate-700 hover:border-indigo-500 text-slate-200`}>
                    <History className="w-3.5 h-3.5" aria-hidden="true" /> Export History
                  </button>
                </div>
                {csv.notice && (
                  <p role={csv.notice.kind === 'error' ? 'alert' : 'status'} data-testid="download-notice" className={`text-xs ${csv.notice.kind === 'error' ? 'text-rose-300' : 'text-emerald-300'}`}>{csv.notice.text}</p>
                )}

                {columns.length > 0 && (
                  <div className="overflow-x-auto" data-testid="gen-preview">
                    <p className="text-[11px] text-slate-500 mb-1">First {result.preview_records.length} of {result.num_records.toLocaleString('en-US')} generated rows.</p>
                    <table className="w-full text-xs text-left">
                      <caption className="sr-only">Preview of generated records</caption>
                      <thead className="text-slate-500"><tr>{columns.map((c) => <th key={c} scope="col" className="py-1.5 pr-3 font-medium whitespace-nowrap">{c}</th>)}</tr></thead>
                      <tbody>
                        {result.preview_records.map((r, i) => (
                          <tr key={i} className="border-t border-slate-800 text-slate-300">{columns.map((c) => <td key={c} className="py-1 pr-3 whitespace-nowrap">{String(r[c] ?? '')}</td>)}</tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  );
};
