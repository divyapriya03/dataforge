import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Download,
  History,
  Loader2,
  RefreshCw,
  Search,
  Sparkles,
  XCircle,
} from 'lucide-react';
import { evaluateRun, getExperiments, getModelStatus } from '../services/api';
import { EvaluationPanel } from '../components/EvaluationPanel';
import { useCsvDownload } from '../hooks/useCsvDownload';
import { UNAVAILABLE, fmtPct, fmtText, isNum, statusLabel } from '../services/format';
import type { ApiEvaluation, ApiExperiment, ApiModel } from '../types/api';

interface ExportHistoryProps {
  /** Lets App keep Dashboard / Quality Comparison in sync with what the backend has stored. */
  onExperimentsLoaded?: (experiments: ApiExperiment[]) => void;
  onNavigateToModelLab: () => void;
  /** With a model ID the Data Generator opens with that model selected. */
  onNavigateToGenerator: (modelId?: string) => void;
  onOpenQuality: (experiment: ApiExperiment) => void;
}

type ListState = { phase: 'loading' } | { phase: 'error'; message: string } | { phase: 'ready' };
type DetailState<T> = { phase: 'loading' } | { phase: 'ok'; data: T } | { phase: 'unavailable'; message: string } | { phase: 'error'; message: string };
type Notice = { kind: 'success' | 'error'; text: string };
type SortKey = 'newest' | 'oldest' | 'fidelity' | 'records';

const card = 'bg-slate-900/80 border border-slate-800 rounded-2xl p-5';
const control =
  'bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 outline-none focus-visible:ring-2 focus-visible:ring-indigo-500';
const btn =
  'inline-flex items-center justify-center gap-1.5 text-xs font-semibold rounded-lg px-3 py-1.5 outline-none focus-visible:ring-2 focus-visible:ring-indigo-300 disabled:opacity-50 disabled:cursor-not-allowed';

const StatusPill: React.FC<{ status: string }> = ({ status }) => {
  const style =
    status === 'completed'
      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
      : status === 'output_missing'
      ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
      : 'bg-slate-700/30 text-slate-300 border-slate-600/40';
  return (
    <span className={`inline-block px-2 py-0.5 rounded-full text-[11px] font-semibold border whitespace-nowrap ${style}`}>
      {statusLabel(status)}
    </span>
  );
};

const Row: React.FC<{ label: string; children: React.ReactNode; testId?: string }> = ({ label, children, testId }) => (
  <>
    <dt className="text-slate-500">{label}</dt>
    <dd className="text-slate-200 break-words" data-testid={testId}>{children}</dd>
  </>
);

export const ExportHistory: React.FC<ExportHistoryProps> = ({
  onExperimentsLoaded,
  onNavigateToModelLab,
  onNavigateToGenerator,
  onOpenQuality,
}) => {
  // ---- history list (always from the backend)
  const [experiments, setExperiments] = useState<ApiExperiment[]>([]);
  const [list, setList] = useState<ListState>({ phase: 'loading' });
  const [reloadTick, setReloadTick] = useState(0);

  const load = useCallback(async () => {
    setList({ phase: 'loading' });
    const res = await getExperiments();
    if (res.success && Array.isArray(res.data)) {
      setExperiments(res.data);
      setList({ phase: 'ready' });
      onExperimentsLoaded?.(res.data);
    } else {
      setExperiments([]);
      setList({ phase: 'error', message: res.error ?? 'The server returned an unexpected experiment list.' });
    }
    setReloadTick((t) => t + 1);
  }, [onExperimentsLoaded]);

  useEffect(() => {
    load();
  }, []);

  // ---- filters
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [sortKey, setSortKey] = useState<SortKey>('newest');

  const statuses = useMemo(() => Array.from(new Set(experiments.map((e) => e.status))), [experiments]);

  const visible = useMemo(() => {
    const q = search.trim().toLowerCase();
    let rows = experiments.filter((e) => {
      if (statusFilter !== 'all' && e.status !== statusFilter) return false;
      if (!q) return true;
      return [e.id, e.dataset_id, e.dataset_name, e.model_id, e.model_name, e.output_filename]
        .some((v) => typeof v === 'string' && v.toLowerCase().includes(q));
    });
    // the backend already returns newest first
    if (sortKey === 'oldest') rows = [...rows].reverse();
    else if (sortKey === 'fidelity') rows = [...rows].sort((a, b) => (b.fidelity_score ?? -Infinity) - (a.fidelity_score ?? -Infinity));
    else if (sortKey === 'records') rows = [...rows].sort((a, b) => b.num_records - a.num_records);
    return rows;
  }, [experiments, search, statusFilter, sortKey]);

  const filtersActive = search.trim() !== '' || statusFilter !== 'all';
  const clearFilters = () => {
    setSearch('');
    setStatusFilter('all');
  };

  // ---- selection and details
  const [selectedId, setSelectedId] = useState<string>('');
  const selected = experiments.find((e) => e.id === selectedId) ?? null;
  useEffect(() => {
    if (selectedId && !experiments.some((e) => e.id === selectedId)) setSelectedId('');
  }, [experiments, selectedId]);

  const [evaluation, setEvaluation] = useState<DetailState<ApiEvaluation>>({ phase: 'loading' });
  const [model, setModel] = useState<DetailState<ApiModel>>({ phase: 'loading' });
  const [detailTick, setDetailTick] = useState(0);

  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    setEvaluation({ phase: 'loading' });
    setModel({ phase: 'loading' });
    evaluateRun(selected.id).then((res) => {
      if (cancelled) return;
      if (res.success && res.data) setEvaluation({ phase: 'ok', data: res.data });
      else if (res.status === 404) setEvaluation({ phase: 'unavailable', message: 'This experiment has no saved evaluation result.' });
      else setEvaluation({ phase: 'error', message: res.error ?? 'Could not load the evaluation.' });
    });
    getModelStatus(selected.model_id).then((res) => {
      if (cancelled) return;
      if (res.success && res.data) setModel({ phase: 'ok', data: res.data });
      else if (res.status === 404) setModel({ phase: 'unavailable', message: 'The model record for this experiment is no longer available.' });
      else setModel({ phase: 'error', message: res.error ?? 'Could not load the model.' });
    });
    return () => {
      cancelled = true;
    };
    // reload when the selection changes or the list was refreshed
  }, [selected?.id, selected?.model_id, detailTick, reloadTick]);

  // ---- CSV download (one request per experiment at a time)
  const { download, downloading, notice } = useCsvDownload(() => load()); // a 404 means the file/record is gone: re-read the list

  const downloadButton = (exp: ApiExperiment, label: string, ariaLabel: string) => {
    const busy = downloading.includes(exp.id);
    return (
      <button
        type="button"
        onClick={() => exp.output_available && download(exp.id)}
        disabled={!exp.output_available || busy}
        aria-label={ariaLabel}
        title={exp.output_available ? undefined : 'The generated CSV is no longer on the server.'}
        className={`${btn} bg-indigo-600 hover:bg-indigo-500 text-white`}
      >
        {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden="true" /> : <Download className="w-3.5 h-3.5" aria-hidden="true" />}
        {busy ? 'Downloading…' : label}
      </button>
    );
  };

  return (
    <div className="space-y-6">
      {/* A. Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-900/80 border border-slate-800 p-5 rounded-2xl">
        <div className="min-w-0">
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            <History className="w-5 h-5 text-indigo-400" aria-hidden="true" />
            Export History
          </h2>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl">
            Every successful generation is saved as an experiment together with its evaluation and the generated CSV. Inspect past runs and download their datasets here.
          </p>
        </div>
        <button
          type="button"
          onClick={load}
          disabled={list.phase === 'loading'}
          className={`${btn} border border-slate-700 hover:border-indigo-500 text-slate-300`}
        >
          <RefreshCw className={`w-3.5 h-3.5 ${list.phase === 'loading' ? 'animate-spin' : ''}`} aria-hidden="true" /> Refresh
        </button>
      </div>

      {notice && (
        <div
          role={notice.kind === 'error' ? 'alert' : 'status'}
          data-testid="download-notice"
          className={`text-xs rounded-lg border px-4 py-3 flex items-start gap-2 ${
            notice.kind === 'error' ? 'text-rose-300 bg-rose-500/10 border-rose-500/30' : 'text-emerald-300 bg-emerald-500/10 border-emerald-500/30'
          }`}
        >
          {notice.kind === 'error' ? <XCircle className="w-4 h-4 flex-shrink-0" aria-hidden="true" /> : <CheckCircle2 className="w-4 h-4 flex-shrink-0" aria-hidden="true" />}
          <span>{notice.text}</span>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
        {/* B. List */}
        <section className={`${card} xl:col-span-3 min-w-0`} aria-labelledby="history-heading">
          <h3 id="history-heading" className="text-indigo-400 font-semibold text-sm border-b border-slate-800 pb-3 mb-4">Experiments</h3>

          {list.phase === 'loading' && (
            <p className="text-sm text-slate-400 flex items-center gap-2" role="status">
              <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" /> Loading experiments…
            </p>
          )}

          {list.phase === 'error' && (
            <div role="alert" className="text-sm text-rose-300 bg-rose-500/10 border border-rose-500/30 rounded-lg p-3 flex items-start justify-between gap-3">
              <span>Could not load experiments: {list.message}</span>
              <button type="button" onClick={load} className="text-xs underline whitespace-nowrap rounded outline-none focus-visible:ring-2 focus-visible:ring-indigo-400">Retry</button>
            </div>
          )}

          {list.phase === 'ready' && experiments.length === 0 && (
            <div className="text-sm text-slate-300 bg-slate-950/60 border border-slate-800 rounded-lg p-4" data-testid="empty-history">
              <p className="font-medium">No experiments yet.</p>
              <p className="text-xs text-slate-400 mt-1">An experiment is saved each time you generate synthetic data from a trained model.</p>
              <div className="flex flex-wrap gap-2 mt-3">
                <button type="button" onClick={onNavigateToModelLab} className={`${btn} border border-slate-700 hover:border-indigo-500 text-slate-300`}>
                  <BrainCircuit className="w-3.5 h-3.5" aria-hidden="true" /> Train a model
                </button>
                <button type="button" onClick={() => onNavigateToGenerator()} className={`${btn} border border-slate-700 hover:border-indigo-500 text-slate-300`}>
                  <Sparkles className="w-3.5 h-3.5" aria-hidden="true" /> Go to Data Generator
                </button>
              </div>
            </div>
          )}

          {list.phase === 'ready' && experiments.length > 0 && (
            <>
              {/* filters */}
              <div className="flex flex-wrap items-end gap-3 mb-4">
                <div className="flex-1 min-w-[12rem]">
                  <label htmlFor="history-search" className="block text-[11px] text-slate-400 mb-1">Search experiments</label>
                  <div className="relative">
                    <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-1/2 -translate-y-1/2" aria-hidden="true" />
                    <input
                      id="history-search"
                      type="search"
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      placeholder="Experiment, dataset or model"
                      className={`${control} w-full pl-8`}
                    />
                  </div>
                </div>
                <div>
                  <label htmlFor="history-status" className="block text-[11px] text-slate-400 mb-1">Status</label>
                  <select id="history-status" value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)} className={control}>
                    <option value="all">All statuses</option>
                    {statuses.map((s) => (
                      <option key={s} value={s}>{statusLabel(s)}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label htmlFor="history-sort" className="block text-[11px] text-slate-400 mb-1">Sort by</label>
                  <select id="history-sort" value={sortKey} onChange={(e) => setSortKey(e.target.value as SortKey)} className={control}>
                    <option value="newest">Newest first</option>
                    <option value="oldest">Oldest first</option>
                    <option value="fidelity">Highest fidelity</option>
                    <option value="records">Most records</option>
                  </select>
                </div>
              </div>
              <p className="text-[11px] text-slate-500 mb-2" data-testid="result-count" aria-live="polite">
                Showing {visible.length} of {experiments.length} experiment{experiments.length === 1 ? '' : 's'}
              </p>

              {visible.length === 0 ? (
                <div className="text-sm text-slate-300 bg-slate-950/60 border border-slate-800 rounded-lg p-4" data-testid="no-matches">
                  No experiments match the current filters.{' '}
                  {filtersActive && (
                    <button type="button" onClick={clearFilters} className="underline text-indigo-300 rounded outline-none focus-visible:ring-2 focus-visible:ring-indigo-400">Clear filters</button>
                  )}
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <caption className="sr-only">Saved experiments</caption>
                    <thead className="text-slate-500">
                      <tr>
                        <th scope="col" className="py-2 pr-3 font-medium">Experiment</th>
                        <th scope="col" className="py-2 pr-3 font-medium">Dataset</th>
                        <th scope="col" className="py-2 pr-3 font-medium">Model</th>
                        <th scope="col" className="py-2 pr-3 font-medium text-right">Records</th>
                        <th scope="col" className="py-2 pr-3 font-medium text-right">Fidelity</th>
                        <th scope="col" className="py-2 pr-3 font-medium text-right">Validity</th>
                        <th scope="col" className="py-2 pr-3 font-medium">Status</th>
                        <th scope="col" className="py-2 pr-3 font-medium">Created (UTC)</th>
                        <th scope="col" className="py-2 font-medium"><span className="sr-only">Actions</span></th>
                      </tr>
                    </thead>
                    <tbody>
                      {visible.map((e) => (
                        <tr
                          key={e.id}
                          data-testid={`experiment-row-${e.id}`}
                          className={`border-t border-slate-800 text-slate-300 ${e.id === selectedId ? 'bg-indigo-500/5' : ''}`}
                        >
                          <td className="py-2 pr-3 font-mono">{e.id}</td>
                          <td className="py-2 pr-3">
                            <div>{e.dataset_name ?? 'Dataset record removed'}</div>
                            <div className="font-mono text-[11px] text-slate-500">{e.dataset_id}</div>
                          </td>
                          <td className="py-2 pr-3">
                            <div>{e.model_name ?? 'Model record removed'}</div>
                            <div className="font-mono text-[11px] text-slate-500">{e.model_id}</div>
                          </td>
                          <td className="py-2 pr-3 text-right">{e.num_records.toLocaleString('en-US')}</td>
                          <td className="py-2 pr-3 text-right">{fmtPct(e.fidelity_score)}</td>
                          <td className="py-2 pr-3 text-right">{fmtPct(e.validity_score)}</td>
                          <td className="py-2 pr-3"><StatusPill status={e.status} /></td>
                          <td className="py-2 pr-3 whitespace-nowrap">{fmtText(e.created_at)}</td>
                          <td className="py-2">
                            <div className="flex gap-1.5 justify-end">
                              <button
                                type="button"
                                onClick={() => setSelectedId(e.id)}
                                aria-label={`View details of ${e.id}`}
                                aria-pressed={e.id === selectedId}
                                className={`${btn} border ${e.id === selectedId ? 'border-indigo-500 text-indigo-300' : 'border-slate-700 text-slate-300 hover:border-indigo-500'}`}
                              >
                                View
                              </button>
                              {downloadButton(e, 'Download CSV', `Download CSV for ${e.id}`)}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </section>

        {/* C + D. Details and evaluation */}
        <section className={`${card} xl:col-span-2 min-w-0`} aria-labelledby="details-heading" data-testid="details-panel">
          <h3 id="details-heading" className="text-indigo-400 font-semibold text-sm border-b border-slate-800 pb-3 mb-4">Experiment details</h3>
          {!selected ? (
            <p className="text-sm text-slate-400">Select “View” on an experiment to see its configuration and saved evaluation.</p>
          ) : (
            <div className="space-y-5">
              <dl className="grid grid-cols-[auto,1fr] gap-x-4 gap-y-1.5 text-xs">
                <Row label="Experiment ID" testId="detail-experiment-id"><span className="font-mono">{selected.id}</span></Row>
                <Row label="Status"><StatusPill status={selected.status} /></Row>
                <Row label="Dataset">
                  {selected.dataset_name ?? 'record removed'} <span className="font-mono text-slate-500">({selected.dataset_id})</span>
                </Row>
                <Row label="Model">
                  {selected.model_name ?? 'record removed'} <span className="font-mono text-slate-500">({selected.model_id})</span>
                </Row>
                <Row label="Model type">{fmtText(selected.model_type)}</Row>
                <Row label="Records generated">{selected.num_records.toLocaleString('en-US')}</Row>
                <Row label="Temperature">{selected.temperature}</Row>
                <Row label="Random seed">{isNum(selected.random_seed) ? selected.random_seed : 'not recorded'}</Row>
                <Row label="Created (UTC)">{fmtText(selected.created_at)}</Row>
                <Row label="Output file">
                  {selected.output_available ? (
                    <>{fmtText(selected.output_filename)} <span className="text-emerald-400">· available</span></>
                  ) : (
                    <span className="text-amber-300" data-testid="output-missing">no longer on the server</span>
                  )}
                </Row>
              </dl>

              <div className="flex flex-wrap gap-2">
                {downloadButton(selected, 'Download CSV', 'Download generated CSV')}
                <button type="button" onClick={() => onOpenQuality(selected)} className={`${btn} border border-slate-700 hover:border-indigo-500 text-slate-300`}>
                  <BarChart3 className="w-3.5 h-3.5" aria-hidden="true" /> Open in Quality Comparison
                </button>
                <button
                  type="button"
                  onClick={() => onNavigateToGenerator(model.phase === 'ok' ? selected.model_id : undefined)}
                  className={`${btn} border border-slate-700 hover:border-indigo-500 text-slate-300`}
                >
                  <Sparkles className="w-3.5 h-3.5" aria-hidden="true" /> Generate again
                </button>
                <button type="button" onClick={onNavigateToModelLab} className={`${btn} border border-slate-700 hover:border-indigo-500 text-slate-300`}>
                  <BrainCircuit className="w-3.5 h-3.5" aria-hidden="true" /> Model Lab
                </button>
              </div>

              {/* training configuration */}
              <div data-testid="model-config">
                <h4 className="text-slate-300 text-xs font-semibold mb-1">Training configuration</h4>
                {model.phase === 'loading' && <p className="text-xs text-slate-500">Loading…</p>}
                {(model.phase === 'unavailable' || model.phase === 'error') && (
                  <p role={model.phase === 'error' ? 'alert' : undefined} className="text-xs text-amber-300">{model.message}</p>
                )}
                {model.phase === 'ok' && <ModelConfig model={model.data} />}
              </div>

              {/* evaluation */}
              <div data-testid="evaluation">
                <h4 className="text-slate-300 text-xs font-semibold mb-1">Saved evaluation</h4>
                {evaluation.phase === 'loading' && <p className="text-xs text-slate-500">Loading…</p>}
                {evaluation.phase === 'unavailable' && <p className="text-xs text-amber-300" data-testid="evaluation-unavailable">{evaluation.message}</p>}
                {evaluation.phase === 'error' && (
                  <p role="alert" className="text-xs text-rose-300">
                    Could not load the evaluation: {evaluation.message}{' '}
                    <button type="button" onClick={() => setDetailTick((t) => t + 1)} className="underline rounded outline-none focus-visible:ring-2 focus-visible:ring-indigo-400">Retry</button>
                  </p>
                )}
                {evaluation.phase === 'ok' && <EvaluationPanel ev={evaluation.data} />}
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
};

const ModelConfig: React.FC<{ model: ApiModel }> = ({ model }) => {
  const p = (model.params ?? {}) as Record<string, unknown>;
  const show = (v: unknown) => (v === undefined ? UNAVAILABLE : v === null ? 'not set (random)' : Array.isArray(v) ? v.join(', ') : String(v));
  return (
    <dl className="grid grid-cols-[auto,1fr] gap-x-4 gap-y-1 text-xs">
      <Row label="Model status">{model.status}</Row>
      <Row label="Epochs">{show(p.epochs)}</Row>
      <Row label="Learning rate">{show(p.learning_rate)}</Row>
      <Row label="Batch size">{show(p.batch_size)}</Row>
      <Row label="Latent dimension">{show(p.latent_dim)}</Row>
      <Row label="Hidden layers">{show(p.hidden_layers)}</Row>
      <Row label="KL weight (β)">{show(p.beta_weight)}</Row>
      <Row label="Training seed">{show(p.random_seed)}</Row>
      <Row label="Trained (UTC)">{fmtText(model.completed_at)}</Row>
    </dl>
  );
};
