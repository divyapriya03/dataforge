import React, { useEffect, useRef, useState } from 'react';
import { ArrowRight, CheckCircle2, Database, Loader2, RefreshCw, Upload, XCircle } from 'lucide-react';
import { getDataset, previewPreprocessing, uploadDataset } from '../services/api';
import { UNAVAILABLE, fmtNum, fmtText } from '../services/format';
import { MIN_TRAIN_ROWS } from '../services/trainingConfig';
import type { ApiColumn, ApiDataset, ApiPreprocessResponse } from '../types/api';

type LoadState = { phase: 'loading' } | { phase: 'ready' } | { phase: 'error'; message: string };
type Tab = 'schema' | 'preview' | 'stats';
type UploadState =
  | { phase: 'idle' }
  | { phase: 'uploading'; filename: string }
  | { phase: 'error'; message: string }
  | { phase: 'done'; dataset: ApiDataset };
type CheckState =
  | { phase: 'idle' }
  | { phase: 'running' }
  | { phase: 'ok'; data: ApiPreprocessResponse }
  | { phase: 'error'; message: string };

interface DatasetStudioProps {
  datasets: ApiDataset[];
  datasetsState: LoadState;
  activeDataset: ApiDataset | null;
  /** Column configuration edited here for the active dataset (null = the dataset's stored schema). */
  columnConfig: ApiColumn[] | null;
  onReloadDatasets: () => Promise<unknown>;
  onSelectDataset: (id: string) => void;
  /** Called with the dataset returned by the server after a successful upload. */
  onUploaded: (dataset: ApiDataset) => void | Promise<void>;
  onColumnConfigChange: (datasetId: string, columns: ApiColumn[] | null) => void;
  onNavigateToModelLab: (datasetId: string) => void;
}

const card = 'bg-slate-900/80 border border-slate-800 rounded-2xl p-6';
const control =
  'bg-slate-950 border border-slate-800 rounded-md px-2 py-1 text-xs text-slate-200 outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:opacity-60';
const btn =
  'inline-flex items-center justify-center gap-1.5 text-xs font-semibold rounded-lg px-3 py-2 outline-none focus-visible:ring-2 focus-visible:ring-indigo-300 disabled:opacity-50 disabled:cursor-not-allowed';

export const DatasetStudio: React.FC<DatasetStudioProps> = ({
  datasets,
  datasetsState,
  activeDataset,
  columnConfig,
  onReloadDatasets,
  onSelectDataset,
  onUploaded,
  onColumnConfigChange,
  onNavigateToModelLab,
}) => {
  const [tab, setTab] = useState<Tab>('schema');

  // ---- upload (a real multipart upload; the server validates the file)
  const [upload, setUpload] = useState<UploadState>({ phase: 'idle' });
  const [dragActive, setDragActive] = useState(false);
  const uploading = useRef(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const processFile = async (file: File) => {
    if (uploading.current) return; // one upload at a time
    uploading.current = true;
    setUpload({ phase: 'uploading', filename: file.name });
    try {
      const res = await uploadDataset(file);
      if (res.success && res.data && typeof res.data.id === 'string') {
        setUpload({ phase: 'done', dataset: res.data });
        await onUploaded(res.data);
      } else if (res.success) {
        setUpload({ phase: 'error', message: 'The server returned a response without a dataset ID.' });
      } else {
        setUpload({ phase: 'error', message: res.error ?? 'Upload failed.' });
      }
    } finally {
      uploading.current = false;
      if (fileInput.current) fileInput.current.value = '';
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file) processFile(file);
  };

  // ---- selected dataset: columns (editable copy), 50-row preview from the server, preprocessing check
  const columns: ApiColumn[] = columnConfig ?? activeDataset?.columns ?? [];
  const customized = columnConfig !== null;

  const [preview, setPreview] = useState<{ phase: 'loading' } | { phase: 'ok'; rows: Record<string, unknown>[] } | { phase: 'error'; message: string }>({ phase: 'loading' });
  const [check, setCheck] = useState<CheckState>({ phase: 'idle' });
  const datasetId = activeDataset?.id;

  useEffect(() => {
    setCheck({ phase: 'idle' });
    if (!datasetId) return;
    let cancelled = false;
    setPreview({ phase: 'loading' });
    getDataset(datasetId).then((res) => {
      if (cancelled) return;
      setPreview(res.success && res.data ? { phase: 'ok', rows: res.data.preview ?? [] } : { phase: 'error', message: res.error ?? 'Could not load the preview.' });
    });
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  const editColumn = (name: string, patch: Partial<ApiColumn>) => {
    if (!activeDataset) return;
    const next = columns.map((c) => (c.name === name ? { ...c, ...patch } : c));
    setCheck({ phase: 'idle' });
    onColumnConfigChange(activeDataset.id, next);
  };

  const runCheck = async () => {
    if (!activeDataset || check.phase === 'running') return;
    setCheck({ phase: 'running' });
    const res = await previewPreprocessing(activeDataset.id, columns);
    setCheck(res.success && res.data ? { phase: 'ok', data: res.data } : { phase: 'error', message: res.error ?? 'Preprocessing check failed.' });
  };

  const usable = columns.filter((c) => c.included && c.selectedType !== 'drop').length;
  const trainBlock = !activeDataset
    ? ''
    : activeDataset.rowCount < MIN_TRAIN_ROWS
    ? `This dataset has ${activeDataset.rowCount} row(s); training needs at least ${MIN_TRAIN_ROWS}.`
    : usable === 0
    ? 'No column is selected for training.'
    : '';

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-900/80 border border-slate-800 p-5 rounded-2xl">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            <Database className="w-5 h-5 text-indigo-400" aria-hidden="true" /> Dataset Studio
          </h2>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl">
            Upload a CSV, review how the backend read it, adjust which columns are used and how missing values are handled, and check the preprocessing before training.
          </p>
        </div>
        <button type="button" onClick={() => onReloadDatasets()} className={`${btn} border border-slate-700 hover:border-indigo-500 text-slate-300`}>
          <RefreshCw className="w-3.5 h-3.5" aria-hidden="true" /> Refresh datasets
        </button>
      </div>

      {/* Upload */}
      <section className={card} aria-labelledby="upload-heading">
        <h3 id="upload-heading" className="text-indigo-400 font-semibold text-sm border-b border-slate-800 pb-3 mb-4">1. Upload a CSV</h3>
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-xl p-6 text-center transition ${dragActive ? 'border-indigo-500 bg-indigo-500/5' : 'border-slate-700'}`}
        >
          <Upload className="w-6 h-6 text-slate-500 mx-auto mb-2" aria-hidden="true" />
          <p className="text-xs text-slate-400 mb-3">Drop a .csv file here, or choose one. The server checks the file and rejects empty, unreadable or too-large files.</p>
          <label htmlFor="csv-file" className={`${btn} bg-indigo-600 hover:bg-indigo-500 text-white cursor-pointer ${upload.phase === 'uploading' ? 'opacity-60 pointer-events-none' : ''}`}>
            {upload.phase === 'uploading' ? <Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden="true" /> : <Upload className="w-3.5 h-3.5" aria-hidden="true" />}
            {upload.phase === 'uploading' ? `Uploading ${upload.filename}…` : 'Choose CSV file'}
          </label>
          <input id="csv-file" ref={fileInput} type="file" accept=".csv,text/csv" onChange={handleFileInput} disabled={upload.phase === 'uploading'} className="sr-only" />
        </div>

        {upload.phase === 'error' && (
          <p role="alert" data-testid="upload-error" className="mt-3 text-xs text-rose-300 bg-rose-500/10 border border-rose-500/30 rounded-lg p-3 flex gap-2">
            <XCircle className="w-4 h-4 flex-shrink-0" aria-hidden="true" /> Upload failed: {upload.message}
          </p>
        )}
        {upload.phase === 'done' && (
          <div role="status" data-testid="upload-success" className="mt-3 text-xs text-emerald-300 bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-3 flex flex-wrap items-center justify-between gap-2">
            <span className="flex gap-2">
              <CheckCircle2 className="w-4 h-4 flex-shrink-0" aria-hidden="true" />
              <span>
                Uploaded <strong>{upload.dataset.filename}</strong> as <span className="font-mono">{upload.dataset.id}</span> ({upload.dataset.rowCount.toLocaleString('en-US')} rows, {upload.dataset.columnCount} columns).
              </span>
            </span>
            <button type="button" onClick={() => onNavigateToModelLab(upload.dataset.id)} className={`${btn} bg-emerald-600 hover:bg-emerald-500 text-white`}>
              Continue to Model Lab <ArrowRight className="w-3.5 h-3.5" aria-hidden="true" />
            </button>
          </div>
        )}
      </section>

      {/* Datasets */}
      <section className={card} aria-labelledby="datasets-heading">
        <h3 id="datasets-heading" className="text-indigo-400 font-semibold text-sm border-b border-slate-800 pb-3 mb-4">2. Dataset</h3>

        {datasetsState.phase === 'loading' && (
          <p role="status" className="text-sm text-slate-400 flex items-center gap-2"><Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" /> Loading datasets…</p>
        )}
        {datasetsState.phase === 'error' && (
          <div role="alert" className="text-sm text-rose-300 bg-rose-500/10 border border-rose-500/30 rounded-lg p-3 flex justify-between gap-3">
            <span>Could not load datasets: {datasetsState.message}</span>
            <button type="button" onClick={() => onReloadDatasets()} className="text-xs underline whitespace-nowrap">Retry</button>
          </div>
        )}
        {datasetsState.phase === 'ready' && datasets.length === 0 && (
          <div data-testid="no-datasets" className="text-sm text-slate-300 bg-slate-950/60 border border-slate-800 rounded-lg p-4">
            <p className="font-medium">No datasets yet.</p>
            <p className="text-xs text-slate-400 mt-1">Upload a CSV above to get started.</p>
          </div>
        )}

        {datasetsState.phase === 'ready' && activeDataset && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-end gap-4">
              <div>
                <label htmlFor="studio-dataset" className="block text-[11px] text-slate-400 mb-1">Dataset</label>
                <select id="studio-dataset" value={activeDataset.id} onChange={(e) => onSelectDataset(e.target.value)} className={`${control} min-w-[16rem]`}>
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>{d.filename} ({d.id})</option>
                  ))}
                </select>
              </div>
              <dl className="grid grid-cols-[auto,1fr] gap-x-3 gap-y-0.5 text-xs" data-testid="dataset-meta">
                <dt className="text-slate-500">ID</dt><dd className="font-mono text-slate-200">{activeDataset.id}</dd>
                <dt className="text-slate-500">Rows × columns</dt><dd className="text-slate-200">{activeDataset.rowCount.toLocaleString('en-US')} × {activeDataset.columnCount}</dd>
                <dt className="text-slate-500">Uploaded (UTC)</dt><dd className="text-slate-200">{fmtText(activeDataset.uploadDate)}</dd>
                <dt className="text-slate-500">Used for training</dt><dd className="text-slate-200">{usable} of {columns.length} columns</dd>
              </dl>
            </div>

            <div role="tablist" aria-label="Dataset views" className="flex gap-1 border-b border-slate-800">
              {(['schema', 'preview', 'stats'] as Tab[]).map((t) => (
                <button
                  key={t}
                  role="tab"
                  type="button"
                  aria-selected={tab === t}
                  onClick={() => setTab(t)}
                  className={`px-3 py-2 text-xs font-semibold rounded-t-md outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 ${tab === t ? 'text-white border-b-2 border-indigo-500' : 'text-slate-400 hover:text-slate-200'}`}
                >
                  {t === 'schema' ? 'Columns' : t === 'preview' ? 'Preview' : 'Statistics'}
                </button>
              ))}
            </div>

            {tab === 'schema' && (
              <div role="tabpanel" className="space-y-3">
                <p className="text-[11px] text-slate-500">
                  Type and missing-value handling were inferred by the server; change them if needed. Identifier-like columns are excluded by default.
                  {customized ? ' You have edited this configuration; Model Lab will train with your version.' : ' Model Lab uses this configuration unless you edit it.'}
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <caption className="sr-only">Column configuration</caption>
                    <thead className="text-slate-500">
                      <tr>
                        <th scope="col" className="py-2 pr-3 font-medium">Use</th>
                        <th scope="col" className="py-2 pr-3 font-medium">Column</th>
                        <th scope="col" className="py-2 pr-3 font-medium">Detected</th>
                        <th scope="col" className="py-2 pr-3 font-medium">Treat as</th>
                        <th scope="col" className="py-2 pr-3 font-medium">Missing</th>
                        <th scope="col" className="py-2 font-medium">Missing-value handling</th>
                      </tr>
                    </thead>
                    <tbody>
                      {columns.map((c) => (
                        <tr key={c.name} className="border-t border-slate-800 text-slate-300" data-testid={`col-row-${c.name}`}>
                          <td className="py-2 pr-3">
                            <input
                              type="checkbox"
                              aria-label={`Use column ${c.name}`}
                              checked={c.included && c.selectedType !== 'drop'}
                              onChange={(e) =>
                                editColumn(
                                  c.name,
                                  e.target.checked
                                    ? {
                                        included: true,
                                        selectedType: c.selectedType === 'drop' ? (c.inferredType === 'numeric' ? 'numeric' : 'categorical') : c.selectedType,
                                        imputationStrategy: c.imputationStrategy === 'drop' ? (c.inferredType === 'numeric' ? 'median' : 'mode') : c.imputationStrategy,
                                      }
                                    : { included: false },
                                )
                              }
                            />
                          </td>
                          <td className="py-2 pr-3 font-medium">{c.name}</td>
                          <td className="py-2 pr-3 text-slate-400">{c.inferredType}</td>
                          <td className="py-2 pr-3">
                            <select
                              aria-label={`Treat ${c.name} as`}
                              value={c.selectedType}
                              onChange={(e) => editColumn(c.name, { selectedType: e.target.value, included: e.target.value !== 'drop' })}
                              className={control}
                            >
                              <option value="numeric">numeric</option>
                              <option value="categorical">categorical</option>
                              <option value="drop">exclude</option>
                            </select>
                          </td>
                          <td className="py-2 pr-3">{c.nullCount} ({c.nullPercentage}%)</td>
                          <td className="py-2">
                            <select
                              aria-label={`Missing-value handling for ${c.name}`}
                              value={c.imputationStrategy}
                              onChange={(e) => editColumn(c.name, { imputationStrategy: e.target.value })}
                              className={control}
                            >
                              <option value="median">median</option>
                              <option value="mean">mean</option>
                              <option value="mode">most frequent</option>
                              <option value="drop">remove rows</option>
                            </select>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {customized && (
                  <button type="button" onClick={() => onColumnConfigChange(activeDataset.id, null)} className="text-xs underline text-indigo-300">
                    Reset to the server's detected configuration
                  </button>
                )}
              </div>
            )}

            {tab === 'preview' && (
              <div role="tabpanel">
                {preview.phase === 'loading' && <p className="text-xs text-slate-400">Loading preview…</p>}
                {preview.phase === 'error' && <p role="alert" className="text-xs text-rose-300">{preview.message}</p>}
                {preview.phase === 'ok' && preview.rows.length === 0 && (
                  <p className="text-xs text-amber-300" data-testid="no-preview">No preview rows are available (the dataset file may be missing on the server).</p>
                )}
                {preview.phase === 'ok' && preview.rows.length > 0 && (
                  <div className="overflow-x-auto" data-testid="preview-table">
                    <p className="text-[11px] text-slate-500 mb-1">First {preview.rows.length} rows, as stored by the server.</p>
                    <table className="w-full text-xs text-left">
                      <caption className="sr-only">Dataset preview</caption>
                      <thead className="text-slate-500">
                        <tr>{Object.keys(preview.rows[0]).map((k) => <th key={k} scope="col" className="py-1.5 pr-3 font-medium whitespace-nowrap">{k}</th>)}</tr>
                      </thead>
                      <tbody>
                        {preview.rows.map((r, i) => (
                          <tr key={i} className="border-t border-slate-800 text-slate-300">
                            {Object.keys(preview.rows[0]).map((k) => <td key={k} className="py-1 pr-3 whitespace-nowrap">{String(r[k] ?? '')}</td>)}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            {tab === 'stats' && (
              <div role="tabpanel" className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <caption className="sr-only">Column statistics</caption>
                  <thead className="text-slate-500">
                    <tr>
                      <th scope="col" className="py-1.5 pr-3 font-medium">Column</th>
                      <th scope="col" className="py-1.5 pr-3 font-medium">Min</th>
                      <th scope="col" className="py-1.5 pr-3 font-medium">Max</th>
                      <th scope="col" className="py-1.5 pr-3 font-medium">Mean</th>
                      <th scope="col" className="py-1.5 pr-3 font-medium">Std</th>
                      <th scope="col" className="py-1.5 font-medium">Categories (first 20)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {activeDataset.columns.map((c) => (
                      <tr key={c.name} className="border-t border-slate-800 text-slate-300">
                        <td className="py-1.5 pr-3 font-medium">{c.name}</td>
                        <td className="py-1.5 pr-3">{fmtNum(c.min, 2)}</td>
                        <td className="py-1.5 pr-3">{fmtNum(c.max, 2)}</td>
                        <td className="py-1.5 pr-3">{fmtNum(c.mean, 2)}</td>
                        <td className="py-1.5 pr-3">{fmtNum(c.std, 2)}</td>
                        <td className="py-1.5">{c.categories && c.categories.length ? c.categories.join(', ') : UNAVAILABLE}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </section>

      {/* Preprocessing */}
      {activeDataset && datasetsState.phase === 'ready' && (
        <section className={card} aria-labelledby="prep-heading">
          <h3 id="prep-heading" className="text-indigo-400 font-semibold text-sm border-b border-slate-800 pb-3 mb-4">3. Preprocessing check</h3>
          <p className="text-xs text-slate-400 mb-3">
            A dry run of the server's preprocessing (missing values, scaling, one-hot encoding) with the columns above. Training prepares the data itself with the same rules; this shows what it will do.
          </p>
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={runCheck} disabled={check.phase === 'running'} className={`${btn} border border-slate-700 hover:border-indigo-500 text-slate-200`}>
              {check.phase === 'running' ? <Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden="true" /> : null}
              {check.phase === 'running' ? 'Checking…' : 'Run preprocessing check'}
            </button>
            <button type="button" onClick={() => onNavigateToModelLab(activeDataset.id)} disabled={!!trainBlock} className={`${btn} bg-indigo-600 hover:bg-indigo-500 text-white`}>
              Continue to Model Lab <ArrowRight className="w-3.5 h-3.5" aria-hidden="true" />
            </button>
          </div>
          {trainBlock && <p className="text-xs text-amber-300 mt-2">{trainBlock}</p>}

          {check.phase === 'error' && (
            <p role="alert" data-testid="check-error" className="mt-3 text-xs text-rose-300 bg-rose-500/10 border border-rose-500/30 rounded-lg p-3">Preprocessing check failed: {check.message}</p>
          )}
          {check.phase === 'ok' && (
            <div data-testid="check-result" className="mt-3 text-xs space-y-1 bg-slate-950/60 border border-slate-800 rounded-lg p-3">
              <p className="text-slate-200">
                {check.data.processed_dimensions} model input features from {check.data.rows_used ?? 'an unknown number of'} rows ·{' '}
                <span className={check.data.ready_for_training ? 'text-emerald-400' : 'text-amber-300'}>{check.data.ready_for_training ? 'ready for training' : 'not ready for training'}</span>
              </p>
              {check.data.dropped_columns.length > 0 && (
                <p className="text-slate-400">Not used: {check.data.dropped_columns.map((c) => `${c.name} (${c.reason})`).join('; ')}</p>
              )}
              {check.data.warnings.map((w) => <p key={w} className="text-amber-300">⚠ {w}</p>)}
              {check.data.warnings.length === 0 && <p className="text-slate-500">No warnings.</p>}
            </div>
          )}
        </section>
      )}
    </div>
  );
};
