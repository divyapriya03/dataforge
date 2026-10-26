import React, { useEffect, useState } from 'react';
import { BarChart3, Download, History, Loader2, Sparkles } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { evaluateRun } from '../services/api';
import { useCsvDownload } from '../hooks/useCsvDownload';
import { EvaluationPanel } from '../components/EvaluationPanel';
import { UNAVAILABLE, fmtNum, fmtText, isNum, statusLabel } from '../services/format';
import type { ApiEvaluation, ApiExperiment } from '../types/api';

type LoadState = { phase: 'loading' } | { phase: 'ready' } | { phase: 'error'; message: string };
type EvalState =
  | { phase: 'loading' }
  | { phase: 'ok'; data: ApiEvaluation }
  | { phase: 'unavailable' }
  | { phase: 'error'; message: string };

interface QualityComparisonProps {
  experiments: ApiExperiment[];
  experimentsState: LoadState;
  selectedExperimentId: string;
  onSelectExperiment: (id: string) => void;
  onReloadExperiments: () => void;
  onNavigateToGenerator: () => void;
  onNavigateToHistory: () => void;
}

const card = 'bg-slate-900/80 border border-slate-800 rounded-2xl p-6';
const btn = 'inline-flex items-center justify-center gap-1.5 text-xs font-semibold rounded-lg px-3 py-2 outline-none focus-visible:ring-2 focus-visible:ring-indigo-300 disabled:opacity-50 disabled:cursor-not-allowed';

export const QualityComparison: React.FC<QualityComparisonProps> = ({
  experiments,
  experimentsState,
  selectedExperimentId,
  onSelectExperiment,
  onReloadExperiments,
  onNavigateToGenerator,
  onNavigateToHistory,
}) => {
  const experiment = experiments.find((e) => e.id === selectedExperimentId) ?? experiments[0] ?? null;
  const [evaluation, setEvaluation] = useState<EvalState>({ phase: 'loading' });
  const [tick, setTick] = useState(0);
  const csv = useCsvDownload(onReloadExperiments);

  useEffect(() => {
    if (!experiment) return;
    let cancelled = false;
    setEvaluation({ phase: 'loading' });
    evaluateRun(experiment.id).then((res) => {
      if (cancelled) return;
      if (res.success && res.data && res.data.experiment_id === experiment.id) setEvaluation({ phase: 'ok', data: res.data });
      else if (res.success) setEvaluation({ phase: 'error', message: 'The server returned an evaluation for a different experiment.' });
      else if (res.status === 404) setEvaluation({ phase: 'unavailable' });
      else setEvaluation({ phase: 'error', message: res.error ?? 'Could not load the evaluation.' });
    });
    return () => {
      cancelled = true;
    };
  }, [experiment?.id, tick]);

  const ev = evaluation.phase === 'ok' ? evaluation.data : null;
  const categorical = Object.entries(ev?.categorical_metrics ?? {}).filter(([, m]) => m.real_proportions && m.synthetic_proportions);
  const summary = ev?.summary_comparisons ?? [];

  return (
    <div className="space-y-6">
      <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-2xl">
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-indigo-400" aria-hidden="true" /> Quality Comparison
        </h2>
        <p className="text-xs text-slate-400 mt-1 max-w-2xl">
          How closely a generated dataset matches the data it was learned from, using the evaluation the server saved when the data was generated.
        </p>
      </div>

      {experimentsState.phase === 'loading' && (
        <p role="status" className="text-sm text-slate-400 flex items-center gap-2"><Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" /> Loading experiments…</p>
      )}
      {experimentsState.phase === 'error' && (
        <div role="alert" className="text-sm text-rose-300 bg-rose-500/10 border border-rose-500/30 rounded-lg p-3 flex justify-between gap-3">
          <span>Could not load experiments: {experimentsState.message}</span>
          <button type="button" onClick={onReloadExperiments} className="text-xs underline whitespace-nowrap">Retry</button>
        </div>
      )}

      {experimentsState.phase === 'ready' && !experiment && (
        <div className={card} data-testid="no-experiments">
          <p className="text-sm text-slate-200 font-medium">No experiments to compare yet.</p>
          <p className="text-xs text-slate-400 mt-1">Generate synthetic data first; each generation is saved with its evaluation.</p>
          <button type="button" onClick={onNavigateToGenerator} className={`${btn} mt-3 bg-indigo-600 hover:bg-indigo-500 text-white`}>
            <Sparkles className="w-3.5 h-3.5" aria-hidden="true" /> Go to Data Generator
          </button>
        </div>
      )}

      {experiment && (
        <>
          <section className={card} aria-labelledby="qc-exp-heading">
            <h3 id="qc-exp-heading" className="text-indigo-400 font-semibold text-sm border-b border-slate-800 pb-3 mb-4">Experiment</h3>
            <div className="flex flex-wrap items-end gap-4">
              <div>
                <label htmlFor="qc-experiment" className="block text-[11px] text-slate-400 mb-1">Experiment</label>
                <select
                  id="qc-experiment"
                  value={experiment.id}
                  onChange={(e) => onSelectExperiment(e.target.value)}
                  className="bg-slate-950 border border-slate-800 rounded-md px-2 py-1.5 text-xs text-slate-200 outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 min-w-[18rem]"
                >
                  {experiments.map((e) => (
                    <option key={e.id} value={e.id}>{e.id} · {e.dataset_name ?? e.dataset_id} · {e.num_records.toLocaleString('en-US')} rows</option>
                  ))}
                </select>
              </div>
              <dl className="grid grid-cols-[auto,1fr] gap-x-3 gap-y-0.5 text-xs" data-testid="qc-experiment-info">
                <dt className="text-slate-500">Experiment ID</dt><dd className="font-mono text-slate-200">{experiment.id}</dd>
                <dt className="text-slate-500">Dataset</dt><dd className="text-slate-200">{experiment.dataset_name ?? 'record removed'} <span className="font-mono text-slate-500">({experiment.dataset_id})</span></dd>
                <dt className="text-slate-500">Model</dt><dd className="text-slate-200">{experiment.model_name ?? 'record removed'} <span className="font-mono text-slate-500">({experiment.model_id})</span></dd>
                <dt className="text-slate-500">Synthetic rows</dt><dd className="text-slate-200">{experiment.num_records.toLocaleString('en-US')}</dd>
                <dt className="text-slate-500">Created (UTC)</dt><dd className="text-slate-200">{fmtText(experiment.created_at)}</dd>
                <dt className="text-slate-500">Status</dt><dd className="text-slate-200">{statusLabel(experiment.status)}</dd>
              </dl>
              <div className="flex gap-2">
                <button
                  type="button"
                  disabled={!experiment.output_available || csv.downloading.includes(experiment.id)}
                  onClick={() => csv.download(experiment.id)}
                  className={`${btn} bg-indigo-600 hover:bg-indigo-500 text-white`}
                >
                  <Download className="w-3.5 h-3.5" aria-hidden="true" /> Download this CSV
                </button>
                <button type="button" onClick={onNavigateToHistory} className={`${btn} border border-slate-700 hover:border-indigo-500 text-slate-200`}>
                  <History className="w-3.5 h-3.5" aria-hidden="true" /> Export History
                </button>
              </div>
            </div>
            {csv.notice && (
              <p role={csv.notice.kind === 'error' ? 'alert' : 'status'} data-testid="download-notice" className={`mt-3 text-xs ${csv.notice.kind === 'error' ? 'text-rose-300' : 'text-emerald-300'}`}>{csv.notice.text}</p>
            )}
          </section>

          <section className={card} aria-labelledby="qc-eval-heading">
            <h3 id="qc-eval-heading" className="text-indigo-400 font-semibold text-sm border-b border-slate-800 pb-3 mb-4">Saved evaluation</h3>
            {evaluation.phase === 'loading' && <p role="status" className="text-sm text-slate-400 flex items-center gap-2"><Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" /> Loading evaluation…</p>}
            {evaluation.phase === 'unavailable' && (
              <p data-testid="qc-no-evaluation" className="text-sm text-amber-300">This experiment has no saved evaluation result, so there is nothing to compare.</p>
            )}
            {evaluation.phase === 'error' && (
              <p role="alert" className="text-sm text-rose-300">
                Could not load the evaluation: {evaluation.message}{' '}
                <button type="button" onClick={() => setTick((t) => t + 1)} className="underline">Retry</button>
              </p>
            )}
            {ev && (
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                <EvaluationPanel ev={ev} />

                <div className="space-y-5 text-xs min-w-0">
                  <div>
                    <h5 className="text-slate-400 font-semibold mb-1">Numeric columns: real vs synthetic summary</h5>
                    {summary.length === 0 ? (
                      <p className="text-slate-500" data-testid="qc-no-summary">No numeric summary is stored for this experiment.</p>
                    ) : (
                      <div className="overflow-x-auto">
                        <table className="w-full text-left" data-testid="qc-summary">
                          <caption className="sr-only">Real and synthetic means and standard deviations</caption>
                          <thead className="text-slate-500">
                            <tr>
                              <th scope="col" className="font-medium pr-3">Column</th>
                              <th scope="col" className="font-medium pr-3">Mean (real / synthetic)</th>
                              <th scope="col" className="font-medium pr-3">Std (real / synthetic)</th>
                              <th scope="col" className="font-medium">KS</th>
                            </tr>
                          </thead>
                          <tbody>
                            {summary.map((r) => (
                              <tr key={r.feature} className="border-t border-slate-800 text-slate-300" data-testid={`qc-summary-${r.feature}`}>
                                <td className="py-1 pr-3">{r.feature}</td>
                                <td className="py-1 pr-3">{fmtNum(r.realMean, 2)} / {fmtNum(r.syntheticMean, 2)}</td>
                                <td className="py-1 pr-3">{fmtNum(r.realStd, 2)} / {fmtNum(r.syntheticStd, 2)}</td>
                                <td className="py-1">{fmtNum(r.ksStatistic)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>

                  <div>
                    <h5 className="text-slate-400 font-semibold mb-1">Categorical columns: category proportions</h5>
                    {categorical.length === 0 ? (
                      <p className="text-slate-500" data-testid="qc-no-proportions">No category proportions are stored for this experiment.</p>
                    ) : (
                      categorical.map(([name, m]) => {
                        const cats = Array.from(new Set([...Object.keys(m.real_proportions!), ...Object.keys(m.synthetic_proportions!)]));
                        const data = cats
                          .map((c) => ({ category: c, real: m.real_proportions![c], synthetic: m.synthetic_proportions![c] }))
                          .filter((d) => isNum(d.real) && isNum(d.synthetic));
                        return (
                          <div key={name} className="mb-4" data-testid={`qc-proportions-${name}`}>
                            <div className="text-slate-300 mb-1">{name} <span className="text-slate-500">(TVD {fmtNum(m.tvd)})</span></div>
                            <div className="h-40" aria-label={`Category proportions of ${name}`}>
                              <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={data} margin={{ top: 5, right: 10, left: -15, bottom: 0 }}>
                                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                  <XAxis dataKey="category" stroke="#64748b" fontSize={10} />
                                  <YAxis stroke="#64748b" fontSize={10} domain={[0, 1]} />
                                  <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', fontSize: 11 }} />
                                  <Legend wrapperStyle={{ fontSize: 11 }} />
                                  <Bar dataKey="real" name="Real" fill="#38bdf8" />
                                  <Bar dataKey="synthetic" name="Synthetic" fill="#818cf8" />
                                </BarChart>
                              </ResponsiveContainer>
                            </div>
                          </div>
                        );
                      })
                    )}
                  </div>

                  <div className="bg-slate-950/60 border border-slate-800 rounded-lg p-3 space-y-1 text-[11px] text-slate-400" data-testid="qc-interpretation">
                    <p className="text-slate-300 font-semibold">How to read these numbers</p>
                    <p>KS statistic and total variation distance range from 0 (identical) to 1; lower is closer. “Passed” means below 0.15 and “Warning” below 0.25, a convention of this project, not a statistical test result.</p>
                    <p>Correlation differences compare pairwise linear relationships between numeric columns only. The matrices themselves are not stored, so only the summary differences are shown ({UNAVAILABLE} when fewer than two numeric columns varied).</p>
                    <p>The fidelity score is a weighted heuristic of these statistics. It says nothing about privacy or about how useful the data is for a particular task.</p>
                  </div>
                </div>
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
};
