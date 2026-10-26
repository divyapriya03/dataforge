import React from 'react';
import { AlertTriangle } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { UNAVAILABLE, fmtNum, fmtPct, isNum } from '../services/format';
import type { ApiEvaluation } from '../types/api';

const FALLBACK_LIMITATION =
  'Fidelity is a heuristic summary of per-column and pairwise statistics. It is not a guarantee of data quality or privacy.';

const MetricStatus: React.FC<{ status?: string }> = ({ status }) => {
  const style = status === 'Passed' ? 'text-emerald-400' : status === 'Warning' ? 'text-amber-400' : status === 'Failed' ? 'text-rose-400' : 'text-slate-400';
  return <span className={style}>{status ?? UNAVAILABLE}</span>;
};

const Row: React.FC<{ label: string; children: React.ReactNode; testId?: string }> = ({ label, children, testId }) => (
  <>
    <dt className="text-slate-500">{label}</dt>
    <dd className="text-slate-200 break-words" data-testid={testId}>{children}</dd>
  </>
);

/**
 * Saved evaluation of one experiment, exactly as stored by the backend (never recomputed here).
 * Missing values are shown as "not available"; a real 0 is shown as a number.
 */
export const EvaluationPanel: React.FC<{ ev: ApiEvaluation }> = ({ ev }) => {
  const numeric = Object.entries(ev.ks_metrics ?? {});
  const categorical = Object.entries(ev.categorical_metrics ?? {});
  const components = Object.entries(ev.fidelity_components ?? {});
  const weights = ev.fidelity_weights ?? {};

  // Only real, finite distances are charted; missing values are left out rather than drawn as 0.
  const chartData = [
    ...numeric.map(([name, m]) => ({ name, value: m.ks_statistic, kind: 'KS' })),
    ...categorical.map(([name, m]) => ({ name, value: m.tvd, kind: 'TVD' })),
  ].filter((d): d is { name: string; value: number; kind: string } => isNum(d.value));
  const omitted = numeric.length + categorical.length - chartData.length;

  return (
    <div className="space-y-4 text-xs">
      <div className="grid grid-cols-2 gap-2">
        <div className="bg-slate-950/60 border border-slate-800 rounded-lg p-3">
          <div className="text-slate-500">Fidelity score (heuristic)</div>
          <div className="text-lg font-bold text-white" data-testid="eval-fidelity">{fmtPct(ev.overall_fidelity_score)}</div>
        </div>
        <div className="bg-slate-950/60 border border-slate-800 rounded-lg p-3">
          <div className="text-slate-500">Validity</div>
          <div className="text-lg font-bold text-white" data-testid="eval-validity">{fmtPct(ev.domain_validity_percent)}</div>
        </div>
      </div>

      <dl className="grid grid-cols-[auto,1fr] gap-x-4 gap-y-1">
        <Row label="Average KS statistic" testId="eval-avg-ks">{fmtNum(ev.average_ks_statistic)}</Row>
        <Row label="Correlation difference (mean │ΔR│)" testId="eval-corr-mad">{fmtNum(ev.correlation_mean_abs_delta)}</Row>
        <Row label="Correlation difference (Frobenius / n)" testId="eval-corr-frob">{fmtNum(ev.correlation_frobenius_delta)}</Row>
      </dl>
      {(ev.correlation_frobenius_delta === null || ev.correlation_frobenius_delta === undefined) && (
        <p className="text-slate-500">Correlations were not computed (fewer than two numeric columns varied in both datasets).</p>
      )}

      {components.length > 0 && (
        <div>
          <h5 className="text-slate-400 font-semibold mb-1">Fidelity components</h5>
          <table className="w-full text-left">
            <caption className="sr-only">Fidelity score components</caption>
            <thead className="text-slate-500"><tr><th scope="col" className="font-medium pr-3">Component</th><th scope="col" className="font-medium pr-3">Score</th><th scope="col" className="font-medium">Weight</th></tr></thead>
            <tbody>
              {components.map(([name, value]) => (
                <tr key={name} className="border-t border-slate-800 text-slate-300">
                  <td className="py-1 pr-3 capitalize">{name}</td>
                  <td className="py-1 pr-3">{isNum(value) ? value.toFixed(1) : 'not computed'}</td>
                  <td className="py-1">{isNum(weights[name]) ? `${Math.round(weights[name] * 100)}%` : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div>
        <h5 className="text-slate-400 font-semibold mb-1">Numeric columns (KS statistic, lower is closer)</h5>
        {numeric.length === 0 ? (
          <p className="text-slate-500" data-testid="no-numeric">No numeric columns were evaluated.</p>
        ) : (
          <table className="w-full text-left">
            <caption className="sr-only">Numeric column metrics</caption>
            <thead className="text-slate-500"><tr><th scope="col" className="font-medium pr-3">Column</th><th scope="col" className="font-medium pr-3">KS</th><th scope="col" className="font-medium pr-3">p-value</th><th scope="col" className="font-medium pr-3">Wasserstein</th><th scope="col" className="font-medium">Status</th></tr></thead>
            <tbody>
              {numeric.map(([name, m]) => (
                <tr key={name} className="border-t border-slate-800 text-slate-300" data-testid={`ks-row-${name}`}>
                  <td className="py-1 pr-3">{name}</td>
                  <td className="py-1 pr-3">{fmtNum(m.ks_statistic)}</td>
                  <td className="py-1 pr-3">{fmtNum(m.p_value)}</td>
                  <td className="py-1 pr-3">{fmtNum(m.wasserstein_distance)}</td>
                  <td className="py-1"><MetricStatus status={m.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div>
        <h5 className="text-slate-400 font-semibold mb-1">Categorical columns (total variation distance, lower is closer)</h5>
        {categorical.length === 0 ? (
          <p className="text-slate-500" data-testid="no-categorical">No categorical comparison is stored for this experiment (none evaluated, or created before it was recorded).</p>
        ) : (
          <table className="w-full text-left">
            <caption className="sr-only">Categorical column metrics</caption>
            <thead className="text-slate-500"><tr><th scope="col" className="font-medium pr-3">Column</th><th scope="col" className="font-medium pr-3">TVD</th><th scope="col" className="font-medium pr-3">Status</th><th scope="col" className="font-medium">Invalid categories</th></tr></thead>
            <tbody>
              {categorical.map(([name, m]) => (
                <tr key={name} className="border-t border-slate-800 text-slate-300" data-testid={`tvd-row-${name}`}>
                  <td className="py-1 pr-3">{name}</td>
                  <td className="py-1 pr-3">{fmtNum(m.tvd)}</td>
                  <td className="py-1 pr-3"><MetricStatus status={m.status} /></td>
                  <td className="py-1">{m.invalid_categories && m.invalid_categories.length > 0 ? m.invalid_categories.join(', ') : 'none'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {chartData.length > 0 && (
        <div>
          <div className="h-44" aria-label="Distance from the real data per column">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 5, right: 10, left: -15, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#64748b" fontSize={10} />
                <YAxis stroke="#64748b" fontSize={10} domain={[0, 1]} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', fontSize: 11 }} />
                <Bar dataKey="value" name="Distance (KS or TVD)" fill="#818cf8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          {omitted > 0 && <p className="text-slate-500" data-testid="chart-omitted">{omitted} column(s) without a stored value are not charted.</p>}
        </div>
      )}

      {ev.warnings && ev.warnings.length > 0 && (
        <ul className="space-y-1" data-testid="eval-warnings">
          {ev.warnings.map((w) => (
            <li key={w} className="text-amber-300 flex gap-1.5"><AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-px" aria-hidden="true" />{w}</li>
          ))}
        </ul>
      )}

      <p className="text-[11px] text-slate-500" data-testid="eval-limitations">{ev.limitations || FALLBACK_LIMITATION}</p>
    </div>
  );
};
