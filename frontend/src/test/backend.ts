// Test-only helpers: an in-memory stand-in for the FastAPI backend, installed as global fetch.
// The application itself never imports this file.
import { vi } from 'vitest';
import type { ApiDataset, ApiEvaluation, ApiExperiment, ApiModel } from '../types/api';

export type Handler = (req: {
  body: any;
  /** The raw request body (e.g. FormData for uploads). */
  raw?: BodyInit | null;
  /** Values of `:name` segments in the route key. */
  params: Record<string, string>;
  signal?: AbortSignal | null;
}) => Response | Promise<Response>;

export const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

function matchRoute(routes: Record<string, Handler>, key: string): { handler: Handler; params: Record<string, string> } | null {
  if (routes[key]) return { handler: routes[key], params: {} };
  const [method, path] = key.split(' ');
  for (const [pattern, handler] of Object.entries(routes)) {
    const [pm, pp] = pattern.split(' ');
    if (pm !== method || !pp.includes(':')) continue;
    const a = pp.split('/');
    const b = path.split('/');
    if (a.length !== b.length) continue;
    const params: Record<string, string> = {};
    if (a.every((seg, i) => (seg.startsWith(':') ? ((params[seg.slice(1)] = decodeURIComponent(b[i])), true) : seg === b[i]))) return { handler, params };
  }
  return null;
}

/** Install a fetch mock. Keys look like "GET /api/datasets" or "GET /api/evaluate/:id". Unknown routes answer 404 and are recorded. */
export function installBackend(routes: Record<string, Handler>) {
  const calls: { key: string; body: any }[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(typeof input === 'string' ? input : input instanceof URL ? input.href : input.url);
    const key = `${(init?.method ?? 'GET').toUpperCase()} ${url.pathname}`;
    const body = typeof init?.body === 'string' ? JSON.parse(init.body) : undefined;
    calls.push({ key, body });
    const match = matchRoute(routes, key);
    if (!match) return json({ detail: `no test route for ${key}` }, 404);
    return match.handler({ body, raw: init?.body, params: match.params, signal: init?.signal });
  });
  vi.stubGlobal('fetch', fetchMock);
  return { calls, fetchMock, count: (key: string) => calls.filter((c) => c.key === key).length };
}

export const column = (name: string, included = true) => ({
  name,
  inferredType: included ? 'numeric' : 'id',
  selectedType: included ? 'numeric' : 'drop',
  nullCount: 0,
  nullPercentage: 0,
  imputationStrategy: included ? 'median' : 'drop',
  included,
});

export const makeDataset = (over: Partial<ApiDataset> = {}): ApiDataset => ({
  id: 'ds_a1b2c3d4',
  filename: 'students.csv',
  rowCount: 250,
  columnCount: 3,
  uploadDate: '2026-10-24 05:10:34',
  columns: [column('student_id', false), column('study_hours'), column('math_score')],
  preview: [],
  ...over,
});

export const makeModel = (over: Partial<ApiModel> = {}): ApiModel => ({
  model_id: 'vae_0f1e2d3c',
  dataset_id: 'ds_a1b2c3d4',
  name: 'VAE-students-Z8',
  status: 'completed',
  current_epoch: 2,
  total_epochs: 2,
  final_loss: 6.6525,
  recon_loss: 0.1711,
  kl_loss: 3.2,
  loss_history: [
    { epoch: 1, totalLoss: 9.1, reconLoss: 0.31, klLoss: 2.9 },
    { epoch: 2, totalLoss: 6.6525, reconLoss: 0.1711, klLoss: 3.2 },
  ],
  model_type: 'numpy-vae',
  params: {
    dataset_id: 'ds_a1b2c3d4',
    model_name: null,
    latent_dim: 8,
    hidden_layers: [128, 64],
    learning_rate: 0.001,
    batch_size: 32,
    epochs: 2,
    beta_weight: 1,
    random_seed: 5,
  },
  error_message: null,
  created_at: '2026-10-24 05:11:00',
  completed_at: '2026-10-24 05:11:07',
  ...over,
});

export const healthOk = () =>
  json({ status: 'online', message: 'ok', version: '1.0.0', deep_learning_engine: 'VAE Neural Engine (Active)' });

export const makeExperiment = (over: Partial<ApiExperiment> = {}): ApiExperiment => ({
  id: 'gen_aaaa1111',
  dataset_id: 'ds_a1b2c3d4',
  model_id: 'vae_0f1e2d3c',
  dataset_name: 'students.csv',
  model_name: 'VAE-students-Z8',
  model_type: 'numpy-vae',
  num_records: 200,
  temperature: 1,
  fidelity_score: 86.6,
  validity_score: 100,
  random_seed: 7,
  output_filename: 'synthetic_students_gen_aaaa1111.csv',
  status: 'completed',
  output_available: true,
  created_at: '2026-10-24 05:20:00',
  ...over,
});

export const makeEvaluation = (over: Partial<ApiEvaluation> = {}): ApiEvaluation => ({
  experiment_id: 'gen_aaaa1111',
  overall_fidelity_score: 86.6,
  domain_validity_percent: 100,
  average_ks_statistic: 0.0792,
  correlation_frobenius_delta: 0.1186,
  correlation_mean_abs_delta: 0.0956,
  ks_metrics: {
    study_hours: { ks_statistic: 0.0652, p_value: 0.7123, wasserstein_distance: 0.5123, status: 'Passed' },
    math_score: { ks_statistic: 0, p_value: 1, wasserstein_distance: 0, status: 'Passed' },
  },
  categorical_metrics: { gender: { tvd: 0.008, status: 'Passed', invalid_categories: [] } },
  fidelity_components: { numeric: 96.7, categorical: 99.2, correlation: 52.2, validity: 100 },
  fidelity_weights: { numeric: 0.4, categorical: 0.2, correlation: 0.2, validity: 0.2 },
  warnings: [],
  limitations: 'Fidelity is a heuristic composite; it provides no privacy guarantee.',
  ...over,
});

export const csvResponse = (text = 'gender,score\nFemale,81.5\nMale,70.0\n') =>
  new Response(text, {
    status: 200,
    headers: { 'Content-Type': 'text/csv; charset=utf-8', 'Content-Disposition': 'attachment; filename="synthetic_x.csv"' },
  });
