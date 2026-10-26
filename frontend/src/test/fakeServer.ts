// Test-only: a small stateful stand-in for the FastAPI backend, so the whole UI workflow can be driven in jsdom.
// It follows the same request/response contracts as the real API (verified separately by live.contract.test.ts
// and the backend tests) but contains no ML: values are simple and deterministic.
import type { ApiDataset, ApiEvaluation, ApiExperiment, ApiModel } from '../types/api';
import { column, csvResponse, healthOk, installBackend, json, makeEvaluation, type Handler } from './backend';

const readText = (file: Blob) =>
  new Promise<string>((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = reject;
    r.readAsText(file);
  });

export function installFakeServer(seed: { datasets?: ApiDataset[]; models?: ApiModel[]; experiments?: ApiExperiment[] } = {}) {
  const state = {
    datasets: [...(seed.datasets ?? [])],
    models: [...(seed.models ?? [])],
    experiments: [...(seed.experiments ?? [])],
    evaluations: {} as Record<string, ApiEvaluation>,
    csv: {} as Record<string, string>,
    n: 0,
  };

  const routes: Record<string, Handler> = {
    'GET /api/health': () => healthOk(),
    'GET /api/datasets': () => json(state.datasets),
    'GET /api/datasets/:id': ({ params }) => {
      const d = state.datasets.find((x) => x.id === params.id);
      return d ? json(d) : json({ detail: 'Dataset not found' }, 404);
    },
    'POST /api/datasets/upload': async ({ raw }) => {
      const file = (raw as FormData).get('file') as File;
      if (!file.name.toLowerCase().endsWith('.csv')) return json({ detail: 'Only CSV files (.csv) are supported.' }, 400);
      const text = await readText(file);
      if (!text.trim()) return json({ detail: 'Failed to parse CSV: The file is empty: no header or data found.' }, 400);
      const lines = text.trim().split('\n');
      const header = lines[0].split(',');
      const d: ApiDataset = {
        id: `ds_up${String(++state.n).padStart(5, '0')}`,
        filename: file.name,
        rowCount: lines.length - 1,
        columnCount: header.length,
        uploadDate: '2026-10-24 07:00:00',
        columns: header.map((h) => column(h.trim())),
        preview: lines.slice(1, 4).map((l) => Object.fromEntries(l.split(',').map((v, i) => [header[i], v]))),
      };
      state.datasets.unshift(d);
      return json(d);
    },
    'POST /api/preprocess': ({ body }) => {
      const d = state.datasets.find((x) => x.id === body.dataset_id);
      if (!d) return json({ detail: 'Dataset not found' }, 404);
      const used = body.columns.filter((c: any) => c.included && c.selectedType !== 'drop');
      return json({
        dataset_id: d.id,
        processed_dimensions: used.length,
        feature_names: used.map((c: any) => c.name),
        ready_for_training: d.rowCount >= 10,
        rows_used: d.rowCount,
        dropped_columns: body.columns
          .filter((c: any) => !(c.included && c.selectedType !== 'drop'))
          .map((c: any) => ({ name: c.name, reason: 'excluded by configuration' })),
        warnings: [],
      });
    },
    'POST /api/models/train': ({ body }) => {
      const d = state.datasets.find((x) => x.id === body.dataset_id);
      if (!d) return json({ detail: 'Dataset not found' }, 404);
      if (d.rowCount < 10) return json({ detail: `At least 10 usable rows are required to train; this dataset has ${d.rowCount}.` }, 400);
      const id = `vae_t${String(++state.n).padStart(5, '0')}`;
      const epochs = body.epochs ?? 50;
      const m: ApiModel = {
        model_id: id,
        dataset_id: d.id,
        name: body.model_name ?? `VAE-${d.filename.replace('.csv', '')}-Z${body.latent_dim ?? 8}`,
        status: 'completed',
        current_epoch: epochs,
        total_epochs: epochs,
        final_loss: 5.5,
        recon_loss: 0.2,
        kl_loss: 3.1,
        loss_history: Array.from({ length: Math.min(epochs, 5) }, (_, i) => ({ epoch: i + 1, totalLoss: 9 - i, reconLoss: 0.5 - i * 0.05, klLoss: 2 + i * 0.2 })),
        model_type: 'numpy-vae',
        params: { ...body, columns: undefined, hidden_layers: body.hidden_layers ?? [128, 64] },
        error_message: null,
        created_at: '2026-10-24 07:01:00',
        completed_at: '2026-10-24 07:01:05',
      };
      state.models.unshift(m);
      return json(m);
    },
    'GET /api/models': () => json(state.models),
    'GET /api/models/status/:id': ({ params }) => {
      const m = state.models.find((x) => x.model_id === params.id);
      return m ? json(m) : json({ detail: 'Model not found' }, 404);
    },
    'POST /api/generate': ({ body }) => {
      const m = state.models.find((x) => x.model_id === body.model_id);
      if (!m) return json({ detail: 'Model not found' }, 404);
      if (body.dataset_id && body.dataset_id !== m.dataset_id) {
        return json({ detail: `Model '${m.model_id}' was trained on dataset '${m.dataset_id}', not '${body.dataset_id}'.` }, 400);
      }
      if (body.num_records > 50000) return json({ detail: 'num_records must not exceed 50000.' }, 422);
      const d = state.datasets.find((x) => x.id === m.dataset_id);
      const id = `gen_t${String(++state.n).padStart(5, '0')}`;
      const fidelity = 80 + state.n;
      const file = `synthetic_${(d?.filename ?? 'data.csv').replace('.csv', '')}_${id}.csv`;
      state.csv[id] = `marker,value\n${id},${body.num_records}\n`;
      state.experiments.unshift({
        id,
        dataset_id: m.dataset_id,
        model_id: m.model_id,
        dataset_name: d?.filename ?? null,
        model_name: m.name,
        model_type: m.model_type,
        num_records: body.num_records,
        temperature: body.temperature ?? 1,
        fidelity_score: fidelity,
        validity_score: 100,
        random_seed: body.random_seed ?? 42,
        output_filename: file,
        status: 'completed',
        output_available: true,
        created_at: `2026-10-24 07:02:${String(10 + state.n).padStart(2, '0')}`,
      });
      state.evaluations[id] = makeEvaluation({
        experiment_id: id,
        overall_fidelity_score: fidelity,
        categorical_metrics: {
          gender: {
            tvd: 0.02,
            status: 'Passed',
            invalid_categories: [],
            real_proportions: { Female: 0.52, Male: 0.48 },
            synthetic_proportions: { Female: 0.5, Male: 0.5 },
          },
        },
        summary_comparisons: [
          { feature: 'study_hours', realMean: 15.34, syntheticMean: 15.1, realStd: 5.1, syntheticStd: 4.8, ksStatistic: 0.0652, pValue: 0.71, status: 'Passed' },
        ],
      });
      return json({
        experiment_id: id,
        dataset_id: m.dataset_id,
        model_id: m.model_id,
        num_records: body.num_records,
        temperature: body.temperature ?? 1,
        output_filename: file,
        fidelity_score: fidelity,
        domain_validity_percent: 100,
        preview_records: [
          { gender: 'Female', study_hours: 14.2 },
          { gender: 'Male', study_hours: 16.8 },
        ],
      });
    },
    'GET /api/experiments': () => json(state.experiments),
    'GET /api/evaluate/:id': ({ params }) => {
      const e = state.evaluations[params.id];
      return e ? json(e) : json({ detail: 'Experiment not found' }, 404);
    },
    'GET /api/export/:id': ({ params }) => {
      const csv = state.csv[params.id];
      return csv !== undefined ? csvResponse(csv) : json({ detail: 'Synthetic CSV file not found' }, 404);
    },
  };

  return { state, ...installBackend(routes) };
}
