// @vitest-environment node
// Contract test against a RUNNING backend (no mocks). Skipped unless VITE_LIVE_API=1.
//
// It trains a small real model, so run it only against a scratch backend, e.g.
//   DATAFORGE_DATA_DIR=<copy>/data DATAFORGE_DB_PATH=<copy>/dataforge.db python -m uvicorn app.main:app --port 8003
//   VITE_LIVE_API=1 VITE_API_BASE_URL=http://127.0.0.1:8003 npx vitest run src/test/live.contract.test.ts
import { describe, expect, it } from 'vitest';
import {
  downloadExperimentCsv,
  evaluateRun,
  generateSynthetic,
  getDatasets,
  getExperiments,
  getHealth,
  getModelStatus,
  getModels,
  previewPreprocessing,
  startTraining,
  uploadDataset,
} from '../services/api';
import { validateTrainingForm, DEFAULT_FORM } from '../services/trainingConfig';
import { apiModelToRecord } from '../services/mappers';

/** A dataset with enough rows to train on: an existing one if the server has one, otherwise one uploaded by the test. */
async function ensureDataset() {
  const existing = (await getDatasets()).data!.find((d) => d.rowCount >= 10);
  if (existing) return existing;
  const NL = String.fromCharCode(10);
  const rows = Array.from({ length: 40 }, (_, k) => `${k % 2 ? 'Male' : 'Female'},${(10 + k * 0.5).toFixed(1)},${50 + (k % 9)}`);
  const up = await uploadDataset(new File([['gender,hours,score', ...rows].join(NL) + NL], 'live_base.csv', { type: 'text/csv' }));
  return up.data!;
}

describe.skipIf(!import.meta.env.VITE_LIVE_API)('live backend contract', () => {
  it('health, datasets, preprocessing dry run, training, model list and status', async () => {
    const health = await getHealth();
    expect(health.success).toBe(true);

    const datasets = await getDatasets();
    expect(datasets.success).toBe(true);
    const ds = await ensureDataset();
    expect(ds).toBeTruthy();
    expect(ds.columns[0]).toHaveProperty('included');

    const prep = await previewPreprocessing(ds.id, ds.columns);
    expect(prep.success).toBe(true);
    expect(prep.data!.ready_for_training).toBe(true);

    // exactly the body the ModelLab form produces
    const { request } = validateTrainingForm({ ...DEFAULT_FORM, epochs: '5', randomSeed: '1' }, ds.id);
    const trained = await startTraining(request!);
    expect(trained.success).toBe(true);
    const m = trained.data!;
    expect(m.status).toBe('completed');
    expect(m.model_id).toMatch(/^vae_/);
    expect(m.dataset_id).toBe(ds.id);
    expect(m.loss_history).toHaveLength(5);
    expect(m.params).toMatchObject({ epochs: 5, random_seed: 1, hidden_layers: [128, 64], learning_rate: 0.001 });
    expect(m.created_at && m.completed_at).toBeTruthy();

    const list = await getModels();
    expect(list.data!.find((x) => x.model_id === m.model_id)).toBeTruthy();
    const status = await getModelStatus(m.model_id);
    expect(status.data).toEqual(m);

    const record = apiModelToRecord(m);
    expect(record.id).toBe(m.model_id);
    expect(record.params).toMatchObject({ epochs: 5, hiddenLayers: '128, 64', latentDim: 8 });
  }, 60000);

  it('reports real backend errors', async () => {
    const unknown = await startTraining({ dataset_id: 'ds_does_not_exist' });
    expect(unknown).toMatchObject({ success: false, kind: 'http', status: 404, error: 'Dataset not found' });

    const invalid = await startTraining({ dataset_id: 'x', hidden_layers: [64] });
    expect(invalid).toMatchObject({ success: false, kind: 'http', status: 422 });
    expect(invalid.error).toContain('hidden_layers');
  });

  it('history, saved evaluation and CSV download for real experiments', async () => {
    const ds = await ensureDataset();
    const model = (await startTraining({ dataset_id: ds.id, epochs: 5, random_seed: 3 })).data!;
    const g1 = (await generateSynthetic({ model_id: model.model_id, num_records: 30, random_seed: 1 })).data!;
    const g2 = (await generateSynthetic({ model_id: model.model_id, num_records: 50, random_seed: 2 })).data!;

    const list = (await getExperiments()).data!;
    const e1 = list.find((e) => e.id === g1.experiment_id)!;
    const e2 = list.find((e) => e.id === g2.experiment_id)!;
    expect(e1).toMatchObject({ dataset_id: ds.id, model_id: model.model_id, num_records: 30, status: 'completed', output_available: true, model_type: 'numpy-vae' });
    expect(e2.num_records).toBe(50);
    expect(list.indexOf(e2)).toBeLessThan(list.indexOf(e1)); // newest first
    expect(JSON.stringify(list)).not.toContain('output_file_path');

    const ev = (await evaluateRun(e1.id)).data!;
    expect(ev.overall_fidelity_score).toBe(g1.fidelity_score);
    expect(Object.keys(ev.ks_metrics).length).toBeGreaterThan(0);
    expect(ev.limitations).toBeTruthy();

    const csv1 = (await downloadExperimentCsv(e1.id)).data!;
    const csv2 = (await downloadExperimentCsv(e2.id)).data!;
    const t1 = await csv1.blob.text();
    const t2 = await csv2.blob.text();
    expect(csv1.filename).toBe(`dataforge_experiment_${e1.id}.csv`);
    expect(t1.trim().split('\n')).toHaveLength(31); // header + 30 rows
    expect(t2.trim().split('\n')).toHaveLength(51); // header + 50 rows: each experiment gets its own file
    expect(t1.split('\n')[0]).toBe(t2.split('\n')[0]);

    expect(await downloadExperimentCsv('gen_does_not_exist')).toMatchObject({ success: false, kind: 'http', status: 404 });
  }, 60000);

  it('uploads through the real client, trains with edited columns, and rejects bad files with the server message', async () => {
    const rows = Array.from({ length: 40 }, (_, k) => `${k % 2 ? 'Male' : 'Female'},${(10 + k * 0.5).toFixed(1)},${50 + (k % 9)}`);
    const csv = ['gender,hours,score', ...rows].join('\n') + '\n';
    const up = await uploadDataset(new File([csv], 'live_students.csv', { type: 'text/csv' }));
    expect(up.success).toBe(true);
    const ds = up.data!;
    expect(ds.id).toMatch(/^ds_/);
    expect(ds).toMatchObject({ filename: 'live_students.csv', rowCount: 40, columnCount: 3 });
    expect(ds.uploadDate).toMatch(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/);

    const empty = await uploadDataset(new File([''], 'empty.csv', { type: 'text/csv' }));
    expect(empty).toMatchObject({ success: false, kind: 'http', status: 400 });
    expect(empty.error).toContain('empty');
    const notCsv = await uploadDataset(new File(['a,b\n1,2\n'], 'notes.txt', { type: 'text/plain' }));
    expect(notCsv).toMatchObject({ success: false, kind: 'http', status: 400 });
    expect(notCsv.error).toContain('Only CSV');

    const all = (await getDatasets()).data!;
    expect(all.filter((d) => d.filename === 'empty.csv' || d.filename === 'notes.txt')).toHaveLength(0);
    expect(all.find((d) => d.id === ds.id)?.rowCount).toBe(40);

    // edited configuration: exclude "score"; the server must honour it
    const columns = ds.columns.map((c) => (c.name === 'score' ? { ...c, included: false, selectedType: 'drop' } : c));
    const prep = (await previewPreprocessing(ds.id, columns)).data!;
    expect(prep.dropped_columns.map((c) => c.name)).toContain('score');
    const model = (await startTraining({ dataset_id: ds.id, epochs: 5, random_seed: 1, columns })).data!;
    const gen = (await generateSynthetic({ model_id: model.model_id, dataset_id: ds.id, num_records: 20, random_seed: 3 })).data!;
    expect(Object.keys(gen.preview_records[0]).sort()).toEqual(['gender', 'hours']);
    expect(gen.dataset_id).toBe(ds.id);
    const ev = (await evaluateRun(gen.experiment_id)).data!;
    expect(Object.keys(ev.ks_metrics)).toEqual(['hours']);
    const csvOut = (await downloadExperimentCsv(gen.experiment_id)).data!;
    expect((await csvOut.blob.text()).trim().split(/\r?\n/)[0]).toBe('gender,hours');
  }, 60000);
});
