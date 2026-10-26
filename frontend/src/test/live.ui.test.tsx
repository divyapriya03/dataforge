// End-to-end UI test against a RUNNING backend: the real React app (rendered in jsdom) talks to a real FastAPI
// process over HTTP. Skipped unless VITE_LIVE_UI=1. The only test double is a fetch bridge that converts jsdom's
// FormData/AbortSignal into what Node's fetch accepts; every request still goes to the real server.
//
// Phase 1 uploads, trains, generates, evaluates, downloads and checks history against the server's own answers.
// Phase 2 (run after restarting the backend, with the expectations printed by phase 1) checks that the records
// and the CSV survived the restart. See TESTING_AND_QA_REPORT.md for the exact commands.
import React from 'react';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App';
import { API_BASE_URL } from '../services/api';
import { captureDownloads, readBlob } from './downloads';
import type { ApiEvaluation, ApiExperiment } from '../types/api';

const LIVE = !!import.meta.env.VITE_LIVE_UI;
const PHASE = import.meta.env.VITE_LIVE_PHASE || '1';
const nodeFetch: typeof fetch = globalThis.fetch.bind(globalThis);
const STEP = { timeout: 30000 };

// FNV-1a hash, enough to compare two CSV texts
const hash = (text: string) => {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 0x01000193) >>> 0;
  return h.toString(16);
};

const readBuffer = (b: Blob) =>
  new Promise<ArrayBuffer>((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as ArrayBuffer);
    r.onerror = reject;
    r.readAsArrayBuffer(b);
  });

async function bridgedFetch(input: RequestInfo | URL, init?: RequestInit) {
  try {
    return await bridgedFetchInner(input, init);
  } catch (e) {
    console.log(`BRIDGE_ERROR ${String(input)} ${(e as Error).stack?.split(String.fromCharCode(10)).slice(0, 4).join(' | ')}`);
    throw e;
  }
}

async function bridgedFetchInner(input: RequestInfo | URL, init?: RequestInit) {
  let body: BodyInit | null | undefined = init?.body;
  let headers = init?.headers;
  if (typeof FormData !== 'undefined' && body instanceof FormData) {
    // serialize the multipart body by hand: jsdom's FormData/Blob are not accepted by Node's fetch
    const CRLF = String.fromCharCode(13, 10);
    const boundary = `----dataforge${Math.random().toString(16).slice(2)}`;
    const enc = new TextEncoder();
    const chunks: Uint8Array[] = [];
    for (const [key, value] of body.entries()) {
      const isFile = typeof value !== 'string';
      const head = `--${boundary}${CRLF}Content-Disposition: form-data; name="${key}"${isFile ? `; filename="${(value as File).name}"` : ''}${CRLF}${isFile ? `Content-Type: ${(value as File).type || 'application/octet-stream'}${CRLF}` : ''}${CRLF}`;
      chunks.push(enc.encode(head));
      chunks.push(isFile ? new Uint8Array(await readBuffer(value as Blob)) : enc.encode(value as string));
      chunks.push(enc.encode(CRLF));
    }
    chunks.push(enc.encode(`--${boundary}--${CRLF}`));
    const bytes = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
    let offset = 0;
    for (const c of chunks) {
      bytes.set(c, offset);
      offset += c.length;
    }
    body = bytes;
    headers = { ...(headers as Record<string, string> | undefined), 'Content-Type': `multipart/form-data; boundary=${boundary}` };
  }
  const { signal: _ignored, ...rest } = init ?? {};
  return nodeFetch(input, { ...rest, headers, body });
}

const api = async <T,>(path: string) => (await (await nodeFetch(`${API_BASE_URL}${path}`)).json()) as T;
const apiText = async (path: string) => (await nodeFetch(`${API_BASE_URL}${path}`)).text();

const sampleCsv = () => {
  const rows = Array.from({ length: 60 }, (_, i) => {
    const hours = 5 + (i % 20) * 0.5;
    return `${i % 2 ? 'Male' : 'Female'},${['Freshman', 'Junior', 'Senior'][i % 3]},${hours.toFixed(1)},${(40 + hours * 2.5 + (i % 7)).toFixed(1)}`;
  });
  return ['gender,grade,hours,score', ...rows].join(String.fromCharCode(10)) + String.fromCharCode(10);
};

describe.skipIf(!LIVE)(`live UI against a real backend (phase ${PHASE})`, () => {
  let saved: ReturnType<typeof captureDownloads>;
  beforeAll(() => {
    vi.stubGlobal('fetch', bridgedFetch);
  });
  afterAll(() => vi.unstubAllGlobals());
  beforeEach(() => {
    saved = captureDownloads();
  });

  it.skipIf(PHASE !== '1')('phase 1: upload -> preprocess -> train -> generate -> evaluate -> download -> history', async () => {
    window.location.hash = '#/dataset-studio';
    render(<App />);

    // health: the sidebar shows the backend as connected
    await waitFor(() => expect(screen.getByText('Connected')).toBeInTheDocument(), STEP);

    // upload a real CSV through the file input
    const csvText = sampleCsv();
    await userEvent.upload(await screen.findByLabelText(/choose csv file/i), new File([csvText], 'live_students.csv', { type: 'text/csv' }));
    const ok = await screen.findByTestId('upload-success', undefined, STEP);
    const datasetId = within(ok).getByText(/^ds_/).textContent!;
    const serverDatasets = await api<{ id: string; rowCount: number; columnCount: number }[]>('/api/datasets');
    expect(serverDatasets.find((d) => d.id === datasetId)).toMatchObject({ rowCount: 60, columnCount: 4 });
    await waitFor(() => expect(screen.getByTestId('dataset-meta')).toHaveTextContent(datasetId), STEP);

    // preprocessing check: the UI shows what the server answered
    await userEvent.click(screen.getByRole('button', { name: /run preprocessing check/i }));
    const check = await screen.findByTestId('check-result', undefined, STEP);
    expect(check).toHaveTextContent('from 60 rows · ready for training');
    const dims = Number(/(\d+) model input features/.exec(check.textContent!)![1]);
    expect(dims).toBe(2 + 3 + 1 + 1); // gender(2) + grade(3) + hours + score

    // train through Model Lab with supported parameters
    await userEvent.click(within(ok).getByRole('button', { name: /continue to model lab/i }));
    await waitFor(() => expect(screen.getByLabelText('Dataset to train on')).toHaveValue(datasetId), STEP);
    const epochs = screen.getByLabelText('Epochs');
    await userEvent.clear(epochs);
    await userEvent.type(epochs, '60');
    await userEvent.type(screen.getByLabelText('Random seed (optional)'), '5');
    await userEvent.click(screen.getByRole('button', { name: 'Start Training' }));
    await screen.findByTestId('status-completed', undefined, STEP);
    const modelId = screen.getByTestId('result-model-id').textContent!;
    const serverModel = await api<{ model_id: string; status: string; dataset_id: string; params: { epochs: number; random_seed: number } }>(`/api/models/status/${modelId}`);
    expect(serverModel).toMatchObject({ model_id: modelId, status: 'completed', dataset_id: datasetId });
    expect(serverModel.params).toMatchObject({ epochs: 60, random_seed: 5 });
    expect(screen.getByTestId('result-dataset-id')).toHaveTextContent(datasetId);

    // generate a known number of rows with that model
    await userEvent.click(screen.getByRole('button', { name: /use this model in data generator/i }));
    await waitFor(() => expect(screen.getByLabelText('Trained model')).toHaveValue(modelId), STEP);
    const records = screen.getByLabelText('Number of records');
    await userEvent.clear(records);
    await userEvent.type(records, '40');
    await userEvent.click(screen.getByRole('button', { name: /generate synthetic data/i }));
    await screen.findByTestId('gen-result', undefined, STEP);
    const experimentId = screen.getByTestId('gen-experiment-id').textContent!;
    const preview = screen.getByTestId('gen-preview');
    expect(within(preview).getAllByRole('row')).toHaveLength(1 + 40);
    expect(['gender', 'grade', 'hours', 'score'].every((c) => preview.textContent!.includes(c))).toBe(true);

    // the evaluation shown equals the server's stored evaluation
    const evaluation = await api<ApiEvaluation>(`/api/evaluate/${experimentId}`);
    expect(screen.getByTestId('gen-fidelity')).toHaveTextContent(`${evaluation.overall_fidelity_score.toFixed(1)}%`);

    // download the CSV from the generator: identical to what the server serves, correct shape
    await userEvent.click(screen.getByRole('button', { name: /download csv/i }));
    await waitFor(() => expect(saved).toHaveLength(1), STEP);
    const downloaded = await readBlob(saved[0].blob);
    const serverCsv = await apiText(`/api/export/${experimentId}`);
    expect(downloaded).toBe(serverCsv);
    expect(saved[0].filename).toBe(`dataforge_experiment_${experimentId}.csv`);
    const lines = downloaded.trim().split(/\r?\n/);
    expect(lines).toHaveLength(41);
    expect(lines[0]).toBe('gender,grade,hours,score');
    const csvRows = lines.slice(1).map((l) => l.split(','));
    expect(csvRows.every((r) => r.length === 4 && ['Female', 'Male'].includes(r[0]) && ['Freshman', 'Junior', 'Senior'].includes(r[1]) && Number.isFinite(Number(r[2])) && Number.isFinite(Number(r[3])))).toBe(true);

    // quality comparison shows the same numbers as the API
    await userEvent.click(screen.getByRole('button', { name: /compare quality/i }));
    const shown = await screen.findByTestId('eval-fidelity', undefined, STEP);
    expect(shown).toHaveTextContent(`${evaluation.overall_fidelity_score.toFixed(1)}%`);
    expect(screen.getByTestId('qc-experiment-info')).toHaveTextContent(experimentId);
    expect(screen.getByTestId('eval-avg-ks')).toHaveTextContent((evaluation.average_ks_statistic as number).toFixed(4));

    // export history: the same experiment, model, dataset, record count and scores as the API
    await userEvent.click(screen.getByRole('button', { name: /6\. Export & History/ }));
    const row = await screen.findByTestId(`experiment-row-${experimentId}`, undefined, STEP);
    const serverExperiment = (await api<ApiExperiment[]>('/api/experiments')).find((e) => e.id === experimentId)!;
    expect(serverExperiment).toMatchObject({ dataset_id: datasetId, model_id: modelId, num_records: 40, status: 'completed', output_available: true });
    expect(row).toHaveTextContent(datasetId);
    expect(row).toHaveTextContent(modelId);
    expect(row).toHaveTextContent(`${serverExperiment.fidelity_score!.toFixed(1)}%`);
    await userEvent.click(within(row).getByRole('button', { name: new RegExp(`download csv for ${experimentId}`, 'i') }));
    await waitFor(() => expect(saved).toHaveLength(2), STEP);
    expect(await readBlob(saved[1].blob)).toBe(serverCsv);

    // expectations for phase 2 (after the backend restart)
    console.log(`LIVE_UI_EXPECT ${JSON.stringify({ experimentId, modelId, datasetId, csvHash: hash(serverCsv), fidelity: evaluation.overall_fidelity_score })}`);
  }, 120000);

  it.skipIf(PHASE !== '2')('phase 2: after a backend restart the records and the CSV are still there and consistent', async () => {
    const experimentId = import.meta.env.VITE_LIVE_EXP_ID as string;
    const expectedHash = import.meta.env.VITE_LIVE_CSV_HASH as string;
    const expectedFidelity = Number(import.meta.env.VITE_LIVE_FIDELITY);
    expect(experimentId).toBeTruthy();

    window.location.hash = '#/history';
    render(<App />);
    const row = await screen.findByTestId(`experiment-row-${experimentId}`, undefined, STEP);
    expect(row).toHaveTextContent(`${expectedFidelity.toFixed(1)}%`);
    expect(row).toHaveTextContent('Completed');

    await userEvent.click(within(row).getByRole('button', { name: /view details/i }));
    expect(await screen.findByTestId('eval-fidelity', undefined, STEP)).toHaveTextContent(`${expectedFidelity.toFixed(1)}%`);
    expect(screen.getByTestId('model-config')).toHaveTextContent('Epochs60');

    await userEvent.click(within(row).getByRole('button', { name: new RegExp(`download csv for ${experimentId}`, 'i') }));
    await waitFor(() => expect(saved).toHaveLength(1), STEP);
    expect(hash(await readBlob(saved[0].blob))).toBe(expectedHash);

    // the persisted model still generates
    const models = await api<{ model_id: string; status: string }[]>('/api/models');
    expect(models.some((m) => m.status === 'completed')).toBe(true);
  }, 120000);
});
