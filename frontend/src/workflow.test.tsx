// Drives the whole UI through the workflow with a stateful stand-in for the backend:
// upload -> preprocessing check -> train -> generate -> quality comparison -> export history -> CSV download.
// It asserts that the same identifiers are used at every step. The real backend contracts are covered by
// live.contract.test.ts and the backend tests.
import React from 'react';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import { installFakeServer } from './test/fakeServer';
import { captureDownloads, readBlob } from './test/downloads';
import { json } from './test/backend';

afterEach(() => vi.unstubAllGlobals());

let saved: ReturnType<typeof captureDownloads>;
beforeEach(() => {
  saved = captureDownloads();
});

const students = (rows: number) => ['gender,study_hours,score', ...Array.from({ length: rows }, (_, i) => `${i % 2 ? 'Male' : 'Female'},${10 + i},${50 + i}`)].join('\n') + '\n';
const csv = (content: string, name = 'students.csv') => new File([content], name, { type: 'text/csv' });

describe('complete workflow in the UI', () => {
  it('moves one dataset through every page with consistent identifiers', async () => {
    const server = installFakeServer();
    window.location.hash = '#/dataset-studio';
    render(<App />);

    // 1. nothing is pre-loaded: no sample or mock dataset
    expect(await screen.findByTestId('no-datasets')).toBeInTheDocument();

    // 2. upload a real file; the id comes from the server
    await userEvent.upload(screen.getByLabelText(/choose csv file/i), csv(students(12)));
    const ok = await screen.findByTestId('upload-success');
    expect(ok).toHaveTextContent('ds_up00001');
    expect(ok).toHaveTextContent('12 rows, 3 columns');
    expect(window.location.hash).toBe('#/dataset-studio?dataset=ds_up00001');
    await waitFor(() => expect(screen.getByTestId('dataset-meta')).toHaveTextContent('ds_up00001'));

    // 3. preprocessing check with an edited configuration (exclude "score")
    await userEvent.click(screen.getByLabelText('Use column score'));
    await userEvent.click(screen.getByRole('button', { name: /run preprocessing check/i }));
    expect(await screen.findByTestId('check-result')).toHaveTextContent('2 model input features from 12 rows · ready for training');
    expect(screen.getByTestId('check-result')).toHaveTextContent('score (excluded by configuration)');

    // 4. Model Lab: the same dataset is selected, the edited columns are sent with the training request
    await userEvent.click(within(screen.getByTestId('upload-success')).getByRole('button', { name: /continue to model lab/i }));
    expect(window.location.hash).toBe('#/model-lab?dataset=ds_up00001');
    await waitFor(() => expect(screen.getByLabelText('Dataset to train on')).toHaveValue('ds_up00001'));
    expect(await screen.findByTestId('custom-columns-note')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Start Training' }));
    await screen.findByTestId('status-completed');
    const modelId = screen.getByTestId('result-model-id').textContent!;
    expect(modelId).toBe('vae_t00002');
    expect(screen.getByTestId('result-dataset-id')).toHaveTextContent('ds_up00001');
    const trainBody = server.calls.find((c) => c.key === 'POST /api/models/train')!.body;
    expect(trainBody.dataset_id).toBe('ds_up00001');
    expect(trainBody.columns.find((c: any) => c.name === 'score').included).toBe(false);
    expect(trainBody.columns.find((c: any) => c.name === 'study_hours').included).toBe(true);

    // 5. Data Generator: the trained model is preselected
    await userEvent.click(screen.getByRole('button', { name: /use this model in data generator/i }));
    expect(window.location.hash).toBe(`#/generator?model=${modelId}`);
    await waitFor(() => expect(screen.getByLabelText('Trained model')).toHaveValue(modelId));
    expect(screen.getByTestId('model-dataset')).toHaveTextContent('students.csv (ds_up00001)');
    const records = screen.getByLabelText('Number of records');
    await userEvent.clear(records);
    await userEvent.type(records, '25');
    await userEvent.click(screen.getByRole('button', { name: /generate synthetic data/i }));
    await screen.findByTestId('gen-result');
    const experimentId = screen.getByTestId('gen-experiment-id').textContent!;
    expect(experimentId).toBe('gen_t00003');
    const genBody = server.calls.find((c) => c.key === 'POST /api/generate')!.body;
    expect(genBody).toMatchObject({ model_id: modelId, dataset_id: 'ds_up00001', num_records: 25 });
    const fidelity = screen.getByTestId('gen-fidelity').textContent!;
    expect(fidelity).toBe('83.0%');

    // 6. CSV straight from the generator: the file of THIS experiment
    await userEvent.click(screen.getByRole('button', { name: /download csv/i }));
    await waitFor(() => expect(saved).toHaveLength(1));
    expect(saved[0].filename).toBe(`dataforge_experiment_${experimentId}.csv`);
    expect(await readBlob(saved[0].blob)).toBe(`marker,value\n${experimentId},25\n`);

    // 7. Quality comparison for the same experiment
    await userEvent.click(screen.getByRole('button', { name: /compare quality/i }));
    expect(window.location.hash).toBe(`#/quality?experiment=${experimentId}`);
    expect(await screen.findByTestId('eval-fidelity')).toHaveTextContent(fidelity);
    expect(screen.getByTestId('qc-experiment-info')).toHaveTextContent(experimentId);
    expect(screen.getByTestId('qc-experiment-info')).toHaveTextContent('students.csv');
    expect(server.count(`GET /api/evaluate/${experimentId}`)).toBe(1);

    // 8. Export history shows the same experiment with the same figures; its download is the same file
    await userEvent.click(screen.getByRole('button', { name: /6\. Export & History/ }));
    const row = await screen.findByTestId(`experiment-row-${experimentId}`);
    expect(row).toHaveTextContent('students.csv');
    expect(row).toHaveTextContent(modelId);
    expect(row).toHaveTextContent('25');
    expect(row).toHaveTextContent(fidelity);
    await userEvent.click(within(row).getByRole('button', { name: new RegExp(`download csv for ${experimentId}`, 'i') }));
    await waitFor(() => expect(saved).toHaveLength(2));
    expect(await readBlob(saved[1].blob)).toBe(await readBlob(saved[0].blob));
    expect(server.calls.filter((c) => c.key.startsWith('GET /api/export/')).every((c) => c.key === `GET /api/export/${experimentId}`)).toBe(true);
  });

  it('refreshes history after a generation without a manual reload', async () => {
    const server = installFakeServer();
    window.location.hash = '#/dataset-studio';
    render(<App />);
    await userEvent.upload(await screen.findByLabelText(/choose csv file/i), csv(students(12)));
    await screen.findByTestId('upload-success');
    const before = server.count('GET /api/experiments');
    await userEvent.click(screen.getByRole('button', { name: /3\. Model Lab/ }));
    await userEvent.click(await screen.findByRole('button', { name: 'Start Training' }));
    await userEvent.click(await screen.findByRole('button', { name: /use this model in data generator/i }));
    await userEvent.click(await screen.findByRole('button', { name: /generate synthetic data/i }));
    await screen.findByTestId('gen-result');
    await waitFor(() => expect(server.count('GET /api/experiments')).toBeGreaterThan(before));
    await userEvent.click(screen.getByRole('button', { name: /export history/i }));
    expect(await screen.findByTestId('experiment-row-gen_t00003')).toBeInTheDocument();
  });
});

describe('workflow error paths', () => {
  it('reports invalid uploads from the server and adds no dataset', async () => {
    const server = installFakeServer();
    window.location.hash = '#/dataset-studio';
    render(<App />);
    await screen.findByTestId('no-datasets');
    await userEvent.upload(screen.getByLabelText(/choose csv file/i), csv('   ', 'empty.csv'));
    expect(await screen.findByTestId('upload-error')).toHaveTextContent('The file is empty');
    await userEvent.upload(screen.getByLabelText(/choose csv file/i), new File(['a,b\n1,2\n'], 'notes.txt', { type: 'text/plain' }), { applyAccept: false });
    await waitFor(() => expect(screen.getByTestId('upload-error')).toHaveTextContent('Only CSV files'));
    expect(server.state.datasets).toHaveLength(0);
    expect(screen.getByTestId('no-datasets')).toBeInTheDocument();
  });

  it('does not allow training on a dataset that is too small', async () => {
    const server = installFakeServer();
    window.location.hash = '#/dataset-studio';
    render(<App />);
    await userEvent.upload(await screen.findByLabelText(/choose csv file/i), csv(students(3), 'tiny.csv'));
    await screen.findByTestId('upload-success');
    await userEvent.click(screen.getByRole('button', { name: /3\. Model Lab/ }));
    await screen.findByTestId('dataset-info');
    expect(screen.getByRole('button', { name: 'Start Training' })).toBeDisabled();
    expect(screen.getByText(/needs at least 10/i)).toBeInTheDocument();
    expect(server.count('POST /api/models/train')).toBe(0);
  });

  it('shows a generation failure without creating history', async () => {
    const server = installFakeServer();
    window.location.hash = '#/dataset-studio';
    render(<App />);
    await userEvent.upload(await screen.findByLabelText(/choose csv file/i), csv(students(12)));
    await screen.findByTestId('upload-success');
    await userEvent.click(screen.getByRole('button', { name: /3\. Model Lab/ }));
    await userEvent.click(await screen.findByRole('button', { name: 'Start Training' }));
    await userEvent.click(await screen.findByRole('button', { name: /use this model in data generator/i }));
    // the server now fails to generate
    const original = (globalThis.fetch as typeof fetch);
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) =>
      String(input).endsWith('/api/generate') ? json({ detail: 'The model\'s weights file is missing on the server.' }, 409) : original(input, init)));
    await userEvent.click(await screen.findByRole('button', { name: /generate synthetic data/i }));
    expect(await screen.findByTestId('gen-error')).toHaveTextContent("weights file is missing");
    expect(screen.queryByTestId('gen-result')).not.toBeInTheDocument();
    expect(server.state.experiments).toHaveLength(0);
    await userEvent.click(screen.getByRole('button', { name: /6\. Export & History/ }));
    expect(await screen.findByTestId('empty-history')).toBeInTheDocument();
  });

  it('shows empty history and an empty quality page for a fresh server', async () => {
    installFakeServer();
    window.location.hash = '#/history';
    render(<App />);
    expect(await screen.findByTestId('empty-history')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /5\. Quality Compare/ }));
    expect(await screen.findByTestId('no-experiments')).toBeInTheDocument();
  });

  it('survives a backend that disappears mid-session', async () => {
    installFakeServer();
    window.location.hash = '#/history';
    render(<App />);
    await screen.findByTestId('empty-history');
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    await userEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    expect(await screen.findByText(/Could not load experiments/)).toBeInTheDocument();
  });
});
