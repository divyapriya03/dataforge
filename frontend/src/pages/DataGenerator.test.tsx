import React from 'react';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DataGenerator, validateGeneration } from './DataGenerator';
import { apiModelToRecord } from '../services/mappers';
import { csvResponse, installBackend, json, makeDataset, makeModel, type Handler } from '../test/backend';
import { captureDownloads, readBlob } from '../test/downloads';

afterEach(() => vi.unstubAllGlobals());

let saved: ReturnType<typeof captureDownloads>;
beforeEach(() => {
  saved = captureDownloads();
});

const generated = (over: Record<string, unknown> = {}) => ({
  experiment_id: 'gen_feed0001',
  dataset_id: 'ds_a1b2c3d4',
  model_id: 'vae_0f1e2d3c',
  num_records: 1000,
  temperature: 1,
  output_filename: 'synthetic_students_gen_feed0001.csv',
  fidelity_score: 86.6,
  domain_validity_percent: 100,
  preview_records: [
    { gender: 'Female', study_hours: 14.2 },
    { gender: 'Male', study_hours: 16.8 },
  ],
  ...over,
});

type Props = React.ComponentProps<typeof DataGenerator>;
function renderGenerator(props: Partial<Props> = {}, routes: Record<string, Handler> = {}) {
  const backend = installBackend({ 'POST /api/generate': () => json(generated()), ...routes });
  const p: Props = {
    models: [apiModelToRecord(makeModel())],
    datasets: [makeDataset()],
    selectedModelId: 'vae_0f1e2d3c',
    onSelectModel: vi.fn(),
    onGenerated: vi.fn(),
    onNavigateToModelLab: vi.fn(),
    onNavigateToQuality: vi.fn(),
    onNavigateToHistory: vi.fn(),
    ...props,
  };
  render(<DataGenerator {...p} />);
  return { p, backend };
}

const generate = () => userEvent.click(screen.getByRole('button', { name: /generate synthetic data|generating/i }));
const setField = async (label: string, value: string) => {
  const input = screen.getByLabelText(label);
  await userEvent.clear(input);
  if (value) await userEvent.type(input, value);
};

describe('validateGeneration', () => {
  const ok = { numRecords: '100', temperature: '1', randomSeed: '' };
  it('builds the request the backend expects', () => {
    expect(validateGeneration(ok, 'vae_1', 'ds_1').request).toEqual({ model_id: 'vae_1', dataset_id: 'ds_1', num_records: 100, temperature: 1 });
    expect(validateGeneration({ ...ok, randomSeed: '0' }, 'vae_1', 'ds_1').request?.random_seed).toBe(0);
  });
  it.each([
    [{ numRecords: '9' }, 'numRecords'],
    [{ numRecords: '50001' }, 'numRecords'],
    [{ numRecords: '12.5' }, 'numRecords'],
    [{ numRecords: '' }, 'numRecords'],
    [{ temperature: '0.05' }, 'temperature'],
    [{ temperature: '2.6' }, 'temperature'],
    [{ temperature: 'hot' }, 'temperature'],
    [{ randomSeed: '-1' }, 'randomSeed'],
    [{ randomSeed: '4294967296' }, 'randomSeed'],
    [{ randomSeed: '1.5' }, 'randomSeed'],
  ])('rejects %j', (patch, field) => {
    const res = validateGeneration({ ...ok, ...patch }, 'vae_1', 'ds_1');
    expect(res.errors[field as keyof typeof res.errors]).toBeTruthy();
    expect(res.request).toBeUndefined();
  });
  it('accepts the boundaries', () => {
    for (const patch of [{ numRecords: '10' }, { numRecords: '50000' }, { temperature: '0.1' }, { temperature: '2.5' }, { randomSeed: '4294967295' }]) {
      expect(validateGeneration({ ...ok, ...patch }, 'vae_1', 'ds_1').request).toBeDefined();
    }
  });
});

describe('DataGenerator', () => {
  it('lists only completed backend models and names the dataset they were trained on', () => {
    const failed = apiModelToRecord(makeModel({ model_id: 'vae_bad00001', name: 'broken', status: 'failed' }));
    renderGenerator({ models: [apiModelToRecord(makeModel()), failed] });
    const options = screen.getAllByRole('option').map((o) => o.textContent);
    expect(options).toEqual(['VAE-students-Z8 (vae_0f1e2d3c)']);
    expect(screen.getByTestId('model-dataset')).toHaveTextContent('students.csv (ds_a1b2c3d4)');
  });

  it('shows an empty state with a link to Model Lab when nothing can generate', async () => {
    const { p } = renderGenerator({ models: [apiModelToRecord(makeModel({ status: 'failed' }))] });
    expect(screen.getByTestId('no-models')).toHaveTextContent('No trained models yet.');
    expect(screen.queryByRole('button', { name: /generate synthetic data/i })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /go to model lab/i }));
    expect(p.onNavigateToModelLab).toHaveBeenCalled();
  });

  it('lets the user choose a model', async () => {
    const other = apiModelToRecord(makeModel({ model_id: 'vae_beef0002', name: 'second' }));
    const { p } = renderGenerator({ models: [apiModelToRecord(makeModel()), other] });
    await userEvent.selectOptions(screen.getByLabelText('Trained model'), 'vae_beef0002');
    expect(p.onSelectModel).toHaveBeenCalledWith('vae_beef0002');
  });

  it.each([
    ['Number of records', '5', /between 10 and 50,000/],
    ['Number of records', '60000', /between 10 and 50,000/],
    ['Temperature', '9', /between 0.1 and 2.5/],
    ['Random seed (optional)', '-4', /whole number between 0/],
  ])('rejects %s = %s before calling the backend', async (label, value, message) => {
    const { backend } = renderGenerator();
    await setField(label, value);
    await generate();
    expect(await screen.findByText(message)).toBeInTheDocument();
    expect(screen.getByLabelText(label)).toHaveAttribute('aria-invalid', 'true');
    expect(backend.count('POST /api/generate')).toBe(0);
  });

  it('sends the model and its dataset, then shows what the server returned', async () => {
    const { p, backend } = renderGenerator();
    await setField('Number of records', '250');
    await setField('Random seed (optional)', '7');
    await generate();

    const result = await screen.findByTestId('gen-result');
    expect(backend.calls.find((c) => c.key === 'POST /api/generate')!.body).toEqual({
      model_id: 'vae_0f1e2d3c',
      dataset_id: 'ds_a1b2c3d4',
      num_records: 250,
      temperature: 1,
      random_seed: 7,
    });
    expect(screen.getByTestId('gen-experiment-id')).toHaveTextContent('gen_feed0001');
    expect(screen.getByTestId('gen-fidelity')).toHaveTextContent('86.6%');
    expect(screen.getByTestId('gen-validity')).toHaveTextContent('100.0%');
    expect(result).toHaveTextContent('synthetic_students_gen_feed0001.csv');
    const preview = screen.getByTestId('gen-preview');
    expect(preview).toHaveTextContent('14.2');
    expect(preview).toHaveTextContent('Female');
    expect(result).toHaveTextContent(/not a guarantee of data quality or privacy/);
    expect(p.onGenerated).toHaveBeenCalledWith(generated());
  });

  it('omits the seed when left empty (the server default applies)', async () => {
    const { backend } = renderGenerator();
    await generate();
    await screen.findByTestId('gen-result');
    expect(backend.calls[0].body).not.toHaveProperty('random_seed');
  });

  it('shows a waiting state and ignores a second submit while generating', async () => {
    let release!: (r: Response) => void;
    const { backend } = renderGenerator({}, { 'POST /api/generate': () => new Promise<Response>((r) => (release = r)) });
    await generate();
    expect(await screen.findByTestId('gen-running')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /generating/i })).toBeDisabled();
    fireEvent.submit(screen.getByRole('form', { name: 'Generation setup' }));
    expect(backend.count('POST /api/generate')).toBe(1);
    await act(async () => release(json(generated())));
    await screen.findByTestId('gen-result');
    expect(backend.count('POST /api/generate')).toBe(1);
  });

  it.each([
    [404, 'Model not found'],
    [409, 'The model\'s weights file is missing on the server.'],
    [422, 'num_records must not exceed 100.'],
    [500, 'Generation failed because of an internal error. See the server log.'],
  ])('reports a %i response as a failure without creating an experiment', async (status, detail) => {
    const { p } = renderGenerator({}, { 'POST /api/generate': () => json({ detail }, status) });
    await generate();
    expect(await screen.findByTestId('gen-error')).toHaveTextContent(`Generation failed: ${detail}`);
    expect(screen.queryByTestId('gen-result')).not.toBeInTheDocument();
    expect(p.onGenerated).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: /generate synthetic data/i })).toBeEnabled();
  });

  it('flags an unknown outcome after a network failure and rejects a malformed answer', async () => {
    const { p } = renderGenerator({}, { 'POST /api/generate': () => Promise.reject(new TypeError('Failed to fetch')) });
    await generate();
    expect(await screen.findByTestId('gen-error')).toHaveTextContent(/outcome is unknown; check Export History/);

    vi.unstubAllGlobals();
    installBackend({ 'POST /api/generate': () => json({ hello: 'world' }) });
    await generate();
    await waitFor(() => expect(screen.getByTestId('gen-error')).toHaveTextContent('without an experiment ID'));
    expect(p.onGenerated).not.toHaveBeenCalled();
  });

  it('downloads the CSV of the generated experiment', async () => {
    const { backend } = renderGenerator({}, { 'GET /api/export/gen_feed0001': () => csvResponse('marker\ngen_feed0001\n') });
    await generate();
    await screen.findByTestId('gen-result');
    await userEvent.click(screen.getByRole('button', { name: /download csv/i }));
    await waitFor(() => expect(saved).toHaveLength(1));
    expect(backend.count('GET /api/export/gen_feed0001')).toBe(1);
    expect(saved[0].filename).toBe('dataforge_experiment_gen_feed0001.csv');
    expect(await readBlob(saved[0].blob)).toBe('marker\ngen_feed0001\n');
  });

  it('shows download failures', async () => {
    renderGenerator({}, { 'GET /api/export/gen_feed0001': () => json({ detail: 'Synthetic CSV file not found' }, 404) });
    await generate();
    await screen.findByTestId('gen-result');
    await userEvent.click(screen.getByRole('button', { name: /download csv/i }));
    expect(await screen.findByTestId('download-notice')).toHaveTextContent('Download of gen_feed0001 failed: Synthetic CSV file not found');
    expect(saved).toHaveLength(0);
  });

  it('links to the quality comparison and history with the experiment ID', async () => {
    const { p } = renderGenerator();
    await generate();
    await screen.findByTestId('gen-result');
    await userEvent.click(screen.getByRole('button', { name: /compare quality/i }));
    expect(p.onNavigateToQuality).toHaveBeenCalledWith('gen_feed0001');
    await userEvent.click(screen.getByRole('button', { name: /export history/i }));
    expect(p.onNavigateToHistory).toHaveBeenCalled();
  });
});

describe('DataGenerator: leaving the page while generating', () => {
  it('blocks a second generation after coming back', async () => {
    let release!: (r: Response) => void;
    const { backend, p } = renderGenerator({}, { 'POST /api/generate': () => new Promise<Response>((r) => (release = r)) });
    await generate();
    await screen.findByTestId('gen-running');
    cleanup();
    render(<DataGenerator {...p} />); // same props and the same fetch mock
    expect(screen.getByRole('button', { name: /generating/i })).toBeDisabled();
    fireEvent.submit(screen.getByRole('form', { name: 'Generation setup' }));
    expect(backend.count('POST /api/generate')).toBe(1);
    await act(async () => release(json(generated())));
    await waitFor(() => expect(p.onGenerated).toHaveBeenCalledTimes(1));
  });
});
