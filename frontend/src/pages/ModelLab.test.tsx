import React from 'react';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ModelLab } from './ModelLab';
import { TRAIN_TIMEOUT_MS } from '../services/api';
import { apiModelToRecord } from '../services/mappers';
import { column, installBackend, json, makeDataset, makeModel, type Handler } from '../test/backend';

afterEach(() => vi.unstubAllGlobals());

type LabProps = React.ComponentProps<typeof ModelLab>;

function renderLab(props: Partial<LabProps> = {}) {
  const p: LabProps = {
    models: [],
    activeModel: null,
    backendOnline: true,
    backendChecking: false,
    onModelTrained: vi.fn(),
    onRefreshModels: vi.fn().mockResolvedValue(null),
    onSelectModel: vi.fn(),
    onNavigateToGenerator: vi.fn(),
    ...props,
  };
  render(<ModelLab {...p} />);
  return p;
}

const withDatasets = (datasets = [makeDataset()], extra: Record<string, Handler> = {}) =>
  installBackend({ 'GET /api/datasets': () => json(datasets), ...extra });

const startButton = () => screen.getByRole('button', { name: /start training|training…/i });
const waitForDatasets = () => screen.findByTestId('dataset-info');

describe('ModelLab: datasets', () => {
  it('loads the dataset list from the backend and shows rows and columns', async () => {
    withDatasets([makeDataset(), makeDataset({ id: 'ds_second01', filename: 'sales.csv', rowCount: 1200, columnCount: 5 })]);
    renderLab();
    expect(screen.getByText(/loading datasets/i)).toBeInTheDocument();
    await waitForDatasets();
    const select = screen.getByLabelText('Dataset to train on');
    expect(within(select).getAllByRole('option').map((o) => o.textContent)).toEqual([
      'students.csv (ds_a1b2c3d4)',
      'sales.csv (ds_second01)',
    ]);
    expect(screen.getByTestId('dataset-info')).toHaveTextContent('250 rows · 3 columns · 2 usable for training (1 excluded');
  });

  it('shows an empty state and blocks training when there are no datasets', async () => {
    withDatasets([]);
    renderLab();
    expect(await screen.findByText('No datasets found.')).toBeInTheDocument();
    expect(startButton()).toBeDisabled();
  });

  it('shows API errors and lets the user retry', async () => {
    let fail = true;
    installBackend({ 'GET /api/datasets': () => (fail ? json({ detail: 'database locked' }, 500) : json([makeDataset()])) });
    renderLab();
    expect(await screen.findByRole('alert')).toHaveTextContent('Could not load datasets: database locked');
    expect(startButton()).toBeDisabled();
    fail = false;
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
    await waitForDatasets();
    expect(startButton()).toBeEnabled();
  });

  it('shows a network failure when the backend is unreachable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    renderLab();
    expect(await screen.findByRole('alert')).toHaveTextContent('Could not reach the backend');
  });

  it('lets the user pick another dataset and trains on the selected one', async () => {
    const backend = withDatasets(
      [makeDataset(), makeDataset({ id: 'ds_second01', filename: 'sales.csv', rowCount: 1200, columnCount: 5 })],
      { 'POST /api/models/train': () => json(makeModel({ dataset_id: 'ds_second01' })) },
    );
    renderLab();
    await waitForDatasets();
    await userEvent.selectOptions(screen.getByLabelText('Dataset to train on'), 'ds_second01');
    expect(screen.getByTestId('dataset-info')).toHaveTextContent('1,200 rows · 5 columns');
    await userEvent.click(startButton());
    await screen.findByTestId('status-completed');
    expect(backend.calls.find((c) => c.key === 'POST /api/models/train')?.body.dataset_id).toBe('ds_second01');
  });

  it('blocks training for datasets that are too small or have no usable columns', async () => {
    withDatasets([makeDataset({ rowCount: 4 })]);
    renderLab();
    await waitForDatasets();
    expect(screen.getByText(/at least 10/i)).toBeInTheDocument();
    expect(startButton()).toBeDisabled();
  });

  it('blocks a dataset whose columns are all excluded', async () => {
    withDatasets([makeDataset({ columns: [column('id_col', false)] })]);
    renderLab();
    await waitForDatasets();
    expect(screen.getByText(/no usable columns/i)).toBeInTheDocument();
    expect(startButton()).toBeDisabled();
  });

  it('runs the preprocessing dry run and shows the real response', async () => {
    const backend = withDatasets([makeDataset()], {
      'POST /api/preprocess': () =>
        json({
          dataset_id: 'ds_a1b2c3d4',
          processed_dimensions: 11,
          feature_names: [],
          ready_for_training: true,
          rows_used: 250,
          dropped_columns: [{ name: 'student_id', reason: 'excluded by configuration' }],
          warnings: ["Column 'k' is constant (7.0); it will be reproduced as a constant."],
        }),
    });
    renderLab();
    await waitForDatasets();
    await userEvent.click(screen.getByRole('button', { name: /check preprocessing/i }));
    const out = await screen.findByTestId('preprocess-result');
    expect(out).toHaveTextContent('11 model input features from 250 rows · ready for training');
    expect(out).toHaveTextContent('student_id (excluded by configuration)');
    expect(out).toHaveTextContent('is constant');
    expect(backend.calls.find((c) => c.key === 'POST /api/preprocess')?.body.columns).toHaveLength(3);
  });
});

describe('ModelLab: configuration and validation', () => {
  it('starts with the backend defaults and explains the fixed architecture', async () => {
    withDatasets();
    renderLab();
    await waitForDatasets();
    expect(screen.getByLabelText('Epochs')).toHaveValue('50');
    expect(screen.getByLabelText('Learning rate')).toHaveValue('0.001');
    expect(screen.getByLabelText('Batch size')).toHaveValue('32');
    expect(screen.getByLabelText('Latent dimension')).toHaveValue('8');
    expect(screen.getByLabelText('Hidden layer 1')).toHaveValue('128');
    expect(screen.getByLabelText('Hidden layer 2')).toHaveValue('64');
    expect(screen.getByLabelText('Random seed (optional)')).toHaveValue('');
    expect(screen.getByText('two', { selector: 'strong' })).toBeInTheDocument();
    expect(screen.getByText(/variational autoencoder/i)).toBeInTheDocument();
    expect(screen.queryByLabelText(/activation/i)).not.toBeInTheDocument();
  });

  it.each([
    ['Epochs', '0', /Epochs must be between 1 and 500/],
    ['Epochs', '501', /Epochs must be between 1 and 500/],
    ['Epochs', '2.5', /whole number/],
    ['Learning rate', '0', /Learning rate must be greater than 0/],
    ['Batch size', '4', /Batch size must be between 8/],
    ['Hidden layer 1', '1', /Hidden layer 1 must be between 2 and 1,024/],
    ['Random seed (optional)', '-3', /Random seed must be between 0/],
  ])('rejects %s = %s without calling the backend', async (label, value, message) => {
    const backend = withDatasets();
    renderLab();
    await waitForDatasets();
    const input = screen.getByLabelText(label);
    await userEvent.clear(input);
    await userEvent.type(input, value);
    await userEvent.click(startButton());
    expect(await screen.findByText(message)).toBeInTheDocument();
    expect(input).toHaveAttribute('aria-invalid', 'true');
    expect(backend.count('POST /api/models/train')).toBe(0);
    expect(screen.getByTestId('status-ready')).toBeInTheDocument();
  });

  it('clears a field error as soon as the user edits the field', async () => {
    withDatasets();
    renderLab();
    await waitForDatasets();
    const epochs = screen.getByLabelText('Epochs');
    await userEvent.clear(epochs);
    await userEvent.click(startButton());
    expect(await screen.findByText('Epochs is required.')).toBeInTheDocument();
    await userEvent.type(epochs, '5');
    expect(screen.queryByText('Epochs is required.')).not.toBeInTheDocument();
  });
});

describe('ModelLab: training', () => {
  it('sends the real request, then shows the identifiers and values the backend returned', async () => {
    const model = makeModel({ model_id: 'vae_abcd1234', name: 'My real model' });
    const backend = withDatasets([makeDataset()], { 'POST /api/models/train': () => json(model) });
    const props = renderLab();
    await waitForDatasets();

    await userEvent.clear(screen.getByLabelText('Epochs'));
    await userEvent.type(screen.getByLabelText('Epochs'), '2');
    await userEvent.type(screen.getByLabelText('Random seed (optional)'), '5');
    await userEvent.type(screen.getByLabelText('Model name (optional)'), 'My real model');
    await userEvent.click(startButton());

    await screen.findByTestId('status-completed');
    expect(backend.calls.find((c) => c.key === 'POST /api/models/train')?.body).toEqual({
      dataset_id: 'ds_a1b2c3d4',
      epochs: 2,
      learning_rate: 0.001,
      batch_size: 32,
      latent_dim: 8,
      hidden_layers: [128, 64],
      beta_weight: 1,
      random_seed: 5,
      model_name: 'My real model',
    });

    const result = screen.getByTestId('training-result');
    expect(within(result).getByTestId('result-model-id')).toHaveTextContent('vae_abcd1234');
    expect(within(result).getByTestId('result-dataset-id')).toHaveTextContent('ds_a1b2c3d4');
    expect(result).toHaveTextContent('numpy-vae');
    expect(result).toHaveTextContent('2026-10-24 05:11:07 UTC');
    expect(result).toHaveTextContent('≈ 7s (server timestamps)');
    expect(result).toHaveTextContent('Total6.6525');
    expect(result).toHaveTextContent('Random seed5');
    expect(result).toHaveTextContent('Hidden layers128, 64');
    expect(props.onModelTrained).toHaveBeenCalledWith(model);
    expect(props.onRefreshModels).toHaveBeenCalled();
  });

  it('marks values the backend did not send as unavailable instead of inventing them', async () => {
    withDatasets([makeDataset()], {
      'POST /api/models/train': () =>
        json(makeModel({ final_loss: null, recon_loss: null, kl_loss: null, completed_at: null, created_at: null, params: null, model_type: null, loss_history: [] })),
    });
    renderLab();
    await waitForDatasets();
    await userEvent.click(startButton());
    const result = await screen.findByTestId('training-result');
    expect(result).toHaveTextContent('Totalunavailable');
    expect(result).toHaveTextContent('Completed (server)unavailable');
    expect(result).toHaveTextContent('Durationunavailable');
    expect(result).toHaveTextContent('Epochsunavailable');
    expect(result).toHaveTextContent('Model typeunavailable');
  });

  it('shows an honest waiting state (no fake progress) and prevents duplicate submissions', async () => {
    let release!: (r: Response) => void;
    const backend = withDatasets([makeDataset()], {
      'POST /api/models/train': () => new Promise<Response>((resolve) => (release = resolve)),
    });
    const props = renderLab();
    await waitForDatasets();

    await userEvent.click(startButton());
    const waiting = await screen.findByTestId('status-training');
    expect(waiting).toHaveTextContent(/no progress bar or epoch counter/i);
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    expect(startButton()).toBeDisabled();
    expect(startButton()).toHaveTextContent('Training…');
    expect(screen.getByLabelText('Epochs')).toBeDisabled();

    // a second submit (e.g. pressing Enter) must not send another request
    fireEvent.submit(screen.getByRole('form', { name: 'Training setup' }));
    await userEvent.click(startButton());
    expect(backend.count('POST /api/models/train')).toBe(1);

    await act(async () => release(json(makeModel())));
    await screen.findByTestId('status-completed');
    expect(backend.count('POST /api/models/train')).toBe(1);
    expect(props.onModelTrained).toHaveBeenCalledTimes(1);
    expect(startButton()).toBeEnabled();
  });

  it('shows backend validation errors (422) and does not report success', async () => {
    withDatasets([makeDataset()], {
      'POST /api/models/train': () =>
        json({ detail: [{ loc: ['body', 'hidden_layers'], msg: 'Value error, hidden_layers must list exactly two layer sizes' }] }, 422),
    });
    const props = renderLab();
    await waitForDatasets();
    await userEvent.click(startButton());
    const failed = await screen.findByTestId('status-failed');
    expect(failed).toHaveTextContent('hidden_layers: Value error, hidden_layers must list exactly two layer sizes');
    expect(screen.queryByTestId('training-result')).not.toBeInTheDocument();
    expect(screen.queryByTestId('status-completed')).not.toBeInTheDocument();
    expect(props.onModelTrained).not.toHaveBeenCalled();
    expect(props.onRefreshModels).toHaveBeenCalled(); // the server may have stored a failed run
    expect(startButton()).toBeEnabled();
  });

  it.each([
    [400, 'Data pipeline setup error: No usable columns selected'],
    [404, 'Dataset not found'],
    [422, 'Training diverged at epoch 3 (loss is not finite). Try a smaller learning rate.'],
    [500, 'Training failed because of an internal error.'],
  ])('shows a %i response from the server as a failure', async (status, detail) => {
    withDatasets([makeDataset()], { 'POST /api/models/train': () => json({ detail }, status) });
    const props = renderLab();
    await waitForDatasets();
    await userEvent.click(startButton());
    expect(await screen.findByTestId('status-failed')).toHaveTextContent(detail);
    expect(props.onModelTrained).not.toHaveBeenCalled();
  });

  it('handles a network failure during training and says the outcome is unknown', async () => {
    withDatasets([makeDataset()], { 'POST /api/models/train': () => Promise.reject(new TypeError('Failed to fetch')) });
    const props = renderLab();
    await waitForDatasets();
    await userEvent.click(startButton());
    const failed = await screen.findByTestId('status-failed');
    expect(failed).toHaveTextContent('Could not reach the backend');
    expect(failed).toHaveTextContent(/result is unknown/i);
    expect(props.onModelTrained).not.toHaveBeenCalled();
  });

  it('gives up after the configured timeout and tells the user to refresh', async () => {
    withDatasets([makeDataset()], {
      'POST /api/models/train': ({ signal }) =>
        new Promise<Response>((_, reject) => signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))),
    });
    const props = renderLab();
    await waitForDatasets();
    vi.useFakeTimers();
    fireEvent.click(startButton());
    expect(screen.getByTestId('status-training')).toBeInTheDocument();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(TRAIN_TIMEOUT_MS + 1000);
    });
    const failed = screen.getByTestId('status-failed');
    expect(failed).toHaveTextContent(/No response within \d+ seconds/);
    expect(failed).toHaveTextContent(/Refresh models/);
    expect(props.onModelTrained).not.toHaveBeenCalled();
  });

  it('rejects an unexpected response format', async () => {
    withDatasets([makeDataset()], { 'POST /api/models/train': () => json({ hello: 'world' }) });
    const props = renderLab();
    await waitForDatasets();
    await userEvent.click(startButton());
    expect(await screen.findByTestId('status-failed')).toHaveTextContent('without a model_id');
    expect(props.onModelTrained).not.toHaveBeenCalled();
  });

  it('does not treat a non-completed status as success', async () => {
    withDatasets([makeDataset()], { 'POST /api/models/train': () => json(makeModel({ status: 'failed' })) });
    const props = renderLab();
    await waitForDatasets();
    await userEvent.click(startButton());
    expect(await screen.findByTestId('status-failed')).toHaveTextContent('instead of "completed"');
    expect(props.onModelTrained).not.toHaveBeenCalled();
  });

  it('offers the trained model to the Data Generator', async () => {
    withDatasets([makeDataset()], { 'POST /api/models/train': () => json(makeModel({ model_id: 'vae_feed0001' })) });
    const props = renderLab();
    await waitForDatasets();
    await userEvent.click(startButton());
    await userEvent.click(await screen.findByRole('button', { name: /use this model in data generator/i }));
    expect(props.onSelectModel).toHaveBeenCalledWith('vae_feed0001');
    expect(props.onNavigateToGenerator).toHaveBeenCalled();
  });
});

describe('ModelLab: header and model list', () => {
  it('shows the backend connection state', async () => {
    withDatasets();
    renderLab({ backendOnline: false });
    expect(screen.getByTestId('backend-status')).toHaveTextContent(/unreachable/i);
  });

  it('lists the models from the backend and selects one', async () => {
    withDatasets();
    const done = apiModelToRecord(makeModel());
    const failed = apiModelToRecord(makeModel({ model_id: 'vae_bad00001', name: 'broken', status: 'failed', final_loss: null, loss_history: [], error_message: 'diverged' }));
    const props = renderLab({ models: [done, failed] });
    expect(await screen.findByTestId('model-row-vae_0f1e2d3c')).toHaveTextContent('completed');
    expect(screen.getByTestId('model-row-vae_bad00001')).toHaveTextContent('failed');
    expect(within(screen.getByTestId('model-row-vae_bad00001')).queryByRole('button')).not.toBeInTheDocument();
    await userEvent.click(within(screen.getByTestId('model-row-vae_0f1e2d3c')).getByRole('button', { name: 'Select' }));
    expect(props.onSelectModel).toHaveBeenCalledWith('vae_0f1e2d3c');
  });

  it('refreshes the model list on request and reports refresh failures', async () => {
    withDatasets();
    const props = renderLab({ onRefreshModels: vi.fn().mockResolvedValue('boom') });
    await userEvent.click(screen.getByRole('button', { name: /refresh models/i }));
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Could not refresh the model list: boom'));
    expect(props.onRefreshModels).toHaveBeenCalled();
  });
});


describe('ModelLab: leaving the page while a training request is running', () => {
  it('still blocks a second training after coming back, and refreshes the model list when the first one ends', async () => {
    let release!: (r: Response) => void;
    const backend = withDatasets([makeDataset()], { 'POST /api/models/train': () => new Promise<Response>((resolve) => (release = resolve)) });
    const first = renderLab();
    await waitForDatasets();
    await userEvent.click(startButton());
    await screen.findByTestId('status-training');

    cleanup(); // the user navigates to another page; the request keeps running on the server
    const second = renderLab();
    await waitForDatasets();
    expect(startButton()).toBeDisabled();
    expect(screen.getByTestId('status-training-elsewhere')).toHaveTextContent(/still running/i);
    fireEvent.submit(screen.getByRole('form', { name: 'Training setup' }));
    expect(backend.count('POST /api/models/train')).toBe(1);

    await act(async () => release(json(makeModel())));
    await waitFor(() => expect(startButton()).toBeEnabled());
    expect(first.onModelTrained).toHaveBeenCalledTimes(1); // the original page still records the result
    await waitFor(() => expect(second.onRefreshModels).toHaveBeenCalled());
    expect(backend.count('POST /api/models/train')).toBe(1);
  });
});
