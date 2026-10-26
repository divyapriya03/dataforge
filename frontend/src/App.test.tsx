import React from 'react';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import { buildHash, parseHash } from './routing';
import { healthOk, installBackend, json, makeDataset, makeEvaluation, makeExperiment, makeModel, type Handler } from './test/backend';

afterEach(() => vi.unstubAllGlobals());

const backend = (extra: Record<string, Handler> = {}) =>
  installBackend({
    'GET /api/health': () => healthOk(),
    'GET /api/models': () => json([makeModel()]),
    'GET /api/datasets': () => json([makeDataset()]),
    'GET /api/experiments': () => json([]),
    ...extra,
  });

const A = makeExperiment();
const B = makeExperiment({ id: 'gen_bbbb2222', dataset_name: 'sales.csv', fidelity_score: 71.2, created_at: '2026-10-23 01:00:00' });
const withExperiments = (extra: Record<string, Handler> = {}) =>
  backend({
    'GET /api/experiments': () => json([A, B]),
    'GET /api/evaluate/gen_aaaa1111': () => json(makeEvaluation()),
    'GET /api/evaluate/gen_bbbb2222': () => json(makeEvaluation({ experiment_id: 'gen_bbbb2222', overall_fidelity_score: 71.2 })),
    'GET /api/models/status/vae_0f1e2d3c': () => json(makeModel()),
    ...extra,
  });

describe('route helpers', () => {
  it('parses and builds hashes', () => {
    expect(parseHash('')).toEqual({ tab: 'dashboard', params: {} });
    expect(parseHash('#/model-lab?dataset=ds_1')).toEqual({ tab: 'model', params: { dataset: 'ds_1' } });
    expect(parseHash('#/quality/?experiment=gen_1&x=')).toEqual({ tab: 'quality', params: { experiment: 'gen_1' } });
    expect(parseHash('#/nope')).toEqual({ tab: 'notfound', params: {} });
    expect(buildHash('generator', { model: 'vae_1' })).toBe('#/generator?model=vae_1');
    expect(buildHash('history', { model: undefined })).toBe('#/history');
  });
});

describe('navigation', () => {
  it('opens every page from the sidebar and highlights the active one', async () => {
    backend();
    render(<App />);
    const pages: [RegExp, string, RegExp][] = [
      [/2\. Dataset Studio/, '#/dataset-studio', /Dataset Studio$/],
      [/3\. Model Lab/, '#/model-lab', /^ModelLab$/],
      [/4\. Data Generator/, '#/generator', /^Data Generator$/],
      [/5\. Quality Compare/, '#/quality', /^Quality Comparison$/],
      [/6\. Export & History/, '#/history', /^Export History$/],
      [/1\. Dashboard/, '#/dashboard', /DataForge: Tabular Synthetic Data Generation/],
    ];
    for (const [button, hash, heading] of pages) {
      await userEvent.click(screen.getByRole('button', { name: button }));
      expect(await screen.findByRole('heading', { name: heading })).toBeInTheDocument();
      expect(window.location.hash.startsWith(hash)).toBe(true);
      expect(screen.getByRole('button', { name: button })).toHaveClass('bg-indigo-600');
    }
  });

  it.each([
    ['#/dataset-studio', /Dataset Studio$/],
    ['#/model-lab', /^ModelLab$/],
    ['#/generator', /^Data Generator$/],
    ['#/quality', /^Quality Comparison$/],
    ['#/history', /^Export History$/],
  ] as [string, RegExp][])('supports direct navigation to %s', async (hash, heading) => {
    backend();
    window.location.hash = hash;
    render(<App />);
    expect(await screen.findByRole('heading', { name: heading })).toBeInTheDocument();
  });

  it('shows a not-found page for unknown routes and recovers', async () => {
    backend();
    window.location.hash = '#/does-not-exist';
    render(<App />);
    expect(await screen.findByTestId('not-found')).toHaveTextContent('Page not found');
    for (const name of [/1\. Dashboard/, /3\. Model Lab/]) expect(screen.getByRole('button', { name })).not.toHaveClass('bg-indigo-600');
    await userEvent.click(screen.getByRole('button', { name: /go to the dashboard/i }));
    expect(screen.queryByTestId('not-found')).not.toBeInTheDocument();
    expect(window.location.hash).toBe('#/dashboard');
  });

  it('follows browser hash changes', async () => {
    backend();
    window.location.hash = '#/model-lab';
    render(<App />);
    await screen.findByRole('heading', { name: 'ModelLab' });
    window.location.hash = '#/history';
    expect(await screen.findByRole('heading', { name: 'Export History' })).toBeInTheDocument();
  });

  it('shows the backend connection state', async () => {
    backend();
    window.location.hash = '#/model-lab';
    render(<App />);
    await waitFor(() => expect(screen.getByTestId('backend-status')).toHaveTextContent('Backend connected'));
  });
});

describe('identifiers travel with the navigation', () => {
  it('opens the experiment named in the URL', async () => {
    const api = withExperiments();
    window.location.hash = '#/quality?experiment=gen_bbbb2222';
    render(<App />);
    expect(await screen.findByTestId('eval-fidelity')).toHaveTextContent('71.2%');
    expect(screen.getByRole('combobox', { name: 'Experiment' })).toHaveValue('gen_bbbb2222');
    expect(screen.getByTestId('qc-experiment-info')).toHaveTextContent('gen_bbbb2222');
    expect(api.count('GET /api/evaluate/gen_aaaa1111')).toBe(0);
  });

  it('writes the selection back into the URL', async () => {
    withExperiments();
    window.location.hash = '#/quality';
    render(<App />);
    await screen.findByTestId('eval-fidelity');
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Experiment' }), 'gen_bbbb2222');
    expect(window.location.hash).toBe('#/quality?experiment=gen_bbbb2222');
    await waitFor(() => expect(screen.getByTestId('eval-fidelity')).toHaveTextContent('71.2%'));
  });

  it('opens the generator with the model named in the URL', async () => {
    backend({ 'GET /api/models': () => json([makeModel(), makeModel({ model_id: 'vae_beef0002', name: 'Second model' })]) });
    window.location.hash = '#/generator?model=vae_beef0002';
    render(<App />);
    await waitFor(() => expect(screen.getByLabelText('Trained model')).toHaveValue('vae_beef0002'));
  });

  it('opens Model Lab with the dataset named in the URL', async () => {
    backend({ 'GET /api/datasets': () => json([makeDataset(), makeDataset({ id: 'ds_second01', filename: 'sales.csv' })]) });
    window.location.hash = '#/model-lab?dataset=ds_second01';
    render(<App />);
    await waitFor(() => expect(screen.getByLabelText('Dataset to train on')).toHaveValue('ds_second01'));
  });

  it('keeps the current experiment when moving between pages through the sidebar', async () => {
    withExperiments();
    window.location.hash = '#/quality?experiment=gen_bbbb2222';
    render(<App />);
    await screen.findByTestId('eval-fidelity');
    await userEvent.click(screen.getByRole('button', { name: /6\. Export & History/ }));
    await screen.findByRole('heading', { name: 'Export History' });
    await userEvent.click(screen.getByRole('button', { name: /5\. Quality Compare/ }));
    expect(window.location.hash).toBe('#/quality?experiment=gen_bbbb2222');
  });
});

describe('models and experiments come from the backend', () => {
  it('lists backend models in the generator and no mock models', async () => {
    backend({ 'GET /api/models': () => json([makeModel(), makeModel({ model_id: 'vae_bad00001', name: 'broken', status: 'failed' })]) });
    window.location.hash = '#/generator';
    render(<App />);
    expect(await screen.findByRole('option', { name: /VAE-students-Z8 \(vae_0f1e2d3c\)/ })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: /broken/ })).not.toBeInTheDocument();
    expect(screen.queryByText(/Student-Z8/i)).not.toBeInTheDocument();
  });

  it('shows empty states instead of placeholder data', async () => {
    backend({ 'GET /api/models': () => json([]), 'GET /api/datasets': () => json([]) });
    window.location.hash = '#/dashboard';
    render(<App />);
    expect(await screen.findByTestId('dashboard-no-runs')).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/93\.5|99\.8|student_performance/);
  });

  it('shows real experiment figures on the dashboard', async () => {
    withExperiments();
    window.location.hash = '#/dashboard';
    render(<App />);
    expect(await screen.findByText('gen_aaaa1111')).toBeInTheDocument();
    expect(screen.getByText('400')).toBeInTheDocument(); // 200 + 200 records
    expect(screen.getByText('78.9%')).toBeInTheDocument(); // mean of 86.6 and 71.2
  });

  it('surfaces backend outages instead of failing silently', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    window.location.hash = '#/quality';
    render(<App />);
    expect(await screen.findByRole('alert')).toHaveTextContent('Could not load experiments');
  });
});

describe('Export History links', () => {
  it('opens the selected experiment in Quality Comparison', async () => {
    withExperiments();
    window.location.hash = '#/history';
    render(<App />);
    await userEvent.click(within(await screen.findByTestId('experiment-row-gen_bbbb2222')).getByRole('button', { name: /view details/i }));
    await userEvent.click(screen.getByRole('button', { name: /open in quality comparison/i }));
    expect(window.location.hash).toBe('#/quality?experiment=gen_bbbb2222');
    await waitFor(() => expect(screen.getByTestId('eval-fidelity')).toHaveTextContent('71.2%'));
  });

  it('opens the generator with the model of the experiment selected', async () => {
    withExperiments();
    window.location.hash = '#/history';
    render(<App />);
    await userEvent.click(within(await screen.findByTestId('experiment-row-gen_aaaa1111')).getByRole('button', { name: /view details/i }));
    await waitFor(() => expect(screen.getByTestId('model-config')).toHaveTextContent('Epochs'));
    await userEvent.click(screen.getByRole('button', { name: /generate again/i }));
    expect(window.location.hash).toBe('#/generator?model=vae_0f1e2d3c');
    await waitFor(() => expect(screen.getByLabelText('Trained model')).toHaveValue('vae_0f1e2d3c'));
  });

  it('links to Model Lab and back through the sidebar', async () => {
    withExperiments();
    window.location.hash = '#/history';
    render(<App />);
    await userEvent.click(within(await screen.findByTestId('experiment-row-gen_aaaa1111')).getByRole('button', { name: /view details/i }));
    await userEvent.click(screen.getByRole('button', { name: /^Model Lab$/ }));
    expect(await screen.findByRole('heading', { name: 'ModelLab' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /6\. Export & History/ }));
    expect(await screen.findByRole('heading', { name: 'Export History' })).toBeInTheDocument();
  });
});
