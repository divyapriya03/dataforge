import React from 'react';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { QualityComparison } from './QualityComparison';
import { csvResponse, installBackend, json, makeEvaluation, makeExperiment, type Handler } from '../test/backend';
import { captureDownloads, readBlob } from '../test/downloads';

afterEach(() => vi.unstubAllGlobals());

let saved: ReturnType<typeof captureDownloads>;
beforeEach(() => {
  saved = captureDownloads();
});

const A = makeExperiment();
const B = makeExperiment({ id: 'gen_bbbb2222', dataset_name: 'sales.csv', dataset_id: 'ds_second01', num_records: 900, fidelity_score: 71.2 });

const rich = (id: string, fidelity: number) =>
  makeEvaluation({
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

type Props = React.ComponentProps<typeof QualityComparison>;
function renderQuality(props: Partial<Props> = {}, routes: Record<string, Handler> = {}) {
  const backend = installBackend({
    'GET /api/evaluate/gen_aaaa1111': () => json(rich('gen_aaaa1111', 86.6)),
    'GET /api/evaluate/gen_bbbb2222': () => json(rich('gen_bbbb2222', 71.2)),
    ...routes,
  });
  const p: Props = {
    experiments: [A, B],
    experimentsState: { phase: 'ready' },
    selectedExperimentId: 'gen_aaaa1111',
    onSelectExperiment: vi.fn(),
    onReloadExperiments: vi.fn(),
    onNavigateToGenerator: vi.fn(),
    onNavigateToHistory: vi.fn(),
    ...props,
  };
  const view = render(<QualityComparison {...p} />);
  return { p, backend, ...view };
}

describe('QualityComparison', () => {
  it('loads and shows the saved evaluation of the selected experiment only', async () => {
    const { backend } = renderQuality();
    expect(await screen.findByTestId('eval-fidelity')).toHaveTextContent('86.6%');
    const info = screen.getByTestId('qc-experiment-info');
    expect(info).toHaveTextContent('gen_aaaa1111');
    expect(info).toHaveTextContent('students.csv');
    expect(info).toHaveTextContent('VAE-students-Z8');
    expect(backend.count('GET /api/evaluate/gen_aaaa1111')).toBe(1);
    expect(backend.count('GET /api/evaluate/gen_bbbb2222')).toBe(0);
    expect(screen.getByTestId('eval-avg-ks')).toHaveTextContent('0.0792');
    expect(screen.getByTestId('ks-row-study_hours')).toHaveTextContent('0.0652');
    expect(screen.getByTestId('tvd-row-gender')).toHaveTextContent('0.0200');
  });

  it('shows real-vs-synthetic summaries and category proportions from the stored evaluation', async () => {
    renderQuality();
    await screen.findByTestId('eval-fidelity');
    expect(screen.getByTestId('qc-summary-study_hours')).toHaveTextContent('15.34 / 15.10');
    expect(screen.getByTestId('qc-summary-study_hours')).toHaveTextContent('5.10 / 4.80');
    expect(screen.getByTestId('qc-proportions-gender')).toHaveTextContent('gender');
    expect(screen.getByTestId('qc-interpretation')).toHaveTextContent(/says nothing about privacy/);
  });

  it('contains none of the old hardcoded placeholder numbers', async () => {
    renderQuality();
    await screen.findByTestId('eval-fidelity');
    expect(document.body.textContent).not.toMatch(/93\.8|99\.8|Frobenius Norm: 0\.041/);
  });

  it('reports stored values that are missing instead of showing zeros', async () => {
    renderQuality(
      {},
      {
        'GET /api/evaluate/gen_aaaa1111': () =>
          json(
            makeEvaluation({
              average_ks_statistic: null,
              correlation_frobenius_delta: null,
              correlation_mean_abs_delta: null,
              categorical_metrics: {},
              summary_comparisons: [],
              ks_metrics: { broken: { ks_statistic: null, p_value: null, wasserstein_distance: null, status: 'Failed' } },
            }),
          ),
      },
    );
    await screen.findByTestId('eval-fidelity');
    expect(screen.getByTestId('eval-avg-ks')).toHaveTextContent('not available');
    expect(screen.getByTestId('ks-row-broken')).not.toHaveTextContent('0.0000');
    expect(screen.getByTestId('qc-no-summary')).toBeInTheDocument();
    expect(screen.getByTestId('qc-no-proportions')).toBeInTheDocument();
  });

  it('explains when the experiment has no saved evaluation', async () => {
    renderQuality({}, { 'GET /api/evaluate/gen_aaaa1111': () => json({ detail: 'Experiment not found' }, 404) });
    expect(await screen.findByTestId('qc-no-evaluation')).toHaveTextContent('no saved evaluation result');
    expect(screen.queryByTestId('eval-fidelity')).not.toBeInTheDocument();
  });

  it('shows evaluation errors with retry', async () => {
    let fail = true;
    const { backend } = renderQuality({}, { 'GET /api/evaluate/gen_aaaa1111': () => (fail ? json({ detail: 'boom' }, 500) : json(rich('gen_aaaa1111', 86.6))) });
    expect(await screen.findByRole('alert')).toHaveTextContent('Could not load the evaluation: boom');
    fail = false;
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(await screen.findByTestId('eval-fidelity')).toBeInTheDocument();
    expect(backend.count('GET /api/evaluate/gen_aaaa1111')).toBe(2);
  });

  it('refuses an evaluation that belongs to a different experiment', async () => {
    renderQuality({}, { 'GET /api/evaluate/gen_aaaa1111': () => json(rich('gen_zzzz9999', 12.3)) });
    expect(await screen.findByRole('alert')).toHaveTextContent('different experiment');
    expect(screen.queryByTestId('eval-fidelity')).not.toBeInTheDocument();
  });

  it('switches experiments through the parent and loads that experiment', async () => {
    const { p, backend, rerender } = renderQuality();
    await screen.findByTestId('eval-fidelity');
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Experiment' }), 'gen_bbbb2222');
    expect(p.onSelectExperiment).toHaveBeenCalledWith('gen_bbbb2222');
    rerender(<QualityComparison {...p} selectedExperimentId="gen_bbbb2222" />);
    await waitFor(() => expect(screen.getByTestId('eval-fidelity')).toHaveTextContent('71.2%'));
    expect(screen.getByTestId('qc-experiment-info')).toHaveTextContent('sales.csv');
    expect(backend.count('GET /api/evaluate/gen_bbbb2222')).toBe(1);
  });

  it('falls back to the newest experiment when the selected one does not exist', async () => {
    const { backend } = renderQuality({ selectedExperimentId: 'gen_gone0000' });
    expect(await screen.findByTestId('eval-fidelity')).toBeInTheDocument();
    expect(backend.count('GET /api/evaluate/gen_aaaa1111')).toBe(1);
  });

  it('shows loading, empty and list-error states', async () => {
    const retry = vi.fn();
    const { p, rerender } = renderQuality({ experiments: [], experimentsState: { phase: 'loading' } });
    expect(screen.getByText(/loading experiments/i)).toBeInTheDocument();
    rerender(<QualityComparison {...p} experimentsState={{ phase: 'ready' }} />);
    const empty = screen.getByTestId('no-experiments');
    await userEvent.click(within(empty).getByRole('button', { name: /data generator/i }));
    expect(p.onNavigateToGenerator).toHaveBeenCalled();
    rerender(<QualityComparison {...p} onReloadExperiments={retry} experimentsState={{ phase: 'error', message: 'db locked' }} />);
    expect(screen.getByRole('alert')).toHaveTextContent('Could not load experiments: db locked');
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(retry).toHaveBeenCalled();
  });

  it('downloads the CSV of the displayed experiment only', async () => {
    const { backend } = renderQuality({}, { 'GET /api/export/gen_aaaa1111': () => csvResponse('marker\ngen_aaaa1111\n') });
    await screen.findByTestId('eval-fidelity');
    await userEvent.click(screen.getByRole('button', { name: /download this csv/i }));
    await waitFor(() => expect(saved).toHaveLength(1));
    expect(backend.count('GET /api/export/gen_aaaa1111')).toBe(1);
    expect(backend.calls.filter((c) => c.key.startsWith('GET /api/export/')).length).toBe(1);
    expect(saved[0].filename).toBe('dataforge_experiment_gen_aaaa1111.csv');
    expect(await readBlob(saved[0].blob)).toBe('marker\ngen_aaaa1111\n');
  });

  it('disables the download when the CSV is no longer on the server', async () => {
    renderQuality({ experiments: [makeExperiment({ status: 'output_missing', output_available: false })] });
    await screen.findByTestId('eval-fidelity'); // the stored evaluation is still available
    expect(screen.getByRole('button', { name: /download this csv/i })).toBeDisabled();
    expect(screen.getByTestId('qc-experiment-info')).toHaveTextContent('Output file missing');
  });
});
