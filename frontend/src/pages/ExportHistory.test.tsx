import React from 'react';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ExportHistory } from './ExportHistory';
import {
  csvResponse,
  installBackend,
  json,
  makeEvaluation,
  makeExperiment,
  makeModel,
  type Handler,
} from '../test/backend';

afterEach(() => vi.unstubAllGlobals());

// ---- browser download capture: records what the page hands to the browser
let saved: { filename: string; blob: Blob }[] = [];
beforeEach(() => {
  saved = [];
  const blobs = new Map<string, Blob>();
  let n = 0;
  (URL as unknown as { createObjectURL: (b: Blob) => string }).createObjectURL = (b: Blob) => {
    const url = `blob:test/${n++}`;
    blobs.set(url, b);
    return url;
  };
  (URL as unknown as { revokeObjectURL: (u: string) => void }).revokeObjectURL = () => {};
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    saved.push({ filename: this.download, blob: blobs.get(this.href)! });
  });
});

const A = makeExperiment();
const B = makeExperiment({
  id: 'gen_bbbb2222',
  dataset_id: 'ds_second01',
  dataset_name: 'sales.csv',
  model_id: 'vae_beef0002',
  model_name: 'VAE-sales-Z4',
  num_records: 1500,
  fidelity_score: 71.2,
  validity_score: 98.4,
  created_at: '2026-10-24 06:30:00',
  output_filename: 'synthetic_sales_gen_bbbb2222.csv',
});
const GONE = makeExperiment({
  id: 'gen_cccc3333',
  status: 'output_missing',
  output_available: false,
  created_at: '2026-10-23 01:00:00',
  output_filename: 'synthetic_students_gen_cccc3333.csv',
});

type Props = React.ComponentProps<typeof ExportHistory>;
function renderHistory(props: Partial<Props> = {}) {
  const p: Props = {
    onExperimentsLoaded: vi.fn(),
    onNavigateToModelLab: vi.fn(),
    onNavigateToGenerator: vi.fn(),
    onOpenQuality: vi.fn(),
    ...props,
  };
  render(<ExportHistory {...p} />);
  return p;
}

const withHistory = (list: unknown[] = [A, B, GONE], extra: Record<string, Handler> = {}) =>
  installBackend({
    'GET /api/experiments': () => json(list),
    'GET /api/evaluate/gen_aaaa1111': () => json(makeEvaluation()),
    'GET /api/models/status/vae_0f1e2d3c': () => json(makeModel()),
    ...extra,
  });

const row = (id: string) => screen.findByTestId(`experiment-row-${id}`);
const readBlob = (b: Blob) =>
  new Promise<string>((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = reject;
    r.readAsText(b);
  });

describe('ExportHistory: list', () => {
  it('loads experiments from the backend and shows only what the API returned', async () => {
    withHistory();
    const props = renderHistory();
    expect(screen.getByText(/loading experiments/i)).toBeInTheDocument();
    const a = await row('gen_aaaa1111');
    expect(a).toHaveTextContent('students.csv');
    expect(a).toHaveTextContent('ds_a1b2c3d4');
    expect(a).toHaveTextContent('VAE-students-Z8');
    expect(a).toHaveTextContent('vae_0f1e2d3c');
    expect(a).toHaveTextContent('200'); // records
    expect(a).toHaveTextContent('86.6%'); // fidelity
    expect(a).toHaveTextContent('100.0%'); // validity
    expect(a).toHaveTextContent('2026-10-24 05:20:00');
    expect(screen.getByTestId('result-count')).toHaveTextContent('Showing 3 of 3 experiments');
    expect(props.onExperimentsLoaded).toHaveBeenCalledWith([A, B, GONE]);
    // backend order (newest first) is preserved
    const ids = screen.getAllByTestId(/experiment-row-/).map((r) => r.getAttribute('data-testid'));
    expect(ids).toEqual(['experiment-row-gen_aaaa1111', 'experiment-row-gen_bbbb2222', 'experiment-row-gen_cccc3333']);
  });

  it('shows an informative empty state with working links', async () => {
    withHistory([]);
    const props = renderHistory();
    const empty = await screen.findByTestId('empty-history');
    expect(empty).toHaveTextContent('No experiments yet.');
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
    await userEvent.click(within(empty).getByRole('button', { name: /train a model/i }));
    await userEvent.click(within(empty).getByRole('button', { name: /data generator/i }));
    expect(props.onNavigateToModelLab).toHaveBeenCalled();
    expect(props.onNavigateToGenerator).toHaveBeenCalledWith();
  });

  it('shows an error with retry when the list cannot be loaded', async () => {
    let fail = true;
    installBackend({ 'GET /api/experiments': () => (fail ? json({ detail: 'database is locked' }, 500) : json([A])) });
    renderHistory();
    expect(await screen.findByRole('alert')).toHaveTextContent('Could not load experiments: database is locked');
    fail = false;
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
    await row('gen_aaaa1111');
  });

  it('reports an unreachable backend', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    renderHistory();
    expect(await screen.findByRole('alert')).toHaveTextContent('Could not reach the backend');
  });

  it('refreshes the list on request', async () => {
    const backend = withHistory([A]);
    renderHistory();
    await row('gen_aaaa1111');
    await userEvent.click(screen.getByRole('button', { name: /refresh/i }));
    await waitFor(() => expect(backend.count('GET /api/experiments')).toBe(2));
  });

  it('shows statuses and marks experiments whose CSV is gone', async () => {
    withHistory();
    renderHistory();
    expect(await row('gen_aaaa1111')).toHaveTextContent('Completed');
    const gone = await row('gen_cccc3333');
    expect(gone).toHaveTextContent('Output file missing');
    expect(within(gone).getByRole('button', { name: /download csv for gen_cccc3333/i })).toBeDisabled();
    expect(within(await row('gen_aaaa1111')).getByRole('button', { name: /download csv for gen_aaaa1111/i })).toBeEnabled();
  });

  it('displays missing fields as unavailable instead of inventing them', async () => {
    withHistory([makeExperiment({ id: 'gen_nulls', dataset_name: null, model_name: null, fidelity_score: null, validity_score: null, created_at: '' })]);
    renderHistory();
    const r = await row('gen_nulls');
    expect(r).toHaveTextContent('Dataset record removed');
    expect(r).toHaveTextContent('Model record removed');
    expect(r.textContent).toContain('not available');
    expect(r).not.toHaveTextContent('0.0%');
  });
});

describe('ExportHistory: filters and search', () => {
  it('searches by experiment id, dataset name and model name', async () => {
    withHistory();
    renderHistory();
    await row('gen_aaaa1111');
    const search = screen.getByLabelText('Search experiments');

    await userEvent.type(search, 'sales');
    expect(screen.queryByTestId('experiment-row-gen_aaaa1111')).not.toBeInTheDocument();
    expect(screen.getByTestId('experiment-row-gen_bbbb2222')).toBeInTheDocument();
    expect(screen.getByTestId('result-count')).toHaveTextContent('Showing 1 of 3');

    await userEvent.clear(search);
    await userEvent.type(search, 'GEN_CCCC');
    expect(screen.getByTestId('experiment-row-gen_cccc3333')).toBeInTheDocument();
    expect(screen.getByTestId('result-count')).toHaveTextContent('Showing 1 of 3');

    await userEvent.clear(search);
    await userEvent.type(search, 'vae_beef0002');
    expect(screen.getByTestId('experiment-row-gen_bbbb2222')).toBeInTheDocument();
  });

  it('filters by status', async () => {
    withHistory();
    renderHistory();
    await row('gen_aaaa1111');
    const status = screen.getByLabelText('Status');
    expect(within(status).getAllByRole('option').map((o) => o.textContent)).toEqual(['All statuses', 'Completed', 'Output file missing']);
    await userEvent.selectOptions(status, 'output_missing');
    expect(screen.getAllByTestId(/experiment-row-/)).toHaveLength(1);
    expect(screen.getByTestId('experiment-row-gen_cccc3333')).toBeInTheDocument();
  });

  it('sorts by creation time, fidelity and record count', async () => {
    withHistory();
    renderHistory();
    await row('gen_aaaa1111');
    const order = () => screen.getAllByTestId(/experiment-row-/).map((r) => r.getAttribute('data-testid')!.replace('experiment-row-', ''));
    const sort = screen.getByLabelText('Sort by');
    await userEvent.selectOptions(sort, 'oldest');
    expect(order()).toEqual(['gen_cccc3333', 'gen_bbbb2222', 'gen_aaaa1111']);
    await userEvent.selectOptions(sort, 'fidelity');
    expect(order()).toEqual(['gen_aaaa1111', 'gen_cccc3333', 'gen_bbbb2222']);
    await userEvent.selectOptions(sort, 'records');
    expect(order()[0]).toBe('gen_bbbb2222');
    await userEvent.selectOptions(sort, 'newest');
    expect(order()).toEqual(['gen_aaaa1111', 'gen_bbbb2222', 'gen_cccc3333']);
  });

  it('shows a no-match state and clears the filters', async () => {
    withHistory();
    renderHistory();
    await row('gen_aaaa1111');
    await userEvent.type(screen.getByLabelText('Search experiments'), 'zzz-nothing');
    const none = screen.getByTestId('no-matches');
    expect(none).toHaveTextContent('No experiments match');
    await userEvent.click(within(none).getByRole('button', { name: /clear filters/i }));
    expect(screen.getAllByTestId(/experiment-row-/)).toHaveLength(3);
  });
});

describe('ExportHistory: details and evaluation', () => {
  it('loads details for the selected experiment from the real endpoints', async () => {
    const backend = withHistory();
    renderHistory();
    await userEvent.click(within(await row('gen_aaaa1111')).getByRole('button', { name: /view details of gen_aaaa1111/i }));

    const panel = screen.getByTestId('details-panel');
    expect(within(panel).getByTestId('detail-experiment-id')).toHaveTextContent('gen_aaaa1111');
    expect(panel).toHaveTextContent('students.csv (ds_a1b2c3d4)');
    expect(panel).toHaveTextContent('VAE-students-Z8 (vae_0f1e2d3c)');
    expect(panel).toHaveTextContent('numpy-vae');
    expect(panel).toHaveTextContent('synthetic_students_gen_aaaa1111.csv · available');
    await waitFor(() => expect(within(panel).getByTestId('model-config')).toHaveTextContent('Hidden layers128, 64'));
    expect(within(panel).getByTestId('model-config')).toHaveTextContent('Training seed5');
    expect(within(panel).getByTestId('model-config')).toHaveTextContent('2026-10-24 05:11:07');

    const ev = await screen.findByTestId('eval-fidelity');
    expect(ev).toHaveTextContent('86.6%');
    expect(screen.getByTestId('eval-validity')).toHaveTextContent('100.0%');
    expect(screen.getByTestId('eval-avg-ks')).toHaveTextContent('0.0792');
    expect(screen.getByTestId('eval-corr-mad')).toHaveTextContent('0.0956');
    expect(screen.getByTestId('eval-corr-frob')).toHaveTextContent('0.1186');
    expect(screen.getByTestId('ks-row-study_hours')).toHaveTextContent('0.0652');
    expect(screen.getByTestId('tvd-row-gender')).toHaveTextContent('0.0080');
    expect(screen.getByTestId('eval-limitations')).toHaveTextContent('no privacy guarantee');
    expect(backend.count('GET /api/evaluate/gen_aaaa1111')).toBe(1);
    expect(backend.count('GET /api/models/status/vae_0f1e2d3c')).toBe(1);
    expect(backend.count('GET /api/evaluate/gen_bbbb2222')).toBe(0); // only the selected experiment is fetched
  });

  it('distinguishes a real zero from a missing metric and never charts missing values', async () => {
    withHistory([A], {
      'GET /api/evaluate/gen_aaaa1111': () =>
        json(
          makeEvaluation({
            average_ks_statistic: null,
            correlation_frobenius_delta: null,
            correlation_mean_abs_delta: null,
            ks_metrics: {
              exact: { ks_statistic: 0, p_value: 1, wasserstein_distance: 0, status: 'Passed' },
              broken: { ks_statistic: null, p_value: null, wasserstein_distance: null, status: 'Failed' },
            },
            categorical_metrics: {},
            fidelity_components: { numeric: 50, categorical: null, correlation: null, validity: 100 },
          }),
        ),
    });
    renderHistory();
    await userEvent.click(within(await row('gen_aaaa1111')).getByRole('button', { name: /view details/i }));
    await screen.findByTestId('eval-fidelity');

    expect(screen.getByTestId('ks-row-exact')).toHaveTextContent('0.0000'); // a real zero is shown as a number
    const missing = screen.getByTestId('ks-row-broken');
    expect(missing).toHaveTextContent('not available');
    expect(missing).not.toHaveTextContent('0.0000');
    expect(screen.getByTestId('eval-avg-ks')).toHaveTextContent('not available');
    expect(screen.getByTestId('eval-corr-frob')).toHaveTextContent('not available');
    expect(screen.getByText(/correlations were not computed/i)).toBeInTheDocument();
    expect(screen.getByTestId('no-categorical')).toBeInTheDocument();
    expect(screen.getByTestId('chart-omitted')).toHaveTextContent('1 column(s) without a stored value are not charted.');
    expect(screen.getAllByText('not computed').length).toBe(2);
  });

  it('explains when an experiment has no saved evaluation', async () => {
    withHistory([A], { 'GET /api/evaluate/gen_aaaa1111': () => json({ detail: 'Experiment not found' }, 404) });
    renderHistory();
    await userEvent.click(within(await row('gen_aaaa1111')).getByRole('button', { name: /view details/i }));
    expect(await screen.findByTestId('evaluation-unavailable')).toHaveTextContent('no saved evaluation result');
    expect(screen.queryByTestId('eval-fidelity')).not.toBeInTheDocument();
  });

  it('shows evaluation errors with retry', async () => {
    let fail = true;
    withHistory([A], { 'GET /api/evaluate/gen_aaaa1111': () => (fail ? json({ detail: 'boom' }, 500) : json(makeEvaluation())) });
    renderHistory();
    await userEvent.click(within(await row('gen_aaaa1111')).getByRole('button', { name: /view details/i }));
    expect(await screen.findByText(/Could not load the evaluation: boom/)).toBeInTheDocument();
    fail = false;
    await userEvent.click(within(screen.getByTestId('evaluation')).getByRole('button', { name: 'Retry' }));
    expect(await screen.findByTestId('eval-fidelity')).toBeInTheDocument();
  });

  it('copes with a removed model record and a removed output file', async () => {
    withHistory([GONE], {
      'GET /api/evaluate/gen_cccc3333': () => json(makeEvaluation({ experiment_id: 'gen_cccc3333' })),
      'GET /api/models/status/vae_0f1e2d3c': () => json({ detail: 'Model not found' }, 404),
    });
    const props = renderHistory();
    await userEvent.click(within(await row('gen_cccc3333')).getByRole('button', { name: /view details/i }));
    const panel = screen.getByTestId('details-panel');
    expect(await within(panel).findByTestId('output-missing')).toHaveTextContent('no longer on the server');
    expect(within(panel).getByRole('button', { name: 'Download generated CSV' })).toBeDisabled();
    expect(await within(panel).findByText(/model record for this experiment is no longer available/i)).toBeInTheDocument();
    expect(await screen.findByTestId('eval-fidelity')).toBeInTheDocument(); // the stored evaluation is still shown
    await userEvent.click(within(panel).getByRole('button', { name: /generate again/i }));
    expect(props.onNavigateToGenerator).toHaveBeenCalledWith(undefined); // no model to preselect
  });

  it('never prints internal filesystem paths', async () => {
    withHistory();
    renderHistory();
    await userEvent.click(within(await row('gen_aaaa1111')).getByRole('button', { name: /view details/i }));
    await screen.findByTestId('eval-fidelity');
    expect(document.body.textContent).not.toMatch(/[A-Z]:\\|\/Users\/|\/home\//);
  });
});

describe('ExportHistory: CSV download', () => {
  it('downloads the CSV of the right experiment with its real content', async () => {
    const backend = withHistory([A, B], {
      'GET /api/export/gen_aaaa1111': () => csvResponse('gender,score\nFemale,81.5\n'),
      'GET /api/export/gen_bbbb2222': () => csvResponse('region,sales\nNorth,10\n'),
    });
    renderHistory();
    await userEvent.click(within(await row('gen_bbbb2222')).getByRole('button', { name: /download csv for gen_bbbb2222/i }));

    await waitFor(() => expect(saved).toHaveLength(1));
    expect(backend.count('GET /api/export/gen_bbbb2222')).toBe(1);
    expect(backend.count('GET /api/export/gen_aaaa1111')).toBe(0);
    expect(saved[0].filename).toBe('dataforge_experiment_gen_bbbb2222.csv');
    expect(await readBlob(saved[0].blob)).toBe('region,sales\nNorth,10\n'); // bytes as sent by the backend
    expect(await screen.findByTestId('download-notice')).toHaveTextContent('Downloaded dataforge_experiment_gen_bbbb2222.csv.');
  });

  it('downloads from the details panel too', async () => {
    withHistory([A], { 'GET /api/export/gen_aaaa1111': () => csvResponse('x\n1\n') });
    renderHistory();
    await userEvent.click(within(await row('gen_aaaa1111')).getByRole('button', { name: /view details/i }));
    await userEvent.click(within(screen.getByTestId('details-panel')).getByRole('button', { name: 'Download generated CSV' }));
    await waitFor(() => expect(saved).toHaveLength(1));
    expect(saved[0].filename).toBe('dataforge_experiment_gen_aaaa1111.csv');
  });

  it('shows a clear message when the export fails and refreshes the list on 404', async () => {
    const backend = withHistory([A], { 'GET /api/export/gen_aaaa1111': () => json({ detail: 'Synthetic CSV file not found' }, 404) });
    renderHistory();
    await userEvent.click(within(await row('gen_aaaa1111')).getByRole('button', { name: /download csv for/i }));
    const notice = await screen.findByTestId('download-notice');
    expect(notice).toHaveTextContent('Download of gen_aaaa1111 failed: Synthetic CSV file not found');
    expect(notice).toHaveAttribute('role', 'alert');
    expect(saved).toHaveLength(0);
    await waitFor(() => expect(backend.count('GET /api/experiments')).toBe(2));
  });

  it('handles network failures and non-CSV answers without saving anything', async () => {
    withHistory([A, B], {
      'GET /api/export/gen_aaaa1111': () => Promise.reject(new TypeError('Failed to fetch')),
      'GET /api/export/gen_bbbb2222': () => json({ file: 'x' }),
    });
    renderHistory();
    await userEvent.click(within(await row('gen_aaaa1111')).getByRole('button', { name: /download csv for/i }));
    expect(await screen.findByTestId('download-notice')).toHaveTextContent('Could not reach the backend');
    await userEvent.click(within(await row('gen_bbbb2222')).getByRole('button', { name: /download csv for/i }));
    await waitFor(() => expect(screen.getByTestId('download-notice')).toHaveTextContent('did not return a CSV'));
    expect(saved).toHaveLength(0);
  });

  it('ignores repeated clicks while a download is in progress', async () => {
    let release!: (r: Response) => void;
    const backend = withHistory([A], { 'GET /api/export/gen_aaaa1111': () => new Promise<Response>((r) => (release = r)) });
    renderHistory();
    const button = within(await row('gen_aaaa1111')).getByRole('button', { name: /download csv for/i });
    await userEvent.click(button);
    await waitFor(() => expect(button).toBeDisabled());
    expect(button).toHaveTextContent('Downloading…');
    await userEvent.click(button);
    await userEvent.dblClick(button);
    expect(backend.count('GET /api/export/gen_aaaa1111')).toBe(1);

    release(csvResponse());
    await waitFor(() => expect(saved).toHaveLength(1));
    await waitFor(() => expect(button).toBeEnabled());
    expect(backend.count('GET /api/export/gen_aaaa1111')).toBe(1);
  });

  it('does not request a file that is known to be missing', async () => {
    const backend = withHistory([GONE]);
    renderHistory();
    await userEvent.click(within(await row('gen_cccc3333')).getByRole('button', { name: /download csv for/i }));
    expect(backend.count('GET /api/export/gen_cccc3333')).toBe(0);
  });
});

describe('ExportHistory: navigation links', () => {
  it('opens the experiment in Quality Comparison, generates with its model, and goes to Model Lab', async () => {
    withHistory([A]);
    const props = renderHistory();
    await userEvent.click(within(await row('gen_aaaa1111')).getByRole('button', { name: /view details/i }));
    const panel = screen.getByTestId('details-panel');
    await waitFor(() => expect(within(panel).getByTestId('model-config')).toHaveTextContent('Epochs'));

    await userEvent.click(within(panel).getByRole('button', { name: /open in quality comparison/i }));
    expect(props.onOpenQuality).toHaveBeenCalledWith(A);
    await userEvent.click(within(panel).getByRole('button', { name: /generate again/i }));
    expect(props.onNavigateToGenerator).toHaveBeenCalledWith('vae_0f1e2d3c');
    await userEvent.click(within(panel).getByRole('button', { name: /model lab/i }));
    expect(props.onNavigateToModelLab).toHaveBeenCalled();
  });
});
