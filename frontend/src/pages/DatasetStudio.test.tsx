import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DatasetStudio } from './DatasetStudio';
import { column, installBackend, json, makeDataset, type Handler } from '../test/backend';
import type { ApiColumn } from '../types/api';

afterEach(() => vi.unstubAllGlobals());

type Props = React.ComponentProps<typeof DatasetStudio>;

function renderStudio(props: Partial<Props> = {}, routes: Record<string, Handler> = {}) {
  const ds = makeDataset();
  const backend = installBackend({
    'GET /api/datasets/ds_a1b2c3d4': () => json({ ...ds, preview: [{ study_hours: 12.5, math_score: 70 }, { study_hours: 9, math_score: 55 }] }),
    ...routes,
  });
  const p: Props = {
    datasets: [ds],
    datasetsState: { phase: 'ready' },
    activeDataset: ds,
    columnConfig: null,
    onReloadDatasets: vi.fn().mockResolvedValue([]),
    onSelectDataset: vi.fn(),
    onUploaded: vi.fn(),
    onColumnConfigChange: vi.fn(),
    onNavigateToModelLab: vi.fn(),
    ...props,
  };
  const view = render(<DatasetStudio {...p} />);
  return { p, backend, ...view };
}

const csvFile = (content: string, name = 'students.csv') => new File([content], name, { type: 'text/csv' });
const chooseFile = (file: File) => userEvent.upload(screen.getByLabelText(/choose csv file/i), file);

describe('DatasetStudio: datasets', () => {
  it('shows the real dataset metadata and the server-detected columns', async () => {
    renderStudio();
    const meta = screen.getByTestId('dataset-meta');
    expect(meta).toHaveTextContent('ds_a1b2c3d4');
    expect(meta).toHaveTextContent('250 × 3');
    expect(meta).toHaveTextContent('2026-10-24 05:10:34');
    expect(meta).toHaveTextContent('2 of 3 columns');
    expect(within(screen.getByTestId('col-row-student_id')).getByLabelText('Use column student_id')).not.toBeChecked();
    expect(within(screen.getByTestId('col-row-study_hours')).getByLabelText('Use column study_hours')).toBeChecked();
  });

  it('shows loading, empty and error states', async () => {
    const reload = vi.fn().mockResolvedValue([]);
    const { rerender, p } = renderStudio({ datasetsState: { phase: 'loading' }, datasets: [], activeDataset: null, onReloadDatasets: reload });
    expect(screen.getByText(/loading datasets/i)).toBeInTheDocument();
    rerender(<DatasetStudio {...p} datasetsState={{ phase: 'ready' }} datasets={[]} activeDataset={null} />);
    expect(screen.getByTestId('no-datasets')).toHaveTextContent('No datasets yet.');
    rerender(<DatasetStudio {...p} datasetsState={{ phase: 'error', message: 'database is locked' }} datasets={[]} activeDataset={null} />);
    expect(screen.getByRole('alert')).toHaveTextContent('Could not load datasets: database is locked');
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(reload).toHaveBeenCalled();
  });

  it('selects another dataset', async () => {
    const other = makeDataset({ id: 'ds_second01', filename: 'sales.csv' });
    const { p } = renderStudio({ datasets: [makeDataset(), other] });
    await userEvent.selectOptions(screen.getByLabelText('Dataset'), 'ds_second01');
    expect(p.onSelectDataset).toHaveBeenCalledWith('ds_second01');
  });

  it('loads the server preview for the dataset', async () => {
    const { backend } = renderStudio();
    await userEvent.click(screen.getByRole('tab', { name: 'Preview' }));
    const table = await screen.findByTestId('preview-table');
    expect(table).toHaveTextContent('12.5');
    expect(table).toHaveTextContent('First 2 rows');
    expect(backend.count('GET /api/datasets/ds_a1b2c3d4')).toBe(1);
  });

  it('says when no preview is available instead of inventing rows', async () => {
    renderStudio({}, { 'GET /api/datasets/ds_a1b2c3d4': () => json(makeDataset({ preview: [] })) });
    await userEvent.click(screen.getByRole('tab', { name: 'Preview' }));
    expect(await screen.findByTestId('no-preview')).toBeInTheDocument();
  });

  it('shows statistics and marks missing ones unavailable', async () => {
    renderStudio({ activeDataset: makeDataset({ columns: [{ ...column('x'), min: 0, max: 9.5, mean: null, std: null, categories: null }] }) });
    await userEvent.click(screen.getByRole('tab', { name: 'Statistics' }));
    const row = screen.getByText('x').closest('tr')!;
    expect(row).toHaveTextContent('0.00');
    expect(row).toHaveTextContent('9.50');
    expect(row.textContent!.match(/not available/g)!.length).toBe(3);
  });
});

describe('DatasetStudio: upload', () => {
  it('uploads a real file and reports the server-assigned dataset ID', async () => {
    const created = makeDataset({ id: 'ds_new00001', filename: 'students.csv', rowCount: 12, columnCount: 3 });
    const { p, backend } = renderStudio({}, { 'POST /api/datasets/upload': () => json(created) });
    await chooseFile(csvFile('a,b\n1,2\n'));
    const ok = await screen.findByTestId('upload-success');
    expect(ok).toHaveTextContent('ds_new00001');
    expect(ok).toHaveTextContent('12 rows, 3 columns');
    expect(backend.count('POST /api/datasets/upload')).toBe(1);
    expect(p.onUploaded).toHaveBeenCalledWith(created);
    await userEvent.click(within(ok).getByRole('button', { name: /continue to model lab/i }));
    expect(p.onNavigateToModelLab).toHaveBeenCalledWith('ds_new00001');
  });

  it.each([
    [400, 'Failed to parse CSV: The file is empty: no header or data found.'],
    [400, 'Only CSV files (.csv) are supported.'],
    [413, 'File is larger than the 25 MB limit.'],
    [500, 'Saving the dataset failed because of an internal error. See the server log.'],
  ])('shows the server message for a %i response', async (status, detail) => {
    const { p } = renderStudio({}, { 'POST /api/datasets/upload': () => json({ detail }, status) });
    await chooseFile(csvFile('x'));
    expect(await screen.findByTestId('upload-error')).toHaveTextContent(`Upload failed: ${detail}`);
    expect(screen.queryByTestId('upload-success')).not.toBeInTheDocument();
    expect(p.onUploaded).not.toHaveBeenCalled();
  });

  it('reports network failures and malformed responses', async () => {
    const { p } = renderStudio({}, { 'POST /api/datasets/upload': () => Promise.reject(new TypeError('Failed to fetch')) });
    await chooseFile(csvFile('x'));
    expect(await screen.findByTestId('upload-error')).toHaveTextContent('Could not reach the backend');

    vi.unstubAllGlobals();
    installBackend({ 'POST /api/datasets/upload': () => json({ hello: 'world' }) });
    await chooseFile(csvFile('y'));
    await waitFor(() => expect(screen.getByTestId('upload-error')).toHaveTextContent('without a dataset ID'));
    expect(p.onUploaded).not.toHaveBeenCalled();
  });

  it('accepts a dropped file and ignores a second upload while one is running', async () => {
    let release!: (r: Response) => void;
    const { backend } = renderStudio({}, { 'POST /api/datasets/upload': () => new Promise<Response>((r) => (release = r)) });
    const zone = screen.getByText(/drop a .csv file here/i).closest('div')!;
    fireEvent.drop(zone, { dataTransfer: { files: [csvFile('a\n1\n')] } });
    await screen.findByText(/uploading students.csv/i);
    expect(document.getElementById('csv-file')).toBeDisabled();
    fireEvent.drop(zone, { dataTransfer: { files: [csvFile('a\n1\n', 'second.csv')] } });
    expect(backend.count('POST /api/datasets/upload')).toBe(1);
    await act(async () => release(json(makeDataset({ id: 'ds_new00002' }))));
    expect(await screen.findByTestId('upload-success')).toHaveTextContent('ds_new00002');
  });
});

describe('DatasetStudio: column configuration and preprocessing', () => {
  it('reports edits to the column configuration (used later by Model Lab)', async () => {
    const { p } = renderStudio();
    await userEvent.click(screen.getByLabelText('Use column math_score'));
    const unchecked = (p.onColumnConfigChange as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(unchecked[0]).toBe('ds_a1b2c3d4');
    expect((unchecked[1] as ApiColumn[]).find((c) => c.name === 'math_score')).toMatchObject({ included: false });
    expect((unchecked[1] as ApiColumn[]).find((c) => c.name === 'study_hours')).toMatchObject({ included: true });

    await userEvent.selectOptions(screen.getByLabelText('Treat study_hours as'), 'categorical');
    await userEvent.selectOptions(screen.getByLabelText('Missing-value handling for study_hours'), 'mean');
    const calls = (p.onColumnConfigChange as ReturnType<typeof vi.fn>).mock.calls;
    expect((calls[1][1] as ApiColumn[]).find((c) => c.name === 'study_hours')).toMatchObject({ selectedType: 'categorical', included: true });
    expect((calls[2][1] as ApiColumn[]).find((c) => c.name === 'study_hours')).toMatchObject({ imputationStrategy: 'mean' });
  });

  it('shows an edited configuration and can reset it', async () => {
    const edited = makeDataset().columns.map((c) => (c.name === 'math_score' ? { ...c, included: false } : c));
    const { p } = renderStudio({ columnConfig: edited });
    expect(screen.getByLabelText('Use column math_score')).not.toBeChecked();
    expect(screen.getByTestId('dataset-meta')).toHaveTextContent('1 of 3 columns');
    expect(screen.getByText(/Model Lab will train with your version/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /reset to the server/i }));
    expect(p.onColumnConfigChange).toHaveBeenCalledWith('ds_a1b2c3d4', null);
  });

  it('runs the preprocessing check with the edited columns and shows warnings and dropped columns', async () => {
    const edited = makeDataset().columns.map((c) => (c.name === 'math_score' ? { ...c, included: false } : c));
    const { backend } = renderStudio(
      { columnConfig: edited },
      {
        'POST /api/preprocess': () =>
          json({
            dataset_id: 'ds_a1b2c3d4',
            processed_dimensions: 1,
            feature_names: ['study_hours'],
            ready_for_training: true,
            rows_used: 250,
            dropped_columns: [{ name: 'student_id', reason: 'excluded by configuration' }],
            warnings: ["Column 'study_hours' has 6 non-numeric value(s) treated as missing."],
          }),
      },
    );
    await userEvent.click(screen.getByRole('button', { name: /run preprocessing check/i }));
    const out = await screen.findByTestId('check-result');
    expect(out).toHaveTextContent('1 model input features from 250 rows · ready for training');
    expect(out).toHaveTextContent('student_id (excluded by configuration)');
    expect(out).toHaveTextContent('treated as missing');
    const sent = backend.calls.find((c) => c.key === 'POST /api/preprocess')!.body;
    expect(sent.dataset_id).toBe('ds_a1b2c3d4');
    expect(sent.columns.find((c: ApiColumn) => c.name === 'math_score').included).toBe(false);
  });

  it('shows preprocessing errors from the server', async () => {
    renderStudio({}, { 'POST /api/preprocess': () => json({ detail: "Preprocessing failed: Column 'x' has no valid numeric values; exclude it or mark it categorical." }, 400) });
    await userEvent.click(screen.getByRole('button', { name: /run preprocessing check/i }));
    expect(await screen.findByTestId('check-error')).toHaveTextContent('has no valid numeric values');
  });

  it('does not offer to train on a dataset the backend would reject', async () => {
    renderStudio({ activeDataset: makeDataset({ rowCount: 4 }) });
    expect(screen.getByText(/training needs at least 10/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /continue to model lab/i })).toBeDisabled();
  });

  it('continues to Model Lab with the dataset ID', async () => {
    const { p } = renderStudio();
    await userEvent.click(screen.getByRole('button', { name: /continue to model lab/i }));
    expect(p.onNavigateToModelLab).toHaveBeenCalledWith('ds_a1b2c3d4');
  });
});
