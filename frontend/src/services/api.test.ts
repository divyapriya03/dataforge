import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_BASE_URL, downloadExperimentCsv, formatErrorDetail, getDatasets, startTraining } from './api';
import { csvResponse, installBackend, json, makeModel } from '../test/backend';

afterEach(() => vi.unstubAllGlobals());

const readBlob = (b: Blob) =>
  new Promise<string>((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = reject;
    r.readAsText(b);
  });

describe('api client', () => {
  it('uses the configured base URL', async () => {
    const { fetchMock } = installBackend({ 'GET /api/datasets': () => json([]) });
    await getDatasets();
    expect(String(fetchMock.mock.calls[0][0])).toBe(`${API_BASE_URL}/api/datasets`);
    expect(API_BASE_URL).toMatch(/^https?:\/\/[^/]+$/);
  });

  it('returns parsed data on success', async () => {
    installBackend({ 'POST /api/models/train': () => json(makeModel()) });
    const res = await startTraining({ dataset_id: 'ds_1' });
    expect(res.success).toBe(true);
    expect(res.data?.model_id).toBe('vae_0f1e2d3c');
  });

  it('extracts string error details', async () => {
    installBackend({ 'POST /api/models/train': () => json({ detail: 'Dataset not found' }, 404) });
    const res = await startTraining({ dataset_id: 'x' });
    expect(res).toMatchObject({ success: false, kind: 'http', status: 404, error: 'Dataset not found' });
  });

  it('formats FastAPI validation errors (a list of {loc, msg})', async () => {
    installBackend({
      'POST /api/models/train': () =>
        json({ detail: [{ loc: ['body', 'hidden_layers'], msg: 'Value error, must list exactly two layer sizes' }] }, 422),
    });
    const res = await startTraining({ dataset_id: 'x', hidden_layers: [64] });
    expect(res.error).toBe('hidden_layers: Value error, must list exactly two layer sizes');
  });

  it('falls back to a generic message when the error body is not JSON', async () => {
    installBackend({ 'GET /api/datasets': () => new Response('<html>Bad gateway</html>', { status: 502, statusText: 'Bad Gateway' }) });
    const res = await getDatasets();
    expect(res).toMatchObject({ success: false, kind: 'http', status: 502 });
    expect(res.error).toContain('502');
  });

  it('flags a 200 response that is not JSON', async () => {
    installBackend({ 'GET /api/datasets': () => new Response('oops', { status: 200 }) });
    expect(await getDatasets()).toMatchObject({ success: false, kind: 'format' });
  });

  it('reports network failures', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    const res = await getDatasets();
    expect(res).toMatchObject({ success: false, kind: 'network' });
    expect(res.error).toContain('Could not reach the backend');
  });

  it('aborts and reports a timeout', async () => {
    installBackend({
      'POST /api/models/train': ({ signal }) =>
        new Promise<Response>((_, reject) => signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))),
    });
    const res = await startTraining({ dataset_id: 'x' }, 30);
    expect(res).toMatchObject({ success: false, kind: 'timeout' });
  });
});

describe('downloadExperimentCsv', () => {
  it('requests the experiment export and returns the CSV bytes untouched with a meaningful file name', async () => {
    const backend = installBackend({ 'GET /api/export/gen_ab12': () => csvResponse('a,b\n1,2\n') });
    const res = await downloadExperimentCsv('gen_ab12');
    expect(backend.count('GET /api/export/gen_ab12')).toBe(1);
    expect(res.success).toBe(true);
    expect(res.data?.filename).toBe('dataforge_experiment_gen_ab12.csv');
    expect(await readBlob(res.data!.blob)).toBe('a,b\n1,2\n');
  });

  it('turns error responses into readable messages', async () => {
    installBackend({ 'GET /api/export/gen_x': () => json({ detail: 'Synthetic CSV file not found' }, 404) });
    expect(await downloadExperimentCsv('gen_x')).toMatchObject({ success: false, kind: 'http', status: 404, error: 'Synthetic CSV file not found' });
  });

  it('rejects a response that is not CSV', async () => {
    installBackend({ 'GET /api/export/gen_x': () => json({ path: 'C:/secret/file.csv' }) });
    const res = await downloadExperimentCsv('gen_x');
    expect(res).toMatchObject({ success: false, kind: 'format' });
    expect(res.error).toContain('did not return a CSV');
  });

  it('reports network failures', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    expect(await downloadExperimentCsv('gen_x')).toMatchObject({ success: false, kind: 'network' });
  });

  it('encodes the experiment id in the URL', async () => {
    const backend = installBackend({});
    await downloadExperimentCsv('a/b c');
    expect(String(backend.fetchMock.mock.calls[0][0])).toBe(`${API_BASE_URL}/api/export/a%2Fb%20c`);
  });
});

describe('formatErrorDetail', () => {
  it('handles odd shapes', () => {
    expect(formatErrorDetail(undefined, 'fallback')).toBe('fallback');
    expect(formatErrorDetail([], 'fallback')).toBe('fallback');
    expect(formatErrorDetail(['a', 'b'], 'fallback')).toBe('a; b');
  });
});

describe('API base URL configuration', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  it('uses VITE_API_BASE_URL for every request and trims a trailing slash', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://example.invalid:9999/');
    vi.resetModules();
    const fresh = await import('./api');
    expect(fresh.API_BASE_URL).toBe('http://example.invalid:9999');
    const backend = installBackend({});
    await fresh.getDatasets();
    await fresh.getExperiments();
    await fresh.downloadExperimentCsv('gen_1');
    await fresh.generateSynthetic({ model_id: 'vae_1', num_records: 20 });
    expect(backend.fetchMock.mock.calls.map((c) => String(c[0]))).toEqual([
      'http://example.invalid:9999/api/datasets',
      'http://example.invalid:9999/api/experiments',
      'http://example.invalid:9999/api/export/gen_1',
      'http://example.invalid:9999/api/generate',
    ]);
  });

  it('falls back to the documented default when the variable is empty', async () => {
    vi.stubEnv('VITE_API_BASE_URL', '');
    vi.resetModules();
    expect((await import('./api')).API_BASE_URL).toBe('http://127.0.0.1:8000');
  });

  it('names the configured address in the error when it is wrong (nothing listens there)', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://127.0.0.1:9');
    vi.resetModules();
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('fetch failed')));
    const fresh = await import('./api');
    const res = await fresh.getDatasets();
    expect(res).toMatchObject({ success: false, kind: 'network' });
    expect(res.error).toContain('http://127.0.0.1:9');
  });
});
