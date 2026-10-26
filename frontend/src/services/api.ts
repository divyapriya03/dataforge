// src/services/api.ts
// Single API client for the DataForge frontend (native fetch). Every page uses these functions;
// no component builds URLs itself.
//
// Base URL: VITE_API_BASE_URL (see frontend/.env), default http://127.0.0.1:8000.

import type {
  ApiDataset,
  ApiGenerateResponse,
  GenerateRequest,
  ApiEvaluation,
  ApiExperiment,
  ApiHealth,
  ApiModel,
  ApiPreprocessResponse,
  ApiColumn,
  TrainRequest,
} from '../types/api';

export const API_BASE_URL: string = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');

/** How long the UI waits for the (synchronous) training request. Override with VITE_TRAIN_TIMEOUT_MS. */
export const TRAIN_TIMEOUT_MS: number = Number(import.meta.env.VITE_TRAIN_TIMEOUT_MS) || 10 * 60 * 1000;

export type ApiErrorKind = 'network' | 'timeout' | 'http' | 'format';

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  /** Why the call failed (only when success is false). */
  kind?: ApiErrorKind;
  /** HTTP status when the server answered. */
  status?: number;
}

interface RequestOptions extends RequestInit {
  timeoutMs?: number;
}

/** FastAPI returns `detail` as a string, or as a list of {loc, msg} for 422 validation errors. */
export function formatErrorDetail(detail: unknown, fallback: string): string {
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((d) => {
        if (d && typeof d === 'object' && 'msg' in d) {
          const loc = Array.isArray((d as { loc?: unknown[] }).loc)
            ? (d as { loc: unknown[] }).loc.filter((p) => p !== 'body').join('.')
            : '';
          const msg = String((d as { msg: unknown }).msg);
          return loc ? `${loc}: ${msg}` : msg;
        }
        return typeof d === 'string' ? d : '';
      })
      .filter(Boolean);
    if (parts.length) return parts.join('; ');
  }
  return fallback;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<ApiResponse<T>> {
  const { timeoutMs, ...init } = options;
  const controller = new AbortController();
  let timedOut = false;
  const timer = timeoutMs
    ? setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, timeoutMs)
    : null;

  try {
    const resp = await fetch(`${API_BASE_URL}${path}`, { ...init, signal: controller.signal });
    const text = await resp.text();
    let body: unknown = undefined;
    let parsed = true;
    if (text) {
      try {
        body = JSON.parse(text);
      } catch {
        parsed = false;
      }
    }
    if (!resp.ok) {
      const detail = parsed && body && typeof body === 'object' ? (body as { detail?: unknown }).detail : undefined;
      return {
        success: false,
        kind: 'http',
        status: resp.status,
        error: formatErrorDetail(detail, `Request failed (${resp.status} ${resp.statusText || 'error'}).`),
      };
    }
    if (!parsed || body === undefined) {
      return { success: false, kind: 'format', status: resp.status, error: 'The server returned a response that is not valid JSON.' };
    }
    return { success: true, data: body as T, status: resp.status };
  } catch (e) {
    if (timedOut) {
      return { success: false, kind: 'timeout', error: `No response within ${Math.round((timeoutMs ?? 0) / 1000)} seconds.` };
    }
    return { success: false, kind: 'network', error: `Could not reach the backend at ${API_BASE_URL}. ${(e as Error).message}` };
  } finally {
    if (timer) clearTimeout(timer);
  }
}

const jsonPost = (body: unknown, extra: RequestOptions = {}): RequestOptions => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
  ...extra,
});

// ----- Health -----
export const getHealth = () => request<ApiHealth>('/api/health', { timeoutMs: 5000 });

// ----- Datasets -----
export const getDatasets = () => request<ApiDataset[]>('/api/datasets');
export const getDataset = (id: string) => request<ApiDataset>(`/api/datasets/${encodeURIComponent(id)}`);

/** Upload a CSV (multipart/form-data). */
export async function uploadDataset(file: File) {
  const form = new FormData();
  form.append('file', file);
  return request<ApiDataset>('/api/datasets/upload', { method: 'POST', body: form });
}

/** Dry-run of the preprocessing pipeline for a column configuration. */
export const previewPreprocessing = (datasetId: string, columns: ApiColumn[]) =>
  request<ApiPreprocessResponse>('/api/preprocess', jsonPost({ dataset_id: datasetId, columns }));

// ----- Models / training -----
/** Trains synchronously on the server: the response arrives when training has finished (or failed). */
export const startTraining = (body: TrainRequest, timeoutMs: number = TRAIN_TIMEOUT_MS) =>
  request<ApiModel>('/api/models/train', jsonPost(body, { timeoutMs }));

export const getModels = () => request<ApiModel[]>('/api/models');
export const getModelStatus = (modelId: string) =>
  request<ApiModel>(`/api/models/status/${encodeURIComponent(modelId)}`);

// ----- Generation / evaluation / history -----
/** Generation is synchronous too, but fast; the UI still gives up after this long. */
export const GENERATE_TIMEOUT_MS: number = Number(import.meta.env.VITE_GENERATE_TIMEOUT_MS) || 5 * 60 * 1000;

export const generateSynthetic = (body: GenerateRequest, timeoutMs: number = GENERATE_TIMEOUT_MS) =>
  request<ApiGenerateResponse>('/api/generate', jsonPost(body, { timeoutMs }));

export const evaluateRun = (experimentId: string) =>
  request<ApiEvaluation>(`/api/evaluate/${encodeURIComponent(experimentId)}`);

export const getExperiments = () => request<ApiExperiment[]>('/api/experiments');

/** How long the UI waits for a CSV download before giving up. */
export const DOWNLOAD_TIMEOUT_MS = 2 * 60 * 1000;

export interface CsvDownload {
  blob: Blob;
  filename: string;
}

/**
 * GET /api/export/{experiment_id}: the backend streams the experiment's generated CSV (text/csv attachment).
 * The bytes are returned untouched; the file name is derived from the experiment ID, not from server paths.
 */
export async function downloadExperimentCsv(experimentId: string): Promise<ApiResponse<CsvDownload>> {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, DOWNLOAD_TIMEOUT_MS);
  try {
    const resp = await fetch(`${API_BASE_URL}/api/export/${encodeURIComponent(experimentId)}`, { signal: controller.signal });
    if (!resp.ok) {
      let detail: unknown;
      try {
        detail = (await resp.json())?.detail;
      } catch {
        /* body was not JSON */
      }
      return {
        success: false,
        kind: 'http',
        status: resp.status,
        error: formatErrorDetail(detail, `Download failed (${resp.status} ${resp.statusText || 'error'}).`),
      };
    }
    const type = resp.headers.get('content-type') ?? '';
    if (!type.toLowerCase().includes('csv')) {
      return { success: false, kind: 'format', status: resp.status, error: `The server did not return a CSV file (content type "${type || 'none'}").` };
    }
    const blob = await resp.blob();
    return { success: true, status: resp.status, data: { blob, filename: `dataforge_experiment_${experimentId}.csv` } };
  } catch (e) {
    if (timedOut) return { success: false, kind: 'timeout', error: `The download did not finish within ${DOWNLOAD_TIMEOUT_MS / 1000} seconds.` };
    return { success: false, kind: 'network', error: `Could not reach the backend at ${API_BASE_URL}. ${(e as Error).message}` };
  } finally {
    clearTimeout(timer);
  }
}
