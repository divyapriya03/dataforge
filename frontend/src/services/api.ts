// src/services/api.ts
// Centralized API client for the DataForge frontend.
// Uses the native fetch API with async/await.
// All requests assume the backend is running at http://127.0.0.1:8000/api

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// Helper to handle JSON responses and error handling
async function request<T>(
  url: string,
  options: RequestInit = {}
): Promise<ApiResponse<T>> {
  try {
    const resp = await fetch(url, {
      credentials: 'include',
      ...options,
    });
    const json = await resp.json();
    if (!resp.ok) {
      return { success: false, error: json?.detail ?? resp.statusText };
    }
    return { success: true, data: json as T };
  } catch (e) {
    return { success: false, error: (e as Error).message };
  }
}

// ----- Dataset Endpoints -----
export async function getDatasets() {
  return request<any[]>('http://127.0.0.1:8000/api/datasets');
}

export async function getDataset(id: string) {
  return request<any>(`http://127.0.0.1:8000/api/datasets/${id}`);
}

// Upload a CSV file (multipart/form-data)
export async function uploadDataset(file: File) {
  const form = new FormData();
  form.append('file', file);
  return request<any>('http://127.0.0.1:8000/api/datasets/upload', {
    method: 'POST',
    body: form,
  });
}

// ----- Model / Training Endpoints -----
export async function startTraining(datasetId: string, params: any) {
  return request<any>(`http://127.0.0.1:8000/api/models/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id: datasetId, hyperparameters: params }),
  });
}

export async function getModelStatus(modelId: string) {
  return request<any>(`http://127.0.0.1:8000/api/models/status/${modelId}`);
}

// ----- Generation Endpoints -----
export async function generateSynthetic(modelId: string, settings: any) {
  return request<any>('http://127.0.0.1:8000/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId, ...settings }),
  });
}

// ----- Evaluation Endpoints -----
export async function evaluateRun(experimentId: string) {
  return request<any>(`http://127.0.0.1:8000/api/evaluate/${experimentId}`);
}

// ----- Experiment / Export Endpoints -----
export async function getExperiments() {
  return request<any[]>('http://127.0.0.1:8000/api/experiments');
}

export async function downloadExport(experimentId: string) {
  // Returns a Blob that can be saved client‑side.
  const resp = await fetch(`http://127.0.0.1:8000/api/export/${experimentId}`);
  if (!resp.ok) {
    throw new Error('Export failed');
  }
  return await resp.blob();
}

// Additional convenience helpers can be added as the UI evolves.
