// Maps backend payloads to the UI types used by the shared pages (App state, Navbar, DataGenerator).

import type { ApiModel } from '../types/api';
import type { ModelRecord } from '../types';

function num(value: unknown): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

/**
 * ApiModel -> ModelRecord. `id` is the backend model_id, which is what POST /api/generate expects.
 * Values the backend does not provide (e.g. final loss of a failed model) become 0 in the record;
 * ModelLab itself always reads the ApiModel and shows "unavailable" for missing values.
 */
export function apiModelToRecord(m: ApiModel): ModelRecord {
  const p = (m.params ?? {}) as Record<string, unknown>;
  const layers = Array.isArray(p.hidden_layers) ? (p.hidden_layers as unknown[]).join(', ') : '';
  const status: ModelRecord['status'] =
    m.status === 'completed' ? 'completed' : m.status === 'failed' ? 'failed' : m.status === 'training' ? 'training' : 'idle';
  return {
    id: m.model_id,
    datasetId: m.dataset_id,
    name: m.name,
    createdAt: m.created_at ?? '',
    params: {
      latentDim: num(p.latent_dim),
      hiddenLayers: layers,
      learningRate: num(p.learning_rate),
      batchSize: num(p.batch_size),
      epochs: num(p.epochs ?? m.total_epochs),
      betaWeight: num(p.beta_weight),
      activation: 'LeakyReLU', // fixed in the backend VAE
    },
    finalLoss: m.final_loss ?? 0,
    finalReconLoss: m.recon_loss ?? 0,
    finalKlLoss: m.kl_loss ?? 0,
    status,
    lossHistory: m.loss_history ?? [],
  };
}

/** Insert or replace a model by id, newest first. */
export function upsertModel(list: ModelRecord[], model: ModelRecord): ModelRecord[] {
  return [model, ...list.filter((m) => m.id !== model.id)];
}
