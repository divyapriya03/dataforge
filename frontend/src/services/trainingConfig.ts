// Training form model: defaults, limits and validation.
// Limits and defaults are copied from TrainModelRequest in backend/app/schemas/api_models.py.

import type { TrainRequest } from '../types/api';

export const LIMITS = {
  epochs: { min: 1, max: 500 },
  learningRate: { exclusiveMin: 0, max: 1 },
  batchSize: { min: 8, max: 4096 },
  latentDim: { min: 2, max: 64 },
  hiddenLayer: { min: 2, max: 1024 },
  betaWeight: { min: 0, max: 100 },
  randomSeed: { min: 0, max: 2 ** 32 - 1 },
  modelNameMaxLength: 100,
} as const;

/** The server needs at least this many usable rows to train (MIN_TRAIN_ROWS in backend/app/core/config.py). */
export const MIN_TRAIN_ROWS = 10;

/** All fields are kept as strings, exactly as typed in the inputs. */
export interface TrainingFormValues {
  epochs: string;
  learningRate: string;
  batchSize: string;
  latentDim: string;
  hiddenLayer1: string;
  hiddenLayer2: string;
  betaWeight: string;
  randomSeed: string; // optional
  modelName: string; // optional
}

export type TrainingFormErrors = Partial<Record<keyof TrainingFormValues, string>>;

/** Backend defaults (the values the server uses when a field is omitted). */
export const DEFAULT_FORM: TrainingFormValues = {
  epochs: '50',
  learningRate: '0.001',
  batchSize: '32',
  latentDim: '8',
  hiddenLayer1: '128',
  hiddenLayer2: '64',
  betaWeight: '1',
  randomSeed: '',
  modelName: '',
};

function parseNumber(raw: string): number | null {
  const text = raw.trim();
  if (text === '') return null;
  const n = Number(text);
  return Number.isFinite(n) ? n : NaN;
}

function integerField(
  raw: string,
  label: string,
  min: number,
  max: number,
): { value?: number; error?: string } {
  const n = parseNumber(raw);
  if (n === null) return { error: `${label} is required.` };
  if (Number.isNaN(n) || !Number.isInteger(n)) return { error: `${label} must be a whole number.` };
  if (n < min || n > max) return { error: `${label} must be between ${min.toLocaleString('en-US')} and ${max.toLocaleString('en-US')}.` };
  return { value: n };
}

export function validateTrainingForm(
  values: TrainingFormValues,
  datasetId: string,
): { errors: TrainingFormErrors; request?: TrainRequest } {
  const errors: TrainingFormErrors = {};
  const body: TrainRequest = { dataset_id: datasetId };

  const epochs = integerField(values.epochs, 'Epochs', LIMITS.epochs.min, LIMITS.epochs.max);
  if (epochs.error) errors.epochs = epochs.error;
  else body.epochs = epochs.value;

  const lr = parseNumber(values.learningRate);
  if (lr === null) errors.learningRate = 'Learning rate is required.';
  else if (Number.isNaN(lr)) errors.learningRate = 'Learning rate must be a number.';
  else if (lr <= LIMITS.learningRate.exclusiveMin || lr > LIMITS.learningRate.max) {
    errors.learningRate = 'Learning rate must be greater than 0 and at most 1.';
  } else body.learning_rate = lr;

  const batch = integerField(values.batchSize, 'Batch size', LIMITS.batchSize.min, LIMITS.batchSize.max);
  if (batch.error) errors.batchSize = batch.error;
  else body.batch_size = batch.value;

  const latent = integerField(values.latentDim, 'Latent dimension', LIMITS.latentDim.min, LIMITS.latentDim.max);
  if (latent.error) errors.latentDim = latent.error;
  else body.latent_dim = latent.value;

  const h1 = integerField(values.hiddenLayer1, 'Hidden layer 1', LIMITS.hiddenLayer.min, LIMITS.hiddenLayer.max);
  const h2 = integerField(values.hiddenLayer2, 'Hidden layer 2', LIMITS.hiddenLayer.min, LIMITS.hiddenLayer.max);
  if (h1.error) errors.hiddenLayer1 = h1.error;
  if (h2.error) errors.hiddenLayer2 = h2.error;
  if (h1.value !== undefined && h2.value !== undefined) body.hidden_layers = [h1.value, h2.value];

  const beta = parseNumber(values.betaWeight);
  if (beta === null) errors.betaWeight = 'KL weight is required.';
  else if (Number.isNaN(beta)) errors.betaWeight = 'KL weight must be a number.';
  else if (beta < LIMITS.betaWeight.min || beta > LIMITS.betaWeight.max) {
    errors.betaWeight = `KL weight must be between ${LIMITS.betaWeight.min} and ${LIMITS.betaWeight.max}.`;
  } else body.beta_weight = beta;

  if (values.randomSeed.trim() !== '') {
    const seed = integerField(values.randomSeed, 'Random seed', LIMITS.randomSeed.min, LIMITS.randomSeed.max);
    if (seed.error) errors.randomSeed = seed.error;
    else body.random_seed = seed.value;
  }

  const name = values.modelName.trim();
  if (name.length > LIMITS.modelNameMaxLength) {
    errors.modelName = `Model name must be at most ${LIMITS.modelNameMaxLength} characters.`;
  } else if (name) body.model_name = name;

  if (!datasetId) return { errors };
  return Object.keys(errors).length ? { errors } : { errors, request: body };
}
