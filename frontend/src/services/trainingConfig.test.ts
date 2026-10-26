import { describe, expect, it } from 'vitest';
import { DEFAULT_FORM, validateTrainingForm, type TrainingFormValues } from './trainingConfig';

const form = (over: Partial<TrainingFormValues> = {}): TrainingFormValues => ({ ...DEFAULT_FORM, ...over });

describe('validateTrainingForm', () => {
  it('turns the defaults into the request body the backend expects', () => {
    const { errors, request } = validateTrainingForm(DEFAULT_FORM, 'ds_1');
    expect(errors).toEqual({});
    expect(request).toEqual({
      dataset_id: 'ds_1',
      epochs: 50,
      learning_rate: 0.001,
      batch_size: 32,
      latent_dim: 8,
      hidden_layers: [128, 64],
      beta_weight: 1,
    });
  });

  it('includes the optional seed and name only when given', () => {
    const { request } = validateTrainingForm(form({ randomSeed: '7', modelName: '  My model ' }), 'ds_1');
    expect(request).toMatchObject({ random_seed: 7, model_name: 'My model' });
    expect(validateTrainingForm(form({ randomSeed: '0' }), 'ds_1').request?.random_seed).toBe(0);
  });

  it('requires a dataset', () => {
    expect(validateTrainingForm(DEFAULT_FORM, '').request).toBeUndefined();
  });

  it.each([
    ['epochs', '0'],
    ['epochs', '501'],
    ['epochs', '10.5'],
    ['epochs', ''],
    ['epochs', 'abc'],
    ['learningRate', '0'],
    ['learningRate', '-0.1'],
    ['learningRate', '1.5'],
    ['learningRate', 'x'],
    ['batchSize', '7'],
    ['batchSize', '4097'],
    ['latentDim', '1'],
    ['latentDim', '65'],
    ['hiddenLayer1', '1'],
    ['hiddenLayer2', '1025'],
    ['hiddenLayer2', ''],
    ['betaWeight', '-1'],
    ['betaWeight', '101'],
    ['randomSeed', '-1'],
    ['randomSeed', '1.5'],
    ['randomSeed', '4294967296'],
    ['modelName', 'x'.repeat(101)],
  ] as [keyof TrainingFormValues, string][])('rejects %s = %j (outside the backend limits)', (field, value) => {
    const { errors, request } = validateTrainingForm(form({ [field]: value }), 'ds_1');
    expect(errors[field]).toBeTruthy();
    expect(request).toBeUndefined();
  });

  it.each([
    ['epochs', '1'],
    ['epochs', '500'],
    ['learningRate', '1'],
    ['learningRate', '0.0001'],
    ['batchSize', '8'],
    ['batchSize', '4096'],
    ['latentDim', '2'],
    ['latentDim', '64'],
    ['hiddenLayer1', '2'],
    ['hiddenLayer2', '1024'],
    ['betaWeight', '0'],
    ['betaWeight', '100'],
    ['randomSeed', '4294967295'],
  ] as [keyof TrainingFormValues, string][])('accepts the boundary value %s = %s', (field, value) => {
    expect(validateTrainingForm(form({ [field]: value }), 'ds_1').errors[field]).toBeUndefined();
  });

  it('reports every invalid field at once', () => {
    const { errors } = validateTrainingForm(form({ epochs: '0', batchSize: '1', latentDim: '' }), 'ds_1');
    expect(Object.keys(errors).sort()).toEqual(['batchSize', 'epochs', 'latentDim']);
  });
});
