export interface Hyperparameters {
  latentDim: number;
  hiddenLayers: string;
  learningRate: number;
  batchSize: number;
  epochs: number;
  betaWeight: number;
  activation: string;
}

export interface TrainingLossPoint {
  epoch: number;
  totalLoss: number;
  reconLoss: number;
  klLoss: number;
}

export interface ModelRecord {
  id: string;
  datasetId: string;
  name: string;
  createdAt: string;
  params: Hyperparameters;
  finalLoss: number;
  finalReconLoss: number;
  finalKlLoss: number;
  status: 'training' | 'completed' | 'failed' | 'idle';
  lossHistory: TrainingLossPoint[];
}

