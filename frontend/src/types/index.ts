export interface ColumnMeta {
  name: string;
  inferredType: 'numeric' | 'categorical' | 'id';
  selectedType: 'numeric' | 'categorical' | 'drop';
  nullCount: number;
  nullPercentage: number;
  imputationStrategy: 'median' | 'mean' | 'mode' | 'drop';
  included: boolean;
  min?: number;
  max?: number;
  mean?: number;
  std?: number;
  categories?: string[];
}

export interface DatasetInfo {
  id: string;
  name: string;
  rowCount: number;
  columnCount: number;
  uploadDate: string;
  columns: ColumnMeta[];
  data: Record<string, any>[];
}

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

export interface GenerationSettings {
  modelId: string;
  numRecords: number;
  temperature: number;
  randomSeed: number;
}

export interface ExperimentRun {
  id: string;
  datasetName: string;
  modelName: string;
  numRecords: number;
  temperature: number;
  fidelityScore: number;
  domainValidity: number;
  avgKsStat: number;
  createdAt: string;
  records: Record<string, any>[];
}

export interface FeatureDistribution {
  value: string;
  realCount: number;
  syntheticCount: number;
}

export interface SummaryStatComparison {
  feature: string;
  realMean: number;
  syntheticMean: number;
  realStd: number;
  syntheticStd: number;
  ksStatistic: number;
  pValue: number;
  status: 'Passed' | 'Warning' | 'Failed';
}
