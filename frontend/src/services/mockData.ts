import { DatasetInfo, ExperimentRun, ModelRecord, SummaryStatComparison } from '../types';

export const sampleStudentData: Record<string, any>[] = [
  { student_id: "STU001", gender: "Female", grade_level: "Senior", study_hours: 18.5, attendance_rate: 96.2, math_score: 88, reading_score: 92, writing_score: 90 },
  { student_id: "STU002", gender: "Male", grade_level: "Junior", study_hours: 12.0, attendance_rate: 88.0, math_score: 74, reading_score: 70, writing_score: 72 },
  { student_id: "STU003", gender: "Female", grade_level: "Sophomore", study_hours: 22.1, attendance_rate: 98.5, math_score: 95, reading_score: 94, writing_score: 96 },
  { student_id: "STU004", gender: "Male", grade_level: "Senior", study_hours: 8.4, attendance_rate: 79.1, math_score: 62, reading_score: 65, writing_score: 60 },
  { student_id: "STU005", gender: "Female", grade_level: "Freshman", study_hours: 15.0, attendance_rate: 91.4, math_score: 81, reading_score: 83, writing_score: 82 },
  { student_id: "STU006", gender: "Male", grade_level: "Sophomore", study_hours: 16.8, attendance_rate: 93.0, math_score: 84, reading_score: 80, writing_score: 85 },
  { student_id: "STU007", gender: "Female", grade_level: "Junior", study_hours: 10.5, attendance_rate: 84.5, math_score: 69, reading_score: 73, writing_score: 71 },
  { student_id: "STU008", gender: "Male", grade_level: "Freshman", study_hours: 20.2, attendance_rate: 97.0, math_score: 91, reading_score: 89, writing_score: 93 },
  { student_id: "STU009", gender: "Female", grade_level: "Senior", study_hours: 14.1, attendance_rate: 89.8, math_score: 78, reading_score: 81, writing_score: 79 },
  { student_id: "STU010", gender: "Male", grade_level: "Junior", study_hours: 6.5, attendance_rate: 74.0, math_score: 55, reading_score: 58, writing_score: 54 },
  { student_id: "STU011", gender: "Female", grade_level: "Sophomore", study_hours: 19.3, attendance_rate: 95.8, math_score: 89, reading_score: 91, writing_score: 88 },
  { student_id: "STU012", gender: "Male", grade_level: "Freshman", study_hours: 13.7, attendance_rate: 86.5, math_score: 76, reading_score: 75, writing_score: 77 },
];

export const initialStudentDataset: DatasetInfo = {
  id: "ds_student_01",
  name: "student_performance.csv",
  rowCount: 1000,
  columnCount: 8,
  uploadDate: "2026-10-12 14:30",
  columns: [
    { name: "student_id", inferredType: "id", selectedType: "drop", nullCount: 0, nullPercentage: 0, imputationStrategy: "drop", included: false },
    { name: "gender", inferredType: "categorical", selectedType: "categorical", nullCount: 0, nullPercentage: 0, imputationStrategy: "mode", included: true, categories: ["Female", "Male"] },
    { name: "grade_level", inferredType: "categorical", selectedType: "categorical", nullCount: 0, nullPercentage: 0, imputationStrategy: "mode", included: true, categories: ["Freshman", "Sophomore", "Junior", "Senior"] },
    { name: "study_hours", inferredType: "numeric", selectedType: "numeric", nullCount: 4, nullPercentage: 0.4, imputationStrategy: "median", included: true, min: 4.0, max: 28.0, mean: 15.3, std: 4.8 },
    { name: "attendance_rate", inferredType: "numeric", selectedType: "numeric", nullCount: 0, nullPercentage: 0, imputationStrategy: "mean", included: true, min: 65.0, max: 100.0, mean: 89.4, std: 7.1 },
    { name: "math_score", inferredType: "numeric", selectedType: "numeric", nullCount: 0, nullPercentage: 0, imputationStrategy: "median", included: true, min: 42.0, max: 100.0, mean: 78.5, std: 11.8 },
    { name: "reading_score", inferredType: "numeric", selectedType: "numeric", nullCount: 0, nullPercentage: 0, imputationStrategy: "median", included: true, min: 45.0, max: 100.0, mean: 79.8, std: 11.2 },
    { name: "writing_score", inferredType: "numeric", selectedType: "numeric", nullCount: 0, nullPercentage: 0, imputationStrategy: "median", included: true, min: 40.0, max: 100.0, mean: 79.1, std: 11.5 },
  ],
  data: sampleStudentData
};

export const sampleLossCurve = [
  { epoch: 1, totalLoss: 0.452, reconLoss: 0.421, klLoss: 0.031 },
  { epoch: 5, totalLoss: 0.284, reconLoss: 0.252, klLoss: 0.032 },
  { epoch: 10, totalLoss: 0.178, reconLoss: 0.149, klLoss: 0.029 },
  { epoch: 15, totalLoss: 0.122, reconLoss: 0.098, klLoss: 0.024 },
  { epoch: 20, totalLoss: 0.089, reconLoss: 0.071, klLoss: 0.018 },
  { epoch: 25, totalLoss: 0.068, reconLoss: 0.054, klLoss: 0.014 },
  { epoch: 30, totalLoss: 0.055, reconLoss: 0.044, klLoss: 0.011 },
  { epoch: 35, totalLoss: 0.046, reconLoss: 0.038, klLoss: 0.008 },
  { epoch: 40, totalLoss: 0.041, reconLoss: 0.034, klLoss: 0.007 },
  { epoch: 45, totalLoss: 0.038, reconLoss: 0.032, klLoss: 0.006 },
  { epoch: 50, totalLoss: 0.035, reconLoss: 0.030, klLoss: 0.005 },
];

export const initialModels: ModelRecord[] = [
  {
    id: "vae_model_z8",
    datasetId: "ds_student_01",
    name: "VAE-Student-Z8 (50 Epochs)",
    createdAt: "2026-10-12 15:10",
    params: {
      latentDim: 8,
      hiddenLayers: "128, 64",
      learningRate: 0.001,
      batchSize: 32,
      epochs: 50,
      betaWeight: 1.0,
      activation: "LeakyReLU"
    },
    finalLoss: 0.035,
    finalReconLoss: 0.030,
    finalKlLoss: 0.005,
    status: "completed",
    lossHistory: sampleLossCurve
  },
  {
    id: "vae_model_z16",
    datasetId: "ds_student_01",
    name: "VAE-Student-Z16 (30 Epochs)",
    createdAt: "2026-10-12 16:05",
    params: {
      latentDim: 16,
      hiddenLayers: "256, 128",
      learningRate: 0.0008,
      batchSize: 64,
      epochs: 30,
      betaWeight: 1.2,
      activation: "ReLU"
    },
    finalLoss: 0.042,
    finalReconLoss: 0.035,
    finalKlLoss: 0.007,
    status: "completed",
    lossHistory: sampleLossCurve.slice(0, 7)
  }
];

export const sampleSyntheticRecords: Record<string, any>[] = [
  { row_id: "#1", gender: "Female", grade_level: "Senior", study_hours: 17.8, attendance_rate: 95.4, math_score: 86.4, reading_score: 90.2, writing_score: 88.7, validity: "Valid ✓" },
  { row_id: "#2", gender: "Male", grade_level: "Junior", study_hours: 11.6, attendance_rate: 87.2, math_score: 72.8, reading_score: 71.5, writing_score: 70.9, validity: "Valid ✓" },
  { row_id: "#3", gender: "Female", grade_level: "Sophomore", study_hours: 21.4, attendance_rate: 97.9, math_score: 93.6, reading_score: 93.1, writing_score: 94.5, validity: "Valid ✓" },
  { row_id: "#4", gender: "Male", grade_level: "Senior", study_hours: 9.1, attendance_rate: 80.3, math_score: 64.2, reading_score: 66.0, writing_score: 62.4, validity: "Valid ✓" },
  { row_id: "#5", gender: "Female", grade_level: "Freshman", study_hours: 14.8, attendance_rate: 90.7, math_score: 80.1, reading_score: 82.4, writing_score: 81.3, validity: "Valid ✓" },
  { row_id: "#6", gender: "Male", grade_level: "Sophomore", study_hours: 16.2, attendance_rate: 92.4, math_score: 82.9, reading_score: 79.4, writing_score: 83.8, validity: "Valid ✓" },
  { row_id: "#7", gender: "Female", grade_level: "Junior", study_hours: 10.9, attendance_rate: 85.1, math_score: 70.3, reading_score: 74.2, writing_score: 72.0, validity: "Valid ✓" },
  { row_id: "#8", gender: "Male", grade_level: "Freshman", study_hours: 19.8, attendance_rate: 96.5, math_score: 90.2, reading_score: 87.9, writing_score: 91.8, validity: "Valid ✓" },
];

export const initialExperiments: ExperimentRun[] = [
  {
    id: "gen_20261012_01",
    datasetName: "student_performance.csv",
    modelName: "VAE-Student-Z8 (50 Epochs)",
    numRecords: 2000,
    temperature: 1.0,
    fidelityScore: 94.2,
    domainValidity: 99.8,
    avgKsStat: 0.038,
    createdAt: "2026-10-12 16:30",
    records: sampleSyntheticRecords
  },
  {
    id: "gen_20261012_02",
    datasetName: "student_performance.csv",
    modelName: "VAE-Student-Z16 (30 Epochs)",
    numRecords: 5000,
    temperature: 1.2,
    fidelityScore: 91.6,
    domainValidity: 99.2,
    avgKsStat: 0.052,
    createdAt: "2026-10-12 16:55",
    records: sampleSyntheticRecords
  }
];

export const statComparisons: SummaryStatComparison[] = [
  { feature: "math_score", realMean: 78.5, syntheticMean: 78.1, realStd: 11.8, syntheticStd: 11.5, ksStatistic: 0.036, pValue: 0.82, status: "Passed" },
  { feature: "reading_score", realMean: 79.8, syntheticMean: 79.4, realStd: 11.2, syntheticStd: 11.0, ksStatistic: 0.041, pValue: 0.74, status: "Passed" },
  { feature: "writing_score", realMean: 79.1, syntheticMean: 78.7, realStd: 11.5, syntheticStd: 11.3, ksStatistic: 0.039, pValue: 0.79, status: "Passed" },
  { feature: "study_hours", realMean: 15.3, syntheticMean: 15.1, realStd: 4.8, syntheticStd: 4.7, ksStatistic: 0.048, pValue: 0.65, status: "Passed" },
  { feature: "attendance_rate", realMean: 89.4, syntheticMean: 89.1, realStd: 7.1, syntheticStd: 6.9, ksStatistic: 0.032, pValue: 0.89, status: "Passed" },
];
