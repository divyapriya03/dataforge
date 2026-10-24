import React, { useState, useEffect } from 'react';
import { 
  Play, 
  Square, 
  BrainCircuit, 
  Sparkles, 
  TrendingDown, 
  Layers, 
  Settings2, 
  CheckCircle2, 
  Activity, 
  ArrowRight 
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';
import { DatasetInfo, Hyperparameters, ModelRecord, TrainingLossPoint } from '../types';

interface ModelLabProps {
  dataset: DatasetInfo;
  activeModel: ModelRecord | null;
  onSaveModel: (model: ModelRecord) => void;
  onNavigateToGenerator: () => void;
}

export const ModelLab: React.FC<ModelLabProps> = ({
  dataset,
  activeModel,
  onSaveModel,
  onNavigateToGenerator
}) => {
  const [params, setParams] = useState<Hyperparameters>({
    latentDim: activeModel?.params.latentDim || 8,
    hiddenLayers: activeModel?.params.hiddenLayers || "128, 64",
    learningRate: activeModel?.params.learningRate || 0.001,
    batchSize: activeModel?.params.batchSize || 32,
    epochs: activeModel?.params.epochs || 50,
    betaWeight: activeModel?.params.betaWeight || 1.0,
    activation: activeModel?.params.activation || "LeakyReLU"
  });

  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [currentEpoch, setCurrentEpoch] = useState<number>(activeModel ? activeModel.params.epochs : 0);
  const [lossData, setLossData] = useState<TrainingLossPoint[]>(activeModel ? activeModel.lossHistory : []);

  // Live training simulation loop (with mathematically realistic loss curves)
  useEffect(() => {
    let interval: any = null;
    if (isTraining && currentEpoch < params.epochs) {
      interval = setInterval(() => {
        setCurrentEpoch((prev) => {
          const next = prev + 1;
          const decay = Math.exp(-next / 15);
          const recon = +(0.025 + 0.40 * decay + (Math.random() * 0.004 - 0.002)).toFixed(4);
          const kl = +(0.005 + 0.028 * Math.exp(-next / 8) + (Math.random() * 0.001 - 0.0005)).toFixed(4);
          const total = +(recon + params.betaWeight * kl).toFixed(4);

          setLossData((prevData) => [
            ...prevData,
            { epoch: next, totalLoss: total, reconLoss: recon, klLoss: kl }
          ]);

          if (next >= params.epochs) {
            setIsTraining(false);
            const finishedModel: ModelRecord = {
              id: `vae_${Date.now()}`,
              datasetId: dataset.id,
              name: `VAE-${dataset.name.split('.')[0]}-Z${params.latentDim}`,
              createdAt: new Date().toISOString().replace('T', ' ').substring(0, 16),
              params,
              finalLoss: total,
              finalReconLoss: recon,
              finalKlLoss: kl,
              status: 'completed',
              lossHistory: [...lossData, { epoch: next, totalLoss: total, reconLoss: recon, klLoss: kl }]
            };
            onSaveModel(finishedModel);
          }
          return next;
        });
      }, 150);
    }
    return () => clearInterval(interval);
  }, [isTraining, currentEpoch, params, dataset]);

  const handleStartTraining = () => {
    setLossData([]);
    setCurrentEpoch(0);
    setIsTraining(true);
  };

  const handleStopTraining = () => {
    setIsTraining(false);
  };

  const progressPercentage = Math.round((currentEpoch / params.epochs) * 100);

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="flex items-center justify-between bg-slate-900/80 border border-slate-800 p-5 rounded-2xl">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">
            Model Lab: Variational Autoencoder (VAE) Training Suite
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Configure neural network bottleneck capacity, learning rates, and ELBO loss weights.
          </p>
        </div>
        {activeModel && (
          <button
            onClick={onNavigateToGenerator}
            className="inline-flex items-center space-x-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-4 py-2 rounded-lg shadow-md shadow-indigo-600/30 transition"
          >
            <span>Proceed to Data Generator</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Hyperparameter Controls Form */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-5">
          <div className="flex items-center space-x-2 text-indigo-400 font-semibold text-sm border-b border-slate-800 pb-3">
            <Settings2 className="w-4 h-4" />
            <span>Hyperparameter Settings</span>
          </div>

          <div className="space-y-4 text-xs">
            {/* Latent Bottleneck Dimension */}
            <div>
              <div className="flex justify-between text-slate-300 font-medium mb-1">
                <span>Latent Bottleneck Dimension (z)</span>
                <span className="text-indigo-400 font-bold">{params.latentDim}</span>
              </div>
              <input
                type="range"
                min="2"
                max="32"
                step="2"
                value={params.latentDim}
                onChange={(e) => setParams({ ...params, latentDim: parseInt(e.target.value) })}
                className="w-full accent-indigo-500 bg-slate-950 h-1.5 rounded cursor-pointer"
                disabled={isTraining}
              />
              <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
                <span>2 (Compressed)</span>
                <span>8 (Balanced)</span>
                <span>32 (High Capacity)</span>
              </div>
            </div>

            {/* Hidden Layers */}
            <div>
              <label className="block text-slate-300 font-medium mb-1">Hidden Layer Architecture</label>
              <select
                value={params.hiddenLayers}
                onChange={(e) => setParams({ ...params, hiddenLayers: e.target.value })}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 outline-none focus:ring-1 focus:ring-indigo-500"
                disabled={isTraining}
              >
                <option value="128, 64">[128, 64] — Default MLP Encoder</option>
                <option value="256, 128">[256, 128] — Deep Wide Encoder</option>
                <option value="64, 32">[64, 32] — Lightweight Fast MLP</option>
              </select>
            </div>

            {/* Activation Function */}
            <div>
              <label className="block text-slate-300 font-medium mb-1">Activation Function</label>
              <select
                value={params.activation}
                onChange={(e) => setParams({ ...params, activation: e.target.value })}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 outline-none focus:ring-1 focus:ring-indigo-500"
                disabled={isTraining}
              >
                <option value="LeakyReLU">LeakyReLU (α = 0.2)</option>
                <option value="ReLU">ReLU</option>
                <option value="ELU">ELU</option>
              </select>
            </div>

            {/* Learning Rate & Batch Size */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-slate-300 font-medium mb-1">Learning Rate</label>
                <input
                  type="number"
                  step="0.0001"
                  value={params.learningRate}
                  onChange={(e) => setParams({ ...params, learningRate: parseFloat(e.target.value) })}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-slate-200 outline-none"
                  disabled={isTraining}
                />
              </div>
              <div>
                <label className="block text-slate-300 font-medium mb-1">Batch Size</label>
                <select
                  value={params.batchSize}
                  onChange={(e) => setParams({ ...params, batchSize: parseInt(e.target.value) })}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-slate-200 outline-none"
                  disabled={isTraining}
                >
                  <option value="16">16</option>
                  <option value="32">32</option>
                  <option value="64">64</option>
                  <option value="128">128</option>
                </select>
              </div>
            </div>

            {/* Epochs & Beta Weight */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-slate-300 font-medium mb-1">Epochs</label>
                <input
                  type="number"
                  min="5"
                  max="200"
                  value={params.epochs}
                  onChange={(e) => setParams({ ...params, epochs: parseInt(e.target.value) })}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-slate-200 outline-none"
                  disabled={isTraining}
                />
              </div>
              <div>
                <label className="block text-slate-300 font-medium mb-1">Beta Weight (β-VAE)</label>
                <input
                  type="number"
                  step="0.1"
                  value={params.betaWeight}
                  onChange={(e) => setParams({ ...params, betaWeight: parseFloat(e.target.value) })}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-slate-200 outline-none"
                  disabled={isTraining}
                />
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="pt-2 space-y-2">
            {!isTraining ? (
              <button
                onClick={handleStartTraining}
                className="w-full flex items-center justify-center space-x-2 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-2.5 rounded-xl shadow-lg shadow-indigo-600/30 text-xs transition"
              >
                <Play className="w-4 h-4 fill-current" />
                <span>Start VAE Training</span>
              </button>
            ) : (
              <button
                onClick={handleStopTraining}
                className="w-full flex items-center justify-center space-x-2 bg-rose-600 hover:bg-rose-500 text-white font-semibold py-2.5 rounded-xl text-xs transition"
              >
                <Square className="w-4 h-4 fill-current" />
                <span>Interrupt Training</span>
              </button>
            )}
          </div>
        </div>

        {/* Right Column: Live Training Monitor & Loss Curve Chart (2 Cols) */}
        <div className="lg:col-span-2 bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center space-x-2 text-indigo-400 font-semibold text-sm">
              <Activity className="w-4 h-4" />
              <span>Real-Time Loss Dynamics</span>
            </div>
            <div className="flex items-center space-x-3 text-xs">
              <span className="text-slate-400">Epoch:</span>
              <span className="font-mono font-bold text-white">
                {currentEpoch} / {params.epochs} ({progressPercentage}%)
              </span>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
            <div 
              className="bg-gradient-to-r from-indigo-500 to-violet-500 h-full transition-all duration-150"
              style={{ width: `${progressPercentage}%` }}
            />
          </div>

          {/* Real-time Recharts Line Chart */}
          <div className="h-72 w-full pt-2">
            {lossData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lossData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="epoch" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} 
                  />
                  <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                  <Line type="monotone" dataKey="totalLoss" name="Total ELBO Loss" stroke="#6366f1" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="reconLoss" name="Reconstruction MSE" stroke="#10b981" strokeWidth={1.8} dot={false} />
                  <Line type="monotone" dataKey="klLoss" name="KL Divergence" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-slate-500 space-y-2">
                <BrainCircuit className="w-8 h-8 opacity-40" />
                <p className="text-xs">Click "Start VAE Training" to view live optimization curves</p>
              </div>
            )}
          </div>

          {/* Loss Metrics Badges */}
          {lossData.length > 0 && (
            <div className="grid grid-cols-3 gap-3 pt-2 text-xs">
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                <span className="text-slate-400 text-[11px]">Final Total Loss</span>
                <p className="text-base font-bold text-indigo-400 mt-0.5">
                  {lossData[lossData.length - 1].totalLoss}
                </p>
              </div>
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                <span className="text-slate-400 text-[11px]">Reconstruction MSE</span>
                <p className="text-base font-bold text-emerald-400 mt-0.5">
                  {lossData[lossData.length - 1].reconLoss}
                </p>
              </div>
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                <span className="text-slate-400 text-[11px]">KL Regularizer</span>
                <p className="text-base font-bold text-amber-400 mt-0.5">
                  {lossData[lossData.length - 1].klLoss}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
