import React from 'react';
import { 
  Database, 
  BrainCircuit, 
  Sparkles, 
  Award, 
  ArrowRight, 
  Play, 
  Upload, 
  ExternalLink, 
  Layers, 
  TrendingUp, 
  CheckCircle2, 
  FileText 
} from 'lucide-react';
import { ModelRecord } from '../types';
import type { ApiDataset, ApiExperiment } from '../types/api';
import type { TabId } from '../routing';

interface DashboardProps {
  datasets: ApiDataset[];
  models: ModelRecord[];
  experiments: ApiExperiment[];
  activeDataset: ApiDataset | null;
  activeModel: ModelRecord | null;
  onNavigate: (tab: TabId) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({
  datasets,
  models,
  experiments,
  activeDataset,
  activeModel,
  onNavigate
}) => {
  const totalSyntheticRows = experiments.reduce((acc, exp) => acc + exp.num_records, 0);
  const scored = experiments.filter((exp) => typeof exp.fidelity_score === 'number');
  const validated = experiments.filter((exp) => typeof exp.validity_score === 'number');
  const mean = (xs: number[]) => (xs.length ? (xs.reduce((a, b) => a + b, 0) / xs.length).toFixed(1) : null);
  const avgFidelity = mean(scored.map((e) => e.fidelity_score as number));
  const avgValidity = mean(validated.map((e) => e.validity_score as number));

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-gradient-to-r from-indigo-900/60 via-slate-900/90 to-purple-900/50 border border-indigo-500/20 rounded-2xl p-6 relative overflow-hidden shadow-xl">
        <div className="relative z-10 max-w-2xl space-y-2">
          <div className="inline-flex items-center space-x-2 bg-indigo-500/20 text-indigo-300 text-xs px-3 py-1 rounded-full border border-indigo-500/30 font-medium">
            <Sparkles className="w-3.5 h-3.5" />
            <span>AI23531 Generative Deep Learning Project</span>
          </div>
          <h2 className="text-2xl font-bold text-white tracking-tight">
            DataForge: Tabular Synthetic Data Generation
          </h2>
          <p className="text-sm text-slate-300 leading-relaxed">
            Generate high-fidelity artificial tabular datasets using Variational Autoencoders (VAEs). 
            Preserve continuous non-linear correlations, safeguard raw records, and validate statistical parity.
          </p>
          <div className="pt-2 flex flex-wrap gap-3">
            <button
              onClick={() => onNavigate('studio')}
              className="inline-flex items-center space-x-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-4 py-2 rounded-lg shadow-lg shadow-indigo-600/30 transition"
            >
              <Upload className="w-3.5 h-3.5" />
              <span>Upload CSV Dataset</span>
            </button>
            <button
              onClick={() => onNavigate('model')}
              className="inline-flex items-center space-x-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold px-4 py-2 rounded-lg border border-slate-700 transition"
            >
              <BrainCircuit className="w-3.5 h-3.5 text-violet-400" />
              <span>Configure VAE Lab</span>
            </button>
            <button
              onClick={() => onNavigate('generator')}
              className="inline-flex items-center space-x-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold px-4 py-2 rounded-lg border border-slate-700 transition"
            >
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              <span>Sample Synthetic Data</span>
            </button>
          </div>
        </div>
      </div>

      {/* KPI Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-xl space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Uploaded Datasets</span>
            <Database className="w-4 h-4 text-cyan-400" />
          </div>
          <p className="text-2xl font-extrabold text-white">{datasets.length}</p>
          <p className="text-[11px] text-cyan-400 font-medium">Active: {activeDataset ? activeDataset.filename : 'None'}</p>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-xl space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Models Trained</span>
            <BrainCircuit className="w-4 h-4 text-violet-400" />
          </div>
          <p className="text-2xl font-extrabold text-white">{models.length}</p>
          <p className="text-[11px] text-violet-400 font-medium">Active: {activeModel ? activeModel.name : 'None'}</p>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-xl space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Synthetic Rows</span>
            <TrendingUp className="w-4 h-4 text-emerald-400" />
          </div>
          <p className="text-2xl font-extrabold text-white">{totalSyntheticRows.toLocaleString()}</p>
          <p className="text-[11px] text-emerald-400 font-medium">Generated across {experiments.length} runs</p>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-xl space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Avg Quality Fidelity</span>
            <Award className="w-4 h-4 text-amber-400" />
          </div>
          <p className="text-2xl font-extrabold text-white">{avgFidelity !== null ? `${avgFidelity}%` : 'no data'}</p>
          <p className="text-[11px] text-amber-400 font-medium">Validity: {avgValidity !== null ? `${avgValidity}%` : 'no data'} · heuristic scores, averaged over saved runs</p>
        </div>
      </div>

      {/* Main Content Grid: Recent Runs & DL Architecture Info */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Experiments (2 Cols) */}
        <div className="lg:col-span-2 bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-white text-base">Recent Experiment Runs</h3>
            <button
              onClick={() => onNavigate('history')}
              className="text-xs text-indigo-400 hover:text-indigo-300 font-medium flex items-center space-x-1"
            >
              <span>View All Runs</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/70 text-slate-400 uppercase tracking-wider text-[10px]">
                <tr>
                  <th className="p-3">Run ID</th>
                  <th className="p-3">Dataset</th>
                  <th className="p-3">Model</th>
                  <th className="p-3">Records</th>
                  <th className="p-3">Fidelity</th>
                  <th className="p-3">Validity</th>
                  <th className="p-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {experiments.slice(0, 8).map((exp) => (
                  <tr key={exp.id} className="hover:bg-slate-800/30 transition">
                    <td className="p-3 font-mono text-slate-300 font-medium">{exp.id}</td>
                    <td className="p-3 text-slate-200">{exp.dataset_name ?? exp.dataset_id}</td>
                    <td className="p-3 text-slate-400">{exp.model_name ?? exp.model_id}</td>
                    <td className="p-3 font-semibold text-white">{exp.num_records.toLocaleString()}</td>
                    <td className="p-3">
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-950/60 text-emerald-400 border border-emerald-800/40">
                        {typeof exp.fidelity_score === 'number' ? `${exp.fidelity_score}%` : 'not available'}
                      </span>
                    </td>
                    <td className="p-3">
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold bg-indigo-950/60 text-indigo-400 border border-indigo-800/40">
                        {typeof exp.validity_score === 'number' ? `${exp.validity_score}%` : 'not available'}
                      </span>
                    </td>
                    <td className="p-3 text-right">
                      <button
                        onClick={() => onNavigate('quality')}
                        className="text-indigo-400 hover:text-indigo-300 hover:underline font-semibold"
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {experiments.length === 0 && (
              <p className="p-3 text-xs text-slate-500" data-testid="dashboard-no-runs">No experiments yet. Generate synthetic data to see runs here.</p>
            )}
          </div>
        </div>

        {/* Deep Learning Syllabus & Theory Card */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center space-x-2 text-indigo-400 font-semibold text-sm">
            <Layers className="w-4 h-4" />
            <span>Deep Learning Formulation</span>
          </div>

          <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 space-y-2 text-xs">
            <div className="text-slate-400 font-mono">Evidence Lower Bound (ELBO):</div>
            <div className="text-indigo-300 font-mono font-medium p-2 bg-slate-900 rounded border border-slate-800">
              L = w · ‖x − x̂‖² + β · D_KL(q(z|x) || p(z))
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed pt-1">
              Balances feature reconstruction accuracy against continuous latent Gaussian distribution smoothness.
            </p>
          </div>

          <div className="space-y-2.5 text-xs">
            <div className="flex items-start space-x-2">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 mt-0.5 flex-shrink-0" />
              <div>
                <span className="text-slate-200 font-semibold">Unit I (MLP Networks):</span>
                <p className="text-slate-400 text-[11px]">Encoder & Decoder Dense layers with LeakyReLU activations.</p>
              </div>
            </div>
            <div className="flex items-start space-x-2">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 mt-0.5 flex-shrink-0" />
              <div>
                <span className="text-slate-200 font-semibold">Unit II (Training):</span>
                <p className="text-slate-400 text-[11px]">Adam optimization, mini-batch gradient descent & loss tracking.</p>
              </div>
            </div>
            <div className="flex items-start space-x-2">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 mt-0.5 flex-shrink-0" />
              <div>
                <span className="text-slate-200 font-semibold">Unit V (Generative DL):</span>
                <p className="text-slate-400 text-[11px]">VAE latent space sampling z = μ + σ ⊙ ε, where ε ~ N(0, I).</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
