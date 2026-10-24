import React, { useState } from 'react';
import { 
  BarChart3, 
  Award, 
  CheckCircle2, 
  AlertTriangle, 
  Sliders, 
  TrendingUp, 
  ArrowRight, 
  Download 
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';
import { DatasetInfo, ExperimentRun, SummaryStatComparison } from '../types';
import { statComparisons } from '../services/mockData';

interface QualityComparisonProps {
  dataset: DatasetInfo;
  activeExperiment: ExperimentRun | null;
  onNavigateToHistory: () => void;
}

export const QualityComparison: React.FC<QualityComparisonProps> = ({
  dataset,
  activeExperiment,
  onNavigateToHistory
}) => {
  const [selectedFeature, setSelectedFeature] = useState<string>('math_score');

  // Distribution bins generator for comparison chart
  const getDistributionData = (feature: string) => {
    return [
      { bin: '40-50', realCount: 45, syntheticCount: 42 },
      { bin: '50-60', realCount: 110, syntheticCount: 115 },
      { bin: '60-70', realCount: 240, syntheticCount: 235 },
      { bin: '70-80', realCount: 320, syntheticCount: 328 },
      { bin: '80-90', realCount: 210, syntheticCount: 204 },
      { bin: '90-100', realCount: 75, syntheticCount: 76 },
    ];
  };

  const distributionData = getDistributionData(selectedFeature);
  const fidelity = activeExperiment ? activeExperiment.fidelityScore : 93.8;
  const validity = activeExperiment ? activeExperiment.domainValidity : 99.8;

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="flex items-center justify-between bg-slate-900/80 border border-slate-800 p-5 rounded-2xl">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">
            Quality Comparison: Empirical Fidelity & Parity
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Evaluate statistical similarity between real and synthesized features using two-sample KS tests and correlation deltas.
          </p>
        </div>
        <button
          onClick={onNavigateToHistory}
          className="inline-flex items-center space-x-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-4 py-2 rounded-lg shadow-md shadow-indigo-600/30 transition"
        >
          <span>Export Certified CSV</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Top Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-gradient-to-br from-indigo-950/70 to-slate-900 border border-indigo-500/30 p-5 rounded-2xl space-y-1">
          <div className="flex items-center justify-between text-indigo-300 text-xs font-medium">
            <span>Overall Data Fidelity Score</span>
            <Award className="w-4 h-4 text-indigo-400" />
          </div>
          <p className="text-3xl font-extrabold text-white">{fidelity}%</p>
          <div className="flex items-center text-[11px] text-emerald-400 font-semibold space-x-1">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>High Fidelity Synthetic Quality</span>
          </div>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-2xl space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Domain Boundary Validity</span>
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          </div>
          <p className="text-3xl font-extrabold text-white">{validity}%</p>
          <p className="text-[11px] text-emerald-400 font-medium">Zero Out-of-Bounds Violations</p>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-2xl space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Avg Kolmogorov-Smirnov Stat</span>
            <TrendingUp className="w-4 h-4 text-cyan-400" />
          </div>
          <p className="text-3xl font-extrabold text-white">0.039</p>
          <p className="text-[11px] text-cyan-400 font-medium">p &gt; 0.05 (Distributions Match)</p>
        </div>
      </div>

      {/* Main Charts: Overlaid Distribution & Correlation Delta */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Feature Distribution Overlay Chart */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center space-x-2 text-indigo-400 font-semibold text-sm">
              <BarChart3 className="w-4 h-4" />
              <span>Feature Density Comparison</span>
            </div>
            <select
              value={selectedFeature}
              onChange={(e) => setSelectedFeature(e.target.value)}
              className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1 text-slate-200 text-xs outline-none"
            >
              <option value="math_score">math_score</option>
              <option value="reading_score">reading_score</option>
              <option value="writing_score">writing_score</option>
              <option value="study_hours">study_hours</option>
              <option value="attendance_rate">attendance_rate</option>
            </select>
          </div>

          <div className="h-64 w-full pt-1">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={distributionData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="bin" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} 
                />
                <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '8px' }} />
                <Bar dataKey="realCount" name="Real Ground Truth" fill="#6366f1" radius={[4, 4, 0, 0]} />
                <Bar dataKey="syntheticCount" name="VAE Synthetic Data" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-slate-400 text-center">
            Overlaid histograms confirming continuous distribution match across density bins.
          </p>
        </div>

        {/* Correlation Delta Matrix */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center space-x-2 text-indigo-400 font-semibold text-sm">
              <TrendingUp className="w-4 h-4" />
              <span>Correlation Matrix Delta (|Real - Synthetic|)</span>
            </div>
            <span className="text-[11px] font-mono text-emerald-400">Frobenius Norm: 0.041</span>
          </div>

          <div className="overflow-x-auto pt-2">
            <table className="w-full text-center text-xs font-mono">
              <thead>
                <tr className="text-slate-400 text-[11px]">
                  <th className="p-2 text-left">Feature</th>
                  <th className="p-2">math</th>
                  <th className="p-2">read</th>
                  <th className="p-2">write</th>
                  <th className="p-2">study</th>
                  <th className="p-2">atten</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                <tr>
                  <td className="p-2 text-left text-slate-300 font-sans font-semibold">math</td>
                  <td className="p-2 bg-slate-950 text-slate-500">0.00</td>
                  <td className="p-2 bg-indigo-950/40 text-emerald-400">0.02</td>
                  <td className="p-2 bg-indigo-950/40 text-emerald-400">0.03</td>
                  <td className="p-2 bg-indigo-950/40 text-emerald-400">0.04</td>
                  <td className="p-2 bg-indigo-950/40 text-emerald-400">0.02</td>
                </tr>
                <tr>
                  <td className="p-2 text-left text-slate-300 font-sans font-semibold">read</td>
                  <td className="p-2 bg-indigo-950/40 text-emerald-400">0.02</td>
                  <td className="p-2 bg-slate-950 text-slate-500">0.00</td>
                  <td className="p-2 bg-indigo-950/40 text-emerald-400">0.02</td>
                  <td className="p-2 bg-indigo-950/40 text-emerald-400">0.03</td>
                  <td className="p-2 bg-indigo-950/40 text-emerald-400">0.01</td>
                </tr>
                <tr>
                  <td className="p-2 text-left text-slate-300 font-sans font-semibold">write</td>
                  <td className="p-2 bg-indigo-950/40 text-emerald-400">0.03</td>
                  <td className="p-2 bg-indigo-950/40 text-emerald-400">0.02</td>
                  <td className="p-2 bg-slate-950 text-slate-500">0.00</td>
                  <td className="p-2 bg-indigo-950/40 text-emerald-400">0.02</td>
                  <td className="p-2 bg-indigo-950/40 text-emerald-400">0.03</td>
                </tr>
                <tr>
                  <td className="p-2 text-left text-slate-300 font-sans font-semibold">study</td>
                  <td className="p-2 bg-indigo-950/40 text-emerald-400">0.04</td>
                  <td className="p-2 bg-indigo-950/40 text-emerald-400">0.03</td>
                  <td className="p-2 bg-indigo-950/40 text-emerald-400">0.02</td>
                  <td className="p-2 bg-slate-950 text-slate-500">0.00</td>
                  <td className="p-2 bg-indigo-950/40 text-emerald-400">0.04</td>
                </tr>
                <tr>
                  <td className="p-2 text-left text-slate-300 font-sans font-semibold">atten</td>
                  <td className="p-2 bg-indigo-950/40 text-emerald-400">0.02</td>
                  <td className="p-2 bg-indigo-950/40 text-emerald-400">0.01</td>
                  <td className="p-2 bg-indigo-950/40 text-emerald-400">0.03</td>
                  <td className="p-2 bg-indigo-950/40 text-emerald-400">0.04</td>
                  <td className="p-2 bg-slate-950 text-slate-500">0.00</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-[11px] text-slate-400 text-center">
            Max feature pair correlation deviation &lt; 0.05, confirming multi-variable dependency preservation.
          </p>
        </div>
      </div>

      {/* Summary Statistics Comparison Table */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-4">
        <h3 className="font-semibold text-white text-base">
          Descriptive Statistics & Kolmogorov-Smirnov Hypothesis Tests
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/70 text-slate-400 uppercase tracking-wider text-[10px]">
              <tr>
                <th className="p-3">Feature</th>
                <th className="p-3">Real Mean (Std)</th>
                <th className="p-3">Synthetic Mean (Std)</th>
                <th className="p-3">Mean Delta</th>
                <th className="p-3">KS Statistic</th>
                <th className="p-3">p-value</th>
                <th className="p-3 text-right">Fidelity Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {statComparisons.map((stat) => (
                <tr key={stat.feature} className="hover:bg-slate-800/30 transition">
                  <td className="p-3 font-mono font-semibold text-cyan-300">{stat.feature}</td>
                  <td className="p-3 text-slate-300">{stat.realMean} (±{stat.realStd})</td>
                  <td className="p-3 text-slate-200 font-semibold">{stat.syntheticMean} (±{stat.syntheticStd})</td>
                  <td className="p-3 font-mono text-emerald-400">
                    {Math.abs(stat.realMean - stat.syntheticMean).toFixed(2)}
                  </td>
                  <td className="p-3 font-mono text-slate-300">{stat.ksStatistic}</td>
                  <td className="p-3 font-mono text-slate-300">{stat.pValue}</td>
                  <td className="p-3 text-right">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded text-[11px] font-semibold bg-emerald-950/60 text-emerald-400 border border-emerald-800/40">
                      <CheckCircle2 className="w-3 h-3 mr-1" /> {stat.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
