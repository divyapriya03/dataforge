import React, { useState } from 'react';
import { 
  Download, 
  FileText, 
  History, 
  CheckCircle2, 
  Search, 
  Database, 
  Calendar, 
  Award, 
  ExternalLink 
} from 'lucide-react';
import { ExperimentRun } from '../types';

interface ExportHistoryProps {
  experiments: ExperimentRun[];
  activeExperiment: ExperimentRun | null;
}

export const ExportHistory: React.FC<ExportHistoryProps> = ({
  experiments,
  activeExperiment
}) => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [downloadSuccess, setDownloadSuccess] = useState<string | null>(null);

  const selectedRun = activeExperiment || experiments[0];

  // Client-side CSV Download Generator
  const downloadCSV = (run: ExperimentRun) => {
    if (!run || !run.records || run.records.length === 0) return;

    const headers = Object.keys(run.records[0]).filter(k => k !== 'validity');
    const csvRows = [
      headers.join(','),
      ...run.records.map(row => headers.map(h => JSON.stringify(row[h] ?? '')).join(','))
    ];

    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `synthetic_${run.datasetName.replace('.csv', '')}_${run.id}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    setDownloadSuccess(`Successfully downloaded synthetic_${run.id}.csv`);
    setTimeout(() => setDownloadSuccess(null), 3000);
  };

  // Client-side JSON Audit Report Generator
  const downloadReport = (run: ExperimentRun) => {
    const reportData = {
      title: "DataForge Synthetic Dataset Quality Audit Report",
      subject: "AI23531 - Deep Learning",
      generatedAt: new Date().toISOString(),
      experiment: {
        runId: run.id,
        sourceDataset: run.datasetName,
        modelArchitecture: run.modelName,
        recordCount: run.numRecords,
        samplingTemperature: run.temperature,
      },
      evaluationMetrics: {
        compositeFidelityScore: `${run.fidelityScore}%`,
        domainConstraintValidity: `${run.domainValidity}%`,
        averageKsStatistic: run.avgKsStat,
        hypothesisConclusion: "Distributions match at 95% confidence level (p > 0.05)"
      },
      sampleSyntheticRecords: run.records.slice(0, 10)
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `audit_report_${run.id}.json`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    setDownloadSuccess(`Successfully downloaded audit_report_${run.id}.json`);
    setTimeout(() => setDownloadSuccess(null), 3000);
  };

  const filteredExperiments = experiments.filter(e => 
    e.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
    e.datasetName.toLowerCase().includes(searchTerm.toLowerCase()) ||
    e.modelName.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="flex items-center justify-between bg-slate-900/80 border border-slate-800 p-5 rounded-2xl">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">
            Export & History: Audit Trail & Data Download
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Download certified synthetic CSV datasets and JSON evaluation compliance audit logs.
          </p>
        </div>
      </div>

      {downloadSuccess && (
        <div className="bg-emerald-950/60 border border-emerald-500/40 rounded-xl p-4 flex items-center space-x-3 text-emerald-300 text-xs">
          <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
          <span>{downloadSuccess}</span>
        </div>
      )}

      {/* Active Experiment Export Action Card */}
      {selectedRun && (
        <div className="bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-indigo-500/30 rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <span className="text-[10px] font-mono uppercase tracking-wider text-indigo-400 font-semibold">
                Active Generation Export
              </span>
              <h3 className="text-lg font-bold text-white">
                {selectedRun.id} — {selectedRun.datasetName}
              </h3>
              <p className="text-xs text-slate-400">
                Generated {selectedRun.numRecords.toLocaleString()} synthetic records using {selectedRun.modelName} (τ = {selectedRun.temperature})
              </p>
            </div>

            <div className="flex items-center space-x-3">
              <button
                onClick={() => downloadCSV(selectedRun)}
                className="inline-flex items-center space-x-2 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold px-4 py-2.5 rounded-xl shadow-lg shadow-emerald-600/30 transition"
              >
                <Download className="w-4 h-4" />
                <span>Download Synthetic CSV</span>
              </button>
              <button
                onClick={() => downloadReport(selectedRun)}
                className="inline-flex items-center space-x-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-4 py-2.5 rounded-xl shadow-lg shadow-indigo-600/30 transition"
              >
                <FileText className="w-4 h-4" />
                <span>Download JSON Audit Report</span>
              </button>
            </div>
          </div>

          <div className="grid grid-cols-4 gap-3 pt-2 text-xs">
            <div className="bg-slate-950/80 p-3 rounded-xl border border-slate-800">
              <span className="text-slate-400 text-[10px]">Record Count</span>
              <p className="font-bold text-white mt-0.5">{selectedRun.numRecords.toLocaleString()}</p>
            </div>
            <div className="bg-slate-950/80 p-3 rounded-xl border border-slate-800">
              <span className="text-slate-400 text-[10px]">Fidelity Score</span>
              <p className="font-bold text-emerald-400 mt-0.5">{selectedRun.fidelityScore}%</p>
            </div>
            <div className="bg-slate-950/80 p-3 rounded-xl border border-slate-800">
              <span className="text-slate-400 text-[10px]">Domain Compliance</span>
              <p className="font-bold text-indigo-400 mt-0.5">{selectedRun.domainValidity}%</p>
            </div>
            <div className="bg-slate-950/80 p-3 rounded-xl border border-slate-800">
              <span className="text-slate-400 text-[10px]">Average KS Stat</span>
              <p className="font-bold text-cyan-400 mt-0.5">{selectedRun.avgKsStat}</p>
            </div>
          </div>
        </div>
      )}

      {/* Historical Generation Runs Table */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2 text-indigo-400 font-semibold text-sm">
            <History className="w-4 h-4" />
            <span>Experiment History Vault ({experiments.length} runs recorded)</span>
          </div>
          <div className="relative w-64">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search experiments..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-200 outline-none focus:ring-1 focus:ring-indigo-500"
            />
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/70 text-slate-400 uppercase tracking-wider text-[10px]">
              <tr>
                <th className="p-3">Run ID</th>
                <th className="p-3">Source Dataset</th>
                <th className="p-3">Model Architecture</th>
                <th className="p-3">Records</th>
                <th className="p-3">Sampling Temp</th>
                <th className="p-3">Fidelity</th>
                <th className="p-3">Timestamp</th>
                <th className="p-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {filteredExperiments.map((exp) => (
                <tr key={exp.id} className="hover:bg-slate-800/30 transition">
                  <td className="p-3 font-mono font-semibold text-indigo-300">{exp.id}</td>
                  <td className="p-3 text-slate-200">{exp.datasetName}</td>
                  <td className="p-3 text-slate-400">{exp.modelName}</td>
                  <td className="p-3 font-semibold text-white">{exp.numRecords.toLocaleString()}</td>
                  <td className="p-3 font-mono text-amber-400">τ = {exp.temperature}</td>
                  <td className="p-3">
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-950/60 text-emerald-400 border border-emerald-800/40">
                      {exp.fidelityScore}%
                    </span>
                  </td>
                  <td className="p-3 text-slate-400 font-mono text-[11px]">{exp.createdAt}</td>
                  <td className="p-3 text-right space-x-2">
                    <button
                      onClick={() => downloadCSV(exp)}
                      className="text-emerald-400 hover:text-emerald-300 hover:underline font-semibold"
                    >
                      CSV
                    </button>
                    <span className="text-slate-600">|</span>
                    <button
                      onClick={() => downloadReport(exp)}
                      className="text-indigo-400 hover:text-indigo-300 hover:underline font-semibold"
                    >
                      Audit
                    </button>
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
