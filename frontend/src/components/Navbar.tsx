import React from 'react';
import { Database, BrainCircuit, Sparkles, RefreshCw } from 'lucide-react';
import { ModelRecord } from '../types';
import type { ApiDataset } from '../types/api';

interface NavbarProps {
  title: string;
  activeDataset: ApiDataset | null;
  activeModel: ModelRecord | null;
  onRefresh: () => void;
  loading: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  title,
  activeDataset,
  activeModel,
  onRefresh,
  loading
}) => {
  return (
    <header className="h-16 border-b border-slate-800 bg-slate-900/40 backdrop-blur px-8 flex items-center justify-between z-10">
      <div className="flex items-center space-x-3">
        <h2 className="text-lg font-bold text-white tracking-tight">{title}</h2>
        <span className="bg-indigo-500/10 text-indigo-400 text-xs px-2.5 py-0.5 rounded-full border border-indigo-500/20 font-medium">
          AI23531 Deep Learning
        </span>
      </div>

      <div className="flex items-center space-x-4">
        {/* Active Dataset Pill */}
        <div className="flex items-center space-x-2 bg-slate-800/70 border border-slate-700/60 px-3 py-1.5 rounded-lg text-xs">
          <Database className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-slate-400">Dataset:</span>
          <span className="font-semibold text-slate-200">
            {activeDataset ? activeDataset.filename : 'None selected'}
          </span>
          {activeDataset && (
            <span className="text-[10px] text-cyan-400 bg-cyan-950/60 px-1.5 py-0.2 rounded border border-cyan-800/40">
              {activeDataset.rowCount} rows
            </span>
          )}
        </div>

        {/* Active Model Pill */}
        <div className="flex items-center space-x-2 bg-slate-800/70 border border-slate-700/60 px-3 py-1.5 rounded-lg text-xs">
          <BrainCircuit className="w-3.5 h-3.5 text-violet-400" />
          <span className="text-slate-400">Model:</span>
          <span className="font-semibold text-slate-200 truncate max-w-[140px]">
            {activeModel ? activeModel.name : 'None trained'}
          </span>
          {activeModel && (
            <span className="text-[10px] text-emerald-400 bg-emerald-950/60 px-1.5 py-0.2 rounded border border-emerald-800/40">
              z={activeModel.params.latentDim}
            </span>
          )}
        </div>

        {/* Refresh Action */}
        <button
          onClick={onRefresh}
          disabled={loading}
          className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 transition"
          title="Refresh Backend Bridge"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>
    </header>
  );
};
