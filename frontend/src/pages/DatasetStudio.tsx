import React, { useState } from 'react';
import { 
  Upload, 
  FileText, 
  Database, 
  CheckCircle, 
  Settings2, 
  Sliders, 
  Layers, 
  BarChart, 
  ArrowRight,
  Sparkles,
  RefreshCw,
  AlertCircle
} from 'lucide-react';
import { ColumnMeta, DatasetInfo } from '../types';
import { uploadDataset } from '../services/api';

interface DatasetStudioProps {
  dataset: DatasetInfo;
  onUpdateDataset: (updated: DatasetInfo) => void;
  onNavigateToModelLab: () => void;
}

export const DatasetStudio: React.FC<DatasetStudioProps> = ({
  dataset,
  onUpdateDataset,
  onNavigateToModelLab
}) => {
  const [activeTab, setActiveTab] = useState<'preview' | 'schema' | 'stats'>('schema');
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [fileName, setFileName] = useState<string>(dataset.name);
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadedDatasetId, setUploadedDatasetId] = useState<string | null>(null);

  const processFile = (file: File) => {
    setFileName(file.name);
    setUploadError(null);
    // 1. Client-side parse for instant UI preview
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      parseCSV(text, file.name);
    };
    reader.readAsText(file);
    // 2. Backend upload for persistent storage + training
    setUploading(true);
    uploadDataset(file)
      .then((apiDataset) => { setUploadedDatasetId(apiDataset.id); setUploading(false); })
      .catch((err: Error) => { setUploadError(`Backend upload: ${err.message}`); setUploading(false); });
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    processFile(file);
  };

  const parseCSV = (csvText: string, name: string) => {
    const lines = csvText.trim().split('\n').filter(l => l.trim().length > 0);
    if (lines.length < 2) return;

    const headers = lines[0].split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
    const rows = lines.slice(1, 201).map(line => {
      const values = line.split(',').map(v => v.trim().replace(/^["']|["']$/g, ''));
      const rowObj: Record<string, any> = {};
      headers.forEach((h, i) => {
        const num = parseFloat(values[i]);
        rowObj[h] = !isNaN(num) && values[i] !== '' ? num : values[i];
      });
      return rowObj;
    });

    const columnsMeta: ColumnMeta[] = headers.map(header => {
      const sampleVals = rows.map(r => r[header]).filter(v => v !== undefined && v !== '');
      const isNum = sampleVals.every(v => typeof v === 'number');
      const isId = header.toLowerCase().includes('id');

      let min: number | undefined = undefined;
      let max: number | undefined = undefined;
      let mean: number | undefined = undefined;
      let std: number | undefined = undefined;
      if (isNum && sampleVals.length > 0) {
        const numVals = sampleVals as number[];
        min = Math.min(...numVals);
        max = Math.max(...numVals);
        const sum = numVals.reduce((a, b) => a + b, 0);
        mean = +(sum / numVals.length).toFixed(1);
        const variance = numVals.reduce((a, b) => a + Math.pow(b - (mean ?? 0), 2), 0) / numVals.length;
        std = +Math.sqrt(variance).toFixed(1);
      }

      return {
        name: header,
        inferredType: isId ? 'id' : isNum ? 'numeric' : 'categorical',
        selectedType: isId ? 'drop' : isNum ? 'numeric' : 'categorical',
        nullCount: rows.length - sampleVals.length,
        nullPercentage: +(((rows.length - sampleVals.length) / rows.length) * 100).toFixed(1),
        imputationStrategy: isNum ? 'median' : 'mode',
        included: !isId,
        min,
        max,
        mean,
        std
      };
    });

    const newDataset: DatasetInfo = {
      id: `ds_${Date.now()}`,
      name: name,
      rowCount: lines.length - 1,
      columnCount: headers.length,
      uploadDate: new Date().toISOString().replace('T', ' ').substring(0, 16),
      columns: columnsMeta,
      data: rows
    };

    onUpdateDataset(newDataset);
  };

  const handleToggleColumn = (colName: string) => {
    const updated = dataset.columns.map(c => 
      c.name === colName ? { ...c, included: !c.included } : c
    );
    onUpdateDataset({ ...dataset, columns: updated });
  };

  const handleTypeChange = (colName: string, newType: 'numeric' | 'categorical' | 'drop') => {
    const updated = dataset.columns.map(c => 
      c.name === colName ? { ...c, selectedType: newType, included: newType !== 'drop' } : c
    );
    onUpdateDataset({ ...dataset, columns: updated });
  };

  const handleImputationChange = (colName: string, strategy: any) => {
    const updated = dataset.columns.map(c => 
      c.name === colName ? { ...c, imputationStrategy: strategy } : c
    );
    onUpdateDataset({ ...dataset, columns: updated });
  };

  const includedColumns = dataset.columns.filter(c => c.included);

  return (
    <div className="space-y-6">
      {/* Upload Dropzone */}
      <div 
        onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragActive(false);
          const file = e.dataTransfer.files?.[0];
          if (file) processFile(file);
        }}
        className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all ${
          dragActive 
            ? 'border-indigo-500 bg-indigo-950/30' 
            : 'border-slate-800 bg-slate-900/60 hover:border-slate-700'
        }`}
      >
        <div className="max-w-md mx-auto space-y-3">
          <div className="w-12 h-12 rounded-xl bg-indigo-500/10 text-indigo-400 mx-auto flex items-center justify-center border border-indigo-500/20">
            <Upload className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-semibold text-white text-base">Drag & drop your tabular CSV dataset</h3>
            <p className="text-xs text-slate-400 mt-1">Supports student records, retail sales, healthcare metrics (Max 25 MB)</p>
          </div>
          <div className="flex items-center justify-center space-x-3 pt-2">
            <label className="cursor-pointer inline-flex items-center space-x-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-4 py-2 rounded-lg shadow-md transition">
              <span>Browse CSV File</span>
              <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
            </label>
          </div>
          <div className="text-[11px] text-slate-500 font-mono pt-1">
            Currently loaded: <span className="text-cyan-400 font-semibold">{fileName}</span> ({dataset.rowCount.toLocaleString()} rows, {dataset.columnCount} columns)
          </div>
        </div>
      </div>

      {/* Dataset Studio Tabs */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-5">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setActiveTab('schema')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                activeTab === 'schema' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              Schema & Feature Configuration ({includedColumns.length} of {dataset.columns.length} active)
            </button>
            <button
              onClick={() => setActiveTab('preview')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                activeTab === 'preview' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              Data Preview ({dataset.data.length} records)
            </button>
            <button
              onClick={() => setActiveTab('stats')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                activeTab === 'stats' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              Summary Statistics
            </button>
          </div>

          <button
            onClick={onNavigateToModelLab}
            className="inline-flex items-center space-x-2 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold px-4 py-2 rounded-lg shadow-md shadow-emerald-600/30 transition"
          >
            <span>Proceed to Model Lab</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Tab 1: Schema Configuration */}
        {activeTab === 'schema' && (
          <div className="space-y-4">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950/70 text-slate-400 uppercase tracking-wider text-[10px]">
                  <tr>
                    <th className="p-3">Include</th>
                    <th className="p-3">Column Name</th>
                    <th className="p-3">Inferred Type</th>
                    <th className="p-3">Role / Encoding</th>
                    <th className="p-3">Missing Values</th>
                    <th className="p-3">Imputation Rule</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {dataset.columns.map((col) => (
                    <tr key={col.name} className={`hover:bg-slate-800/30 transition ${!col.included ? 'opacity-50' : ''}`}>
                      <td className="p-3">
                        <input
                          type="checkbox"
                          checked={col.included}
                          onChange={() => handleToggleColumn(col.name)}
                          className="rounded border-slate-700 text-indigo-600 focus:ring-indigo-500 bg-slate-900 w-4 h-4"
                        />
                      </td>
                      <td className="p-3 font-mono font-semibold text-slate-200">{col.name}</td>
                      <td className="p-3">
                        <span className={`inline-flex px-2 py-0.5 rounded text-[10px] font-semibold ${
                          col.inferredType === 'numeric' ? 'bg-cyan-950 text-cyan-300 border border-cyan-800/40' :
                          col.inferredType === 'categorical' ? 'bg-purple-950 text-purple-300 border border-purple-800/40' :
                          'bg-slate-800 text-slate-400'
                        }`}>
                          {col.inferredType}
                        </span>
                      </td>
                      <td className="p-3">
                        <select
                          value={col.selectedType}
                          onChange={(e) => handleTypeChange(col.name, e.target.value as any)}
                          className="bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-slate-200 text-xs focus:ring-1 focus:ring-indigo-500 outline-none"
                        >
                          <option value="numeric">Numerical (Min-Max [0,1])</option>
                          <option value="categorical">Categorical (One-Hot)</option>
                          <option value="drop">Drop / Exclude</option>
                        </select>
                      </td>
                      <td className="p-3 text-slate-300">
                        {col.nullCount} ({col.nullPercentage}%)
                      </td>
                      <td className="p-3">
                        <select
                          value={col.imputationStrategy}
                          onChange={(e) => handleImputationChange(col.name, e.target.value)}
                          className="bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-slate-200 text-xs focus:ring-1 focus:ring-indigo-500 outline-none"
                        >
                          <option value="median">Median Imputation</option>
                          <option value="mean">Mean Imputation</option>
                          <option value="mode">Mode / Most Frequent</option>
                          <option value="drop">Drop Row</option>
                        </select>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Tab 2: Raw Preview Table */}
        {activeTab === 'preview' && (
          <div className="overflow-x-auto max-h-96">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/80 sticky top-0 text-slate-400 uppercase tracking-wider text-[10px]">
                <tr>
                  {dataset.columns.filter(c => c.included).map((col) => (
                    <th key={col.name} className="p-3">{col.name}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {dataset.data.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/30 transition">
                    {dataset.columns.filter(c => c.included).map((col) => (
                      <td key={col.name} className="p-3 font-mono text-slate-300">{row[col.name]}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Tab 3: Summary Statistics */}
        {activeTab === 'stats' && (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/70 text-slate-400 uppercase tracking-wider text-[10px]">
                <tr>
                  <th className="p-3">Numerical Feature</th>
                  <th className="p-3">Min</th>
                  <th className="p-3">Mean</th>
                  <th className="p-3">Std Dev</th>
                  <th className="p-3">Max</th>
                  <th className="p-3">Scaled Target</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {dataset.columns.filter(c => c.included && c.min !== undefined).map((col) => (
                  <tr key={col.name} className="hover:bg-slate-800/30 transition">
                    <td className="p-3 font-mono font-semibold text-cyan-300">{col.name}</td>
                    <td className="p-3 text-slate-300">{col.min}</td>
                    <td className="p-3 text-slate-200 font-semibold">{col.mean}</td>
                    <td className="p-3 text-slate-300">±{col.std}</td>
                    <td className="p-3 text-slate-300">{col.max}</td>
                    <td className="p-3 text-emerald-400 font-mono">[0.0, 1.0]</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
