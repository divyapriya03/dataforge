// src/components/DataGenerator.tsx
import React, { useState } from 'react';
import { generateSynthetic } from '../services/api';

interface SyntheticRecord {
  [key: string]: any;
}

export default function DataGenerator() {
  const [datasetId, setDatasetId] = useState(''); // to be selected by user
  const [modelId, setModelId] = useState(''); // to be selected by user
  const [numRecords, setNumRecords] = useState(1000);
  const [temperature, setTemperature] = useState(1.0);
  const [seed, setSeed] = useState(42);
  const [preview, setPreview] = useState<SyntheticRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  const handleGenerate = async () => {
    if (!datasetId || !modelId) {
      setError('Dataset and model must be selected');
      return;
    }
    setError('');
    setLoading(true);
    try {
      const resp = await generateSynthetic(modelId, {
        dataset_id: datasetId,
        num_records: numRecords,
        temperature,
        random_seed: seed,
      });
      if (resp.success && resp.data) {
        setPreview(resp.data.preview_records);
      } else {
        setError(resp.error || 'Generation failed');
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">Data Generator</h2>
      {error && <div className="text-red-600 mb-2">{error}</div>}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div>
          <label className="block mb-1">Dataset ID</label>
          <input
            type="text"
            className="border rounded p-2 w-full"
            value={datasetId}
            onChange={(e) => setDatasetId(e.target.value)}
            placeholder="e.g., ds_abcd1234"
          />
        </div>
        <div>
          <label className="block mb-1">Model ID</label>
          <input
            type="text"
            className="border rounded p-2 w-full"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            placeholder="e.g., vae_abcd1234"
          />
        </div>
        <div>
          <label className="block mb-1">Number of records</label>
          <input
            type="number"
            className="border rounded p-2 w-full"
            value={numRecords}
            onChange={(e) => setNumRecords(Number(e.target.value))}
            min={10}
          />
        </div>
        <div>
          <label className="block mb-1">Temperature</label>
          <input
            type="number"
            step="0.1"
            className="border rounded p-2 w-full"
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
            min={0.1}
          />
        </div>
        <div>
          <label className="block mb-1">Random Seed</label>
          <input
            type="number"
            className="border rounded p-2 w-full"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
          />
        </div>
        <div className="flex items-end">
          <button
            className="bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:opacity-50"
            onClick={handleGenerate}
            disabled={loading}
          >
            {loading ? 'Generating...' : 'Generate'}
          </button>
        </div>
      </div>

      {preview.length > 0 && (
        <div className="mt-6">
          <h3 className="font-semibold mb-2">Preview (first {preview.length} records)</h3>
          <div className="overflow-auto max-h-96 border rounded">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-100">
                <tr>
                  {Object.keys(preview[0] || {}).map((col) => (
                    <th key={col} className="px-2 py-1 text-left border">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.map((row, idx) => (
                  <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    {Object.values(row).map((val, i) => (
                      <td key={i} className="px-2 py-1 border">
                        {String(val)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
