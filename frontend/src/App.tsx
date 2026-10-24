import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Navbar } from './components/Navbar';
import { Dashboard } from './pages/Dashboard';
import { DatasetStudio } from './pages/DatasetStudio';
import { ModelLab } from './pages/ModelLab';
import { DataGenerator } from './pages/DataGenerator';
import { QualityComparison } from './pages/QualityComparison';
import { ExportHistory } from './pages/ExportHistory';
import { 
  DatasetInfo, 
  ModelRecord, 
  ExperimentRun 
} from './types';
import { 
  initialStudentDataset, 
  initialModels, 
  initialExperiments 
} from './services/mockData';

export default function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'studio' | 'model' | 'generator' | 'quality' | 'history'>('dashboard');
  
  // Core application state
  const [datasets, setDatasets] = useState<DatasetInfo[]>([initialStudentDataset]);
  const [activeDataset, setActiveDataset] = useState<DatasetInfo>(initialStudentDataset);
  
  const [models, setModels] = useState<ModelRecord[]>(initialModels);
  const [activeModel, setActiveModel] = useState<ModelRecord>(initialModels[0]);
  
  const [experiments, setExperiments] = useState<ExperimentRun[]>(initialExperiments);
  const [activeExperiment, setActiveExperiment] = useState<ExperimentRun>(initialExperiments[0]);

  // Backend Health check
  const [backendOnline, setBackendOnline] = useState<boolean>(false);
  const [backendChecking, setBackendChecking] = useState<boolean>(false);
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

  const checkHealth = async () => {
    setBackendChecking(true);
    try {
      const res = await fetch(`${apiBaseUrl}/api/health`);
      if (res.ok) {
        setBackendOnline(true);
      } else {
        setBackendOnline(false);
      }
    } catch {
      setBackendOnline(false);
    } finally {
      setBackendChecking(false);
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  // Handlers for state updates across modules
  const handleUpdateDataset = (updated: DatasetInfo) => {
    setActiveDataset(updated);
    setDatasets(prev => {
      const idx = prev.findIndex(d => d.id === updated.id);
      if (idx >= 0) {
        const copy = [...prev];
        copy[idx] = updated;
        return copy;
      }
      return [updated, ...prev];
    });
  };

  const handleSaveModel = (newModel: ModelRecord) => {
    setActiveModel(newModel);
    setModels(prev => [newModel, ...prev]);
  };

  const handleGenerationComplete = (newRun: ExperimentRun) => {
    setActiveExperiment(newRun);
    setExperiments(prev => [newRun, ...prev]);
  };

  const pageTitles: Record<string, string> = {
    dashboard: 'Platform Dashboard',
    studio: 'Dataset Studio & Feature Profiling',
    model: 'Model Lab — VAE Training Suite',
    generator: 'Data Generator — Latent Sampling',
    quality: 'Quality Comparison & Statistical Fidelity',
    history: 'Export Vault & Experiment History'
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* Sidebar Navigation */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        backendOnline={backendOnline}
        backendChecking={backendChecking}
        apiBaseUrl={apiBaseUrl}
      />

      {/* Main View Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Navbar */}
        <Navbar
          title={pageTitles[activeTab] || 'DataForge'}
          activeDataset={activeDataset}
          activeModel={activeModel}
          onRefresh={checkHealth}
          loading={backendChecking}
        />

        {/* Scrollable Page Body */}
        <main className="flex-1 overflow-y-auto p-8 bg-slate-950">
          <div className="max-w-7xl mx-auto pb-12">
            {activeTab === 'dashboard' && (
              <Dashboard
                datasets={datasets}
                models={models}
                experiments={experiments}
                activeDataset={activeDataset}
                activeModel={activeModel}
                onNavigate={setActiveTab}
              />
            )}

            {activeTab === 'studio' && (
              <DatasetStudio
                dataset={activeDataset}
                onUpdateDataset={handleUpdateDataset}
                onNavigateToModelLab={() => setActiveTab('model')}
              />
            )}

            {activeTab === 'model' && (
              <ModelLab
                dataset={activeDataset}
                activeModel={activeModel}
                onSaveModel={handleSaveModel}
                onNavigateToGenerator={() => setActiveTab('generator')}
              />
            )}

            {activeTab === 'generator' && (
              <DataGenerator
                dataset={activeDataset}
                models={models}
                activeModel={activeModel}
                onGenerationComplete={handleGenerationComplete}
                onNavigateToQuality={() => setActiveTab('quality')}
                onNavigateToHistory={() => setActiveTab('history')}
              />
            )}

            {activeTab === 'quality' && (
              <QualityComparison
                dataset={activeDataset}
                activeExperiment={activeExperiment}
                onNavigateToHistory={() => setActiveTab('history')}
              />
            )}

            {activeTab === 'history' && (
              <ExportHistory
                experiments={experiments}
                activeExperiment={activeExperiment}
              />
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
