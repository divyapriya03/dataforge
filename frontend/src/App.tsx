import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { Navbar } from './components/Navbar';
import { Dashboard } from './pages/Dashboard';
import { DatasetStudio } from './pages/DatasetStudio';
import { ModelLab } from './pages/ModelLab';
import { DataGenerator } from './pages/DataGenerator';
import { QualityComparison } from './pages/QualityComparison';
import { ExportHistory } from './pages/ExportHistory';
import { ModelRecord } from './types';
import { API_BASE_URL, getDatasets, getExperiments, getHealth, getModels } from './services/api';
import { apiModelToRecord, upsertModel } from './services/mappers';
import { TAB_PARAMS, buildHash, parseHash, type Route, type TabId } from './routing';
import type { ApiColumn, ApiDataset, ApiExperiment, ApiGenerateResponse, ApiModel } from './types/api';

export { TAB_PATHS as TAB_HASHES } from './routing';

type LoadState = { phase: 'loading' } | { phase: 'ready' } | { phase: 'error'; message: string };

const PAGE_TITLES: Record<TabId, string> = {
  dashboard: 'Platform Dashboard',
  studio: 'Dataset Studio & Preprocessing',
  model: 'Model Lab — VAE Training',
  generator: 'Data Generator — Synthetic Records',
  quality: 'Quality Comparison — Saved Evaluation',
  history: 'Export Vault & Experiment History',
};

export default function App() {
  // ---- routing: `#/page?param=id`; identifiers travel in the URL so links and reloads keep them
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    const onHashChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const navigate = (tab: TabId, params: Record<string, string | undefined> = {}) => {
    const hash = buildHash(tab, params);
    setRoute(parseHash(hash));
    if (window.location.hash !== hash) window.location.hash = hash;
  };

  // ---- backend data (nothing is seeded locally)
  const [datasets, setDatasets] = useState<ApiDataset[]>([]);
  const [datasetsState, setDatasetsState] = useState<LoadState>({ phase: 'loading' });
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [experiments, setExperiments] = useState<ApiExperiment[]>([]);
  const [experimentsState, setExperimentsState] = useState<LoadState>({ phase: 'loading' });

  // ---- current selections (kept in sync with the URL parameters)
  const [activeDatasetId, setActiveDatasetId] = useState('');
  const [activeModelId, setActiveModelId] = useState('');
  const [activeExperimentId, setActiveExperimentId] = useState('');
  // Column configurations edited in Dataset Studio, used by Model Lab when training (memory only)
  const [columnConfigs, setColumnConfigs] = useState<Record<string, ApiColumn[]>>({});

  useEffect(() => {
    if (route.params.dataset) setActiveDatasetId(route.params.dataset);
    if (route.params.model) setActiveModelId(route.params.model);
    if (route.params.experiment) setActiveExperimentId(route.params.experiment);
  }, [route]);

  const activeDataset = datasets.find((d) => d.id === activeDatasetId) ?? datasets[0] ?? null;
  const activeModel = models.find((m) => m.id === activeModelId && m.status === 'completed') ?? models.find((m) => m.status === 'completed') ?? null;
  const activeExperiment = experiments.find((e) => e.id === activeExperimentId) ?? experiments[0] ?? null;

  // ---- backend health
  const [backendOnline, setBackendOnline] = useState<boolean>(false);
  const [backendChecking, setBackendChecking] = useState<boolean>(false);
  const checkHealth = async () => {
    setBackendChecking(true);
    const res = await getHealth();
    setBackendOnline(res.success);
    setBackendChecking(false);
  };

  const reloadDatasets = useCallback(async (): Promise<ApiDataset[]> => {
    setDatasetsState({ phase: 'loading' });
    const res = await getDatasets();
    if (res.success && Array.isArray(res.data)) {
      setDatasets(res.data);
      setDatasetsState({ phase: 'ready' });
      return res.data;
    }
    setDatasets([]);
    setDatasetsState({ phase: 'error', message: res.error ?? 'Unexpected response from /api/datasets' });
    return [];
  }, []);

  /** Reload trained models. Resolves to an error message, or null on success. */
  const refreshModels = useCallback(async (): Promise<string | null> => {
    const res = await getModels();
    if (!res.success || !Array.isArray(res.data)) return res.error ?? 'Unexpected response from /api/models';
    setModels(res.data.map(apiModelToRecord));
    return null;
  }, []);

  const applyExperiments = useCallback((list: ApiExperiment[]) => {
    setExperiments(list);
    setExperimentsState({ phase: 'ready' });
  }, []);

  const reloadExperiments = useCallback(async () => {
    setExperimentsState({ phase: 'loading' });
    const res = await getExperiments();
    if (res.success && Array.isArray(res.data)) applyExperiments(res.data);
    else setExperimentsState({ phase: 'error', message: res.error ?? 'Unexpected response from /api/experiments' });
  }, [applyExperiments]);

  useEffect(() => {
    checkHealth();
    reloadDatasets();
    refreshModels();
    reloadExperiments();
  }, []);

  // ---- navigation helpers: carry the relevant identifier along
  const idFor = (param: string | null): string | undefined =>
    param === 'dataset' ? activeDataset?.id : param === 'model' ? activeModel?.id : param === 'experiment' ? activeExperiment?.id : undefined;

  const goTab = (tab: TabId) => {
    const param = TAB_PARAMS[tab];
    navigate(tab, param ? { [param]: idFor(param) } : {});
  };

  const selectDataset = (id: string) => {
    setActiveDatasetId(id);
    if (route.tab === 'studio' || route.tab === 'model') navigate(route.tab, { dataset: id });
  };

  const selectModel = (id: string) => {
    setActiveModelId(id);
    if (route.tab === 'generator') navigate('generator', { model: id });
  };

  const selectExperiment = (id: string) => {
    setActiveExperimentId(id);
    if (route.tab === 'quality') navigate('quality', { experiment: id });
  };

  // ---- events from the pages
  const handleDatasetUploaded = async (dataset: ApiDataset) => {
    await reloadDatasets();
    setActiveDatasetId(dataset.id);
    navigate('studio', { dataset: dataset.id });
  };

  const handleModelTrained = (trained: ApiModel) => {
    setActiveModelId(trained.model_id);
    setModels((prev) => upsertModel(prev, apiModelToRecord(trained)));
  };

  const handleGenerated = (result: ApiGenerateResponse) => {
    setActiveExperimentId(result.experiment_id);
    reloadExperiments(); // history now contains the new experiment
  };

  const handleColumnConfigChange = (datasetId: string, columns: ApiColumn[] | null) => {
    setColumnConfigs((prev) => {
      const next = { ...prev };
      if (columns) next[datasetId] = columns;
      else delete next[datasetId];
      return next;
    });
  };

  const pageTitle = route.tab === 'notfound' ? 'Page not found' : PAGE_TITLES[route.tab];
  const sidebarTab = useMemo(() => (route.tab === 'notfound' ? '' : route.tab), [route.tab]);

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      <Sidebar
        activeTab={sidebarTab}
        setActiveTab={goTab}
        backendOnline={backendOnline}
        backendChecking={backendChecking}
        apiBaseUrl={API_BASE_URL}
      />

      <div className="flex-1 flex flex-col overflow-hidden">
        <Navbar
          title={pageTitle}
          activeDataset={activeDataset}
          activeModel={activeModel}
          onRefresh={() => {
            checkHealth();
            reloadDatasets();
            refreshModels();
            reloadExperiments();
          }}
          loading={backendChecking}
        />

        <main className="flex-1 overflow-y-auto p-8 bg-slate-950">
          <div className="max-w-7xl mx-auto pb-12">
            {route.tab === 'notfound' && (
              <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-8 text-center space-y-3" data-testid="not-found">
                <h2 className="text-xl font-bold text-white">Page not found</h2>
                <p className="text-sm text-slate-400">There is no page at this address.</p>
                <button
                  type="button"
                  onClick={() => navigate('dashboard')}
                  className="inline-flex bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-4 py-2 rounded-lg focus-visible:ring-2 focus-visible:ring-indigo-300 outline-none"
                >
                  Go to the Dashboard
                </button>
              </div>
            )}

            {route.tab === 'dashboard' && (
              <Dashboard
                datasets={datasets}
                models={models}
                experiments={experiments}
                activeDataset={activeDataset}
                activeModel={activeModel}
                onNavigate={goTab}
              />
            )}

            {route.tab === 'studio' && (
              <DatasetStudio
                datasets={datasets}
                datasetsState={datasetsState}
                activeDataset={activeDataset}
                columnConfig={activeDataset ? columnConfigs[activeDataset.id] ?? null : null}
                onReloadDatasets={reloadDatasets}
                onSelectDataset={selectDataset}
                onUploaded={handleDatasetUploaded}
                onColumnConfigChange={handleColumnConfigChange}
                onNavigateToModelLab={(datasetId) => navigate('model', { dataset: datasetId })}
              />
            )}

            {route.tab === 'model' && (
              <ModelLab
                models={models}
                activeModel={activeModel}
                backendOnline={backendOnline}
                backendChecking={backendChecking}
                initialDatasetId={route.params.dataset || activeDataset?.id}
                columnConfigs={columnConfigs}
                onSelectDataset={selectDataset}
                onModelTrained={handleModelTrained}
                onRefreshModels={refreshModels}
                onSelectModel={(id) => setActiveModelId(id)}
                onNavigateToGenerator={(modelId) => navigate('generator', { model: modelId ?? activeModel?.id })}
              />
            )}

            {route.tab === 'generator' && (
              <DataGenerator
                models={models}
                datasets={datasets}
                selectedModelId={activeModel?.id ?? ''}
                onSelectModel={selectModel}
                onGenerated={handleGenerated}
                onNavigateToModelLab={() => goTab('model')}
                onNavigateToQuality={(experimentId) => navigate('quality', { experiment: experimentId })}
                onNavigateToHistory={() => goTab('history')}
              />
            )}

            {route.tab === 'quality' && (
              <QualityComparison
                experiments={experiments}
                experimentsState={experimentsState}
                selectedExperimentId={activeExperiment?.id ?? ''}
                onSelectExperiment={selectExperiment}
                onReloadExperiments={reloadExperiments}
                onNavigateToGenerator={() => goTab('generator')}
                onNavigateToHistory={() => goTab('history')}
              />
            )}

            {route.tab === 'history' && (
              <ExportHistory
                onExperimentsLoaded={applyExperiments}
                onNavigateToModelLab={() => goTab('model')}
                onNavigateToGenerator={(modelId) => navigate('generator', { model: modelId ?? activeModel?.id })}
                onOpenQuality={(exp) => navigate('quality', { experiment: exp.id })}
              />
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
