import React from 'react';
import { 
  Activity, 
  Database, 
  BrainCircuit, 
  Sparkles, 
  BarChart3, 
  History, 
  Cpu, 
  CheckCircle2, 
  XCircle, 
  RefreshCw 
} from 'lucide-react';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: any) => void;
  backendOnline: boolean;
  backendChecking: boolean;
  apiBaseUrl: string;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  backendOnline,
  backendChecking,
  apiBaseUrl
}) => {
  const navItems = [
    { id: 'dashboard', label: '1. Dashboard', icon: Activity, desc: 'Overview & metrics' },
    { id: 'studio', label: '2. Dataset Studio', icon: Database, desc: 'CSV upload & schema' },
    { id: 'model', label: '3. Model Lab', icon: BrainCircuit, desc: 'VAE training & loss' },
    { id: 'generator', label: '4. Data Generator', icon: Sparkles, desc: 'Synthesize records' },
    { id: 'quality', label: '5. Quality Compare', icon: BarChart3, desc: 'Statistical fidelity' },
    { id: 'history', label: '6. Export & History', icon: History, desc: 'CSV & audit reports' },
  ];

  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col justify-between select-none">
      <div>
        {/* Brand Header */}
        <div className="p-5 border-b border-slate-800 flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/25">
            <Cpu className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-lg tracking-wide text-white leading-tight">DataForge</h1>
            <p className="text-[11px] text-indigo-400 font-medium">Deep Learning GenAI</p>
          </div>
        </div>

        {/* Navigation List */}
        <nav className="p-3 space-y-1.5">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-left transition-all group ${
                  isActive 
                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30' 
                    : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/70'
                }`}
              >
                <Icon className={`w-4 h-4 flex-shrink-0 transition-transform group-hover:scale-110 ${
                  isActive ? 'text-white' : 'text-slate-400 group-hover:text-slate-200'
                }`} />
                <div className="truncate">
                  <div className={`text-sm font-semibold ${isActive ? 'text-white' : 'text-slate-300'}`}>
                    {item.label}
                  </div>
                  <div className={`text-[11px] truncate ${isActive ? 'text-indigo-200' : 'text-slate-500'}`}>
                    {item.desc}
                  </div>
                </div>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer System Status */}
      <div className="p-4 border-t border-slate-800 bg-slate-950/40">
        <div className="flex items-center justify-between text-xs mb-1.5">
          <span className="text-slate-400 font-medium">FastAPI Backend</span>
          {backendChecking ? (
            <span className="inline-flex items-center text-amber-400">
              <RefreshCw className="w-3 h-3 animate-spin mr-1" /> Checking
            </span>
          ) : backendOnline ? (
            <span className="inline-flex items-center text-emerald-400 font-medium">
              <CheckCircle2 className="w-3.5 h-3.5 mr-1" /> Connected
            </span>
          ) : (
            <span className="inline-flex items-center text-rose-400 font-medium">
              <XCircle className="w-3.5 h-3.5 mr-1" /> Offline
            </span>
          )}
        </div>
        <div className="bg-slate-900 px-2.5 py-1.5 rounded-md border border-slate-800/80 text-[10px] text-slate-400 font-mono truncate">
          {apiBaseUrl}
        </div>
      </div>
    </aside>
  );
};
