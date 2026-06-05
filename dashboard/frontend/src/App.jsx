import React, { useState, useEffect, useRef } from 'react';
import {
  Shield, AlertTriangle, CheckCircle, Activity,
  Settings, Play, Pause, RefreshCw, Moon, Sun,
  ChevronRight, Search, BarChart3, Clock, Lock
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, BarChart, Bar, Cell
} from 'recharts';
import axios from 'axios';

// Environment-aware URLs
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_BASE = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/stream';

const App = () => {
  const [darkMode, setDarkMode] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const [transactions, setTransactions] = useState([]);
  const [metrics, setMetrics] = useState({
    tp: 0, fp: 0, tn: 0, fn: 0,
    total: 0, fraudDetected: 0
  });
  const [selectedTx, setSelectedTx] = useState(null);
  const [shapData, setShapData] = useState(null);
  const [speed, setSpeed] = useState(1);
  const ws = useRef(null);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  useEffect(() => {
    return () => {
      if (ws.current) ws.current.close();
    };
  }, []);

  const toggleStreaming = () => {
    if (!streaming) {
      if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
        connectWS();
      } else {
        ws.current.send(JSON.stringify({ action: 'start', interval: 1 / speed }));
      }
    } else {
      if (ws.current) {
        ws.current.send(JSON.stringify({ action: 'stop' }));
      }
    }
    setStreaming(!streaming);
  };

  const connectWS = () => {
    ws.current = new WebSocket(WS_URL);
    ws.current.onopen = () => {
      ws.current.send(JSON.stringify({ action: 'start', interval: 1 / speed }));
    };
    ws.current.onmessage = (event) => {
      const tx = JSON.parse(event.data);
      setTransactions(prev => [tx, ...prev].slice(0, 200)); // Increased for auditability
      updateMetrics(tx);
    };
    ws.current.onclose = () => setStreaming(false);
  };

  useEffect(() => {
    if (streaming && ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ action: 'speed', interval: 1 / speed }));
    }
  }, [speed]);

  const updateMetrics = (tx) => {
    setMetrics(prev => {
      const isTP = tx.prediction === 1 && tx.actual === 1;
      const isFP = tx.prediction === 1 && tx.actual === 0;
      const isTN = tx.prediction === 0 && tx.actual === 0;
      const isFN = tx.prediction === 0 && tx.actual === 1;

      return {
        ...prev,
        tp: prev.tp + (isTP ? 1 : 0),
        fp: prev.fp + (isFP ? 1 : 0),
        tn: prev.tn + (isTN ? 1 : 0),
        fn: prev.fn + (isFN ? 1 : 0),
        total: prev.total + 1,
        fraudDetected: prev.fraudDetected + (tx.prediction === 1 ? 1 : 0)
      };
    });
  };

  const handleTxClick = async (tx) => {
    setSelectedTx(tx);
    setShapData(null);
    try {
      const res = await axios.get(`${API_BASE}/explain/${tx.index}`);
      setShapData(res.data);
    } catch (err) {
      console.error("Failed to fetch SHAP", err);
    }
  };

  const precisionVal = metrics.tp + metrics.fp > 0 ? (metrics.tp / (metrics.tp + metrics.fp)) : 0;
  const recallVal = metrics.tp + metrics.fn > 0 ? (metrics.tp / (metrics.tp + metrics.fn)) : 0;
  const f1Val = (2 * precisionVal * recallVal) / (precisionVal + recallVal) || 0;

  return (
    <div className="min-h-screen w-full bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 flex flex-col font-sans transition-colors duration-300">
      {/* Header */}
      <header className="h-16 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between px-6 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/30">
            <Shield className="text-white w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-black tracking-tight">FraudGuard AI</h1>
            <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Real-time Stacking Ensemble Monitoring</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center bg-slate-100 dark:bg-slate-800 rounded-full px-4 py-1.5 gap-3 border border-slate-200 dark:border-slate-700">
            <span className="text-[10px] font-black uppercase tracking-wider text-slate-500">Speed</span>
            <input
              type="range" min="0.5" max="10" step="0.5"
              value={speed} onChange={(e) => setSpeed(parseFloat(e.target.value))}
              className="w-24 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-600"
            />
            <span className="text-[10px] font-mono font-bold w-6 text-indigo-500">{speed}x</span>
          </div>

          <button
            onClick={toggleStreaming}
            className={`flex items-center gap-2 px-6 py-2 rounded-xl font-black text-sm tracking-wide transition-all active:scale-95 ${
              streaming
                ? 'bg-red-50 text-red-600 border border-red-200 hover:bg-red-100 dark:bg-red-900/20 dark:border-red-900/50 dark:text-red-400'
                : 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-xl shadow-indigo-500/25'
            }`}
          >
            {streaming ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}
            {streaming ? 'Stop Simulation' : 'Start Simulation'}
          </button>

          <button
            onClick={() => setDarkMode(!darkMode)}
            className="p-2.5 rounded-xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700 transition-all active:rotate-12"
          >
            {darkMode ? <Sun size={20} className="text-amber-400" /> : <Moon size={20} className="text-indigo-600" />}
          </button>
        </div>
      </header>

      <main className="flex-1 p-6 flex gap-6 overflow-hidden">
        {/* Left Column: Metrics & Controls */}
        <div className="w-80 flex flex-col gap-6 shrink-0">
          <section className="bg-white dark:bg-slate-900 p-6 rounded-[2rem] border border-slate-200 dark:border-slate-800 shadow-xl shadow-slate-200/50 dark:shadow-none">
            <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-5 flex items-center gap-2">
              <Activity size={14} className="text-indigo-500" /> Live Accuracy
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-slate-50 dark:bg-slate-800/50 rounded-2xl border border-slate-100 dark:border-slate-800">
                <p className="text-[10px] font-bold text-slate-500 uppercase mb-1">Precision</p>
                <p className="text-2xl font-black text-indigo-500">{precisionVal.toFixed(3)}</p>
              </div>
              <div className="p-4 bg-slate-50 dark:bg-slate-800/50 rounded-2xl border border-slate-100 dark:border-slate-800">
                <p className="text-[10px] font-bold text-slate-500 uppercase mb-1">Recall</p>
                <p className="text-2xl font-black text-emerald-500">{recallVal.toFixed(3)}</p>
              </div>
              <div className="col-span-2 p-5 bg-indigo-600 rounded-[1.5rem] shadow-lg shadow-indigo-500/30 text-white relative overflow-hidden group">
                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:scale-125 transition-transform duration-500">
                  <Shield size={64} />
                </div>
                <p className="text-[10px] font-black uppercase tracking-widest mb-1 opacity-80">Ensemble F1 Score</p>
                <p className="text-4xl font-black">{f1Val.toFixed(3)}</p>
              </div>
            </div>
          </section>

          <section className="bg-white dark:bg-slate-900 p-6 rounded-[2rem] border border-slate-200 dark:border-slate-800 shadow-xl shadow-slate-200/50 dark:shadow-none">
            <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-5 flex items-center gap-2">
              <BarChart3 size={14} className="text-indigo-500" /> Detection Stats
            </h3>
            <div className="space-y-4">
              <StatRow label="Total Processed" value={metrics.total} color="slate" />
              <StatRow label="Fraud Flagged" value={metrics.fraudDetected} color="red" />
              <StatRow label="True Positives" value={metrics.tp} color="emerald" />
              <StatRow label="False Positives" value={metrics.fp} color="amber" />
            </div>
          </section>

          <section className="bg-slate-900 dark:bg-indigo-950 p-6 rounded-[2rem] text-white shadow-2xl mt-auto border border-white/10">
            <h3 className="text-[10px] font-black opacity-50 uppercase tracking-[0.2em] mb-3">Model Architecture</h3>
            <p className="text-xl font-black mb-1">Stacking Ensemble</p>
            <div className="flex gap-1.5 mb-5">
              {['LR', 'RF', 'XGB'].map(m => (
                <span key={m} className="text-[8px] font-black px-1.5 py-0.5 bg-white/10 rounded uppercase">{m}</span>
              ))}
              <span className="text-[8px] font-black px-1.5 py-0.5 bg-indigo-500 rounded uppercase">Meta: LR</span>
            </div>
            <div className="flex items-center gap-2 bg-white/5 p-3 rounded-xl border border-white/5">
              <Lock size={12} className="text-indigo-400" />
              <span className="text-[9px] font-mono opacity-60">HASH: 42_DETERMINISTIC_SEED</span>
            </div>
          </section>
        </div>

        {/* Middle Column: Streaming Feed */}
        <div className="flex-1 flex flex-col gap-6 overflow-hidden">
          <section className="flex-1 bg-white dark:bg-slate-900 rounded-[2rem] border border-slate-200 dark:border-slate-800 shadow-xl shadow-slate-200/50 dark:shadow-none flex flex-col overflow-hidden">
            <div className="p-5 border-b border-slate-100 dark:border-slate-800 flex items-center justify-between bg-slate-50/50 dark:bg-slate-800/30">
              <h3 className="font-black text-sm tracking-tight flex items-center gap-2">
                <Clock size={16} className="text-indigo-500" /> Live Transaction Feed
              </h3>
              <div className="flex gap-2">
                <span className={`flex items-center gap-1.5 text-[10px] font-black px-3 py-1.5 rounded-full uppercase tracking-widest ${streaming ? 'bg-emerald-50 text-emerald-600 dark:bg-emerald-900/30 dark:text-emerald-400' : 'bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-500'}`}>
                  <div className={`w-2 h-2 rounded-full ${streaming ? 'bg-emerald-500 animate-pulse' : 'bg-slate-400'}`} /> {streaming ? 'System Live' : 'System Standby'}
                </span>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto scrollbar-hide">
              <table className="w-full text-left border-collapse">
                <thead className="sticky top-0 bg-white dark:bg-slate-900 z-10">
                  <tr className="text-[10px] font-black text-slate-400 uppercase tracking-widest border-b border-slate-50 dark:border-slate-800">
                    <th className="px-6 py-4">Timestamp</th>
                    <th className="px-6 py-4">Amount</th>
                    <th className="px-6 py-4">Risk Level</th>
                    <th className="px-6 py-4 text-center">Prediction</th>
                    <th className="px-6 py-4 text-center">Actual</th>
                    <th className="px-6 py-4 text-right">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-50 dark:divide-slate-800/50">
                  {transactions.map((tx, i) => (
                    <tr
                      key={`${tx.index}-${tx.timestamp}`}
                      onClick={() => handleTxClick(tx)}
                      className={`group cursor-pointer hover:bg-indigo-50/50 dark:hover:bg-indigo-900/10 transition-all ${selectedTx?.index === tx.index ? 'bg-indigo-50 dark:bg-indigo-900/20' : ''}`}
                    >
                      <td className="px-6 py-4 text-xs font-mono font-medium text-slate-500">
                        {new Date(tx.timestamp * 1000).toLocaleTimeString()}
                      </td>
                      <td className="px-6 py-4 text-sm font-black">
                        ${tx.features.Amount.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                      </td>
                      <td className="px-6 py-4">
                        <span className={`text-[9px] font-black px-2.5 py-1 rounded-lg uppercase tracking-wider ${
                          tx.risk_level === 'High' ? 'bg-red-50 text-red-600 dark:bg-red-900/40 dark:text-red-400' :
                          tx.risk_level === 'Medium' ? 'bg-amber-50 text-amber-600 dark:bg-amber-900/40 dark:text-amber-400' :
                          'bg-emerald-50 text-emerald-600 dark:bg-emerald-900/40 dark:text-emerald-400'
                        }`}>
                          {tx.risk_level}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-center">
                        {tx.prediction === 1 ? (
                          <div className="flex items-center justify-center text-red-500 bg-red-50 dark:bg-red-900/20 w-8 h-8 rounded-lg mx-auto shadow-sm">
                            <AlertTriangle size={16} fill="currentColor" fillOpacity={0.2} />
                          </div>
                        ) : (
                          <div className="flex items-center justify-center text-emerald-500 bg-emerald-50 dark:bg-emerald-900/20 w-8 h-8 rounded-lg mx-auto shadow-sm">
                            <CheckCircle size={16} fill="currentColor" fillOpacity={0.2} />
                          </div>
                        )}
                      </td>
                      <td className="px-6 py-4 text-center">
                         <span className={`text-[10px] font-black tracking-widest ${tx.actual === 1 ? 'text-red-500 bg-red-100/50 px-2 py-0.5 rounded' : 'text-slate-400'}`}>
                           {tx.actual === 1 ? 'FRAUD' : 'LEGIT'}
                         </span>
                      </td>
                      <td className="px-6 py-4 text-right text-[10px] font-mono font-bold text-slate-400 italic">
                        {tx.latency_ms.toFixed(2)}ms
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {transactions.length === 0 && (
                <div className="h-full flex flex-col items-center justify-center text-slate-300 py-20">
                  <Activity size={48} className="mb-4 opacity-20" />
                  <p className="font-black text-sm uppercase tracking-widest opacity-30 italic">Awaiting Transaction Data...</p>
                </div>
              )}
            </div>
          </section>
        </div>

        {/* Right Column: Investigation / SHAP */}
        <div className="w-96 flex flex-col gap-6 shrink-0">
           <section className="flex-1 bg-white dark:bg-slate-900 rounded-[2rem] border border-slate-200 dark:border-slate-800 shadow-xl shadow-slate-200/50 dark:shadow-none flex flex-col overflow-hidden">
            <div className="p-5 border-b border-slate-100 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-800/30">
              <h3 className="font-black text-sm tracking-tight flex items-center gap-2">
                <Search size={16} className="text-indigo-500" /> Transaction Investigator
              </h3>
            </div>

            <div className="flex-1 p-6 overflow-y-auto scrollbar-hide">
              {selectedTx ? (
                <div className="space-y-8 animate-in fade-in slide-in-from-right-4 duration-500">
                  <div className="flex justify-between items-end">
                    <div>
                      <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Transaction Ref</p>
                      <p className="text-2xl font-mono font-black text-slate-800 dark:text-slate-100">#TX-{selectedTx.index}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Probability</p>
                      <p className={`text-4xl font-black ${(selectedTx.probability > 0.5) ? 'text-red-600' : 'text-indigo-600'}`}>{(selectedTx.probability * 100).toFixed(1)}%</p>
                    </div>
                  </div>

                  <div className="h-px bg-slate-100 dark:bg-slate-800" />

                  <div>
                    <h4 className="text-[10px] font-black text-slate-500 mb-5 uppercase tracking-[0.2em] flex justify-between">
                      <span>SHAP Explainability</span>
                      <span className="text-indigo-500">Base: XGBoost</span>
                    </h4>
                    {shapData ? (
                      <div className="space-y-4">
                         {shapData.feature_names.map((name, i) => {
                           const val = shapData.shap_values[i];
                           if (Math.abs(val) < 0.1) return null; // Only show significant
                           return (
                             <div key={name} className="space-y-1.5">
                               <div className="flex justify-between items-center text-[10px] font-black uppercase tracking-wider">
                                 <span className="text-slate-400">{name}</span>
                                 <span className={val > 0 ? 'text-red-500' : 'text-indigo-500'}>
                                   {val > 0 ? '+' : ''}{val.toFixed(3)}
                                 </span>
                               </div>
                               <div className="h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden flex">
                                 {val > 0 ? (
                                   <div className="h-full bg-red-500 rounded-full ml-auto shadow-[0_0_8px_rgba(239,68,68,0.5)]" style={{ width: `${Math.min(val * 10, 100)}%` }} />
                                 ) : (
                                   <div className="h-full bg-indigo-500 rounded-full shadow-[0_0_8px_rgba(99,102,241,0.5)]" style={{ width: `${Math.min(Math.abs(val) * 10, 100)}%` }} />
                                 )}
                               </div>
                             </div>
                           );
                         })}
                         <p className="text-[9px] text-slate-400 font-bold italic mt-4 px-3 py-2 bg-slate-50 dark:bg-slate-800/50 rounded-lg border border-slate-100 dark:border-slate-800">
                           Red pushes towards <span className="text-red-500">FRAUD</span>, Blue towards <span className="text-indigo-500">LEGIT</span>.
                         </p>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center justify-center py-16 text-slate-400">
                        <div className="relative">
                           <RefreshCw className="animate-spin mb-4 opacity-50" size={32} />
                           <div className="absolute inset-0 flex items-center justify-center">
                             <div className="w-1.5 h-1.5 bg-indigo-500 rounded-full" />
                           </div>
                        </div>
                        <p className="text-[10px] font-black uppercase tracking-[0.2em]">Computing SHAP...</p>
                      </div>
                    )}
                  </div>

                  <div className="bg-slate-900 rounded-[1.5rem] p-6 text-white shadow-xl shadow-slate-900/20">
                    <h4 className="text-[10px] font-black mb-4 uppercase tracking-[0.2em] opacity-40">Forensic Features</h4>
                    <div className="grid grid-cols-2 gap-x-6 gap-y-4">
                       {['V17', 'V14', 'V12', 'V10', 'Amount', 'Time'].map(f => (
                         <div key={f} className="flex flex-col">
                           <span className="text-[9px] font-black opacity-30 mb-0.5">{f}</span>
                           <span className="font-mono font-bold text-xs truncate">
                             {typeof selectedTx.features[f] === 'number' ? selectedTx.features[f].toFixed(4) : selectedTx.features[f]}
                           </span>
                         </div>
                       ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-slate-400 text-center px-10">
                  <div className="w-20 h-20 bg-slate-50 dark:bg-slate-800/50 rounded-[2rem] flex items-center justify-center mb-6 shadow-inner border border-slate-100 dark:border-slate-800">
                    <Search size={32} className="opacity-20" />
                  </div>
                  <p className="font-black text-sm uppercase tracking-widest text-slate-500 mb-2">No Transaction Selected</p>
                  <p className="text-[10px] font-bold opacity-50 leading-relaxed">Select a transaction from the live feed to perform deep forensic analysis and SHAP explanation.</p>
                </div>
              )}
            </div>
           </section>
        </div>
      </main>
    </div>
  );
};

const StatRow = ({ label, value, color }) => {
  const colorMap = {
    slate: 'bg-slate-400 dark:bg-slate-600',
    red: 'bg-red-500',
    emerald: 'bg-emerald-500',
    amber: 'bg-amber-500',
    indigo: 'bg-indigo-500'
  };

  return (
    <div className="flex items-center justify-between group">
      <div className="flex items-center gap-3">
        <div className={`w-2 h-2 rounded-full ${colorMap[color]} shadow-sm group-hover:scale-150 transition-transform`} />
        <span className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">{label}</span>
      </div>
      <span className="text-sm font-black font-mono text-slate-700 dark:text-slate-200">{value.toLocaleString()}</span>
    </div>
  );
}

export default App;
