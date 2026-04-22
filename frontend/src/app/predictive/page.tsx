"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { predictiveService, PredictiveResponse } from "@/lib/services/predictiveService";
import { dataService } from "@/lib/services/dataService";
import { BrainCircuit, RefreshCw, AlertCircle, Cpu, Network } from "lucide-react";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";

const ALL_INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"];
const ALL_FEATURES = [
  "RSI_14", "MACD_hist", "BB_width", "ATR_14", "OBV", "VWAP", 
  "MFI_14", "Stoch_OSC", "Williams_R", "CCI_14", "ADX_14", 
  "CMF_20", "ROC_10", "TRIX_15", "VORTEX_14", "SAR"
];

export default function PredictivePage() {
  const [universe, setUniverse] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [ticker, setTicker] = useState("BTCUSDT");
  const [interval, setInterval] = useState("1h");
  const [direction, setDirection] = useState("Both Sides");
  const [features, setFeatures] = useState<string[]>(["RSI_14", "MACD_hist", "BB_width", "ATR_14", "OBV", "VWAP"]);
  const [regType, setRegType] = useState("RF Classifier");
  const [barType, setBarType] = useState("Time Bars");
  const [labelType, setLabelType] = useState("BoxRange");

  const [testRatio, setTestRatio] = useState(0.3);
  const [rfDepth, setRfDepth] = useState(5);
  const [vifTh, setVifTh] = useState(10.0);
  
  const [data, setData] = useState<PredictiveResponse | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    dataService.getUniverse()
      .then(uni => setUniverse(uni))
      .catch(err => setError("Failed to load universe"))
      .finally(() => setLoading(false));
  }, []);

  const handleRun = async () => {
    try {
      if (features.length === 0) throw new Error("Please select at least one feature");
      
      setIsGenerating(true);
      setError(null);
      const res = await predictiveService.runAnalysis({
        ticker, interval, trade_direction: direction, features,
        reg_type: regType, bar_type: barType, labeler_type: labelType,
        test_ratio: testRatio, rf_max_depth: rfDepth, vif_th: vifTh,
        labeler_params: { vol_window: 10, upper_mult: 2, lower_mult: 2, max_holding: 10, amp_th: 100, max_inactive: 10, window: 20, vote_th: 0.5, threshold: 1.0 }
      });
      setData(res);
    } catch (err: any) {
       setError(err.response?.data?.detail || err.message || "Predictive analysis failed");
       setData(null);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between mt-2 mb-6 gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-indigo-400 to-cyan-200 bg-clip-text text-transparent flex items-center">
            <BrainCircuit className="h-8 w-8 mr-3 text-indigo-400" />
            Predictive Analytics
          </h1>
          <p className="text-muted-foreground mt-1">Machine learning directional prediction & meta-labeling.</p>
        </div>
      </div>

      <AnimatePresence>
        {error && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}>
            <div className="p-4 bg-rose-500/10 border border-rose-500/50 text-rose-400 rounded-lg flex items-center mb-6">
              <AlertCircle className="h-5 w-5 mr-3 shrink-0" />
              {error}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-1 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Model Pipeline</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">Asset</label>
                <select 
                  className="w-full bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:ring-1 focus:ring-indigo-500 outline-none"
                  value={ticker}
                  onChange={e => setTicker(e.target.value)}
                  disabled={loading}
                >
                  {universe ? universe.map(u => <option key={u} value={u}>{u}</option>) : null}
                </select>
              </div>

               <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">Interval</label>
                <select 
                  className="w-full bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:ring-1 focus:ring-indigo-500 outline-none"
                  value={interval}
                  onChange={e => setInterval(e.target.value)}
                >
                  {ALL_INTERVALS ? ALL_INTERVALS.map(u => <option key={u} value={u}>{u}</option>) : null}
                </select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground flex justify-between">
                  <span>Features</span>
                  <span className="text-xs">{features.length} selected</span>
                </label>
                <select 
                  multiple 
                  className="w-full h-32 bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:ring-1 focus:ring-indigo-500 outline-none"
                  value={features}
                  onChange={e => {
                    const options = Array.from(e.target.selectedOptions, option => option.value);
                    setFeatures(options);
                  }}
                >
                  {ALL_FEATURES.map(u => <option key={u} value={u}>{u}</option>)}
                </select>
              </div>

               <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">Labeling Method</label>
                <select 
                  className="w-full bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:ring-1 focus:ring-indigo-500 outline-none"
                  value={labelType}
                  onChange={e => setLabelType(e.target.value)}
                >
                  <option value="BoxRange">Triple Barrier (BoxRange)</option>
                  <option value="Trend">Trend Breakout</option>
                  <option value="Regime">Stationarity Regime</option>
                  <option value="Combine">Combined Ensemble</option>
                  <option value="TailSet">Tail Set</option>
                </select>
              </div>

               <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">Model</label>
                <select 
                  className="w-full bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:ring-1 focus:ring-indigo-500 outline-none"
                  value={regType}
                  onChange={e => setRegType(e.target.value)}
                >
                  <option value="RF Classifier">Random Forest</option>
                  <option value="XGB Classifier">XGBoost</option>
                </select>
              </div>

               <div className="grid grid-cols-2 gap-4">
                 <div className="space-y-2">
                   <label className="text-xs font-medium text-muted-foreground">Max Depth</label>
                   <input type="number" value={rfDepth} onChange={e => setRfDepth(Number(e.target.value))} className="w-full bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500" />
                 </div>
                 <div className="space-y-2">
                   <label className="text-xs font-medium text-muted-foreground">VIF Thresh</label>
                   <input type="number" step="0.5" value={vifTh} onChange={e => setVifTh(Number(e.target.value))} className="w-full bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500" />
                 </div>
               </div>

              <button
                onClick={handleRun}
                disabled={isGenerating || !ticker}
                className="w-full mt-4 bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2 rounded-md transition-colors disabled:opacity-50 flex justify-center items-center shadow-[0_0_15px_rgba(79,70,229,0.3)]"
              >
                {isGenerating ? <RefreshCw className="h-4 w-4 mr-2 animate-spin" /> : <Cpu className="h-4 w-4 mr-2" />}
                Train Model
              </button>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-3 space-y-6">
          <div className="grid grid-cols-3 gap-4">
             <Card>
                <CardContent className="p-5 flex flex-col justify-center items-center h-full">
                  <span className="text-sm text-muted-foreground mb-1">Primary Train N</span>
                  <span className="text-3xl font-mono text-cyan-400 font-bold">{data?.train_size || "-"}</span>
                </CardContent>
             </Card>
             <Card>
                <CardContent className="p-5 flex flex-col justify-center items-center h-full">
                  <span className="text-sm text-muted-foreground mb-1">Meta Train N</span>
                  <span className="text-3xl font-mono text-indigo-400 font-bold">{data?.meta_size || "-"}</span>
                </CardContent>
             </Card>
             <Card>
                <CardContent className="p-5 flex flex-col justify-center items-center h-full">
                  <span className="text-sm text-muted-foreground mb-1">Out-of-Sample N</span>
                  <span className="text-3xl font-mono text-purple-400 font-bold">{data?.test_size || "-"}</span>
                </CardContent>
             </Card>
          </div>

          <Card className="min-h-[400px]">
            <CardHeader className="pb-2">
              <CardTitle>Model Validation</CardTitle>
            </CardHeader>
            <CardContent>
              {!data ? (
                <div className="h-[350px] flex flex-col items-center justify-center text-muted-foreground space-y-4">
                  <Network className="h-12 w-12 opacity-20" />
                  <p>Configure pipeline and train model to view diagnostics.</p>
                </div>
              ) : (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6 pt-4">
                  <div className="glass-panel p-6 rounded-xl flex justify-between items-center border-indigo-500/20">
                     <div className="flex flex-col">
                        <span className="text-muted-foreground text-sm uppercase tracking-wider mb-1">OOS Accuracy</span>
                        <span className="text-4xl font-mono font-bold text-white">{(data.accuracy * 100).toFixed(2)}%</span>
                     </div>
                     <div className="flex flex-col text-right">
                        <span className="text-muted-foreground text-sm uppercase tracking-wider mb-1">Features Used</span>
                        <span className="text-xl font-mono font-bold text-indigo-300">{data.features_used.length} / {features.length}</span>
                     </div>
                  </div>

                  <div>
                     <h3 className="text-lg font-medium mb-3">Classification Report</h3>
                     <div className="overflow-x-auto">
                        <table className="w-full text-sm text-left">
                          <thead className="bg-secondary/50 text-muted-foreground">
                            <tr>
                              <th className="px-4 py-2 font-medium">Class</th>
                              <th className="px-4 py-2 font-medium">Precision</th>
                              <th className="px-4 py-2 font-medium">Recall</th>
                              <th className="px-4 py-2 font-medium">F1-Score</th>
                              <th className="px-4 py-2 font-medium">Support</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-border/50">
                            {Object.entries(data.classification_report).filter(([k]) => k !== 'accuracy' && k !== 'macro avg' && k !== 'weighted avg').map(([key, metrics]: any) => (
                              <tr key={key} className="hover:bg-secondary/20">
                                <td className="px-4 py-2 font-mono font-bold">
                                  {key === "-1" ? <span className="text-rose-400">Short (-1)</span> : key === "1" ? <span className="text-emerald-400">Long (1)</span> : <span className="text-muted-foreground">Neutral (0)</span>}
                                </td>
                                <td className="px-4 py-2">{metrics.precision.toFixed(3)}</td>
                                <td className="px-4 py-2">{metrics.recall.toFixed(3)}</td>
                                <td className="px-4 py-2">{metrics['f1-score']?.toFixed(3) || "N/A"}</td>
                                <td className="px-4 py-2 font-mono text-muted-foreground">{metrics.support}</td>
                              </tr>
                            ))}
                            {['macro avg', 'weighted avg'].map(agg => data.classification_report[agg] && (
                               <tr key={agg} className="bg-indigo-500/10 font-medium">
                                 <td className="px-4 py-2 capitalize">{agg}</td>
                                 <td className="px-4 py-2">{data.classification_report[agg].precision.toFixed(3)}</td>
                                 <td className="px-4 py-2">{data.classification_report[agg].recall.toFixed(3)}</td>
                                 <td className="px-4 py-2">{data.classification_report[agg]['f1-score']?.toFixed(3) || "N/A"}</td>
                                 <td className="px-4 py-2 font-mono text-muted-foreground">{data.classification_report[agg].support}</td>
                               </tr>
                            ))}
                          </tbody>
                        </table>
                     </div>
                  </div>
                </motion.div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
