"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { marketRadarService } from "@/lib/services/marketRadarService";
import { dataService } from "@/lib/services/dataService";
import { Activity, RefreshCw, AlertCircle, BarChart2, Zap, GitMerge } from "lucide-react";
import { cn, formatError } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import dynamic from "next/dynamic";

// Dynamic import for Plotly to avoid SSR issues
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false, loading: () => <div className="h-[400px] flex items-center justify-center text-muted-foreground animate-pulse glass rounded-md">Loading Chart Engine...</div> });

const ALL_INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"];

export default function MarketRadarPage() {
  const [universe, setUniverse] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Tab State
  const [activeTab, setActiveTab] = useState<"snapshot" | "path">("snapshot");

  // Snapshot State
  const [snapSymbols, setSnapSymbols] = useState<string[]>([]);
  const [snapIntervals, setSnapIntervals] = useState<string[]>(["1h"]);
  const [snapData, setSnapData] = useState<any[]>([]);
  const [isSnapping, setIsSnapping] = useState(false);

  // Path State
  const [pathSymbolA, setPathSymbolA] = useState("BTCUSDT");
  const [pathSymbolB, setPathSymbolB] = useState("ETHUSDT");
  const [pathInterval, setPathInterval] = useState("1h");
  const [pathData, setPathData] = useState<any>(null);
  const [isPathing, setIsPathing] = useState(false);

  useEffect(() => {
    dataService.getUniverse()
      .then(uni => {
        setUniverse(uni);
        setSnapSymbols(uni.slice(0, 10)); // Default top 10
      })
      .catch(err => setError(formatError(err)))
      .finally(() => setLoading(false));
  }, []);

  const handleSnapshot = async () => {
    if (snapSymbols.length === 0 || snapIntervals.length === 0) return;
    try {
      setIsSnapping(true);
      setError(null);
      const res = await marketRadarService.getSnapshot({ symbols: snapSymbols, intervals: snapIntervals });
      setSnapData(res.metrics);
    } catch (err: any) {
      setError(formatError(err));
    } finally {
      setIsSnapping(false);
    }
  };

  const handlePath = async () => {
    if (!pathSymbolA || !pathSymbolB || !pathInterval) return;
    try {
      setIsPathing(true);
      setError(null);
      const res = await marketRadarService.getPathAnalysis({ 
        symbol_a: pathSymbolA, 
        symbol_b: pathSymbolB, 
        interval: pathInterval 
      });
      setPathData(res);
    } catch (err: any) {
      setError(formatError(err));
      setPathData(null);
    } finally {
      setIsPathing(false);
    }
  };

  const getMetricColor = (val: number | string, invert = false) => {
    if (typeof val !== 'number') return "text-foreground";
    if (val === 0) return "text-muted-foreground";
    const isGood = invert ? val < 0 : val > 0;
    return isGood ? "text-emerald-400 font-medium" : "text-rose-400 font-medium";
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between mt-2 mb-6 gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-emerald-400 to-teal-200 bg-clip-text text-transparent flex items-center">
            <Activity className="h-8 w-8 mr-3 text-emerald-400" />
            Market Radar
          </h1>
          <p className="text-muted-foreground mt-1">Real-time asset scoring and pairwise path analysis.</p>
        </div>
        
        <div className="flex bg-secondary/30 p-1 rounded-lg border border-border/50">
          <button
            onClick={() => setActiveTab("snapshot")}
            className={cn("px-4 py-2 rounded-md text-sm font-medium transition-all duration-300", activeTab === "snapshot" ? "bg-emerald-500/20 text-emerald-400 shadow-sm" : "text-muted-foreground hover:text-foreground")}
          >
            Market Snapshot
          </button>
          <button
            onClick={() => setActiveTab("path")}
            className={cn("px-4 py-2 rounded-md text-sm font-medium transition-all duration-300", activeTab === "path" ? "bg-emerald-500/20 text-emerald-400 shadow-sm" : "text-muted-foreground hover:text-foreground")}
          >
            Trajectory Paths
          </button>
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

      {activeTab === "snapshot" && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-1 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Filters</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-muted-foreground">Intervals</label>
                  <div className="flex flex-wrap gap-2">
                    {ALL_INTERVALS.map(inv => (
                      <button
                        key={inv}
                        onClick={() => {
                          if (snapIntervals.includes(inv)) setSnapIntervals(snapIntervals.filter(i => i !== inv));
                          else setSnapIntervals([...snapIntervals, inv]);
                        }}
                        className={cn("px-2 py-1 text-xs rounded border transition-colors", snapIntervals.includes(inv) ? "bg-emerald-500/20 border-emerald-500/50 text-emerald-400" : "border-border bg-secondary/30 text-muted-foreground hover:bg-secondary")}
                      >
                        {inv}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-muted-foreground flex justify-between">
                    <span>Symbols</span>
                    <span className="text-xs">{snapSymbols.length} selected</span>
                  </label>
                  <div className="h-48 overflow-y-auto bg-secondary/30 border border-border/50 rounded-md p-2 flex flex-wrap gap-1.5 content-start">
                    {(universe || []).map(u => {
                      const isSelected = snapSymbols.includes(u);
                      return (
                        <button
                          key={u}
                          onClick={() => {
                            if (isSelected) setSnapSymbols(snapSymbols.filter(s => s !== u));
                            else setSnapSymbols([...snapSymbols, u]);
                          }}
                          className={cn(
                            "px-2.5 py-1 text-xs rounded-full border transition-all duration-200 font-medium",
                            isSelected ? "bg-emerald-500/20 border-emerald-500/50 text-emerald-400" : "bg-background/50 border-border/50 text-muted-foreground hover:bg-secondary/80 hover:text-foreground hover:border-border"
                          )}
                        >
                          {u}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <button
                  onClick={handleSnapshot}
                  disabled={isSnapping || snapSymbols.length === 0 || snapIntervals.length === 0}
                  className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-medium py-2 rounded-md transition-colors disabled:opacity-50 flex justify-center items-center shadow-[0_0_15px_rgba(16,185,129,0.3)]"
                >
                  {isSnapping ? <RefreshCw className="h-4 w-4 mr-2 animate-spin" /> : <BarChart2 className="h-4 w-4 mr-2" />}
                  Generate Snapshot
                </button>
              </CardContent>
            </Card>
          </div>

          <div className="lg:col-span-3">
            <Card className="h-full flex flex-col min-h-[600px]">
              <CardHeader>
                <CardTitle>Asset Metrics</CardTitle>
              </CardHeader>
              <CardContent className="flex-1 overflow-auto">
                {!snapData || snapData.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-muted-foreground space-y-4">
                    <Zap className="h-12 w-12 opacity-20" />
                    <p>Select parameters and generate to view metrics.</p>
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left whitespace-nowrap">
                      <thead className="text-xs text-muted-foreground uppercase bg-secondary/50 sticky top-0 backdrop-blur-md">
                        <tr>
                          {Object.keys(snapData[0] || {}).map(k => (
                            <th key={k} className="px-4 py-3 font-semibold">{k}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border/50">
                        {snapData.map((row, i) => (
                          <motion.tr 
                            key={i}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: Math.min(i * 0.05, 0.5) }} 
                            className="hover:bg-secondary/30 transition-colors"
                          >
                            {Object.entries(row).map(([k, v]) => {
                               let displayVal = typeof v === 'number' ? (v % 1 === 0 ? v : v.toFixed(2)) : v;
                               if (v === null) displayVal = "N/A";
                               
                               const isNum = typeof v === 'number';
                               const isPercentCol = k.includes('%') || k.includes('Rate') || k.includes('DD');
                               const invert = k.includes('DD');
                               
                               return (
                                <td key={k} className={cn("px-4 py-3", 
                                  k === 'Symbol' && "font-bold text-blue-300", 
                                  k === 'Interval' && "text-muted-foreground",
                                  isNum && getMetricColor(v as number, invert)
                                )}>
                                  {displayVal as React.ReactNode}
                                </td>
                              );
                            })}
                          </motion.tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </motion.div>
      )}

      {activeTab === "path" && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="grid grid-cols-1 lg:grid-cols-4 gap-6">
           <div className="lg:col-span-1 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Path Config</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-muted-foreground">Asset A</label>
                  <select 
                    className="w-full bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:ring-1 focus:ring-emerald-500 outline-none"
                    value={pathSymbolA}
                    onChange={e => setPathSymbolA(e.target.value)}
                  >
                    {universe.map(u => <option key={u} value={u}>{u}</option>)}
                  </select>
                </div>
                
                <div className="space-y-2">
                  <label className="text-sm font-medium text-muted-foreground">Asset B</label>
                  <select 
                    className="w-full bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:ring-1 focus:ring-emerald-500 outline-none"
                    value={pathSymbolB}
                    onChange={e => setPathSymbolB(e.target.value)}
                  >
                    {universe.map(u => <option key={u} value={u}>{u}</option>)}
                  </select>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-muted-foreground">Interval</label>
                  <select 
                    className="w-full bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:ring-1 focus:ring-emerald-500 outline-none"
                    value={pathInterval}
                    onChange={e => setPathInterval(e.target.value)}
                  >
                    {ALL_INTERVALS.map(u => <option key={u} value={u}>{u}</option>)}
                  </select>
                </div>

                <button
                  onClick={handlePath}
                  disabled={isPathing || !pathSymbolA || !pathSymbolB}
                  className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-medium py-2 rounded-md transition-colors disabled:opacity-50 flex justify-center items-center shadow-[0_0_15px_rgba(16,185,129,0.3)]"
                >
                  {isPathing ? <RefreshCw className="h-4 w-4 mr-2 animate-spin" /> : <GitMerge className="h-4 w-4 mr-2" />}
                  Analyze Path
                </button>
              </CardContent>
            </Card>

            {pathData && pathData.metrics && (
              <Card>
                 <CardHeader>
                  <CardTitle>Path Metrics</CardTitle>
                 </CardHeader>
                 <CardContent>
                    <div className="space-y-3">
                      {Object.entries(pathData.metrics).map(([k, v]) => (
                        <div key={k} className="flex justify-between items-center border-b border-border/50 pb-2">
                          <span className="text-muted-foreground text-sm">{k}</span>
                          <span className="font-mono text-sm font-medium">{typeof v === 'number' ? v.toFixed(4) : String(v)}</span>
                        </div>
                      ))}
                    </div>
                 </CardContent>
              </Card>
            )}
           </div>

           <div className="lg:col-span-3 space-y-6">
              <Card className="min-h-[400px]">
                <CardHeader>
                  <CardTitle>Trajectory Phase Space Plot</CardTitle>
                </CardHeader>
                <CardContent>
                  {pathData ? (
                    <Plot
                      data={[
                        {
                          x: pathData.stats_data?.map((d: any) => d.ret_b),
                          y: pathData.stats_data?.map((d: any) => d.ret_a),
                          type: 'scatter',
                          mode: 'markers',
                          marker: { color: 'rgba(16, 185, 129, 0.6)', size: 6 }
                        }
                      ]}
                      layout={{
                        paper_bgcolor: 'transparent',
                        plot_bgcolor: 'transparent',
                        font: { color: '#94a3b8', family: '"Space Mono", monospace' },
                        height: 400,
                        margin: { t: 20, r: 20, b: 40, l: 40 },
                        xaxis: { title: { text: `${pathSymbolB} Returns` }, gridcolor: 'rgba(51, 65, 85, 0.5)', zerolinecolor: 'rgba(255,255,255,0.2)' },
                        yaxis: { title: { text: `${pathSymbolA} Returns` }, gridcolor: 'rgba(51, 65, 85, 0.5)', zerolinecolor: 'rgba(255,255,255,0.2)' },
                      }}
                      useResizeHandler={true}
                      style={{ width: "100%", height: "100%" }}
                    />
                  ) : (
                    <div className="h-full min-h-[300px] flex items-center justify-center text-muted-foreground">Run analysis to view path phase space</div>
                  )}
                </CardContent>
              </Card>

              <Card className="min-h-[400px]">
                <CardHeader>
                  <CardTitle>Systemic Volatility (EWMA vs Historical)</CardTitle>
                </CardHeader>
                <CardContent>
                  {pathData && pathData.volatility ? (
                    <Plot
                      data={[
                        {
                          x: Array.from({length: pathData.volatility.ewma.length}, (_, i) => i),
                          y: pathData.volatility.ewma,
                          type: 'scatter',
                          mode: 'lines',
                          name: 'EWMA Volatility',
                          line: { color: '#3b82f6', width: 2 }
                        },
                        {
                          x: Array.from({length: pathData.volatility.hist.length}, (_, i) => i),
                          y: pathData.volatility.hist,
                          type: 'scatter',
                          mode: 'lines',
                          name: 'Historical Volatility',
                          line: { color: 'rgba(16, 185, 129, 0.5)', width: 1, dash: 'dash' }
                        }
                      ]}
                      layout={{
                        paper_bgcolor: 'transparent',
                        plot_bgcolor: 'transparent',
                        font: { color: '#94a3b8', family: '"Space Mono", monospace' },
                        margin: { t: 10, r: 10, b: 40, l: 40 },
                        xaxis: { gridcolor: 'rgba(51, 65, 85, 0.5)' },
                        yaxis: { gridcolor: 'rgba(51, 65, 85, 0.5)' },
                        autosize: true,
                        showlegend: false
                      } as any}
                      useResizeHandler={true}
                      style={{ width: "100%", height: "100%" }}
                    />
                  ) : (
                    <div className="h-full min-h-[300px] flex items-center justify-center text-muted-foreground">Run analysis to view systemic volatility</div>
                  )}
                </CardContent>
              </Card>
           </div>
        </motion.div>
      )}

    </div>
  );
}
