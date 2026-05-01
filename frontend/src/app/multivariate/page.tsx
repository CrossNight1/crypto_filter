"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { multivariateService } from "@/lib/services/multivariateService";
import { dataService } from "@/lib/services/dataService";
import { TrendingUp, RefreshCw, AlertCircle, Network } from "lucide-react";
import { cn, formatError } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false, loading: () => <div className="h-[600px] flex items-center justify-center text-muted-foreground animate-pulse glass rounded-md">Loading Matrix Engine...</div> });

const ALL_INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"];

export default function MultivariatePage() {
  const [universe, setUniverse] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [activeTab, setActiveTab] = useState<"matrix" | "decomp">("matrix");

  // Matrix State
  const [mSymbols, setMSymbols] = useState<string[]>([]);
  const [mInterval, setMInterval] = useState("1h");
  const [mStructure, setMStructure] = useState("Correlation");
  const [mMethod, setMMethod] = useState("pearson");
  const [mWindow, setMWindow] = useState(300);
  const [mDataSource, setMDataSource] = useState("return");
  const [mDataStruct, setMDataStruct] = useState("raw");

  const [matrixData, setMatrixData] = useState<any>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  // Decomp State
  const [dSymbols, setDSymbols] = useState<string[]>([]);
  const [dInterval, setDInterval] = useState("1h");
  const [dMethod, setDMethod] = useState("eigen");
  const [dWindow, setDWindow] = useState(100);
  const [dComps, setDComps] = useState(5);
  const [dLinkage, setDLinkage] = useState("complete");
  
  const [decompData, setDecompData] = useState<any>(null);

  useEffect(() => {
    dataService.getUniverse()
      .then(uni => {
        setUniverse(uni);
        const top20 = uni.slice(0, 20);
        setMSymbols(top20);
        setDSymbols(top20);
      })
      .catch(err => setError(formatError(err)))
      .finally(() => setLoading(false));
  }, []);

  const handleMatrix = async () => {
    try {
      setIsGenerating(true);
      setError(null);
      const res = await multivariateService.generateMatrix({
        symbols: mSymbols, interval: mInterval,
        structure: mStructure, method: mMethod,
        window: mWindow, data_source: mDataSource, data_structure: mDataStruct
      });
      setMatrixData(res);
    } catch (err: any) {
      setError(formatError(err));
      setMatrixData(null);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDecomp = async () => {
     try {
       setIsGenerating(true);
       setError(null);
       const res = await multivariateService.runDecomposition({
         symbols: dSymbols, interval: dInterval,
         method: dMethod, window: dWindow,
         n_components: dComps, linkage_method: dLinkage,
         data_source: mDataSource, data_structure: mDataStruct,
         matrix_structure: mStructure, matrix_method: mMethod
       });
       setDecompData(res);
     } catch (err: any) {
       setError(formatError(err));
       setDecompData(null);
     } finally {
       setIsGenerating(false);
     }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between mt-2 mb-6 gap-4">
        <div>
           <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-rose-400 to-pink-200 bg-clip-text text-transparent flex items-center">
             <TrendingUp className="h-8 w-8 mr-3 text-rose-400" />
             Multivariate Analysis
           </h1>
           <p className="text-muted-foreground mt-1">Cross-sectional correlation, decomposition, and structure mapping.</p>
        </div>
        
        <div className="flex bg-secondary/30 p-1 rounded-lg border border-border/50">
          <button
            onClick={() => setActiveTab("matrix")}
            className={cn("px-4 py-2 rounded-md text-sm font-medium transition-all duration-300", activeTab === "matrix" ? "bg-rose-500/20 text-rose-400 shadow-sm" : "text-muted-foreground hover:text-foreground")}
          >
            Matrix Radar
          </button>
          <button
            onClick={() => setActiveTab("decomp")}
            className={cn("px-4 py-2 rounded-md text-sm font-medium transition-all duration-300", activeTab === "decomp" ? "bg-rose-500/20 text-rose-400 shadow-sm" : "text-muted-foreground hover:text-foreground")}
          >
            Decomposition
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

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
         <div className="lg:col-span-1 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Configuration</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                
                {activeTab === "matrix" ? (
                  <>
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-muted-foreground">Structure</label>
                      <select 
                        className="w-full bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:ring-1 focus:ring-rose-500 outline-none"
                        value={mStructure}
                        onChange={e => setMStructure(e.target.value)}
                      >
                        <option value="Correlation">Correlation</option>
                        <option value="Partial Correlation">Partial Correlation</option>
                        <option value="Covariance">Covariance</option>
                        <option value="Arbitrage">Arbitrage Matrix</option>
                      </select>
                    </div>

                    {mStructure === "Arbitrage" ? (
                       <div className="space-y-2">
                        <label className="text-sm font-medium text-muted-foreground">Arb Method</label>
                        <select 
                          className="w-full bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:ring-1 focus:ring-rose-500 outline-none"
                          value={mMethod}
                          onChange={e => setMMethod(e.target.value)}
                        >
                          <option value="cointegration">Cointegration</option>
                          <option value="zscore">Z-Score</option>
                          <option value="arbitrage_score">Arbitrage Score</option>
                        </select>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <label className="text-sm font-medium text-muted-foreground">Correlation Method</label>
                        <select 
                          className="w-full bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:ring-1 focus:ring-rose-500 outline-none"
                          value={mMethod}
                          onChange={e => setMMethod(e.target.value)}
                        >
                          <option value="pearson">Pearson</option>
                          <option value="spearman">Spearman</option>
                          <option value="kendall">Kendall</option>
                        </select>
                      </div>
                    )}
                  </>
                ) : (
                  <>
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-muted-foreground">Method</label>
                      <select 
                        className="w-full bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:ring-1 focus:ring-rose-500 outline-none"
                        value={dMethod}
                        onChange={e => setDMethod(e.target.value)}
                      >
                        <option value="eigen">PCA (Eigenvalue)</option>
                        <option value="rmt">RMT Spectral Filter</option>
                        <option value="cluster">Hierarchical Clustering</option>
                        <option value="mst">Spillover Network</option>
                      </select>
                    </div>

                    {(dMethod === "eigen") && (
                      <div className="space-y-2">
                        <label className="text-xs font-medium text-muted-foreground">Components (K)</label>
                        <input 
                          type="number" 
                          value={dComps}
                          onChange={e => setDComps(Number(e.target.value))}
                          className="w-full bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-rose-500"
                        />
                      </div>
                    )}
                  </>
                )}

                <div className="space-y-2">
                  <label className="text-sm font-medium text-muted-foreground">Interval</label>
                  <select 
                    className="w-full bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:ring-1 focus:ring-rose-500 outline-none"
                    value={activeTab === "matrix" ? mInterval : dInterval}
                    onChange={e => activeTab === "matrix" ? setMInterval(e.target.value) : setDInterval(e.target.value)}
                  >
                    {(ALL_INTERVALS || []).map(u => <option key={u} value={u}>{u}</option>)}
                  </select>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-muted-foreground">Window Size</label>
                  <input 
                    type="number" 
                    value={activeTab === "matrix" ? mWindow : dWindow}
                    onChange={e => activeTab === "matrix" ? setMWindow(Number(e.target.value)) : setDWindow(Number(e.target.value))}
                    className="w-full bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-rose-500"
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-muted-foreground flex justify-between">
                    <span>Symbols</span>
                    <span className="text-xs">{activeTab === "matrix" ? mSymbols.length : dSymbols.length} selected</span>
                  </label>
                  <div className="h-48 overflow-y-auto bg-secondary/30 border border-border/50 rounded-md p-2 flex flex-wrap gap-1.5 content-start">
                    {(universe || []).map(u => {
                      const isSelected = activeTab === "matrix" ? mSymbols.includes(u) : dSymbols.includes(u);
                      return (
                        <button
                          key={u}
                          onClick={() => {
                            if (activeTab === "matrix") {
                              if (isSelected) setMSymbols(mSymbols.filter(s => s !== u));
                              else setMSymbols([...mSymbols, u]);
                            } else {
                              if (isSelected) setDSymbols(dSymbols.filter(s => s !== u));
                              else setDSymbols([...dSymbols, u]);
                            }
                          }}
                          className={cn(
                            "px-2.5 py-1 text-xs rounded-full border transition-all duration-200 font-medium",
                            isSelected ? "bg-rose-500/20 border-rose-500/50 text-rose-400" : "bg-background/50 border-border/50 text-muted-foreground hover:bg-secondary/80 hover:text-foreground hover:border-border"
                          )}
                        >
                          {u}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <button
                  onClick={activeTab === "matrix" ? handleMatrix : handleDecomp}
                  disabled={isGenerating || (activeTab === "matrix" ? mSymbols.length === 0 : dSymbols.length === 0)}
                  className="w-full mt-4 bg-rose-600 hover:bg-rose-700 text-white font-medium py-2 rounded-md transition-colors disabled:opacity-50 flex justify-center items-center shadow-[0_0_15px_rgba(225,29,72,0.3)]"
                >
                  {isGenerating ? <RefreshCw className="h-4 w-4 mr-2 animate-spin" /> : (activeTab === "matrix" ? <Network className="h-4 w-4 mr-2" /> : <TrendingUp className="h-4 w-4 mr-2" />)}
                  {activeTab === "matrix" ? "Generate Matrix" : "Run Decomp"}
                </button>

              </CardContent>
            </Card>
         </div>

         <div className="lg:col-span-3 space-y-6">
            <Card className="min-h-[600px]">
               <CardHeader className="pb-2">
                 <CardTitle>{activeTab === "matrix" ? "Dependence Matrix" : "Decomposition Results"}</CardTitle>
               </CardHeader>
               <CardContent>
                 {activeTab === "matrix" ? (
                   matrixData ? (
                     <div className="h-[650px]">
                       <Plot 
                         data={[{
                            z: matrixData.data,
                            x: matrixData.columns,
                            y: matrixData.index,
                            type: 'heatmap',
                            colorscale: 'RdBu',
                            reversescale: true,
                            zsmooth: false
                         }]}
                         layout={{
                            paper_bgcolor: 'transparent',
                            plot_bgcolor: 'transparent',
                            font: { color: '#94a3b8', family: '"Space Mono", monospace' },
                            margin: { t: 10, r: 10, b: 80, l: 80 },
                            xaxis: { tickangle: 45 },
                            autosize: true
                         }}
                         useResizeHandler={true}
                         style={{ width: "100%", height: "100%" }}
                       />
                     </div>
                   ) : (
                     <div className="h-[600px] flex flex-col items-center justify-center text-muted-foreground space-y-4">
                        <Network className="h-12 w-12 opacity-20" />
                        <p>Generate matrix to view cross-sectional dependence.</p>
                     </div>
                   )
                 ) : (
                   decompData ? (
                     <div className="space-y-4">
                        <div className="glass-panel p-4 rounded-lg">
                          <h3 className="font-semibold text-rose-400 mb-2 font-mono">Method: {decompData.method}</h3>
                          <pre className="text-sm text-foreground overflow-auto max-h-[500px] font-mono">
                            {JSON.stringify(decompData, null, 2)}
                          </pre>
                        </div>
                     </div>
                   ) : (
                     <div className="h-[600px] flex flex-col items-center justify-center text-muted-foreground space-y-4">
                        <TrendingUp className="h-12 w-12 opacity-20" />
                        <p>Run decomposition to extract market structure signals.</p>
                     </div>
                   )
                 )}
               </CardContent>
            </Card>
         </div>
      </div>
    </div>
  );
}
