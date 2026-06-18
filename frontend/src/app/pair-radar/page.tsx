"use client";

import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { pairRadarService, PairResponse } from "@/lib/services/pairRadarService";
import { dataService } from "@/lib/services/dataService";
import { GitMerge, RefreshCw, AlertCircle, Activity, Scaling, TrendingDown, BarChart2, CandlestickChart } from "lucide-react";
import { cn, formatError } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import dynamic from "next/dynamic";
import api from "@/lib/api";

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => <div className="h-[400px] flex items-center justify-center text-muted-foreground animate-pulse glass rounded-md">Loading Chart Engine...</div>,
});

const TvChart = dynamic(() => import("@/components/ui/TvChart"), {
  ssr: false,
  loading: () => <div className="h-[520px] flex items-center justify-center text-muted-foreground animate-pulse glass rounded-md">Loading TradingView Chart...</div>,
});

const ALL_INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"];

interface Candle { time: number; open: number; high: number; low: number; close: number; volume: number; }

export default function PairRadarPage() {
  const [universe, setUniverse] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [symbolA, setSymbolA] = useState("BTCUSDT");
  const [symbolB, setSymbolB] = useState("ETHUSDT");
  const [interval, setInterval] = useState("1h");
  const [mode, setMode] = useState("spread");
  const [rollingWindow, setRollingWindow] = useState(320);
  const [pairWindow, setPairWindow] = useState(500);

  const [activeTab, setActiveTab] = useState<"dashboard" | "candles" | "copula" | "comp">("dashboard");
  const [copulaMode, setCopulaMode] = useState("price");
  const [copulaType, setCopulaType] = useState("t");
  const [copulaParam, setCopulaParam] = useState(2.0);
  const [rWindow, setRWindow] = useState(10);
  const [copulaStationarize, setCopulaStationarize] = useState(false);
  const [copulaEmaWindow, setCopulaEmaWindow] = useState(20);

  // Candle state
  const [candleSymbol, setCandleSymbol] = useState("BTCUSDT");
  const [candleInterval, setCandleInterval] = useState("1h");
  const [candles, setCandles] = useState<Candle[]>([]);
  const [candleLimit, setCandleLimit] = useState(500);
  const [isFetchingCandles, setIsFetchingCandles] = useState(false);
  const [candleError, setCandleError] = useState<string | null>(null);

  const [data, setData] = useState<PairResponse | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    dataService.getUniverse()
      .then((uni) => {
        setUniverse(uni);
        setCandleSymbol(uni[0] || "BTCUSDT");
      })
      .catch((err) => setError(formatError(err)))
      .finally(() => setLoading(false));
  }, []);

  const handleGenerate = async () => {
    try {
      setIsGenerating(true);
      setError(null);
      const res = await pairRadarService.generatePairRadar({
        symbol_a: symbolA, symbol_b: symbolB, interval, mode,
        rolling_window: rollingWindow, pair_window: pairWindow,
        copula_mode: copulaMode, copula_type: copulaType, copula_param: copulaParam,
        r_window: rWindow, copula_stationarize: copulaStationarize, copula_ema_window: copulaEmaWindow
      });
      setData(res);
    } catch (err: any) {
      setError(formatError(err));
      setData(null);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleFetchCandles = async () => {
    try {
      setIsFetchingCandles(true);
      setCandleError(null);
      const res = await api.get("/data/klines", { params: { symbol: candleSymbol, interval: candleInterval, limit: candleLimit } });
      setCandles(res.data.candles);
    } catch (err: any) {
      setCandleError(formatError(err));
    } finally {
      setIsFetchingCandles(false);
    }
  };

  const tabs = [
    { id: "dashboard", label: "Spread Dashboard", icon: BarChart2 },
    { id: "candles", label: "Candle Chart", icon: CandlestickChart },
    { id: "copula", label: "Copula Analysis", icon: Activity },
    { id: "comp", label: "Asset Comparison", icon: Scaling },
  ] as const;

  return (
    <div className="max-w-7xl mx-auto space-y-6">

      {/* ── TOP TOOLBAR ── */}
      <div className="flex flex-col gap-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-purple-400 to-fuchsia-200 bg-clip-text text-transparent flex items-center">
              <GitMerge className="h-8 w-8 mr-3 text-purple-400" />
              Pair Radar
            </h1>
            <p className="text-muted-foreground mt-1">Statistical arbitrage, cointegration &amp; copula analysis.</p>
          </div>

          {/* Tab switcher */}
          <div className="flex bg-secondary/30 p-1 rounded-lg border border-border/50 overflow-x-auto">
            {tabs.map((t) => (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={cn(
                  "px-4 py-2 rounded-md text-sm font-medium whitespace-nowrap transition-all duration-300 flex items-center gap-1.5",
                  activeTab === t.id
                    ? "bg-purple-500/20 text-purple-400 shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                <t.icon className="h-4 w-4" />
                {t.label}
              </button>
            ))}
          </div>
        </div>

        {/* Config bar — always visible */}
        <Card>
          <CardContent className="pt-4 pb-4">
            <div className="flex flex-wrap gap-4 items-end">
              <div className="space-y-1 min-w-[130px]">
                <label className="text-xs font-medium text-muted-foreground">Asset A</label>
                <select className="w-full bg-secondary/30 border border-border rounded-md px-2 py-1.5 text-sm focus:ring-1 focus:ring-purple-500 outline-none" value={symbolA} onChange={e => setSymbolA(e.target.value)} disabled={loading}>
                  {universe.map(u => <option key={u} value={u}>{u}</option>)}
                </select>
              </div>
              <div className="space-y-1 min-w-[130px]">
                <label className="text-xs font-medium text-muted-foreground">Asset B</label>
                <select className="w-full bg-secondary/30 border border-border rounded-md px-2 py-1.5 text-sm focus:ring-1 focus:ring-purple-500 outline-none" value={symbolB} onChange={e => setSymbolB(e.target.value)} disabled={loading}>
                  {universe.map(u => <option key={u} value={u}>{u}</option>)}
                </select>
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium text-muted-foreground">Interval</label>
                <select className="bg-secondary/30 border border-border rounded-md px-2 py-1.5 text-sm focus:ring-1 focus:ring-purple-500 outline-none" value={interval} onChange={e => setInterval(e.target.value)}>
                  {ALL_INTERVALS.map(u => <option key={u} value={u}>{u}</option>)}
                </select>
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium text-muted-foreground">Mode</label>
                <select className="bg-secondary/30 border border-border rounded-md px-2 py-1.5 text-sm focus:ring-1 focus:ring-purple-500 outline-none" value={mode} onChange={e => setMode(e.target.value)}>
                  <option value="spread">OLS Spread</option>
                  <option value="ratio">Ratio A/B</option>
                </select>
              </div>
              <div className="space-y-1 w-24">
                <label className="text-xs font-medium text-muted-foreground">Z Window</label>
                <input type="number" value={rollingWindow} onChange={e => setRollingWindow(Number(e.target.value))} className="w-full bg-secondary/30 border border-border rounded-md px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-purple-500" />
              </div>
              <div className="space-y-1 w-24">
                <label className="text-xs font-medium text-muted-foreground">Chart Len</label>
                <input type="number" value={pairWindow} onChange={e => setPairWindow(Number(e.target.value))} className="w-full bg-secondary/30 border border-border rounded-md px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-purple-500" />
              </div>
              <button
                onClick={handleGenerate}
                disabled={isGenerating || !symbolA || !symbolB}
                className="bg-purple-600 hover:bg-purple-700 text-white font-medium px-5 py-1.5 rounded-md transition-colors disabled:opacity-50 flex items-center gap-2 shadow-[0_0_15px_rgba(147,51,234,0.3)] ml-auto"
              >
                {isGenerating ? <RefreshCw className="h-4 w-4 animate-spin" /> : <GitMerge className="h-4 w-4" />}
                Analyze Pair
              </button>
            </div>
          </CardContent>
        </Card>
      </div>

      <AnimatePresence>
        {error && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}>
            <div className="p-4 bg-rose-500/10 border border-rose-500/50 text-rose-400 rounded-lg flex items-center">
              <AlertCircle className="h-5 w-5 mr-3 shrink-0" />
              {error}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── DASHBOARD TAB ── */}
      {activeTab === "dashboard" && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: "Hedge Ratio (β)", value: data?.metrics?.Coefficient, color: "text-purple-400" },
              { label: "Volatility Ratio", value: data?.metrics?.VolRatio, color: "text-blue-400" },
              { label: "Cointegration p-val", value: data?.metrics?.ADF_P, color: data?.metrics?.ADF_P != null && data.metrics.ADF_P < 0.05 ? "text-emerald-400" : "text-rose-400" },
              { label: "R-Squared", value: data?.metrics?.R2, color: "text-amber-400" },
            ].map((m) => (
              <Card key={m.label}>
                <CardContent className="p-4 flex flex-col justify-center items-center">
                  <span className="text-sm text-muted-foreground mb-1 text-center">{m.label}</span>
                  <span className={cn("text-2xl font-mono font-bold", m.color)}>{m.value != null ? m.value.toFixed(4) : "N/A"}</span>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card className="min-h-[600px]">
            <CardHeader className="pb-2 flex flex-row items-center justify-between">
              <CardTitle>Synthetic Spread Dynamics (OHLC)</CardTitle>
              {data && <span className="text-xs font-mono text-muted-foreground">β: {data.metrics.Coefficient?.toFixed(4)}</span>}
            </CardHeader>
            <CardContent>
              {data && data.chart_data ? (
                <div className="h-[550px]">
                  <Plot
                    data={[
                      { x: data.chart_data.map(d => d.open_time), open: data.chart_data.map(d => d.open), high: data.chart_data.map(d => d.high), low: data.chart_data.map(d => d.low), close: data.chart_data.map(d => d.close), type: "candlestick", name: "Spread", yaxis: "y2", increasing: { line: { color: "#10b981" } }, decreasing: { line: { color: "#f43f5e" } }, showlegend: false },
                      { x: data.chart_data.map(d => d.open_time), y: data.chart_data.map(d => d.bb_up), type: "scatter", mode: "lines", yaxis: "y2", line: { color: "rgba(59, 130, 246, 0.3)", width: 1, dash: "dot" }, showlegend: false },
                      { x: data.chart_data.map(d => d.open_time), y: data.chart_data.map(d => d.bb_dn), type: "scatter", mode: "lines", yaxis: "y2", line: { color: "rgba(59, 130, 246, 0.3)", width: 1, dash: "dot" }, fill: "tonexty", fillcolor: "rgba(59, 130, 246, 0.05)", showlegend: false },
                      { x: data.chart_data.map(d => d.open_time), y: data.chart_data.map(d => d.ma), type: "scatter", mode: "lines", yaxis: "y2", line: { color: "rgba(59, 130, 246, 0.6)", width: 1.5 }, showlegend: false },
                      { x: data.chart_data.map(d => d.open_time), y: data.chart_data.map(d => d.zscore), type: "scatter", mode: "lines", yaxis: "y", name: "Z-Score", line: { color: "#a855f7", width: 2 }, fill: "tozeroy", fillcolor: "rgba(168, 85, 247, 0.1)" },
                      { x: data.chart_data.map(d => d.open_time), y: Array(data.chart_data.length).fill(2), type: "scatter", mode: "lines", yaxis: "y", line: { color: "rgba(244, 63, 94, 0.5)", width: 1, dash: "dash" }, showlegend: false },
                      { x: data.chart_data.map(d => d.open_time), y: Array(data.chart_data.length).fill(-2), type: "scatter", mode: "lines", yaxis: "y", line: { color: "rgba(16, 185, 129, 0.5)", width: 1, dash: "dash" }, showlegend: false },
                    ]}
                    layout={{ paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#94a3b8", family: '"Space Mono", monospace' }, margin: { t: 10, r: 10, b: 40, l: 40 }, xaxis: { gridcolor: "rgba(51, 65, 85, 0.5)", rangeslider: { visible: false } }, yaxis: { domain: [0, 0.25], gridcolor: "rgba(51, 65, 85, 0.5)", title: { text: "Z-Score" } }, yaxis2: { domain: [0.3, 1], gridcolor: "rgba(51, 65, 85, 0.5)", title: { text: "Synthetic Spread" } }, autosize: true, showlegend: false }}
                    useResizeHandler={true} style={{ width: "100%", height: "100%" }}
                  />
                </div>
              ) : (
                <div className="h-[450px] flex flex-col items-center justify-center text-muted-foreground space-y-4">
                  <TrendingDown className="h-12 w-12 opacity-20" />
                  <p>Configure assets and analyze to preview deviation dynamics.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* ── CANDLE CHART TAB ── */}
      {activeTab === "candles" && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
          <Card>
            <CardContent className="pt-4 pb-4">
              <div className="flex flex-wrap gap-4 items-end">
                <div className="space-y-1 min-w-[150px]">
                  <label className="text-xs font-medium text-muted-foreground">Symbol</label>
                  <select className="w-full bg-secondary/30 border border-border rounded-md px-2 py-1.5 text-sm focus:ring-1 focus:ring-purple-500 outline-none" value={candleSymbol} onChange={e => setCandleSymbol(e.target.value)} disabled={loading}>
                    {universe.map(u => <option key={u} value={u}>{u}</option>)}
                  </select>
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium text-muted-foreground">Interval</label>
                  <select className="bg-secondary/30 border border-border rounded-md px-2 py-1.5 text-sm focus:ring-1 focus:ring-purple-500 outline-none" value={candleInterval} onChange={e => setCandleInterval(e.target.value)}>
                    {ALL_INTERVALS.map(u => <option key={u} value={u}>{u}</option>)}
                  </select>
                </div>
                <div className="space-y-1 w-24">
                  <label className="text-xs font-medium text-muted-foreground">Limit</label>
                  <input type="number" value={candleLimit} onChange={e => setCandleLimit(Number(e.target.value))} min={50} max={2000} className="w-full bg-secondary/30 border border-border rounded-md px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-purple-500" />
                </div>
                <button
                  onClick={handleFetchCandles}
                  disabled={isFetchingCandles}
                  className="bg-purple-600 hover:bg-purple-700 text-white font-medium px-5 py-1.5 rounded-md transition-colors disabled:opacity-50 flex items-center gap-2 shadow-[0_0_15px_rgba(147,51,234,0.3)] ml-auto"
                >
                  {isFetchingCandles ? <RefreshCw className="h-4 w-4 animate-spin" /> : <CandlestickChart className="h-4 w-4" />}
                  Load Chart
                </button>
              </div>
              {candleError && (
                <div className="mt-3 p-3 bg-rose-500/10 border border-rose-500/40 text-rose-400 rounded-md text-sm flex items-center gap-2">
                  <AlertCircle className="h-4 w-4 shrink-0" />
                  {candleError}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2">
                <CandlestickChart className="h-5 w-5 text-purple-400" />
                {candleSymbol} — {candleInterval}
                {candles.length > 0 && <span className="text-xs font-mono text-muted-foreground ml-auto">{candles.length} candles</span>}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {candles.length > 0 ? (
                <TvChart candles={candles} symbol={candleSymbol} height={560} showVolume={true} />
              ) : (
                <div className="h-[520px] flex flex-col items-center justify-center text-muted-foreground space-y-4">
                  <CandlestickChart className="h-16 w-16 opacity-20" />
                  <p className="text-center">Select a symbol and interval, then click <strong className="text-purple-400">Load Chart</strong> to view candlestick data.</p>
                  <p className="text-xs text-muted-foreground/60">Data is served from your local cache. Use Data Manager to sync first.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* ── COPULA TAB ── */}
      {activeTab === "copula" && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
          <div className="grid grid-cols-2 gap-4">
            <Card><CardContent className="p-4 flex flex-col justify-center items-center"><span className="text-sm text-muted-foreground mb-1">P(U ≤ u | V=v)</span><span className="text-2xl font-mono text-emerald-400 font-bold">{data?.copula ? data.copula.p_uv.toFixed(4) : "N/A"}</span></CardContent></Card>
            <Card><CardContent className="p-4 flex flex-col justify-center items-center"><span className="text-sm text-muted-foreground mb-1">P(V ≤ v | U=u)</span><span className="text-2xl font-mono text-emerald-400 font-bold">{data?.copula ? data.copula.p_vu.toFixed(4) : "N/A"}</span></CardContent></Card>
          </div>
          <Card className="min-h-[600px]">
            <CardHeader><CardTitle>Copula Dependency {data?.copula?.title_suffix}</CardTitle></CardHeader>
            <CardContent>
              {data && data.copula ? (
                <div className="h-[600px]">
                  <Plot
                    data={[
                      { x: [...data.copula.u].sort((a,b)=>a-b), y: [...data.copula.u].sort((a,b)=>a-b).map((_, i) => data.copula!.dens_a[i]), xaxis: "x", yaxis: "y2", type: "scatter", fill: "tozeroy", hoverinfo: "skip", line: { color: "rgba(6, 182, 212, 0.8)", width: 1 }, fillcolor: "rgba(6, 182, 212, 0.2)", showlegend: false },
                      { x: [...data.copula.v].sort((a,b)=>a-b).map((_, i) => data.copula!.dens_b[i]), y: [...data.copula.v].sort((a,b)=>a-b), xaxis: "x3", yaxis: "y", type: "scatter", fill: "tozerox", hoverinfo: "skip", line: { color: "rgba(6, 182, 212, 0.8)", width: 1 }, fillcolor: "rgba(6, 182, 212, 0.2)", showlegend: false },
                      { x: data.copula.u, y: data.copula.v, xaxis: "x", yaxis: "y", mode: "markers", type: "scatter", marker: { size: 6, color: "rgba(6, 182, 212, 0.5)", line: { width: 0.5, color: "#fff" } }, name: "Joint Distribution" },
                      { x: [data.copula.u_curr], y: [data.copula.v_curr], xaxis: "x", yaxis: "y", mode: "markers", type: "scatter", marker: { size: 14, color: "#f43f5e", symbol: "cross", line: { width: 2, color: "#fff" } }, name: "Current" },
                    ]}
                    layout={{ paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#94a3b8", family: '"Space Mono", monospace' }, autosize: true, grid: { rows: 2, columns: 2, pattern: "independent" }, xaxis: { domain: [0, 0.85], anchor: "y", gridcolor: "rgba(51, 65, 85, 0.5)", range: [-0.05, 1.05], title: `Rank ${symbolA}` }, yaxis: { domain: [0, 0.85], anchor: "x", gridcolor: "rgba(51, 65, 85, 0.5)", range: [-0.05, 1.05], title: `Rank ${symbolB}` }, xaxis3: { domain: [0.85, 1.0], anchor: "y", showgrid: false, zeroline: false, showticklabels: false }, yaxis2: { domain: [0.85, 1.0], anchor: "x", showgrid: false, zeroline: false, showticklabels: false }, showlegend: false, margin: { t: 20, r: 20, b: 40, l: 40 }, shapes: [{ type: "line", xref: "x", yref: "y", x0: data.copula.u_curr, x1: data.copula.u_curr, y0: 0, y1: 1, line: { color: "rgba(255,255,255,0.4)", dash: "dash", width: 1 } }, { type: "line", xref: "x", yref: "y", x0: 0, x1: 1, y0: data.copula.v_curr, y1: data.copula.v_curr, line: { color: "rgba(255,255,255,0.4)", dash: "dash", width: 1 } }] } as any}
                    useResizeHandler={true} style={{ width: "100%", height: "100%" }}
                  />
                </div>
              ) : (
                <div className="h-[500px] flex flex-col items-center justify-center text-muted-foreground space-y-4">
                  <Activity className="h-12 w-12 opacity-20" />
                  <p>Generate analysis to view Copula relationships.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* ── COMPARISON TAB ── */}
      {activeTab === "comp" && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <Card className="min-h-[600px]">
            <CardHeader><CardTitle>Asset Comparison (Volatility Scaled)</CardTitle></CardHeader>
            <CardContent>
              {data && data.comp ? (
                <div className="h-[500px]">
                  <Plot
                    data={[
                      { x: data.comp.ts, y: data.comp.cum_ret_a, type: "scatter", mode: "lines", name: `${symbolA} (Scaled)`, line: { color: "#06b6d4", width: 2 } },
                      { x: data.comp.ts, y: data.comp.cum_ret_b, type: "scatter", mode: "lines", name: symbolB, line: { color: "#f59e0b", width: 2 } },
                    ]}
                    layout={{ paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#94a3b8", family: '"Space Mono", monospace' }, margin: { t: 10, r: 10, b: 40, l: 40 }, xaxis: { gridcolor: "rgba(51, 65, 85, 0.5)" }, yaxis: { gridcolor: "rgba(51, 65, 85, 0.5)" }, showlegend: true, legend: { orientation: "h", y: -0.2 }, autosize: true } as any}
                    useResizeHandler={true} style={{ width: "100%", height: "100%" }}
                  />
                </div>
              ) : (
                <div className="h-[500px] flex flex-col items-center justify-center text-muted-foreground space-y-4">
                  <Scaling className="h-12 w-12 opacity-20" />
                  <p>Generate analysis to view Volatility Scaled Comparison.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      )}

    </div>
  );
}
