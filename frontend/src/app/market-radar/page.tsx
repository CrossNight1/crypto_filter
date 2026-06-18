"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { marketRadarService } from "@/lib/services/marketRadarService";
import { dataService } from "@/lib/services/dataService";
import { Activity, RefreshCw, AlertCircle, BarChart2, Zap, ScatterChart } from "lucide-react";
import { cn, formatError } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => <div className="h-[480px] flex items-center justify-center text-muted-foreground animate-pulse glass rounded-md">Loading Chart Engine...</div>,
});

const ALL_INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"];

const METRIC_OPTIONS = [
  { value: "ewva",               label: "EWVA" },
  { value: "aroon_osc",          label: "Aroon Oscillator" },
  { value: "rsi_norm",           label: "RSI (Normalized)" },
  { value: "atr_norm",           label: "Normalized ATR" },
  { value: "cmf",                label: "Chaikin Money Flow" },
  { value: "vwap_z",             label: "VWAP Z-Score" },
  { value: "rel_strength_z",     label: "RS Z-Score" },
  { value: "vam",                label: "VAM" },
  { value: "skewness",           label: "Skewness" },
  { value: "vol_imbalance",      label: "Vol Imbalance" },
  { value: "volume_imbalance",   label: "Volume Imbalance" },
  { value: "volatility",         label: "Volatility" },
  { value: "fip",                label: "FIP" },
  { value: "max_drawdown",       label: "Max Drawdown" },
  { value: "avg_drawdown",       label: "Avg Drawdown" },
  { value: "breakout_score_dist","label": "Breakout Score v1" },
  { value: "breakout_score_break","label":"Breakout Score v2" },
  { value: "adf_hist",           label: "ADF Regime" },
  { value: "funding_rate",       label: "Funding Rate" },
  { value: "oi_circulating",     label: "OI / Circulating Supply" },
  { value: "top_ls_position_ratio", label: "Top Trader L/S (Position)" },
  { value: "top_ls_account_ratio",  label: "Top Trader L/S (Account)" },
  { value: "global_ls_ratio",    label: "Global L/S Account Ratio" },
  { value: "taker_buysell_ratio", label: "Taker Buy/Sell Volume Ratio" },
  { value: "adl_risk",           label: "ADL Risk Rating" },
];

export default function MarketRadarPage() {
  const [universe, setUniverse] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [activeTab, setActiveTab] = useState<"snapshot" | "scatter">("snapshot");

  // Snapshot state
  const [snapSymbols, setSnapSymbols] = useState<string[]>([]);
  const [snapInterval, setSnapInterval] = useState("1h");
  const [snapData, setSnapData] = useState<any[]>([]);
  const [isSnapping, setIsSnapping] = useState(false);

  // Scatter state
  const [xMetric, setXMetric] = useState("rel_strength_z");
  const [yMetric, setYMetric] = useState("breakout_score_dist");
  const [colorMetric, setColorMetric] = useState("funding_rate");
  const [focusSymbol, setFocusSymbol] = useState("");

  useEffect(() => {
    dataService.getUniverse()
      .then((uni) => {
        setUniverse(uni);
        setSnapSymbols(uni.slice(0, 20));
      })
      .catch((err) => setError(formatError(err)))
      .finally(() => setLoading(false));
  }, []);

  const handleSnapshot = async () => {
    if (snapSymbols.length === 0) return;
    try {
      setIsSnapping(true);
      setError(null);
      const res = await marketRadarService.getSnapshot({ symbols: snapSymbols, intervals: [snapInterval] });
      setSnapData(res.metrics);
    } catch (err: any) {
      setError(formatError(err));
    } finally {
      setIsSnapping(false);
    }
  };

  const getMetricColor = (val: number | string, invert = false) => {
    if (typeof val !== "number") return "text-foreground";
    if (val === 0) return "text-muted-foreground";
    const isGood = invert ? val < 0 : val > 0;
    return isGood ? "text-emerald-400 font-medium" : "text-rose-400 font-medium";
  };

  const formatVal = (v: any) => {
    if (v === null || v === undefined) return "—";
    if (typeof v === "number") return v % 1 === 0 ? String(v) : v.toFixed(3);
    return String(v);
  };

  // Scatter chart data derived from snapData
  const scatterTrace = (() => {
    if (!snapData.length) return null;
    const filtered = snapData.filter(r =>
      typeof r[xMetric] === "number" && typeof r[yMetric] === "number"
    );
    if (!filtered.length) return null;

    const focusIdx = focusSymbol
      ? filtered.findIndex(r => r.symbol?.toUpperCase() === focusSymbol.toUpperCase())
      : -1;

    const colorVals = filtered.map(r => typeof r[colorMetric] === "number" ? r[colorMetric] : 0);

    return {
      x: filtered.map(r => r[xMetric]),
      y: filtered.map(r => r[yMetric]),
      text: filtered.map(r => r.symbol || ""),
      mode: "markers+text" as const,
      type: "scatter" as const,
      textposition: "top center",
      textfont: { size: 9, color: "rgba(148,163,184,0.8)" },
      marker: {
        size: filtered.map((_, i) => i === focusIdx ? 16 : 8),
        color: colorVals,
        colorscale: "Viridis",
        showscale: true,
        colorbar: {
          title: METRIC_OPTIONS.find(m => m.value === colorMetric)?.label || colorMetric,
          tickfont: { color: "#94a3b8", size: 10 },
          titlefont: { color: "#94a3b8", size: 10 },
          thickness: 12,
        },
        line: { color: filtered.map((_, i) => i === focusIdx ? "#facc15" : "rgba(255,255,255,0.15)"), width: filtered.map((_, i) => i === focusIdx ? 2 : 0.5) },
        opacity: filtered.map((_, i) => focusIdx >= 0 ? (i === focusIdx ? 1 : 0.3) : 0.85),
      },
      hovertemplate: "<b>%{text}</b><br>" +
        `${METRIC_OPTIONS.find(m=>m.value===xMetric)?.label}: %{x:.3f}<br>` +
        `${METRIC_OPTIONS.find(m=>m.value===yMetric)?.label}: %{y:.3f}<extra></extra>`,
    };
  })();

  return (
    <div className="max-w-7xl mx-auto space-y-6">

      {/* ── TOP TOOLBAR ── */}
      <div className="flex flex-col gap-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-emerald-400 to-teal-200 bg-clip-text text-transparent flex items-center">
              <Activity className="h-8 w-8 mr-3 text-emerald-400" />
              Market Radar
            </h1>
            <p className="text-muted-foreground mt-1">Real-time asset scoring and scatter analysis.</p>
          </div>

          <div className="flex bg-secondary/30 p-1 rounded-lg border border-border/50">
            <button onClick={() => setActiveTab("snapshot")} className={cn("px-4 py-2 rounded-md text-sm font-medium transition-all duration-300 flex items-center gap-2", activeTab === "snapshot" ? "bg-emerald-500/20 text-emerald-400 shadow-sm" : "text-muted-foreground hover:text-foreground")}>
              <BarChart2 className="h-4 w-4" /> Snapshot Table
            </button>
            <button onClick={() => setActiveTab("scatter")} className={cn("px-4 py-2 rounded-md text-sm font-medium transition-all duration-300 flex items-center gap-2", activeTab === "scatter" ? "bg-emerald-500/20 text-emerald-400 shadow-sm" : "text-muted-foreground hover:text-foreground")}>
              <ScatterChart className="h-4 w-4" /> Scatter Chart
            </button>
          </div>
        </div>

        {/* Config toolbar */}
        <Card>
          <CardContent className="pt-4 pb-4">
            <div className="flex flex-wrap gap-4 items-end">
              <div className="space-y-1">
                <label className="text-xs font-medium text-muted-foreground">Interval</label>
                <select className="bg-secondary/30 border border-border rounded-md px-2 py-1.5 text-sm focus:ring-1 focus:ring-emerald-500 outline-none" value={snapInterval} onChange={e => setSnapInterval(e.target.value)}>
                  {ALL_INTERVALS.map(i => <option key={i} value={i}>{i}</option>)}
                </select>
              </div>

              {activeTab === "scatter" && (
                <>
                  <div className="space-y-1 min-w-[150px]">
                    <label className="text-xs font-medium text-muted-foreground">X Axis</label>
                    <select className="w-full bg-secondary/30 border border-border rounded-md px-2 py-1.5 text-sm focus:ring-1 focus:ring-emerald-500 outline-none" value={xMetric} onChange={e => setXMetric(e.target.value)}>
                      {METRIC_OPTIONS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                    </select>
                  </div>
                  <div className="space-y-1 min-w-[150px]">
                    <label className="text-xs font-medium text-muted-foreground">Y Axis</label>
                    <select className="w-full bg-secondary/30 border border-border rounded-md px-2 py-1.5 text-sm focus:ring-1 focus:ring-emerald-500 outline-none" value={yMetric} onChange={e => setYMetric(e.target.value)}>
                      {METRIC_OPTIONS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                    </select>
                  </div>
                  <div className="space-y-1 min-w-[140px]">
                    <label className="text-xs font-medium text-muted-foreground">Color By</label>
                    <select className="w-full bg-secondary/30 border border-border rounded-md px-2 py-1.5 text-sm focus:ring-1 focus:ring-emerald-500 outline-none" value={colorMetric} onChange={e => setColorMetric(e.target.value)}>
                      {METRIC_OPTIONS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                    </select>
                  </div>
                  <div className="space-y-1 min-w-[120px]">
                    <label className="text-xs font-medium text-muted-foreground">Focus Symbol</label>
                    <input type="text" placeholder="e.g. BTCUSDT" value={focusSymbol} onChange={e => setFocusSymbol(e.target.value.toUpperCase())} className="w-full bg-secondary/30 border border-border rounded-md px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500 uppercase" />
                  </div>
                </>
              )}

              <div className="flex items-end gap-2 ml-auto">
                <div className="space-y-1">
                  <label className="text-xs font-medium text-muted-foreground">Symbols ({snapSymbols.length})</label>
                  <div className="flex gap-2">
                    <button onClick={() => setSnapSymbols(universe.slice(0, 20))} className="px-2 py-1.5 text-xs bg-secondary/50 hover:bg-secondary border border-border rounded-md">Top 20</button>
                    <button onClick={() => setSnapSymbols(universe.slice(0, 50))} className="px-2 py-1.5 text-xs bg-secondary/50 hover:bg-secondary border border-border rounded-md">Top 50</button>
                    <button onClick={() => setSnapSymbols(universe)} className="px-2 py-1.5 text-xs bg-secondary/50 hover:bg-secondary border border-border rounded-md">All</button>
                  </div>
                </div>
                <button
                  onClick={handleSnapshot}
                  disabled={isSnapping || snapSymbols.length === 0}
                  className="bg-emerald-600 hover:bg-emerald-700 text-white font-medium px-5 py-1.5 rounded-md transition-colors disabled:opacity-50 flex items-center gap-2 shadow-[0_0_15px_rgba(16,185,129,0.3)]"
                >
                  {isSnapping ? <RefreshCw className="h-4 w-4 animate-spin" /> : <BarChart2 className="h-4 w-4" />}
                  {isSnapping ? "Calculating…" : "Generate Snapshot"}
                </button>
              </div>
            </div>

            {/* Symbol pill selector */}
            <div className="mt-3 h-24 overflow-y-auto bg-secondary/20 border border-border/40 rounded-md p-2 flex flex-wrap gap-1.5 content-start">
              {universe.map(u => {
                const isSel = snapSymbols.includes(u);
                return (
                  <button
                    key={u}
                    onClick={() => isSel ? setSnapSymbols(snapSymbols.filter(s => s !== u)) : setSnapSymbols([...snapSymbols, u])}
                    className={cn("px-2 py-0.5 text-xs rounded-full border transition-all font-medium", isSel ? "bg-emerald-500/20 border-emerald-500/50 text-emerald-400" : "bg-background/50 border-border/50 text-muted-foreground hover:bg-secondary/80 hover:text-foreground")}
                  >
                    {u}
                  </button>
                );
              })}
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

      {/* ── SNAPSHOT TABLE ── */}
      {activeTab === "snapshot" && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <Card className="min-h-[500px]">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart2 className="h-5 w-5 text-emerald-400" />
                Asset Metrics
                {snapData.length > 0 && <span className="text-xs font-mono text-muted-foreground ml-auto">{snapData.length} symbols</span>}
              </CardTitle>
            </CardHeader>
            <CardContent className="overflow-auto">
              {!snapData.length ? (
                <div className="h-80 flex flex-col items-center justify-center text-muted-foreground space-y-4">
                  <Zap className="h-12 w-12 opacity-20" />
                  <p>Select symbols and generate snapshot to view metrics.</p>
                </div>
              ) : (
                <table className="w-full text-sm text-left whitespace-nowrap">
                  <thead className="text-xs text-muted-foreground uppercase bg-secondary/50 sticky top-0 backdrop-blur-md">
                    <tr>
                      {Object.keys(snapData[0] || {}).map(k => (
                        <th key={k} className="px-4 py-3 font-semibold">{k.replace(/_/g, " ")}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/50">
                    {snapData.map((row, i) => (
                      <motion.tr
                        key={i}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: Math.min(i * 0.03, 0.6) }}
                        className="hover:bg-secondary/30 transition-colors cursor-pointer"
                        onClick={() => setFocusSymbol(row.symbol || "")}
                      >
                        {Object.entries(row).map(([k, v]) => (
                          <td key={k} className={cn(
                            "px-4 py-2.5",
                            k === "symbol" && "font-bold text-blue-300 font-mono",
                            typeof v === "number" && k !== "count" && getMetricColor(v as number, k.includes("drawdown"))
                          )}>
                            {formatVal(v)}
                          </td>
                        ))}
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              )}
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* ── SCATTER CHART ── */}
      {activeTab === "scatter" && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <Card className="min-h-[560px]">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <ScatterChart className="h-5 w-5 text-emerald-400" />
                {METRIC_OPTIONS.find(m => m.value === xMetric)?.label} vs {METRIC_OPTIONS.find(m => m.value === yMetric)?.label}
                {snapData.length > 0 && <span className="text-xs font-mono text-muted-foreground ml-auto">{snapData.length} symbols</span>}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {scatterTrace ? (
                <Plot
                  data={[scatterTrace as any]}
                  layout={{
                    paper_bgcolor: "transparent",
                    plot_bgcolor: "transparent",
                    font: { color: "#94a3b8", family: '"Space Mono", monospace', size: 11 },
                    margin: { t: 20, r: 80, b: 60, l: 60 },
                    height: 520,
                    xaxis: {
                      title: { text: METRIC_OPTIONS.find(m => m.value === xMetric)?.label },
                      gridcolor: "rgba(51, 65, 85, 0.5)",
                      zerolinecolor: "rgba(255,255,255,0.2)",
                    },
                    yaxis: {
                      title: { text: METRIC_OPTIONS.find(m => m.value === yMetric)?.label },
                      gridcolor: "rgba(51, 65, 85, 0.5)",
                      zerolinecolor: "rgba(255,255,255,0.2)",
                    },
                    shapes: [
                      { type: "line", x0: 0, x1: 0, y0: 0, y1: 1, xref: "x", yref: "paper", line: { color: "rgba(255,255,255,0.15)", width: 1, dash: "dot" } },
                      { type: "line", x0: 0, x1: 1, y0: 0, y1: 0, xref: "paper", yref: "y", line: { color: "rgba(255,255,255,0.15)", width: 1, dash: "dot" } },
                    ],
                    autosize: true,
                    showlegend: false,
                  } as any}
                  useResizeHandler
                  style={{ width: "100%", height: "100%" }}
                  config={{ displayModeBar: false, responsive: true }}
                />
              ) : (
                <div className="h-[480px] flex flex-col items-center justify-center text-muted-foreground space-y-4">
                  <ScatterChart className="h-16 w-16 opacity-20" />
                  <p className="text-center">Generate a snapshot first, then switch to Scatter Chart to visualize.</p>
                  <p className="text-xs text-muted-foreground/60">Use the axis dropdowns in the toolbar to change metrics.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      )}

    </div>
  );
}
