"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { diagnosticsService, DiagnosticsResponse } from "@/lib/services/diagnosticsService";
import { dataService } from "@/lib/services/dataService";
import { Search, AlertCircle, Play, TrendingDown, TrendingUp, ShieldAlert, BarChart3, Activity, RefreshCw } from "lucide-react";
import { cn, formatError } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false, loading: () => <div className="h-[300px] flex items-center justify-center text-muted-foreground animate-pulse glass rounded-md">Loading Chart Engine...</div> });

const ALL_INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"];

export default function DiagnosticsPage() {
  const [universe, setUniverse] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [symbol, setSymbol] = useState("BTCUSDT");
  const [interval, setInterval] = useState("1h");
  const [metricWindow, setMetricWindow] = useState(10);
  const [diagWindow, setDiagWindow] = useState(100);
  
  const [data, setData] = useState<DiagnosticsResponse | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    dataService.getUniverse()
      .then(uni => setUniverse(uni))
      .catch(err => setError(formatError(err)))
      .finally(() => setLoading(false));
  }, []);

  const handleRun = async () => {
    try {
      setIsRunning(true);
      setError(null);
      const res = await diagnosticsService.runDiagnostics({
        symbol, interval, metric_window: metricWindow, diag_window: diagWindow
      });
      setData(res);
    } catch (err: any) {
      setError(formatError(err));
      setData(null);
    } finally {
      setIsRunning(false);
    }
  };

  const perf = data?.performance;

  const MetricCard = ({ title, value, unit = "", invert = false, icon: Icon }: any) => {
    const isGood = invert ? value < 0 : value > 0;
    const color = value === 0 || value === null ? "text-muted-foreground" : isGood ? "text-emerald-400" : "text-rose-400";
    
    return (
      <div className="bg-secondary/40 border border-border/50 rounded-xl p-4 flex flex-col justify-between hover:bg-secondary/60 transition-colors">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-muted-foreground truncate mr-2">{title}</span>
          <Icon className={cn("h-4 w-4 opacity-50", color)} />
        </div>
        <div className={cn("text-2xl font-bold font-mono", color)}>
          {value !== null ? (typeof value === "number" && value % 1 !== 0 ? value.toFixed(4) : value) : "N/A"}{unit}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between mt-2 mb-6 gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-amber-400 to-orange-200 bg-clip-text text-transparent flex items-center">
            <Search className="h-8 w-8 mr-3 text-amber-400" />
            Symbol Diagnostics
          </h1>
          <p className="text-muted-foreground mt-1">Deep-dive performance, risk, and structural analysis.</p>
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
              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">Symbol</label>
                <select 
                  className="w-full bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:ring-1 focus:ring-amber-500 outline-none"
                  value={symbol}
                  onChange={e => setSymbol(e.target.value)}
                  disabled={loading}
                >
                  {universe ? universe.map(u => <option key={u} value={u}>{u}</option>) : null}
                </select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">Interval</label>
                <select 
                  className="w-full bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:ring-1 focus:ring-amber-500 outline-none"
                  value={interval}
                  onChange={e => setInterval(e.target.value)}
                >
                  {ALL_INTERVALS ? ALL_INTERVALS.map(u => <option key={u} value={u}>{u}</option>) : null}
                </select>
              </div>
              
              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">Metric Window</label>
                <input 
                  type="number" 
                  value={metricWindow}
                  onChange={e => setMetricWindow(Number(e.target.value))}
                  className="w-full bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
                />
              </div>
              
              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">Diagnostic Window</label>
                <input 
                  type="number" 
                  value={diagWindow}
                  onChange={e => setDiagWindow(Number(e.target.value))}
                  className="w-full bg-secondary/30 border border-border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
                />
              </div>

              <button
                onClick={handleRun}
                disabled={isRunning || !symbol}
                className="w-full mt-4 bg-amber-600 hover:bg-amber-700 text-white font-medium py-2 rounded-md transition-colors disabled:opacity-50 flex justify-center items-center shadow-[0_0_15px_rgba(217,119,6,0.3)]"
              >
                {isRunning ? <RefreshCw className="h-4 w-4 mr-2 animate-spin" /> : <Play className="h-4 w-4 mr-2" />}
                Run Diagnostics
              </button>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-3">
          <Card className="h-full min-h-[500px]">
            <CardHeader>
              <CardTitle>Analysis Results</CardTitle>
            </CardHeader>
            <CardContent>
              {!data ? (
                <div className="h-full min-h-[400px] flex flex-col items-center justify-center text-muted-foreground space-y-4">
                  <BarChart3 className="h-12 w-12 opacity-20" />
                  <p>Configure parameters and run diagnostics to view metrics.</p>
                </div>
              ) : (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
                  
                  <div>
                    <h3 className="text-lg font-medium mb-4 flex items-center">
                      <Activity className="h-5 w-5 mr-2 text-primary" /> Return & Risk Profiles
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-4">
                      <MetricCard title="Sharpe Ratio" value={perf?.sharpe} icon={TrendingUp} />
                      <MetricCard title="Sortino Ratio" value={perf?.sortino} icon={TrendingUp} />
                      <MetricCard title="Omega Ratio" value={perf?.omega} icon={Activity} />
                      <MetricCard title="Volatility (Ann.)" value={perf?.volatility} invert icon={TrendingDown} />
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-medium mb-4 flex items-center">
                      <ShieldAlert className="h-5 w-5 mr-2 text-rose-400" /> Drawdown & Tail Risk
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-4">
                      <MetricCard title="Max Drawdown" value={perf?.maxdd} invert icon={TrendingDown} />
                      <MetricCard title="Avg Drawdown" value={perf?.avgdd} invert icon={TrendingDown} />
                      <MetricCard title="CVaR (95%)" value={perf?.cvar} invert icon={ShieldAlert} />
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-medium mb-4 flex items-center">
                      <BarChart3 className="h-5 w-5 mr-2 text-blue-400" /> Market Microstructure
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-4">
                      <MetricCard title="Beta to BTC" value={perf?.beta} icon={Activity} />
                      <MetricCard title="Alpha to BTC" value={perf?.alpha} icon={TrendingUp} />
                      <MetricCard title="Impact Spread" value={perf?.impact_spread} invert icon={Activity} />
                      <MetricCard title="OB Imbalance" value={perf?.imbalance} icon={Activity} />
                    </div>
                  </div>

                  {data.charts && (
                    <div className="mt-8 space-y-6">
                      {data.charts.ohlcv && (
                        <Card className="min-h-[500px]">
                          <CardHeader className="pb-2">
                            <CardTitle>Price Action & Structural Indicators</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="h-[450px]">
                              <Plot
                                data={[
                                  {
                                    x: data.charts.ohlcv.map(d => d.time),
                                    open: data.charts.ohlcv.map(d => d.open),
                                    high: data.charts.ohlcv.map(d => d.high),
                                    low: data.charts.ohlcv.map(d => d.low),
                                    close: data.charts.ohlcv.map(d => d.close),
                                    type: 'candlestick',
                                    name: 'Price',
                                    increasing: { line: { color: '#10b981' } },
                                    decreasing: { line: { color: '#f43f5e' } }
                                  },
                                  {
                                    x: data.charts.ohlcv.map(d => d.time),
                                    y: data.charts.ohlcv.map(d => d.bb_up),
                                    type: 'scatter', mode: 'lines',
                                    line: { color: 'rgba(59, 130, 246, 0.3)', width: 1, dash: 'dot' },
                                    name: 'BB Upper'
                                  },
                                  {
                                    x: data.charts.ohlcv.map(d => d.time),
                                    y: data.charts.ohlcv.map(d => d.bb_dn),
                                    type: 'scatter', mode: 'lines',
                                    line: { color: 'rgba(59, 130, 246, 0.3)', width: 1, dash: 'dot' },
                                    name: 'BB Lower', fill: 'tonexty', fillcolor: 'rgba(59, 130, 246, 0.05)'
                                  },
                                  {
                                    x: data.charts.ohlcv.map(d => d.time),
                                    y: data.charts.ohlcv.map(d => d.bb_mid),
                                    type: 'scatter', mode: 'lines',
                                    line: { color: 'rgba(59, 130, 246, 0.6)', width: 1.5 },
                                    name: 'MA 20'
                                  }
                                ]}
                                layout={{
                                  paper_bgcolor: 'transparent',
                                  plot_bgcolor: 'transparent',
                                  font: { color: '#94a3b8', family: '"Space Mono", monospace' },
                                  margin: { t: 10, r: 10, b: 40, l: 40 },
                                  xaxis: { gridcolor: 'rgba(51, 65, 85, 0.5)', rangeslider: { visible: false } },
                                  yaxis: { gridcolor: 'rgba(51, 65, 85, 0.5)', fixedrange: false },
                                  autosize: true,
                                  showlegend: false
                                }}
                                useResizeHandler={true}
                                style={{ width: "100%", height: "100%" }}
                              />
                            </div>
                          </CardContent>
                        </Card>
                      )}
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <Card className="min-h-[400px]">
                          <CardHeader>
                            <CardTitle>Factor Metrics (Z-Score)</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <Plot
                              data={[{
                                type: 'barpolar',
                                r: data.charts.metrics.map(m => m.Value),
                                theta: data.charts.metrics.map(m => m.Metric),
                                marker: {
                                  color: data.charts.metrics.map(m => m.Value),
                                  colorscale: 'Spectral',
                                  reversescale: true
                                },
                                opacity: 0.8
                              }]}
                              layout={{
                                polar: {
                                  bgcolor: 'transparent',
                                  radialaxis: { visible: true, gridcolor: 'rgba(255, 255, 255, 0.1)' },
                                  angularaxis: { gridcolor: 'rgba(255, 255, 255, 0.1)' }
                                },
                                paper_bgcolor: 'transparent',
                                plot_bgcolor: 'transparent',
                                font: { color: '#94a3b8', family: '"Space Mono", monospace' },
                                margin: { t: 40, r: 40, b: 40, l: 40 },
                                autosize: true
                              } as any}
                              useResizeHandler={true}
                              style={{ width: "100%", height: "100%", minHeight: "350px" }}
                            />
                          </CardContent>
                        </Card>

                        <Card className="min-h-[400px]">
                          <CardHeader>
                            <CardTitle>Market-Neutral Cum Return (Ex-PC1)</CardTitle>
                          </CardHeader>
                          <CardContent>
                             <Plot
                              data={[{
                                x: Array.from({length: data.charts.mn_cum_ret.length}, (_, i) => i),
                                y: data.charts.mn_cum_ret,
                                type: 'scatter',
                                mode: 'lines',
                                name: 'MN Cum Ret',
                                line: { color: '#06b6d4', width: 2 }
                              }]}
                              layout={{
                                paper_bgcolor: 'transparent',
                                plot_bgcolor: 'transparent',
                                font: { color: '#94a3b8', family: '"Space Mono", monospace' },
                                margin: { t: 20, r: 20, b: 40, l: 40 },
                                xaxis: { gridcolor: 'rgba(51, 65, 85, 0.5)' },
                                yaxis: { gridcolor: 'rgba(51, 65, 85, 0.5)' },
                                autosize: true
                              } as any}
                              useResizeHandler={true}
                              style={{ width: "100%", height: "100%", minHeight: "350px" }}
                            />
                          </CardContent>
                        </Card>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <Card className="min-h-[400px]">
                          <CardHeader>
                            <CardTitle>Forecast: Price (ARIMA)</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <Plot
                              data={[
                                {
                                  y: data.charts.prices.hist,
                                  type: 'scatter',
                                  mode: 'lines',
                                  name: 'History',
                                  line: { color: '#06b6d4', width: 2 }
                                },
                                {
                                  x: Array.from({length: data.charts.prices.forecast.length}, (_, i) => i + data.charts.prices.hist.length - 1),
                                  y: [data.charts.prices.hist[data.charts.prices.hist.length - 1], ...data.charts.prices.forecast.slice(1)],
                                  type: 'scatter',
                                  mode: 'lines',
                                  name: 'Forecast',
                                  line: { color: '#84cc16', width: 2, dash: 'dot' }
                                },
                                {
                                  x: Array.from({length: data.charts.prices.forecast.length * 2}, (_, i) => {
                                      const steps = data.charts.prices.forecast.length;
                                      return i < steps ? i + data.charts.prices.hist.length - 1 : (steps * 2 - 1 - i) + data.charts.prices.hist.length - 1;
                                  }),
                                  y: [...data.charts.prices.ci_lower, ...data.charts.prices.ci_upper.reverse()],
                                  fill: 'toself',
                                  fillcolor: 'rgba(132, 204, 22, 0.1)',
                                  line: { color: 'transparent' },
                                  name: '95% CI',
                                  type: 'scatter'
                                }
                              ]}
                              layout={{
                                paper_bgcolor: 'transparent',
                                plot_bgcolor: 'transparent',
                                font: { color: '#94a3b8', family: '"Space Mono", monospace' },
                                margin: { t: 20, r: 20, b: 40, l: 40 },
                                xaxis: { gridcolor: 'rgba(51, 65, 85, 0.5)' },
                                yaxis: { gridcolor: 'rgba(51, 65, 85, 0.5)' },
                                showlegend: true,
                                legend: { orientation: 'h', y: -0.2 },
                                autosize: true
                              } as any}
                              useResizeHandler={true}
                              style={{ width: "100%", height: "100%", minHeight: "350px" }}
                            />
                          </CardContent>
                        </Card>

                        <Card className="min-h-[400px]">
                          <CardHeader>
                            <CardTitle>Forecast: Volatility (GARCH)</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <Plot
                              data={[
                                {
                                  y: data.charts.volatility.hist,
                                  type: 'scatter',
                                  mode: 'lines',
                                  name: 'History',
                                  line: { color: '#f59e0b', width: 2 }
                                },
                                {
                                  x: Array.from({length: data.charts.volatility.forecast.length}, (_, i) => i + data.charts.volatility.hist.length - 1),
                                  y: [data.charts.volatility.hist[data.charts.volatility.hist.length - 1], ...data.charts.volatility.forecast.slice(1)],
                                  type: 'scatter',
                                  mode: 'lines',
                                  name: 'Forecast',
                                  line: { color: '#f43f5e', width: 2, dash: 'dot' }
                                }
                              ]}
                              layout={{
                                paper_bgcolor: 'transparent',
                                plot_bgcolor: 'transparent',
                                font: { color: '#94a3b8', family: '"Space Mono", monospace' },
                                margin: { t: 20, r: 20, b: 40, l: 40 },
                                xaxis: { gridcolor: 'rgba(51, 65, 85, 0.5)' },
                                yaxis: { gridcolor: 'rgba(51, 65, 85, 0.5)' },
                                showlegend: true,
                                legend: { orientation: 'h', y: -0.2 },
                                autosize: true
                              } as any}
                              useResizeHandler={true}
                              style={{ width: "100%", height: "100%", minHeight: "350px" }}
                            />
                          </CardContent>
                        </Card>
                      </div>
                    </div>
                  )}

                </motion.div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
