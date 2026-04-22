"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { dataService, FileMetadata } from "@/lib/services/dataService";
import { motion, AnimatePresence } from "framer-motion";
import { Database, RefreshCw, Trash2, ShieldAlert, CheckCircle2, HardDrive } from "lucide-react";
import { cn, formatError } from "@/lib/utils";

const ALL_INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"];

export default function DataManagerPage() {
  const [universe, setUniverse] = useState<string[]>([]);
  const [metadata, setMetadata] = useState<FileMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  
  // Form states
  const [selectedMode, setSelectedMode] = useState<"universe" | "custom">("universe");
  const [selectedIntervals, setSelectedIntervals] = useState<string[]>(["1h"]);
  const [customSymbols, setCustomSymbols] = useState<string>("BTCUSDT,ETHUSDT");
  const [isFetching, setIsFetching] = useState(false);
  const [statusMsg, setStatusMsg] = useState<{type: "success" | "error" | "info", text: string} | null>(null);

  const loadData = async () => {
    try {
      setLoading(true);
      const [uni, meta] = await Promise.all([
        dataService.getUniverse(),
        dataService.getMetadata()
      ]);
      setUniverse(uni);
      setMetadata(meta);
    } catch (err) {
      console.error(err);
      setStatusMsg({ type: "error", text: "Failed to load data. Is the API running?" });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleFetch = async () => {
    try {
      setIsFetching(true);
      setStatusMsg({ type: "info", text: "Starting background fetch task..." });
      
      const req = {
        mode: selectedMode,
        intervals: selectedIntervals,
        symbols: selectedMode === "custom" ? customSymbols.split(",").map(s => s.trim().toUpperCase()) : undefined
      };
      
      const res = await dataService.fetchData(req);
      setStatusMsg({ type: "success", text: res.message });
      
      // Auto-refresh metadata after 5 seconds to show new files
      setTimeout(loadData, 5000);
    } catch (err: any) {
      setStatusMsg({ type: "error", text: formatError(err) });
    } finally {
      setIsFetching(false);
    }
  };

  const handleDelete = async (interval: string) => {
    if (!confirm(`Are you sure you want to delete ALL data for interval ${interval}?`)) return;
    try {
      await dataService.deleteCache(interval);
      setStatusMsg({ type: "success", text: `Deleted cache for ${interval}` });
      loadData();
    } catch (err: any) {
      setStatusMsg({ type: "error", text: formatError(err) });
    }
  };

  const totalSize = metadata.reduce((acc, m) => acc + (m.size_mb || 0), 0);

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="flex items-center justify-between mt-2 mb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-blue-400 to-primary bg-clip-text text-transparent">
            Data Manager
          </h1>
          <p className="text-muted-foreground mt-1">Manage local tick data caches and synchronize universe.</p>
        </div>
        <button 
          onClick={loadData}
          disabled={loading}
          className="p-2 bg-secondary hover:bg-secondary/80 rounded-full transition-colors"
        >
          <RefreshCw className={cn("h-5 w-5", loading && "animate-spin")} />
        </button>
      </div>

      <AnimatePresence>
        {statusMsg && (
          <motion.div 
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className={cn(
              "p-4 rounded-lg flex items-center shadow-lg border",
              statusMsg.type === "success" && "bg-emerald-500/10 border-emerald-500/50 text-emerald-400",
              statusMsg.type === "error" && "bg-rose-500/10 border-rose-500/50 text-rose-400",
              statusMsg.type === "info" && "bg-blue-500/10 border-blue-500/50 text-blue-400"
            )}
          >
            {statusMsg.type === "success" && <CheckCircle2 className="h-5 w-5 mr-3" />}
            {statusMsg.type === "error" && <ShieldAlert className="h-5 w-5 mr-3" />}
            {statusMsg.type === "info" && <RefreshCw className="h-5 w-5 mr-3 animate-spin" />}
            {statusMsg.text}
          </motion.div>
        )}
      </AnimatePresence>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 space-y-6">
          {/* Controls */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Database className="h-5 w-5 mr-2 text-primary" />
                Fetch Controls
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <div className="space-y-2">
                <label className="font-medium text-muted-foreground">Mode</label>
                <div className="flex space-x-2">
                  <button 
                    onClick={() => setSelectedMode("universe")}
                    className={cn("flex-1 py-1.5 rounded text-center transition-colors border", selectedMode === "universe" ? "bg-primary/20 border-primary text-primary-foreground" : "border-border hover:bg-secondary")}
                  >
                    Top Universe
                  </button>
                  <button 
                    onClick={() => setSelectedMode("custom")}
                    className={cn("flex-1 py-1.5 rounded text-center transition-colors border", selectedMode === "custom" ? "bg-primary/20 border-primary text-primary-foreground" : "border-border hover:bg-secondary")}
                  >
                    Custom List
                  </button>
                </div>
              </div>

              {selectedMode === "custom" && (
                <div className="space-y-2">
                  <label className="font-medium text-muted-foreground">Symbols (comma-separated)</label>
                  <input 
                    type="text" 
                    value={customSymbols}
                    onChange={e => setCustomSymbols(e.target.value)}
                    className="w-full bg-secondary/50 border border-border rounded-md px-3 py-2 text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                  />
                </div>
              )}

              <div className="space-y-2">
                <label className="font-medium text-muted-foreground">Intervals</label>
                <div className="flex flex-wrap gap-1">
                  {ALL_INTERVALS.map(inv => (
                    <button
                      key={inv}
                      onClick={() => {
                        if (selectedIntervals.includes(inv)) setSelectedIntervals(selectedIntervals.filter(i => i !== inv));
                        else setSelectedIntervals([...selectedIntervals, inv]);
                      }}
                      className={cn(
                        "px-2 py-1 text-xs rounded border transition-colors",
                        selectedIntervals.includes(inv) ? "bg-primary/30 border-primary text-blue-100" : "border-border bg-secondary/30 text-muted-foreground hover:bg-secondary"
                      )}
                    >
                      {inv}
                    </button>
                  ))}
                </div>
              </div>

              <button
                onClick={handleFetch}
                disabled={isFetching || selectedIntervals.length === 0}
                className="w-full mt-4 bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-2 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center shadow-[0_0_15px_rgba(59,130,246,0.5)]"
              >
                {isFetching ? <RefreshCw className="h-4 w-4 mr-2 animate-spin" /> : <Database className="h-4 w-4 mr-2" />}
                Sync Market Data
              </button>
            </CardContent>
          </Card>

          {/* Stats Summary */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <HardDrive className="h-5 w-5 mr-2 text-emerald-400" />
                Storage Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex justify-between items-center mb-4">
                <span className="text-muted-foreground">Total Cache Size</span>
                <span className="text-2xl font-mono text-emerald-400 font-bold">{totalSize.toFixed(2)} MB</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Cached Symbols</span>
                <span className="font-mono text-lg">{universe.length}</span>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-2">
          {/* Metadata Table */}
          <Card className="h-full flex flex-col">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle>File Metadata</CardTitle>
              <div className="text-sm text-muted-foreground">
                {metadata.length} files found
              </div>
            </CardHeader>
            <CardContent className="flex-1 overflow-hidden flex flex-col pt-4">
              <div className="overflow-y-auto pr-2" style={{ maxHeight: "600px" }}>
                <table className="w-full text-sm text-left">
                  <thead className="text-xs text-muted-foreground uppercase bg-secondary/50 sticky top-0 backdrop-blur-md">
                    <tr>
                      <th className="px-4 py-3 font-semibold rounded-tl-md">Ticker</th>
                      <th className="px-4 py-3 font-semibold">Interval</th>
                      <th className="px-4 py-3 font-semibold text-right">Size (MB)</th>
                      <th className="px-4 py-3 font-semibold text-right">Last Modified</th>
                      <th className="px-4 py-3 font-semibold text-center rounded-tr-md">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/50">
                    {loading ? (
                      <tr>
                        <td colSpan={5} className="text-center py-8 text-muted-foreground animate-pulse">Loading data...</td>
                      </tr>
                    ) : metadata.length === 0 ? (
                      <tr>
                        <td colSpan={5} className="text-center py-8 text-muted-foreground">No cache files found.</td>
                      </tr>
                    ) : (
                      metadata.map((file, idx) => (
                        <motion.tr 
                          key={`${file.ticker}-${file.interval}-${idx}`}
                          initial={{ opacity: 0, y: 5 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: idx * 0.02 }}
                          className="hover:bg-secondary/30 transition-colors group"
                        >
                          <td className="px-4 py-3 font-mono text-blue-300 font-medium">{file.ticker}</td>
                          <td className="px-4 py-3">
                            <span className="px-2 py-0.5 bg-secondary text-xs rounded border border-border/50">{file.interval}</span>
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-muted-foreground">{(file.size_mb || 0).toFixed(2)}</td>
                          <td className="px-4 py-3 text-right text-muted-foreground text-xs">{file.last_modified ? new Date(file.last_modified * 1000).toLocaleString() : "N/A"}</td>
                          <td className="px-4 py-3 text-center">
                            <button 
                              onClick={() => handleDelete(file.interval)}
                              className="p-1.5 text-rose-400 hover:bg-rose-500/20 rounded-md transition-colors opacity-0 group-hover:opacity-100"
                              title={`Delete all ${file.interval} data`}
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          </td>
                        </motion.tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
