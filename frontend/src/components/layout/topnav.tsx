"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import { 
  Database, 
  Activity, 
  Search, 
  GitMerge, 
  TrendingUp, 
  Hexagon,
  Send,
  X,
  ChevronDown
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { dataService } from "@/lib/services/dataService";

const routes = [
  { href: "/", label: "Home", exact: true },
  { href: "/data", label: "Data Manager" },
  { href: "/diagnostics", label: "Diagnostics" },
  { href: "/market-radar", label: "Market Radar" },
  { href: "/pair-radar", label: "Pair Radar" },
  { href: "/multivariate", label: "Multivariate" },
  { href: "/predictive", label: "Predictive AI" },
];

export function TopNav() {
  const pathname = usePathname();
  const [universe, setUniverse] = useState<string[]>([]);
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Load symbol universe on mount
  useEffect(() => {
    async function fetchUniverse() {
      try {
        const symbols = await dataService.getUniverse();
        setUniverse(symbols || []);
      } catch (err) {
        console.error("Failed to load symbol universe in top nav", err);
      }
    }
    fetchUniverse();
  }, []);

  // Close dropdown on click outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelectSymbol = (sym: string) => {
    if (!selectedSymbols.includes(sym)) {
      setSelectedSymbols([...selectedSymbols, sym]);
    } else {
      setSelectedSymbols(selectedSymbols.filter(s => s !== sym));
    }
    setSearchQuery("");
  };

  const handleRemoveSymbol = (sym: string) => {
    setSelectedSymbols(selectedSymbols.filter(s => s !== sym));
  };

  const handleLaunch = () => {
    if (selectedSymbols.length === 0) return;

    selectedSymbols.forEach((sym) => {
      const url = `https://www.tradingview.com/chart/?symbol=BINANCE:${sym}.P&interval=60`;
      window.open(url, "_blank");
    });

    setSelectedSymbols([]);
    setIsOpen(false);
  };

  const filteredSymbols = universe.filter(sym => 
    sym.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <header className="fixed top-0 left-0 right-0 h-16 border-b border-border bg-background/80 backdrop-blur-md z-50 flex items-center justify-between px-6">
      {/* Brand logo & title */}
      <div className="flex items-center gap-3 shrink-0">
        <Link href="/" className="flex items-center gap-3 group">
          <Hexagon className="h-6 w-6 text-primary group-hover:scale-110 transition-transform" />
          <span className="font-bold text-lg tracking-wide uppercase bg-gradient-to-r from-primary to-purple-400 bg-clip-text text-transparent">
            Crypto Filter
          </span>
        </Link>
      </div>

      {/* Horizontal Nav Links */}
      <nav className="hidden xl:flex items-center h-full gap-1">
        {routes.map((route) => {
          const isActive = route.exact 
            ? pathname === route.href
            : pathname.startsWith(route.href);

          return (
            <Link
              key={route.href}
              href={route.href}
              className={cn(
                "relative flex items-center px-4 py-2 text-sm font-medium rounded-md transition-colors h-10",
                isActive 
                  ? "text-primary font-semibold" 
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary/40"
              )}
            >
              {isActive && (
                <motion.div
                  layoutId="active-topnav-tab"
                  className="absolute inset-0 bg-primary/10 border-b-2 border-primary rounded-t-sm"
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              )}
              <span className="z-10">{route.label}</span>
            </Link>
          );
        })}
      </nav>

      {/* Compact Nav for smaller viewports */}
      <nav className="flex xl:hidden items-center h-full gap-1 max-w-[50%] overflow-x-auto no-scrollbar">
        {routes.map((route) => {
          const isActive = route.exact 
            ? pathname === route.href
            : pathname.startsWith(route.href);

          return (
            <Link
              key={route.href}
              href={route.href}
              className={cn(
                "relative flex items-center px-2 py-1 text-xs font-medium rounded-md transition-colors shrink-0",
                isActive 
                  ? "text-primary font-semibold" 
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <span className="z-10">{route.label}</span>
            </Link>
          );
        })}
      </nav>

      {/* Launch Assets Widget */}
      <div className="flex items-center gap-2 relative" ref={dropdownRef}>
        {/* Combobox Multi-Select Container */}
        <div className="w-64 relative bg-secondary/50 border border-border rounded-md px-3 py-1.5 flex flex-wrap gap-1.5 items-center cursor-pointer min-h-[38px] hover:border-primary/50 transition-colors"
             onClick={() => setIsOpen(!isOpen)}>
          {selectedSymbols.length === 0 ? (
            <span className="text-muted-foreground text-sm select-none">Launch Assets</span>
          ) : (
            selectedSymbols.map((sym) => (
              <span 
                key={sym} 
                className="bg-primary/20 border border-primary/30 text-primary-foreground text-xs font-semibold px-2 py-0.5 rounded flex items-center gap-1 z-10"
                onClick={(e) => {
                  e.stopPropagation();
                  handleRemoveSymbol(sym);
                }}
              >
                {sym}
                <X className="h-3.5 w-3.5 hover:text-destructive cursor-pointer" />
              </span>
            ))
          )}
          
          <div className="ml-auto shrink-0 text-muted-foreground">
            <ChevronDown className="h-4 w-4" />
          </div>
        </div>

        {/* Action Button */}
        <button 
          onClick={handleLaunch}
          disabled={selectedSymbols.length === 0}
          className={cn(
            "h-[38px] w-[38px] flex items-center justify-center rounded-md border text-sm font-semibold transition-all shrink-0",
            selectedSymbols.length > 0
              ? "bg-primary border-primary text-primary-foreground hover:bg-primary/80 hover:scale-105 cursor-pointer shadow-md shadow-primary/20"
              : "bg-secondary/30 border-border text-muted-foreground cursor-not-allowed"
          )}
          title="Launch Selected Assets in TradingView"
        >
          <Send className="h-4 w-4" />
        </button>

        {/* Dropdown Options */}
        <AnimatePresence>
          {isOpen && (
            <motion.div 
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 5 }}
              transition={{ duration: 0.15 }}
              className="absolute right-0 top-12 w-64 bg-card border border-border rounded-md shadow-2xl overflow-hidden z-50 flex flex-col max-h-72"
            >
              {/* Dropdown Search Input */}
              <div className="p-2 border-b border-border flex items-center gap-2 bg-secondary/20">
                <Search className="h-4 w-4 text-muted-foreground shrink-0" />
                <input 
                  type="text"
                  placeholder="Filter symbols..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="bg-transparent border-0 outline-none w-full text-sm py-1 placeholder-muted-foreground text-foreground focus:ring-0"
                  onClick={(e) => e.stopPropagation()}
                  autoFocus
                />
              </div>

              {/* Symbol Options List */}
              <div className="overflow-y-auto flex-1 py-1 max-h-56">
                {filteredSymbols.length === 0 ? (
                  <div className="px-3 py-2 text-xs text-muted-foreground text-center">
                    No symbols found
                  </div>
                ) : (
                  filteredSymbols.map((sym) => {
                    const isSelected = selectedSymbols.includes(sym);
                    return (
                      <div
                        key={sym}
                        onClick={(e) => {
                          e.stopPropagation();
                          handleSelectSymbol(sym);
                        }}
                        className={cn(
                          "px-3 py-1.5 text-sm cursor-pointer transition-colors flex items-center justify-between",
                          isSelected 
                            ? "bg-primary/10 text-primary font-semibold" 
                            : "text-foreground hover:bg-secondary/60"
                        )}
                      >
                        <span>{sym}</span>
                        {isSelected && (
                          <span className="text-primary text-xs font-bold">✓</span>
                        )}
                      </div>
                    );
                  })
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </header>
  );
}
