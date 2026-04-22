"use client";

import { motion } from "framer-motion";
import { Activity, Database, TrendingUp, Search, GitMerge } from "lucide-react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function Home() {
  const modules = [
    { name: "Data Manager", icon: Database, desc: "Sync and manage market data", href: "/data", color: "text-blue-400" },
    { name: "Market Radar", icon: Activity, desc: "Real-time market snapshot", href: "/market-radar", color: "text-emerald-400" },
    { name: "Diagnostics", icon: Search, desc: "In-depth ticker analysis", href: "/diagnostics", color: "text-amber-400" },
    { name: "Pair Radar", icon: GitMerge, desc: "Statistical arbitrage analysis", href: "/pair-radar", color: "text-purple-400" },
    { name: "Multivariate", icon: TrendingUp, desc: "Decomposition & Correlation", href: "/multivariate", color: "text-rose-400" },
    { name: "Predictive AI", icon: Activity, desc: "Machine learning return predictions", href: "/predictive", color: "text-indigo-400" },
  ];

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-2 mt-8"
      >
        <h1 className="text-4xl font-bold tracking-tight">System Overview</h1>
        <p className="text-muted-foreground text-lg">Select a module to begin quantitative analysis.</p>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {modules.map((m, i) => (
          <motion.div
            key={m.name}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.1 }}
          >
            <Link href={m.href}>
              <Card className="hover:border-primary/50 transition-colors cursor-pointer group h-full">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-3 text-xl">
                    <m.icon className={`h-8 w-8 ${m.color} group-hover:scale-110 transition-transform`} />
                    <span>{m.name}</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">{m.desc}</p>
                </CardContent>
              </Card>
            </Link>
          </motion.div>
        ))}
      </div>
      
      {/* Decorative gradient orb */}
      <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-primary/10 rounded-full blur-[120px] pointer-events-none -z-10" />
    </div>
  );
}
