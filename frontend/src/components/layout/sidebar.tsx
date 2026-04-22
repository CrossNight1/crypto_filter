"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { 
  Database, 
  Activity, 
  Search, 
  GitMerge, 
  TrendingUp, 
  Settings,
  Hexagon
} from "lucide-react";
import { motion } from "framer-motion";

const routes = [
  { href: "/data", label: "Data Manager", icon: Database },
  { href: "/market-radar", label: "Market Radar", icon: Activity },
  { href: "/diagnostics", label: "Diagnostics", icon: Search },
  { href: "/pair-radar", label: "Pair Radar", icon: GitMerge },
  { href: "/multivariate", label: "Multivariate", icon: TrendingUp },
  { href: "/predictive", label: "Predictive AI", icon: Activity },
];

export function Sidebar({ className }: { className?: string }) {
  const pathname = usePathname();

  return (
    <aside className={cn("glass flex flex-col bg-card/80", className)}>
      <div className="h-16 flex items-center px-6 border-b border-border/50">
        <Hexagon className="h-6 w-6 text-primary mr-3" />
        <span className="font-bold text-lg tracking-wide uppercase bg-gradient-to-r from-primary to-purple-400 bg-clip-text text-transparent">
          Crypto Filter
        </span>
      </div>

      <nav className="flex-1 px-4 py-6 space-y-2">
        {routes.map((route) => {
          const isActive = pathname.startsWith(route.href);
          
          return (
            <Link
              key={route.href}
              href={route.href}
              className={cn(
                "group relative flex items-center px-3 py-3 text-sm font-medium rounded-md transition-colors",
                isActive 
                  ? "text-primary-foreground" 
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
              )}
            >
              {isActive && (
                <motion.div
                  layoutId="active-nav"
                  className="absolute inset-0 bg-primary/20 border border-primary/30 rounded-md"
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              )}
              <route.icon className={cn(
                "h-5 w-5 mr-3 shrink-0 transition-colors z-10",
                isActive ? "text-primary" : "text-muted-foreground group-hover:text-foreground"
              )} />
              <span className="z-10">{route.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-border/50">
        <button className="flex items-center w-full px-3 py-2 text-sm font-medium text-muted-foreground rounded-md hover:text-foreground hover:bg-secondary/50 transition-colors">
          <Settings className="h-5 w-5 mr-3 shrink-0" />
          Settings
        </button>
      </div>
    </aside>
  );
}
