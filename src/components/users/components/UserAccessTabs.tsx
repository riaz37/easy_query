"use client";

import React from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Database, Brain } from "lucide-react";
import { useTheme } from "@/store/theme-store";
import { cn } from "@/lib/utils";
import { UserAccessTabsProps } from "../types";

export function UserAccessTabs({
  activeTab,
  onTabChange,
  isDark,
  children
}: UserAccessTabsProps) {
  const theme = useTheme();
  
  return (
    <Tabs value={activeTab || "mssql"} onValueChange={onTabChange} className="w-full">
      <TabsList className={cn(
        "grid w-full grid-cols-2 transition-all duration-200 backdrop-filter backdrop-blur-20",
        theme === "dark" 
          ? "bg-slate-800/70 border border-emerald-500/30" 
          : "bg-white/95 border border-emerald-500/30 shadow-sm"
      )}>
        <TabsTrigger
          value="mssql"
          className={cn(
            "flex items-center gap-2 transition-all duration-200",
            theme === "dark"
              ? "text-gray-300 data-[state=active]:bg-emerald-600 data-[state=active]:text-white data-[state=active]:shadow-lg data-[state=active]:shadow-emerald-500/25 hover:bg-slate-700 hover:text-white"
              : "text-gray-600 data-[state=active]:bg-emerald-500 data-[state=active]:text-white data-[state=active]:shadow-md hover:bg-emerald-50 hover:text-emerald-600"
          )}
        >
          <Database className="h-4 w-4" />
          MSSQL Database Access
        </TabsTrigger>
        <TabsTrigger
          value="vector"
          className={cn(
            "flex items-center gap-2 transition-all duration-200",
            theme === "dark"
              ? "text-gray-300 data-[state=active]:bg-emerald-600 data-[state=active]:text-white data-[state=active]:shadow-lg data-[state=active]:shadow-emerald-500/25 hover:bg-slate-700 hover:text-white"
              : "text-gray-600 data-[state=active]:bg-emerald-500 data-[state=active]:text-white data-[state=active]:shadow-md hover:bg-emerald-50 hover:text-emerald-600"
          )}
        >
          <Brain className="h-4 w-4" />
          Vector Database Access
        </TabsTrigger>
      </TabsList>

      {children}
    </Tabs>
  );
}
