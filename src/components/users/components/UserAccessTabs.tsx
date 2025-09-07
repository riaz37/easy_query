"use client";

import React from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Database, Brain } from "lucide-react";
import { UserAccessTabsProps } from "../types";

export function UserAccessTabs({
  activeTab,
  onTabChange,
  isDark,
  children
}: UserAccessTabsProps) {
  return (
    <Tabs value={activeTab || "mssql"} onValueChange={onTabChange} className="w-full">
      <TabsList className="grid w-full grid-cols-2 transition-all duration-200 bg-slate-800/70 border border-emerald-500/30 backdrop-filter backdrop-blur-20">
        <TabsTrigger
          value="mssql"
          className="flex items-center gap-2 transition-all duration-200 text-gray-300 data-[state=active]:bg-emerald-600 data-[state=active]:text-white data-[state=active]:shadow-lg data-[state=active]:shadow-emerald-500/25 hover:bg-slate-700 hover:text-white"
        >
          <Database className="h-4 w-4" />
          MSSQL Database Access
        </TabsTrigger>
        <TabsTrigger
          value="vector"
          className="flex items-center gap-2 transition-all duration-200 text-gray-300 data-[state=active]:bg-teal-600 data-[state=active]:text-white data-[state=active]:shadow-lg data-[state=active]:shadow-teal-500/25 hover:bg-slate-700 hover:text-white"
        >
          <Brain className="h-4 w-4" />
          Vector Database Access
        </TabsTrigger>
      </TabsList>

      {children}
    </Tabs>
  );
}
