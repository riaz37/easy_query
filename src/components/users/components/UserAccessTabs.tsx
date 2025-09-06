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
      <TabsList className={`grid w-full grid-cols-2 transition-all duration-200 ${
        isDark 
          ? 'bg-slate-800/70 border border-slate-600' 
          : 'bg-gray-100 border border-gray-200'
      }`}>
        <TabsTrigger
          value="mssql"
          className={`flex items-center gap-2 transition-all duration-200 ${
            isDark 
              ? 'text-gray-300 data-[state=active]:bg-blue-600 data-[state=active]:text-white data-[state=active]:shadow-lg data-[state=active]:shadow-blue-500/25 hover:bg-slate-700 hover:text-white' 
              : 'text-gray-700 data-[state=active]:bg-blue-500 data-[state=active]:text-white data-[state=active]:shadow-lg data-[state=active]:shadow-blue-500/25 hover:bg-gray-200 hover:text-gray-900'
          }`}
        >
          <Database className="h-4 w-4" />
          MSSQL Database Access
        </TabsTrigger>
        <TabsTrigger
          value="vector"
          className={`flex items-center gap-2 transition-all duration-200 ${
            isDark 
              ? 'text-gray-300 data-[state=active]:bg-purple-600 data-[state=active]:text-white data-[state=active]:shadow-lg data-[state=active]:shadow-purple-500/25 hover:bg-slate-700 hover:text-white' 
              : 'text-gray-700 data-[state=active]:bg-purple-500 data-[state=active]:text-white data-[state=active]:shadow-lg data-[state=active]:shadow-purple-500/25 hover:bg-gray-200 hover:text-gray-900'
          }`}
        >
          <Brain className="h-4 w-4" />
          Vector Database Access
        </TabsTrigger>
      </TabsList>

      {children}
    </Tabs>
  );
}
