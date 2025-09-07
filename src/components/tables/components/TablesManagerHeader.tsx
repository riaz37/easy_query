"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Database, RefreshCw, Settings } from "lucide-react";

interface TablesManagerHeaderProps {
  onRefreshTables: () => void;
  onReloadDatabase: () => void;
  onSetDatabase: () => void;
  isLoading: boolean;
  isGeneratingTables: boolean;
  isSettingDB: boolean;
  isDark?: boolean;
}

export function TablesManagerHeader({
  onRefreshTables,
  onReloadDatabase,
  onSetDatabase,
  isLoading,
  isGeneratingTables,
  isSettingDB,
  isDark = true,
}: TablesManagerHeaderProps) {
  return (
    <div className="mb-8">
      <div className="flex items-center justify-center mb-6">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-white flex items-center justify-center gap-3">
            <div className="relative">
              <Database className="h-8 w-8 text-emerald-400" />
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-emerald-400 rounded-full animate-pulse" />
            </div>
            Database Tables Management
          </h1>
          <p className="text-gray-300 mt-2">
            Manage table relationships, import data, and visualize database
            structure
          </p>
        </div>
      </div>
    </div>
  );
}
