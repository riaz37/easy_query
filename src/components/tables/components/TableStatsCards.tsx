"use client";

import React from "react";
import { UserCurrentDBTableData } from "@/types/api";

interface TableStatsCardsProps {
  tableData: UserCurrentDBTableData | null;
  isDark?: boolean;
}

export function TableStatsCards({ tableData, isDark = true }: TableStatsCardsProps) {
  const stats = {
    totalTables: tableData?.table_info?.metadata?.total_tables || 0,
    processedTables: tableData?.table_info?.metadata?.processed_tables || 0,
    failedTables: tableData?.table_info?.metadata?.failed_tables || 0,
    unmatchedRules: tableData?.table_info?.unmatched_business_rules?.length || 0,
  };

  const cards = [
    {
      title: "Total Tables",
      value: stats.totalTables,
      color: "emerald",
      gradient: isDark 
        ? "from-slate-800/80 to-slate-700/80 border-slate-600 hover:border-slate-500" 
        : "from-white to-gray-50 border-gray-200 shadow-sm hover:shadow-md",
      textColor: isDark ? "text-gray-300" : "text-gray-600",
      valueColor: isDark ? "text-emerald-400" : "text-emerald-600"
    },
    {
      title: "Processed Tables",
      value: stats.processedTables,
      color: "blue",
      gradient: isDark 
        ? "from-blue-900/30 to-blue-800/20 border-blue-600/50 hover:border-blue-500" 
        : "from-blue-50 to-blue-100 border-blue-200 shadow-sm hover:shadow-md",
      textColor: isDark ? "text-blue-300" : "text-blue-600",
      valueColor: "text-blue-500"
    },
    {
      title: "Failed Tables",
      value: stats.failedTables,
      color: "red",
      gradient: isDark 
        ? "from-red-900/30 to-red-800/20 border-red-600/50 hover:border-red-500" 
        : "from-red-50 to-red-100 border-red-200 shadow-sm hover:shadow-md",
      textColor: isDark ? "text-red-300" : "text-red-600",
      valueColor: "text-red-500"
    },
    {
      title: "Unmatched Rules",
      value: stats.unmatchedRules,
      color: "yellow",
      gradient: isDark 
        ? "from-yellow-900/30 to-yellow-800/20 border-yellow-600/50 hover:border-yellow-500" 
        : "from-yellow-50 to-yellow-100 border-yellow-200 shadow-sm hover:shadow-md",
      textColor: isDark ? "text-yellow-300" : "text-yellow-600",
      valueColor: "text-yellow-500"
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
      {cards.map((card) => (
        <div key={card.title} className="card-enhanced">
          <div className="card-content-enhanced">
            <div className="card-header-enhanced">
              <div className="card-title-enhanced text-sm font-medium">
                {card.title}
              </div>
            </div>
            <div className={`text-2xl font-bold ${card.valueColor}`}>
              {card.value}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
