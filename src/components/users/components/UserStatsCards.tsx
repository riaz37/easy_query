"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { UserStatsCardsProps } from "../types";

export function UserStatsCards({ stats, isDark }: UserStatsCardsProps) {
  const cards = [
    {
      title: "Total Users",
      value: stats.totalUsers,
      color: "emerald",
      gradient: isDark 
        ? "from-slate-800/80 to-slate-700/80 border-slate-600 hover:border-slate-500" 
        : "from-white to-gray-50 border-gray-200 shadow-sm hover:shadow-md",
      textColor: isDark ? "text-gray-300" : "text-gray-600",
      valueColor: isDark ? "text-emerald-400" : "text-emerald-600"
    },
    {
      title: "MSSQL Access",
      value: stats.mssqlUsers,
      color: "blue",
      gradient: isDark 
        ? "from-blue-900/30 to-blue-800/20 border-blue-600/50 hover:border-blue-500" 
        : "from-blue-50 to-blue-100 border-blue-200 shadow-sm hover:shadow-md",
      textColor: isDark ? "text-blue-300" : "text-blue-600",
      valueColor: "text-blue-500"
    },
    {
      title: "Vector DB Access",
      value: stats.vectorDBUsers,
      color: "purple",
      gradient: isDark 
        ? "from-purple-900/30 to-purple-800/20 border-purple-600/50 hover:border-purple-500" 
        : "from-purple-50 to-purple-100 border-purple-200 shadow-sm hover:shadow-md",
      textColor: isDark ? "text-purple-300" : "text-purple-600",
      valueColor: "text-purple-500"
    },
    {
      title: "Full Access",
      value: stats.fullAccessUsers,
      color: "emerald",
      gradient: isDark 
        ? "from-emerald-900/30 to-emerald-800/20 border-emerald-600/50 hover:border-emerald-500" 
        : "from-emerald-50 to-emerald-100 border-emerald-200 shadow-sm hover:shadow-md",
      textColor: isDark ? "text-emerald-300" : "text-emerald-600",
      valueColor: "text-emerald-500"
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
            <div className="text-2xl font-bold text-emerald-400">
              {card.value}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
