"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
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
      textColor: isDark ? "text-gray-300" : "text-gray-700",
      valueColor: isDark ? "text-emerald-400" : "text-emerald-700"
    },
    {
      title: "MSSQL Access",
      value: stats.mssqlUsers,
      color: "emerald",
      gradient: isDark 
        ? "from-emerald-900/30 to-emerald-800/20 border-emerald-600/50 hover:border-emerald-500" 
        : "from-emerald-50 to-emerald-100 border-emerald-200 shadow-sm hover:shadow-md",
      textColor: isDark ? "text-emerald-300" : "text-emerald-700",
      valueColor: isDark ? "text-emerald-400" : "text-emerald-700"
    },
    {
      title: "Vector DB Access",
      value: stats.vectorDBUsers,
      color: "emerald",
      gradient: isDark 
        ? "from-emerald-900/30 to-emerald-800/20 border-emerald-600/50 hover:border-emerald-500" 
        : "from-emerald-50 to-emerald-100 border-emerald-200 shadow-sm hover:shadow-md",
      textColor: isDark ? "text-emerald-300" : "text-emerald-700",
      valueColor: isDark ? "text-emerald-400" : "text-emerald-700"
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
            <div className={cn(
              "text-2xl font-bold",
              card.valueColor
            )}>
              {card.value}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
