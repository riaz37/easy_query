"use client";

import React from "react";
import { Play, FileText, BarChart3, History } from "lucide-react";

interface DatabaseQueryStatsCardsProps {
  isDark?: boolean;
}

export function DatabaseQueryStatsCards({ isDark = true }: DatabaseQueryStatsCardsProps) {
  const cards = [
    {
      title: "Quick Queries",
      description: "Instant results for simple questions",
      icon: Play,
      color: "emerald",
      valueColor: isDark ? "text-emerald-400" : "text-emerald-600"
    },
    {
      title: "AI Reports",
      description: "Comprehensive analysis & insights",
      icon: FileText,
      color: "emerald",
      valueColor: isDark ? "text-emerald-400" : "text-emerald-600"
    },
    {
      title: "Smart Results",
      description: "AI-powered data insights",
      icon: BarChart3,
      color: "green",
      valueColor: isDark ? "text-green-400" : "text-green-600"
    },
    {
      title: "Query History",
      description: "Track and reuse past queries",
      icon: History,
      color: "emerald",
      valueColor: isDark ? "text-emerald-400" : "text-emerald-600"
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
      {cards.map((card) => {
        const IconComponent = card.icon;
        return (
          <div key={card.title} className="card-enhanced">
            <div className="card-content-enhanced">
              <div className="card-header-enhanced">
                <div className="card-title-enhanced text-sm font-medium">
                  {card.title}
                </div>
              </div>
              <div className="mt-4">
                <div className="flex items-center gap-3 mb-3">
                  <div className={`w-8 h-8 bg-${card.color}-500/20 rounded-lg flex items-center justify-center`}>
                    <IconComponent className={`w-4 h-4 ${card.valueColor}`} />
                  </div>
                  <div>
                    <h3 className="text-white font-medium">
                      {card.title}
                    </h3>
                    <p className="text-gray-400 text-sm">
                      {card.description}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
