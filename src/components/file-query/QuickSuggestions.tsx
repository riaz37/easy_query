"use client";

import React from "react";
import { BarChart3 } from "lucide-react";

interface QuickSuggestionsProps {
  className?: string;
}

export function QuickSuggestions({ className = "" }: QuickSuggestionsProps) {
  const suggestions = [
    { text: "Use time references: 'last week', 'this month', 'yesterday'", icon: <BarChart3 className="h-4 w-4 text-green-400" /> },
    { text: "Use time references: 'last week', 'this month', 'yesterday'", icon: <BarChart3 className="h-4 w-4 text-green-400" /> },
    { text: "Use time references: 'last week', 'this month', 'yesterday'", icon: <BarChart3 className="h-4 w-4 text-green-400" /> },
    { text: "Use time references: 'last week', 'this month', 'yesterday'", icon: <BarChart3 className="h-4 w-4 text-green-400" /> },
  ];

  return (
    <div className={`space-y-4 ${className}`}>
      <h3 className="text-xl font-semibold text-white mb-6">
        Quick suggestion
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {suggestions.map((suggestion, index) => (
          <div
            key={index}
            className="p-4"
            style={{
              background:
                "linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)), linear-gradient(246.02deg, rgba(19, 245, 132, 0) 91.9%, rgba(19, 245, 132, 0.2) 114.38%), linear-gradient(59.16deg, rgba(19, 245, 132, 0) 71.78%, rgba(19, 245, 132, 0.2) 124.92%)",
              border: "1.5px solid",
              borderImageSource:
                "linear-gradient(158.39deg, rgba(255, 255, 255, 0.06) 14.19%, rgba(255, 255, 255, 1.5e-05) 50.59%, rgba(255, 255, 255, 1.5e-05) 68.79%, rgba(255, 255, 255, 0.015) 105.18%)",
              borderRadius: "30px",
              backdropFilter: "blur(30px)",
            }}
          >
            <div className="space-y-2">
              <p className="text-sm text-slate-400">
                {suggestion.text}
              </p>
              <div className="flex justify-center">
                <div className="w-6 h-6 bg-green-400 flex items-center justify-center rounded">
                  {suggestion.icon}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
