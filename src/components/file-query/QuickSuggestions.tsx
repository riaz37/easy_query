"use client";

import React from "react";
import Image from "next/image";
import { BarChart3, FileText, Search, Database } from "lucide-react";

interface Suggestion {
  text: string;
  query: string;
  icon: React.ReactNode;
}

interface QuickSuggestionsProps {
  className?: string;
  onQuerySelect?: (query: string) => void;
  suggestions?: Suggestion[];
  title?: string;
}

export function QuickSuggestions({ 
  className = "", 
  onQuerySelect, 
  suggestions: customSuggestions,
  title = "Quick suggestion"
}: QuickSuggestionsProps) {
  const defaultSuggestions = [
    { 
      text: "What are the main topics covered in the uploaded documents?", 
      query: "What are the main topics covered in the uploaded documents?",
      icon: <FileText className="h-4 w-4 text-green-400" /> 
    },
    { 
      text: "Summarize the key findings from the financial reports", 
      query: "Summarize the key findings from the financial reports",
      icon: <BarChart3 className="h-4 w-4 text-green-400" /> 
    },
    { 
      text: "Find all mentions of budget allocations and spending", 
      query: "Find all mentions of budget allocations and spending",
      icon: <Search className="h-4 w-4 text-green-400" /> 
    },
    { 
      text: "Extract data from tables and structured content", 
      query: "Extract data from tables and structured content",
      icon: <Database className="h-4 w-4 text-green-400" /> 
    },
  ];

  const suggestions = customSuggestions || defaultSuggestions;

  const handleSuggestionClick = (query: string) => {
    if (onQuerySelect) {
      onQuerySelect(query);
    }
  };

  return (
    <div className={`space-y-4 ${className}`} style={{ height: "320px" }}>
      <h3 className="text-xl font-semibold text-white mb-6">
        {title}
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {suggestions.map((suggestion, index) => (
          <div
            key={index}
            className="p-4 cursor-pointer hover:scale-105 transition-transform duration-200"
            onClick={() => handleSuggestionClick(suggestion.query)}
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
               {/* Quicksuggest SVG at bottom of each card */}
               <div className="flex justify-start mt-2">
                <Image
                  src="/querydemo.svg"
                  alt="Quick Suggest"
                  width={100}
                  height={100}
                  className="opacity-60"
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
