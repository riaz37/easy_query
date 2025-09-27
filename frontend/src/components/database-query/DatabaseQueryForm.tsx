"use client";

import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Play, Database, AlertCircle, Sparkles, Brain, Zap, Clock } from "lucide-react";
import { ButtonLoader, ProgressLoader, InlineLoader } from "@/components/ui/loading";

interface DatabaseQueryFormProps {
  onSubmit: (query: string) => void;
  loading: boolean;
  hasDatabase: boolean;
  currentQuery?: string;
}

export function DatabaseQueryForm({ onSubmit, loading, hasDatabase, currentQuery }: DatabaseQueryFormProps) {
  const [query, setQuery] = useState("");
  const [localLoading, setLocalLoading] = useState(false);
  const [processingTime, setProcessingTime] = useState(0);

  // Update local query when currentQuery changes
  useEffect(() => {
    if (currentQuery) {
      setQuery(currentQuery);
    }
  }, [currentQuery]);

  // Handle loading state changes
  useEffect(() => {
    if (loading) {
      setLocalLoading(true);
      setProcessingTime(0);
      
      const interval = setInterval(() => {
        setProcessingTime(prev => prev + 1);
      }, 1000);

      return () => clearInterval(interval);
    } else {
      setLocalLoading(false);
      setProcessingTime(0);
    }
  }, [loading]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && hasDatabase) {
      onSubmit(query.trim());
    }
  };


  const handleClear = () => {
    setQuery("");
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="relative">
        <Textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask your question in natural language... (e.g., 'Show me all users from last month')"
          className="w-full h-48 p-4 pr-32 bg-slate-800/50 text-white placeholder-slate-400 focus:outline-none resize-none border-0"
          style={{
            background:
              "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
            borderRadius: "16px",
            outline: "none",
            border: "none",
          }}
          disabled={!hasDatabase || localLoading}
          data-element="query-input"
        />

        {/* Action Buttons */}
        <div className="absolute bottom-3 left-7 right-7 flex justify-start">
          <div className="flex gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={handleClear}
              className="text-xs cursor-pointer"
              style={{
                background:
                  "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
                border:
                  "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
                color: "white",
                borderRadius: "99px",
                height: "48px",
                minWidth: "64px",
              }}
            >
              Clear
            </Button>
            <Button
              type="submit"
              disabled={!query.trim() || !hasDatabase || localLoading}
              className="text-xs cursor-pointer"
              style={{
                background:
                  "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
                border:
                  "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
                color: "var(--p-main, rgba(19, 245, 132, 1))",
                borderRadius: "99px",
                height: "48px",
                minWidth: "64px",
              }}
            >
              {localLoading ? (
                <div className="flex items-center gap-2">
                  <Spinner size="sm" variant="accent-green" />
                  <span>Processing...</span>
                </div>
              ) : (
                "Ask"
              )}
            </Button>
          </div>
        </div>
      </div>

      {!hasDatabase && (
        <div className="flex items-center gap-2 text-yellow-400 text-sm">
          <AlertCircle className="w-4 h-4" />
          Please select a database first
        </div>
      )}

      {/* Loading Progress Indicator */}
      {localLoading && (
        <div className="space-y-3 p-4 bg-emerald-900/20 border border-emerald-400/30 rounded-lg">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2 text-emerald-400">
              <InlineLoader size="sm" variant="accent-green">
                AI Processing Your Query
              </InlineLoader>
            </div>
            <div className="flex items-center gap-2 text-gray-400">
              <Clock className="w-4 h-4" />
              <span>{formatTime(processingTime)}</span>
            </div>
          </div>
          
          <ProgressLoader 
            progress={Math.min((processingTime / 30) * 100, 90)} 
            size="sm"
            variant="accent-green"
            showPercentage={false}
          />
          
          <div className="grid grid-cols-2 gap-2 text-xs text-gray-400">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span>Analyzing question</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
              <span>Connecting to DB</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
              <span>Processing query</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
              <span>Generating results</span>
            </div>
          </div>
        </div>
      )}


      {/* Processing Status */}
      {localLoading && (
        <div className="p-3 bg-green-900/20 border border-green-400/30 rounded-lg">
          <div className="flex items-center gap-2 text-green-400 text-sm">
            <Zap className="w-4 h-4 animate-pulse" />
            <span>Your query is being processed by AI. This may take a few moments...</span>
          </div>
        </div>
      )}
    </form>
  );
} 