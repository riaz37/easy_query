"use client";

import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
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
  const [queryHistory, setQueryHistory] = useState<string[]>([
    "Show me all users from last month",
    "How many orders were placed today?",
    "What are the top 10 products by sales?",
    "Find customers who haven't ordered in 30 days",
    "Show me the revenue breakdown by month",
    "List all employees in the sales department",
  ]);
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

  const handleQuerySelect = (selectedQuery: string) => {
    setQuery(selectedQuery);
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
    <div className="card-enhanced">
      <div className="card-content-enhanced">
        <div className="card-header-enhanced">
          <div className="card-title-enhanced flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-emerald-400" />
            Natural Language Query
          </div>
        </div>
        <div className="mt-4 space-y-4">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask your question in natural language... (e.g., 'Show me all users from last month')"
              className="min-h-[120px] bg-gray-800/50 border-emerald-400/30 text-white placeholder:text-gray-400 resize-none"
              disabled={!hasDatabase || localLoading}
              data-element="query-input"
            />
            {!hasDatabase && (
              <div className="flex items-center gap-2 text-yellow-400 text-sm">
                <AlertCircle className="w-4 h-4" />
                Please select a database first
              </div>
            )}
          </div>

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

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <ButtonLoader
                type="submit"
                disabled={!query.trim() || !hasDatabase || localLoading}
                loading={localLoading}
                text="Processing..."
                size="md"
                variant="accent-green"
                className="min-w-[140px]"
                data-element="query-submit"
              >
                <Play className="w-4 h-4 mr-2" />
                Ask Question
              </ButtonLoader>
              
              {query && !localLoading && (
                <Button
                  type="button"
                  variant="outline"
                  onClick={handleClear}
                  className="border-gray-600 text-gray-300 hover:bg-gray-700"
                >
                  Clear
                </Button>
              )}
            </div>

            <div className="text-sm text-gray-400">
              {query.length} characters
            </div>
          </div>
        </form>

        {/* Quick Query Templates */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400 font-medium">Example Questions:</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {queryHistory.map((template, index) => (
              <Badge
                key={index}
                variant="outline"
                className="cursor-pointer border-emerald-400/30 text-emerald-400 hover:bg-emerald-400/10 hover:border-emerald-400/50 transition-all duration-200"
                onClick={() => handleQuerySelect(template)}
                disabled={localLoading}
              >
                {template.length > 40 ? template.substring(0, 40) + "..." : template}
              </Badge>
            ))}
          </div>
        </div>

        {/* Query Tips */}
        <div className="p-3 bg-emerald-900/20 border border-emerald-400/30 rounded-lg">
          <div className="text-sm text-emerald-300">
            <strong>💡 Natural Language Tips:</strong>
            <ul className="mt-2 space-y-1 text-emerald-200">
              <li>• Ask questions like you're talking to a person</li>
              <li>• Use time references: "last week", "this month", "yesterday"</li>
              <li>• Be specific: "top 10 products" instead of "some products"</li>
              <li>• Use business terms: "revenue", "orders", "customers", "employees"</li>
              <li>• The AI will apply business rules automatically</li>
            </ul>
          </div>
        </div>

        {/* Processing Status */}
        {localLoading && (
          <div className="p-3 bg-green-900/20 border border-green-400/30 rounded-lg">
            <div className="flex items-center gap-2 text-green-400 text-sm">
              <Zap className="w-4 h-4 animate-pulse" />
              <span>Your query is being processed by AI. This may take a few moments...</span>
            </div>
          </div>
        )}
        </div>
      </div>
    </div>
  );
} 