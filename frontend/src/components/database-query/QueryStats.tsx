"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Clock, User, Database, FileText, X, BarChart3, Code } from "lucide-react";

interface QueryStatsProps {
  query: string;
  metadata: {
    timestamp: Date;
    userId: string;
    databaseName: string;
    rowCount: number;
    sqlQuery?: string;
  } | null;
  onClear: () => void;
}

export function QueryStats({ query, metadata, onClear }: QueryStatsProps) {
  if (!metadata) return null;

  const formatTimestamp = (timestamp: Date) => {
    return new Intl.DateTimeFormat("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }).format(timestamp);
  };

  return (
    <Card className="bg-gray-900/50 border-emerald-400/30">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-emerald-400 flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            Query Results
          </CardTitle>
          <Button
            variant="outline"
            size="sm"
            onClick={onClear}
            className="border-red-400/30 text-red-400 hover:bg-red-400/10"
          >
            <X className="w-4 h-4 mr-2" />
            Clear Results
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Query Text */}
        <div className="p-3 bg-emerald-900/20 border border-emerald-400/30 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <FileText className="w-4 h-4 text-emerald-400" />
            <span className="text-emerald-400 font-medium">Query:</span>
          </div>
          <p className="text-white text-sm">{query}</p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-3 bg-gray-800/30 rounded-lg border border-gray-700/50">
            <div className="flex items-center justify-center w-8 h-8 bg-emerald-500/20 rounded-lg mx-auto mb-2">
              <Database className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="text-2xl font-bold text-emerald-400">
              {metadata.databaseName}
            </div>
            <div className="text-xs text-gray-400">Database</div>
          </div>

          <div className="text-center p-3 bg-gray-800/30 rounded-lg border border-gray-700/50">
            <div className="flex items-center justify-center w-8 h-8 bg-green-500/20 rounded-lg mx-auto mb-2">
              <BarChart3 className="w-4 h-4 text-green-400" />
            </div>
            <div className="text-2xl font-bold text-green-400">
              {metadata.rowCount.toLocaleString()}
            </div>
            <div className="text-xs text-gray-400">Results</div>
          </div>

          <div className="text-center p-3 bg-gray-800/30 rounded-lg border border-gray-700/50">
            <div className="flex items-center justify-center w-8 h-8 bg-emerald-500/20 rounded-lg mx-auto mb-2">
              <User className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="text-sm font-medium text-emerald-400 truncate">
              {metadata.userId}
            </div>
            <div className="text-xs text-gray-400">User</div>
          </div>

          <div className="text-center p-3 bg-gray-800/30 rounded-lg border border-gray-700/50">
            <div className="flex items-center justify-center w-8 h-8 bg-orange-500/20 rounded-lg mx-auto mb-2">
              <Clock className="w-4 h-4 text-orange-400" />
            </div>
            <div className="text-sm font-medium text-orange-400">
              {formatTimestamp(metadata.timestamp)}
            </div>
            <div className="text-xs text-gray-400">Executed</div>
          </div>
        </div>

        {/* SQL Query Display */}
        {metadata.sqlQuery && (
          <div className="p-3 bg-green-900/20 border border-green-400/30 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Code className="w-4 h-4 text-green-400" />
              <span className="text-green-400 font-medium">Generated SQL:</span>
            </div>
            <code className="text-white text-sm bg-gray-800/50 p-2 rounded block overflow-x-auto">
              {metadata.sqlQuery}
            </code>
          </div>
        )}

        {/* Additional Info */}
        <div className="flex items-center justify-between text-sm text-gray-400">
          <div className="flex items-center gap-4">
            <span>Query executed successfully</span>
            <Badge variant="outline" className="border-green-400/30 text-green-400">
              ✓ Success
            </Badge>
          </div>
          <div className="text-xs">
            Results are interactive - use the table and charts below to explore your data
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 