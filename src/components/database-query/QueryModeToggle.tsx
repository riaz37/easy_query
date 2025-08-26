"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Play, FileText, Sparkles, BarChart3, Loader2 } from "lucide-react";

interface QueryModeToggleProps {
  mode: 'query' | 'reports';
  onModeChange: (mode: 'query' | 'reports') => void;
  hasDatabase: boolean;
  loading?: boolean;
}

export function QueryModeToggle({ mode, onModeChange, hasDatabase, loading = false }: QueryModeToggleProps) {
  return (
    <div className={`flex items-center gap-2 p-1 rounded-lg border transition-all duration-300 ${
      loading 
        ? 'bg-gray-800/70 border-blue-400/50 opacity-80' 
        : 'bg-gray-800/50 border-blue-400/30'
    }`}>
      <Button
        variant={mode === 'query' ? 'default' : 'ghost'}
        size="sm"
        onClick={() => onModeChange('query')}
        disabled={!hasDatabase || loading}
        className={`${
          mode === 'query'
            ? 'bg-blue-600 hover:bg-blue-700 text-white'
            : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
        } transition-all duration-200 ${
          loading ? 'opacity-60 cursor-not-allowed' : ''
        }`}
      >
        {loading ? (
          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
        ) : (
          <Play className="w-4 h-4 mr-2" />
        )}
        Quick Query
        {mode === 'query' && (
          <Badge variant="secondary" className="ml-2 bg-blue-100 text-blue-800">
            {loading ? 'Processing...' : 'Active'}
          </Badge>
        )}
      </Button>
      
      <Button
        variant={mode === 'reports' ? 'default' : 'ghost'}
        size="sm"
        onClick={() => onModeChange('reports')}
        disabled={!hasDatabase || loading}
        className={`${
          mode === 'reports'
            ? 'bg-purple-600 hover:bg-purple-700 text-white'
            : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
        } transition-all duration-200 ${
          loading ? 'opacity-60 cursor-not-allowed' : ''
        }`}
      >
        {loading ? (
          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
        ) : (
          <FileText className="w-4 h-4 mr-2" />
        )}
        AI Reports
        {mode === 'reports' && (
          <Badge variant="secondary" className="ml-2 bg-purple-100 text-purple-800">
            {loading ? 'Processing...' : 'Active'}
          </Badge>
        )}
      </Button>
    </div>
  );
} 