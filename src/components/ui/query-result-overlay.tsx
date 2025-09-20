"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { 
  CheckCircle, 
  Database, 
  ArrowRight,
  FileText
} from "lucide-react";

interface QueryResultOverlayProps {
  isVisible: boolean;
  onViewResults: () => void;
  queryText?: string;
  queryMode?: 'query' | 'reports';
}

export function QueryResultOverlay({
  isVisible,
  onViewResults,
  queryText,
  queryMode = 'query'
}: QueryResultOverlayProps) {
  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900/95 border border-green-400/30 rounded-lg shadow-2xl p-6 max-w-md w-full">
        {/* Success Icon */}
        <div className="flex items-center justify-center mb-4">
          <div className="p-3 bg-green-500/20 rounded-full">
            <CheckCircle className="w-8 h-8 text-green-400" />
          </div>
        </div>

        {/* Title */}
        <h3 className="text-xl font-semibold text-white text-center mb-2">
          {queryMode === 'reports' ? 'Report Generated!' : 'Query Complete!'}
        </h3>
        
        <p className="text-gray-400 text-center mb-6">
          {queryMode === 'reports' 
            ? 'Your AI report has been generated successfully.' 
            : 'Your database query has been processed successfully.'
          }
        </p>

        {/* Query Preview */}
        {queryText && (
          <div className="mb-6 p-3 bg-gray-800/50 rounded-lg border border-gray-700/50">
            <p className="text-sm text-gray-300 line-clamp-2">
              "{queryText}"
            </p>
          </div>
        )}

        {/* Action Button */}
        <Button
          onClick={onViewResults}
          className="w-full bg-green-600 hover:bg-green-700 text-white py-3"
        >
          {queryMode === 'reports' ? (
            <FileText className="w-5 h-5 mr-2" />
          ) : (
            <Database className="w-5 h-5 mr-2" />
          )}
          {queryMode === 'reports' ? 'View Report' : 'View Results'}
          <ArrowRight className="w-5 h-5 ml-2" />
        </Button>
      </div>
    </div>
  );
}
