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
      <div 
        className="rounded-[32px] shadow-2xl p-6 max-w-md w-full border"
        style={{
          background: `linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)),
                        linear-gradient(246.02deg, rgba(19, 245, 132, 0) 91.9%, rgba(19, 245, 132, 0.2) 114.38%),
                        linear-gradient(59.16deg, rgba(19, 245, 132, 0) 71.78%, rgba(19, 245, 132, 0.2) 124.92%)`,
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          border: "1px solid rgba(255, 255, 255, 0.1)",
        }}
      >
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
          <div 
            className="mb-6 p-4 rounded-lg"
            style={{
              backgroundColor: "var(--item-root-active-bgcolor, #13F58414)",
            }}
          >
            <p className="text-sm text-gray-300 line-clamp-2">
              "{queryText}"
            </p>
          </div>
        )}

        {/* Action Button */}
        <Button
          onClick={onViewResults}
          className="w-full bg-green-600 hover:bg-green-700 text-white py-3 cursor-pointer"
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
