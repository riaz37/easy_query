"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { X, ChevronDown, ChevronUp } from "lucide-react";

interface QueryResultOverlayProps {
  isVisible: boolean;
  onViewResults: () => void;
  onClose?: () => void;
  queryText?: string;
  queryMode?: 'query' | 'reports';
}

export function QueryResultOverlay({
  isVisible,
  onViewResults,
  onClose,
  queryText,
  queryMode = 'query'
}: QueryResultOverlayProps) {
  const [isQueryExpanded, setIsQueryExpanded] = useState(false);
  
  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div 
        className="rounded-[32px] shadow-2xl p-6 max-w-lg w-full border relative"
        style={{
          background: `linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)),
                        linear-gradient(246.02deg, rgba(19, 245, 132, 0) 91.9%, rgba(19, 245, 132, 0.2) 114.38%),
                        linear-gradient(59.16deg, rgba(19, 245, 132, 0) 71.78%, rgba(19, 245, 132, 0.2) 124.92%)`,
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          border: "1px solid rgba(255, 255, 255, 0.1)",
        }}
      >
        {/* Close Button */}
        {onClose && (
          <button
            onClick={onClose}
            className="cursor-pointer z-10"
            style={{
              position: 'absolute',
              top: '1rem',
              right: '1rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '2rem',
              height: '2rem',
              borderRadius: '0.5rem',
              background: 'transparent',
              border: 'none',
              color: '#9ca3af',
              transition: 'all 0.2s ease',
              flexShrink: 0,
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = '#f8f8f8';
              e.currentTarget.style.transform = 'scale(1.1)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = '#9ca3af';
              e.currentTarget.style.transform = 'scale(1)';
            }}
            onMouseDown={(e) => {
              e.currentTarget.style.transform = 'scale(0.95)';
            }}
            onMouseUp={(e) => {
              e.currentTarget.style.transform = 'scale(1.1)';
            }}
          >
            <X className="w-5 h-5" />
          </button>
        )}

        {/* Title */}
        <h3 className="text-xl font-semibold text-white text-center mb-2">
          {queryMode === 'reports' ? 'Report Generated!' : 'Query Complete!'}
        </h3>
        
        <p className="text-gray-400 text-center mb-6 whitespace-nowrap overflow-hidden text-ellipsis">
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
            <div className="flex items-start justify-between gap-2">
              <p className={`text-sm text-gray-300 flex-1 ${!isQueryExpanded ? 'line-clamp-2' : ''}`}>
                "{queryText}"
              </p>
              {queryText.length > 100 && (
                <button
                  onClick={() => setIsQueryExpanded(!isQueryExpanded)}
                  className="flex-shrink-0 text-gray-400 hover:text-white transition-colors ml-2"
                >
                  {isQueryExpanded ? (
                    <ChevronUp className="w-4 h-4" />
                  ) : (
                    <ChevronDown className="w-4 h-4" />
                  )}
                </button>
              )}
            </div>
            {queryText.length > 100 && (
              <button
                onClick={() => setIsQueryExpanded(!isQueryExpanded)}
                className="text-xs text-gray-400 hover:text-white transition-colors mt-2"
              >
                {isQueryExpanded ? 'Show less' : 'Show full query'}
              </button>
            )}
          </div>
        )}

        {/* Action Button */}
        <div className="flex justify-center">
          <Button
            onClick={onViewResults}
            variant="outline"
            className="border-0 text-white hover:bg-emerald-400/10"
            size="sm"
            style={{ 
              borderRadius: "99px", 
              height: "32px", 
              minWidth: "160px",
              background: "rgba(255, 255, 255, 0.04)"
            }}
          >
            {queryMode === 'reports' ? 'View Report' : 'View Results'}
          </Button>
        </div>
      </div>
    </div>
  );
}
