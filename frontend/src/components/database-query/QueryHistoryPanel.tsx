"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Clock, User, Database, X, Play, Trash2 } from "lucide-react";
import { useDatabaseOperations } from "@/lib/hooks/use-database-operations";

interface QueryHistoryPanelProps {
  history: any[];
  loading: boolean;
  onClose: () => void;
  onQuerySelect: (query: string) => void;
}

export function QueryHistoryPanel({ 
  history, 
  loading, 
  onClose, 
  onQuerySelect 
}: QueryHistoryPanelProps) {
  const { clearHistory } = useDatabaseOperations();

  const handleClearHistory = async () => {
    try {
      await clearHistory();
    } catch (error) {
      console.error("Failed to clear history:", error);
    }
  };

  const formatTimestamp = (timestamp: string | Date) => {
    const date = new Date(timestamp);
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  };

  const getStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case "success":
        return "border-green-400/30 text-green-400";
      case "error":
        return "border-red-400/30 text-red-400";
      case "pending":
        return "border-yellow-400/30 text-yellow-400";
      default:
        return "border-gray-400/30 text-gray-400";
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="w-full max-w-2xl max-h-[80vh] card-enhanced">
        <div className="card-content-enhanced">
          <div className="card-header-enhanced">
            <div className="flex items-center justify-between">
              <div className="card-title-enhanced flex items-center gap-2">
                <Clock className="w-5 h-5 text-emerald-400" />
                Query History
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleClearHistory}
                  className="border-red-400/30 text-red-400 hover:bg-red-400/10"
                  disabled={history.length === 0}
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onClose}
                  className="border-emerald-400/30 text-emerald-400 hover:bg-emerald-400/10"
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </div>
          
          <div className="flex-1 min-h-0 max-h-[60vh]">
            {loading ? (
              <div className="flex items-center justify-center h-64">
                <div className="text-center">
                  <div className="w-8 h-8 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                  <p className="text-gray-400">Loading history...</p>
                </div>
              </div>
            ) : history.length === 0 ? (
              <div className="flex items-center justify-center h-64">
                <div className="text-center">
                  <Clock className="w-12 h-12 text-gray-500 mx-auto mb-2" />
                  <p className="text-gray-400">No query history yet</p>
                  <p className="text-gray-500 text-sm">Your queries will appear here</p>
                </div>
              </div>
            ) : (
              <ScrollArea className="h-full">
                <div className="p-4 space-y-3">
                  {history.map((item, index) => (
                    <div
                      key={index}
                      className="p-3 bg-white/5 border border-emerald-500/20 rounded-lg hover:bg-white/10 hover:border-emerald-400/30 transition-all duration-200 cursor-pointer"
                      onClick={() => onQuerySelect(item.query)}
                    >
                      <div className="flex items-start justify-between gap-2 mb-2">
                        <div className="flex-1 min-w-0">
                          <p className="text-white text-sm font-medium truncate">
                            {item.query}
                          </p>
                        </div>
                        <Badge 
                          variant="outline" 
                          size="sm"
                          className={getStatusColor(item.status)}
                        >
                          {item.status || "Unknown"}
                        </Badge>
                      </div>
                      
                      <div className="flex items-center gap-4 text-xs text-gray-400">
                        <div className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {formatTimestamp(item.timestamp)}
                        </div>
                        
                        {item.type && (
                          <div className="flex items-center gap-1">
                            <Database className="w-3 h-3" />
                            {item.type}
                          </div>
                        )}
                        
                        {item.executionTime && (
                          <div className="flex items-center gap-1">
                            <Play className="w-3 h-3" />
                            {item.executionTime}ms
                          </div>
                        )}
                      </div>
                      
                      {item.rowCount !== undefined && (
                        <div className="mt-2 text-xs text-gray-400">
                          {item.rowCount} results
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            )}
          </div>
        </div>
      </div>
    </div>
  );
} 