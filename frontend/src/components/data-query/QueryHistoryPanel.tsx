// Card components removed - now handled by parent component
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { History, Clock, CheckCircle, XCircle, FileText, Database, Trash2 } from "lucide-react";
import { QueryHistoryItem } from "@/store/query-store";

interface QueryHistoryPanelProps {
  history: QueryHistoryItem[];
  onSelect?: (item: QueryHistoryItem) => void;
  onClear?: () => void;
  type?: 'file' | 'database';
  maxItems?: number;
  emptyMessage?: string;
}

export function QueryHistoryPanel({ 
  history, 
  onSelect,
  onClear,
  type = 'file',
  maxItems = 10, 
  emptyMessage
}: QueryHistoryPanelProps) {
  const getIcon = () => {
    return type === 'file' ? <FileText className="h-4 w-4" /> : <Database className="h-4 w-4" />;
  };

  const getEmptyMessage = () => {
    if (emptyMessage) return emptyMessage;
    return type === 'file' 
      ? "No file query history yet" 
      : "No database query history yet";
  };

  const getEmptySubMessage = () => {
    return type === 'file'
      ? "Execute your first file query to see history"
      : "Execute your first database query to see history";
  };

  const getStatusIcon = (status: string) => {
    return status === 'success' ? (
      <CheckCircle className="h-3 w-3 text-green-400" />
    ) : (
      <XCircle className="h-3 w-3 text-red-400" />
    );
  };

  const handleQueryClick = (item: QueryHistoryItem) => {
    if (onSelect) {
      onSelect(item);
    }
  };

  return (
    <div className="space-y-3">
      {/* Header with clear button */}
      {history.length > 0 && onClear && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-gray-400">
            {history.length} quer{history.length !== 1 ? 'ies' : 'y'} in history
          </p>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClear}
            className="text-red-400 hover:text-red-300 hover:bg-red-900/20"
          >
            <Trash2 className="h-3 w-3 mr-1" />
            Clear All
          </Button>
        </div>
      )}

      {/* History List */}
      <div className="space-y-2 max-h-80 overflow-y-auto">
        {history.length > 0 ? (
          history.slice(0, maxItems).map((item) => (
            <div
              key={item.id}
              className={`p-3 bg-gray-800/30 border border-gray-600/30 rounded-lg transition-all duration-200 ${
                onSelect 
                  ? 'cursor-pointer hover:bg-emerald-900/20 hover:border-emerald-400/30' 
                  : ''
              }`}
              onClick={() => onSelect && handleQueryClick(item)}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  {getIcon()}
                  <Badge
                    variant="outline"
                    className={`text-xs ${
                      item.status === 'success' 
                        ? 'border-green-400/30 text-green-400' 
                        : 'border-red-400/30 text-red-400'
                    }`}
                  >
                    {getStatusIcon(item.status)}
                    {item.status}
                  </Badge>
                </div>
                <span className="text-xs text-gray-400 flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {new Date(item.timestamp).toLocaleString()}
                </span>
              </div>
              
              <div className="text-sm font-mono text-white bg-gray-900/50 p-2 rounded border border-gray-600/30 mb-2">
                <div className="truncate" title={item.query}>
                  {item.query}
                </div>
              </div>

              {/* Query metadata */}
              <div className="flex items-center justify-between text-xs text-gray-400">
                <div className="flex items-center gap-3">
                  {item.metadata?.executionTime && (
                    <span>⏱️ {item.metadata.executionTime}ms</span>
                  )}
                  {item.metadata?.resultCount !== undefined && (
                    <span>📊 {item.metadata.resultCount} results</span>
                  )}
                  {item.metadata?.fileIds && (
                    <span>📄 {item.metadata.fileIds.length} files</span>
                  )}
                </div>
                {onSelect && (
                  <span className="text-emerald-400">Click to reuse →</span>
                )}
              </div>
            </div>
          ))
        ) : (
          <div className="text-center py-8">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-800/50 flex items-center justify-center border border-gray-600/30">
              <History className="h-8 w-8 text-gray-400" />
            </div>
            <p className="text-gray-400 font-medium mb-1">{getEmptyMessage()}</p>
            <p className="text-gray-500 text-sm">{getEmptySubMessage()}</p>
          </div>
        )}
      </div>

      {/* Show more indicator */}
      {history.length > maxItems && (
        <div className="text-center py-2">
          <p className="text-xs text-gray-500">
            Showing {maxItems} of {history.length} queries
          </p>
        </div>
      )}
    </div>
  );
} 