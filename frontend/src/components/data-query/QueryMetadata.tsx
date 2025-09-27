import { Badge } from "@/components/ui/badge";
import { Clock, Database, CheckCircle, XCircle, AlertCircle } from "lucide-react";

interface QueryMetadataProps {
  rowCount: number;
  executionTime: number;
  status: 'success' | 'error' | 'pending';
  columns?: string[];
  databaseName?: string;
  className?: string;
}

export function QueryMetadata({
  rowCount,
  executionTime,
  status,
  columns = [],
  databaseName,
  className = ""
}: QueryMetadataProps) {
  const getStatusIcon = () => {
    switch (status) {
      case 'success':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'error':
        return <XCircle className="h-4 w-4 text-red-600" />;
      case 'pending':
        return <AlertCircle className="h-4 w-4 text-yellow-600" />;
      default:
        return <AlertCircle className="h-4 w-4 text-gray-600" />;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'success':
        return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-200';
      case 'error':
        return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-200';
      case 'pending':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-200';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-200';
    }
  };

  return (
    <div className={`grid grid-cols-1 md:grid-cols-3 gap-4 text-sm ${className}`}>
      <div className="flex items-center gap-2">
        <Database className="h-4 w-4 text-muted-foreground" />
        <span className="font-medium">Rows:</span> {(rowCount || 0).toLocaleString()}
      </div>
      
      <div className="flex items-center gap-2">
        <Clock className="h-4 w-4 text-muted-foreground" />
        <span className="font-medium">Execution Time:</span> {(executionTime || 0).toFixed(2)}s
      </div>
      
      <div className="flex items-center gap-2">
        {getStatusIcon()}
        <span className="font-medium">Status:</span> 
        <Badge variant="secondary" className={getStatusColor()}>
          {status}
        </Badge>
      </div>
      
      {databaseName && (
        <div className="flex items-center gap-2 md:col-span-3">
          <Database className="h-4 w-4 text-muted-foreground" />
          <span className="font-medium">Database:</span> {databaseName}
        </div>
      )}
      
      {columns.length > 0 && (
        <div className="flex items-center gap-2 md:col-span-3">
          <span className="font-medium">Columns:</span> 
          <div className="flex flex-wrap gap-1">
            {columns.map((column, index) => (
              <Badge key={index} variant="outline" className="text-xs">
                {column}
              </Badge>
            ))}
          </div>
        </div>
      )}
    </div>
  );
} 