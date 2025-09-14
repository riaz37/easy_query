import React, { useState, useEffect } from "react";
// Card components removed - now handled by parent component
import { Button } from "@/components/ui/button";
import { Database, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { ServiceRegistry } from "@/lib/api";
import { useAuthContext } from "@/components/providers";

interface TableSelectorProps {
  databaseId?: number | null;
  onTableSelect: (tableName: string) => void;
  className?: string;
}

export function TableSelector({
  databaseId,
  onTableSelect,
  className = "",
}: TableSelectorProps) {
  const { user } = useAuthContext();
  const [availableTables, setAvailableTables] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Get user tables from the API using authenticated service
  const fetchUserTables = async () => {
    if (!user?.user_id) {
      setError("Please log in to view tables");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Use endpoint that requires user ID parameter
      const response = await ServiceRegistry.vectorDB.getUserTableNames(
        user.user_id
      );

      if (response.success && response.data && Array.isArray(response.data)) {
        setAvailableTables(response.data);
      } else {
        setAvailableTables([]);
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Failed to fetch tables";
      console.error("Failed to fetch user tables:", error);
      setError(errorMessage);
      toast.error("Failed to load tables", {
        description: errorMessage,
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch tables on mount and when databaseId changes
  useEffect(() => {
    if (user?.user_id) {
      fetchUserTables();
    }
  }, [user?.user_id, databaseId]);


  if (!user?.user_id) {
    return (
      <div className={className}>
        <div className="text-center py-8">
          <Database className="h-12 w-12 mx-auto text-gray-400 mb-2" />
          <p className="text-gray-400 text-sm">
            Please log in to view available tables
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={className}>
      <div className="space-y-4">

        {/* Loading State */}
        {isLoading && (
          <div className="text-center py-6">
            <Loader2 className="h-8 w-8 mx-auto text-green-400 animate-spin mb-2" />
            <p className="text-gray-400 text-sm">Loading tables...</p>
          </div>
        )}

        {/* Error State */}
        {error && !isLoading && (
          <div className="text-center py-6">
            <div className="p-4 bg-red-900/20 border border-red-500/30 rounded-lg">
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* Tables List */}
        {!isLoading && !error && (
          <>
            {availableTables.length === 0 ? (
              <div className="text-center py-6">
                <Database className="h-12 w-12 mx-auto text-gray-400 mb-2" />
                <p className="text-gray-400 text-sm">No tables found</p>
              </div>
            ) : (
              <div className="space-y-2">
                <div className="max-h-60 overflow-y-auto space-y-1">
                  {availableTables.map((table) => (
                    <Button
                      key={table}
                      variant="ghost"
                      size="sm"
                      onClick={() => onTableSelect(table)}
                      className="w-full justify-start text-left p-3 h-auto border border-gray-600/30 hover:bg-green-400/10 hover:border-green-400/30"
                    >
                      <div className="flex items-center gap-2 w-full">
                        <Database className="h-4 w-4 text-green-400" />
                        <span className="text-white font-mono text-sm">
                          {table}
                        </span>
                      </div>
                    </Button>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
