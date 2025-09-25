"use client";

import React, { useState, useCallback, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Loader2,
  AlertCircle,
  Database,
} from "lucide-react";
import { useNewTable } from "@/lib/hooks/use-new-table";
import { toast } from "sonner";
import type { UserTable } from "@/types/api";

interface ExcelStep2SelectDestinationProps {
  userId: string;
  availableTables: any[];
  selectedTable: string;
  onTableSelect: (table: string) => void;
  onNext: () => void;
  onBack: () => void;
}

export function ExcelStep2SelectDestination({
  userId,
  availableTables,
  selectedTable,
  onTableSelect,
  onNext,
  onBack,
}: ExcelStep2SelectDestinationProps) {
  // State for user tables
  const [userTables, setUserTables] = useState<UserTable[]>([]);
  const [isLoadingTables, setIsLoadingTables] = useState(false);
  const [lastTablesUpdate, setLastTablesUpdate] = useState<Date | null>(null);
  const [tableLoadError, setTableLoadError] = useState<string | null>(null);

  const { getUserTables } = useNewTable();

  // Fetch user tables from API
  const fetchUserTables = useCallback(async () => {
    if (!userId) return;

    setIsLoadingTables(true);
    setTableLoadError(null);
    try {
      const response = await getUserTables(userId);
      if (response && response.tables && Array.isArray(response.tables)) {
        setUserTables(response.tables);
        setLastTablesUpdate(new Date());

        if (response.tables.length > 0) {
          toast.success(`Loaded ${response.tables.length} user table(s)`);
        } else {
          toast.info("No user tables found");
        }
      } else {
        console.warn("Invalid response structure:", response);
        toast.error("Invalid response structure from server");
      }
    } catch (error) {
      console.error("Failed to fetch user tables:", error);
      const errorMessage =
        error instanceof Error ? error.message : "Failed to fetch user tables";
      setTableLoadError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setIsLoadingTables(false);
    }
  }, [userId, getUserTables]);

  // Fetch user tables on component mount
  useEffect(() => {
    if (userId) {
      fetchUserTables();
    }
  }, [userId, fetchUserTables]);

  // Transform user table data to match the expected format
  const transformedTables = userTables.map((table) => ({
    table_name: table.table_name,
    full_name: table.table_full_name,
    columns: table.table_schema.columns.map((col) => ({
      column_name: col.name,
      data_type: col.type,
      is_nullable: !col.is_required,
    })),
  }));

  // Use transformed tables if available, otherwise fall back to availableTables prop
  const displayTables =
    transformedTables.length > 0 ? transformedTables : availableTables;

  const handleTableChange = (tableFullName: string) => {
    onTableSelect(tableFullName);
  };

  return (
    <div className="space-y-6">
      {/* Table Selection */}
      <div className="space-y-4">
        

        <div className="space-y-3">
        <div className="flex items-center justify-between">
            <Label className="modal-label-enhanced">Select Table</Label>
          </div>
            <Select value={selectedTable} onValueChange={handleTableChange}>
            <SelectTrigger className="modal-select-enhanced w-full">
                <SelectValue placeholder="Choose a database table" />
              </SelectTrigger>
            <SelectContent className="modal-select-content-enhanced">
                {isLoadingTables ? (
                <SelectItem value="loading" disabled className="dropdown-item">
                  <div className="flex items-center justify-center w-full py-2">
                    <Loader2 className="h-5 w-5 animate-spin text-green-400 mr-2" />
                    <span className="text-gray-400">Loading tables...</span>
                  </div>
                </SelectItem>
              ) : displayTables.length > 0 ? (
                displayTables.map((table) => (
                      <SelectItem 
                        key={table.full_name} 
                        value={table.full_name}
                    className="dropdown-item"
                  >
                    <div className="flex items-center justify-between w-full">
                      <span className="font-medium">{table.full_name}</span>
                          <Badge
                            variant="secondary"
                        className="bg-green-500/20 text-green-400 border-green-500/30 ml-2"
                          >
                            {table.columns.length} cols
                          </Badge>
                        </div>
                      </SelectItem>
                ))
              ) : (
                <SelectItem value="no-tables" disabled className="dropdown-item">
                  <div className="text-center py-4 text-gray-400">
                    <Database className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No tables found</p>
                    <p className="text-xs">Create tables first before importing Excel data</p>
                  </div>
                </SelectItem>
                )}
              </SelectContent>
            </Select>
          </div>

        {/* Error display */}
        {tableLoadError && (
          <Alert
            variant="destructive"
            className="border-red-500/50 bg-red-500/10"
          >
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="text-red-300">
              Failed to load tables: {tableLoadError}
            </AlertDescription>
          </Alert>
        )}
      </div>

      {/* Action Buttons */}
      <div className="modal-footer-enhanced">
        <Button
          onClick={onBack}
          variant="outline"
          className="modal-button-secondary w-full sm:w-auto"
        >
          Back
        </Button>

        <Button
          onClick={onNext}
          disabled={!selectedTable}
          className="modal-button-primary w-full sm:w-auto"
        >
          Continue
        </Button>
      </div>
    </div>
  );
}
