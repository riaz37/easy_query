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
  RefreshCw,
  ArrowLeft,
  ArrowRight,
  AlertCircle,
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
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Label className="font-semibold text-white">Select Table</Label>
          </div>
          <Button
            onClick={fetchUserTables}
            variant="outline"
            size="sm"
            disabled={isLoadingTables}
            className="border-slate-600 hover:bg-slate-700/50"
          >
            {isLoadingTables ? (
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
            ) : (
              <RefreshCw className="h-4 w-4 mr-2" />
            )}
            Refresh
          </Button>
        </div>

        {!lastTablesUpdate && !isLoadingTables ? (
          <div className="p-8 bg-slate-700/30 rounded-xl text-center">
            <p className="text-slate-300 mb-2">Click refresh to load tables</p>
            <Button
              onClick={fetchUserTables}
              variant="outline"
              size="lg"
              className="border-slate-600 hover:bg-slate-700/50"
            >
              <RefreshCw className="h-5 w-5 mr-2" />
              Load Tables
            </Button>
          </div>
        ) : displayTables.length === 0 ? (
          <div className="p-8 bg-slate-700/30 rounded-xl text-center">
            <p className="text-slate-300 mb-2">No tables found</p>
            <p className="text-slate-500 mb-6">
              You need to create tables first before importing Excel data
            </p>
            <div className="flex gap-3 justify-center">
              <Button
                onClick={fetchUserTables}
                variant="outline"
                size="lg"
                className="border-slate-600 hover:bg-slate-700/50"
              >
                <RefreshCw className="h-5 w-5 mr-2" />
                Refresh
              </Button>
              <Button
                variant="outline"
                size="lg"
                className="text-green-400 border-green-400/50 hover:bg-green-400/10"
              >
                Create Table
              </Button>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <Select value={selectedTable} onValueChange={handleTableChange}>
              <SelectTrigger className="modal-input-enhanced">
                <SelectValue placeholder="Choose a database table" />
              </SelectTrigger>
              <SelectContent 
                className="modal-select-content-enhanced"
                style={{
                  background: "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))"
                }}
              >
                {isLoadingTables ? (
                  <div className="flex items-center justify-center p-8">
                    <Loader2 className="h-8 w-8 animate-spin text-green-400" />
                  </div>
                ) : (
                  displayTables.map((table) => {
                    const userTable = userTables.find(
                      (ut) => ut.table_full_name === table.full_name
                    );
                    return (
                      <SelectItem 
                        key={table.full_name} 
                        value={table.full_name}
                        className="hover:bg-slate-700/30 focus:bg-slate-700/30 data-[highlighted]:bg-slate-700/30"
                      >
                        <div className="flex items-center gap-3 p-3 rounded-lg">
                          <div className="flex-1 min-w-0">
                            <div className="font-semibold text-white">
                              {table.full_name}
                            </div>
                            
                          </div>
                          <Badge
                            variant="secondary"
                            className="bg-green-500/20 text-green-400 border-green-500/30"
                          >
                            {table.columns.length} cols
                          </Badge>
                        </div>
                      </SelectItem>
                    );
                  })
                )}
              </SelectContent>
            </Select>
          </div>
        )}

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
      <div className="flex justify-between">
        <Button
          onClick={onBack}
          variant="outline"
          size="lg"
          className="modal-button-secondary"
        >
          <ArrowLeft className="h-5 w-5 mr-2" />
          Back
        </Button>

        <Button
          onClick={onNext}
          disabled={!selectedTable}
          className="modal-button-primary"
        >
          Continue
          <ArrowRight className="h-5 w-5 ml-2" />
        </Button>
      </div>
    </div>
  );
}
