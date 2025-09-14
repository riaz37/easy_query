"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Loader2,
  FileSpreadsheet,
  Database,
  Settings,
  ArrowLeft,
  ArrowRight,
  AlertCircle,
  Trash2,
  X,
  Sparkles,
} from "lucide-react";
import { useExcelToDB } from "@/lib/hooks/use-excel-to-db";
import { useNewTable } from "@/lib/hooks/use-new-table";
import { toast } from "sonner";
import type { UserTable, ExcelToDBGetAIMappingResponse } from "@/types/api";

interface ExcelStep3MappingProps {
  selectedFile: File | null;
  selectedTable: string;
  userId: string;
  onMappingComplete: (data: any) => void;
  onNext: () => void;
  onBack: () => void;
}

export function ExcelStep3Mapping({
  selectedFile,
  selectedTable,
  userId,
  onMappingComplete,
  onNext,
  onBack,
}: ExcelStep3MappingProps) {
  const [aiMappingData, setAIMappingData] =
    useState<ExcelToDBGetAIMappingResponse | null>(null);
  const [customMapping, setCustomMapping] = useState<Record<string, string>>(
    {}
  );
  const [isLoadingMapping, setIsLoadingMapping] = useState(false);
  const [userTables, setUserTables] = useState<UserTable[]>([]);

  const { getAIMapping } = useExcelToDB();
  const { getUserTables } = useNewTable();

  // Helper function to ensure table name includes schema
  const ensureTableFullName = useCallback((tableName: string): string => {
    if (!tableName.includes(".")) {
      return `dbo.${tableName}`;
    }
    return tableName;
  }, []);

  // Fetch user tables to get table structure
  const fetchUserTables = useCallback(async () => {
    if (!userId) return;

    try {
      const response = await getUserTables(userId);
      if (response && response.tables && Array.isArray(response.tables)) {
        setUserTables(response.tables);
      }
    } catch (error) {
      console.error("Failed to fetch user tables:", error);
    }
  }, [userId, getUserTables]);

  // Get AI mapping suggestions
  const handleGetAIMapping = async () => {
    if (!selectedFile || !selectedTable || !userId) {
      toast.error("Please select a file and table first");
      return;
    }

    setIsLoadingMapping(true);
    try {
      const response = await getAIMapping({
        user_id: userId,
        table_full_name: ensureTableFullName(selectedTable),
        excel_file: selectedFile,
      });

      if (
        response &&
        response.all_table_columns &&
        response.all_excel_columns &&
        response.mapping_details
      ) {
        setAIMappingData(response);

        // Initialize custom mapping with AI suggestions, excluding identity columns
        const initialMapping: Record<string, string> = {};
        response.mapping_details.forEach((detail) => {
          if (detail.is_mapped && detail.excel_column && detail.table_column) {
            // Check if this is an identity column
            const isIdentityColumn =
              detail.is_identity ||
              detail.table_column.toLowerCase().includes("id");

            // Only add to mapping if it's not an identity column
            if (!isIdentityColumn) {
              initialMapping[detail.excel_column] = detail.table_column;
            }
          }
        });

        setCustomMapping(initialMapping);
        onMappingComplete(response);
        toast.success("AI mapping suggestions generated successfully");
      } else {
        console.warn("Invalid AI mapping response structure:", response);
        toast.error("Invalid response from AI mapping service");
      }
    } catch (err) {
      console.error("Error getting AI mapping:", err);
      toast.error("Failed to get AI mapping suggestions");
    } finally {
      setIsLoadingMapping(false);
    }
  };

  // Update custom mapping
  const updateMapping = (excelColumn: string, dbColumn: string) => {
    if (dbColumn === "__no_mapping__") {
      setCustomMapping((prev) => {
        const newMapping = { ...prev };
        delete newMapping[excelColumn];
        return newMapping;
      });
    } else {
      setCustomMapping((prev) => ({
        ...prev,
        [excelColumn]: dbColumn,
      }));
    }
  };

  // Remove mapping
  const removeMapping = (excelColumn: string) => {
    setCustomMapping((prev) => {
      const newMapping = { ...prev };
      delete newMapping[excelColumn];
      return newMapping;
    });
  };

  // Get selected table data
  const selectedTableData = userTables.find(
    (table) => table.table_full_name === selectedTable
  );

  // Fetch user tables on mount
  useEffect(() => {
    if (userId) {
      fetchUserTables();
    }
  }, [userId, fetchUserTables]);

  // Auto-generate mapping when component mounts
  useEffect(() => {
    if (selectedFile && selectedTable && userId && !aiMappingData) {
      handleGetAIMapping();
    }
  }, [selectedFile, selectedTable, userId]);

  if (isLoadingMapping) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin text-green-400 mx-auto mb-4" />
          <p className="text-slate-400">Generating AI mapping suggestions...</p>
        </div>
      </div>
    );
  }

  if (!aiMappingData) {
    return (
      <div className="text-center py-12">
        <AlertCircle className="h-12 w-12 text-red-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-slate-300 mb-2">
          Unable to generate mapping
        </h3>
        <p className="text-slate-400 mb-6">
          Please ensure you have selected a valid file and table
        </p>
        <Button
          onClick={handleGetAIMapping}
          className="bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800"
        >
          <Sparkles className="h-5 w-5 mr-2" />
          Try Again
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Mapping Header */}
      <div className="text-center space-y-4">
        {/* Identity column notice */}
        <div className="modal-input-enhanced p-3 border border-red-500/20 rounded-lg">
          <div className="flex items-center gap-2 text-sm text-red-400">
            <AlertCircle className="h-4 w-4" />
            <span>
              <strong>Note:</strong> Identity columns (auto-generated IDs)
              cannot be mapped.
            </span>
          </div>
        </div>
      </div>

      {/* Mapping Table */}
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-6 text-sm font-semibold text-slate-400 pb-3 border-b border-slate-600">
          <div className="flex items-center gap-2">
            <FileSpreadsheet className="h-4 w-4 text-green-400" />
            Excel Column
          </div>
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-green-400" />
            Database Column
          </div>
        </div>

        {aiMappingData.all_excel_columns.map((excelColumn: string) => {
          const mappingDetail = aiMappingData.mapping_details.find(
            (detail) => detail.excel_column === excelColumn
          );
          const currentMapping = customMapping[excelColumn];

          return (
            <div
              key={excelColumn}
              className="grid grid-cols-2 gap-6 items-center p-4 bg-slate-700/30 rounded-xl border border-slate-600/50"
            >
              <div className="flex items-center gap-3">
                <div className="p-2 bg-green-500/20 rounded-lg">
                  <FileSpreadsheet className="h-4 w-4 text-green-400" />
                </div>
                <span className="text-white font-semibold text-lg">
                  {excelColumn}
                </span>
              </div>

              <div className="space-y-2">
                <Select
                  value={currentMapping || ""}
                  onValueChange={(value) => updateMapping(excelColumn, value)}
                >
                  <SelectTrigger className="w-full h-12 border-slate-600 bg-slate-700/50">
                    <SelectValue placeholder="Select DB column" />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-slate-600">
                    <SelectItem value="__no_mapping__">
                      <div className="flex items-center gap-2">
                        <X className="h-4 w-4 text-red-400" />
                        <span>No mapping</span>
                      </div>
                    </SelectItem>
                    {aiMappingData.all_table_columns.map((col) => {
                      // Check if this is an identity column (first column or has 'id' in name)
                      const isIdentityColumn =
                        col === aiMappingData.all_table_columns[0] ||
                        col.toLowerCase().includes("id");

                      return (
                        <SelectItem
                          key={col}
                          value={col}
                          disabled={isIdentityColumn}
                          className={
                            isIdentityColumn
                              ? "opacity-50 cursor-not-allowed"
                              : ""
                          }
                        >
                          <div className="flex items-center gap-2">
                            <Database className="h-4 w-4 text-green-400" />
                            <span
                              className={
                                isIdentityColumn
                                  ? "text-slate-500"
                                  : "text-white"
                              }
                            >
                              {col}
                            </span>
                            {isIdentityColumn && (
                              <Badge
                                variant="outline"
                                className="text-xs bg-red-500/20 text-red-400 border-red-500/30"
                              >
                                Identity
                              </Badge>
                            )}
                          </div>
                        </SelectItem>
                      );
                    })}
                  </SelectContent>
                </Select>

                {mappingDetail && (
                  <div className="flex items-center gap-2 text-xs">
                    <Badge
                      className={`${
                        mappingDetail.mapping_status === "MAPPED"
                          ? "bg-green-500/20 text-green-400 border-green-500/30"
                          : mappingDetail.mapping_status === "IDENTITY"
                          ? "bg-green-500/20 text-green-400 border-green-500/30"
                          : "bg-gray-500/20 text-gray-400 border-gray-500/30"
                      }`}
                    >
                      {mappingDetail.mapping_status}
                    </Badge>
                    {mappingDetail.is_identity && (
                      <Badge className="bg-red-500/20 text-red-400 border-red-500/30">
                        Identity (Auto-generated)
                      </Badge>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Action Buttons */}
      <div className="modal-footer-enhanced">
        <Button
          onClick={onBack}
          variant="outline"
          className="modal-button-secondary"
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back
        </Button>

        <Button
          onClick={onNext}
          disabled={Object.keys(customMapping).length === 0}
          className="modal-button-primary"
        >
          Continue
          <ArrowRight className="h-4 w-4 ml-2" />
        </Button>
      </div>
    </div>
  );
}
