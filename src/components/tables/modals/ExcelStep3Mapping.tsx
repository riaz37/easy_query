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
          className="bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700"
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
        <div className="flex items-center justify-center gap-4 text-sm">
          <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/30">
            {aiMappingData.all_excel_columns.length} Excel columns
          </Badge>
          <Badge className="bg-green-500/20 text-green-400 border-green-500/30">
            {aiMappingData.all_table_columns.length} DB columns
          </Badge>
          <Badge className="bg-purple-500/20 text-purple-400 border-purple-500/30">
            {aiMappingData.mapping_details.filter((d) => d.is_mapped).length}{" "}
            mapped
          </Badge>
        </div>

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

      {/* Mapping Interface */}
      <div className="grid grid-cols-2 gap-8">
        {/* Existing Table Column */}
        <div className="space-y-4">
          <div className="text-center">
            <h3 className="modal-title-enhanced text-base mb-2">
              Existing Table Column
            </h3>
            <p className="modal-description-enhanced text-sm">
              Be Patient Until fully completed
            </p>
          </div>

          <div className="space-y-3">
            {aiMappingData.all_table_columns.map((column, index) => {
              const isIdentityColumn =
                column.toLowerCase().includes("id") || index === 0;
              return (
                <div
                  key={column}
                  className={`modal-input-enhanced p-4 rounded-xl border-2 transition-all duration-300 ${
                    isIdentityColumn
                      ? "border-red-500/30"
                      : "border-green-500/30"
                  }`}
                  style={{
                    background: isIdentityColumn
                      ? "linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)), linear-gradient(246.02deg, rgba(239, 68, 68, 0) 91.9%, rgba(239, 68, 68, 0.2) 114.38%), linear-gradient(59.16deg, rgba(239, 68, 68, 0) 71.78%, rgba(239, 68, 68, 0.2) 124.92%)"
                      : "linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)), linear-gradient(246.02deg, rgba(19, 245, 132, 0) 91.9%, rgba(19, 245, 132, 0.2) 114.38%), linear-gradient(59.16deg, rgba(19, 245, 132, 0) 71.78%, rgba(19, 245, 132, 0.2) 124.92%)"
                  }}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`p-2 rounded-lg ${
                        isIdentityColumn ? "bg-red-500/20" : "bg-green-500/20"
                      }`}
                    >
                      <Database
                        className={`h-4 w-4 ${
                          isIdentityColumn ? "text-red-400" : "text-green-400"
                        }`}
                      />
                    </div>
                    <span
                      className={`font-semibold ${
                        isIdentityColumn ? "text-red-400" : "text-white"
                      }`}
                    >
                      {column}
                    </span>
                    <div
                      className={`ml-auto w-3 h-3 rounded-full ${
                        isIdentityColumn ? "bg-red-500" : "bg-green-500"
                      }`}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Upload File Column */}
        <div className="space-y-4">
          <div className="text-center">
            <h3 className="modal-title-enhanced text-base mb-2">
              Upload File Column
            </h3>
            <p className="modal-description-enhanced text-sm">
              Be Patient Until fully completed
            </p>
          </div>

          <div className="space-y-3">
            {aiMappingData.all_excel_columns.map((excelColumn) => {
              const mappingDetail = aiMappingData.mapping_details.find(
                (detail) => detail.excel_column === excelColumn
              );
              const currentMapping = customMapping[excelColumn];
              const isMismatch =
                mappingDetail && mappingDetail.mapping_status === "MISMATCH";

              return (
                <div
                  key={excelColumn}
                  className={`modal-input-enhanced p-4 rounded-xl border-2 transition-all duration-300 ${
                    isMismatch
                      ? "border-red-500/30"
                      : currentMapping
                      ? "border-green-500/30"
                      : "border-slate-600/50"
                  }`}
                  style={{
                    background: isMismatch
                      ? "linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)), linear-gradient(246.02deg, rgba(239, 68, 68, 0) 91.9%, rgba(239, 68, 68, 0.2) 114.38%), linear-gradient(59.16deg, rgba(239, 68, 68, 0) 71.78%, rgba(239, 68, 68, 0.2) 124.92%)"
                      : currentMapping
                      ? "linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)), linear-gradient(246.02deg, rgba(19, 245, 132, 0) 91.9%, rgba(19, 245, 132, 0.2) 114.38%), linear-gradient(59.16deg, rgba(19, 245, 132, 0) 71.78%, rgba(19, 245, 132, 0.2) 124.92%)"
                      : "linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)), linear-gradient(246.02deg, rgba(107, 114, 128, 0) 91.9%, rgba(107, 114, 128, 0.2) 114.38%), linear-gradient(59.16deg, rgba(107, 114, 128, 0) 71.78%, rgba(107, 114, 128, 0.2) 124.92%)"
                  }}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`p-2 rounded-lg ${
                        isMismatch
                          ? "bg-red-500/20"
                          : currentMapping
                          ? "bg-green-500/20"
                          : "bg-slate-600/50"
                      }`}
                    >
                      <Database
                        className={`h-4 w-4 ${
                          isMismatch
                            ? "text-red-400"
                            : currentMapping
                            ? "text-green-400"
                            : "text-slate-400"
                        }`}
                      />
                    </div>
                    <span
                      className={`font-semibold ${
                        isMismatch
                          ? "text-red-400"
                          : currentMapping
                          ? "text-white"
                          : "text-slate-400"
                      }`}
                    >
                      {isMismatch ? "Mis Match" : excelColumn}
                    </span>
                    <div className="ml-auto flex items-center gap-2">
                      {currentMapping && (
                        <div className="w-3 h-3 rounded-full bg-green-500" />
                      )}
                      {isMismatch && (
                        <div className="w-3 h-3 rounded-full bg-red-500" />
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Mapping Summary */}
      <div 
        className="modal-input-enhanced p-4 rounded-xl border border-slate-600/50"
        style={{
          background: "linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)), linear-gradient(246.02deg, rgba(19, 245, 132, 0) 91.9%, rgba(19, 245, 132, 0.2) 114.38%), linear-gradient(59.16deg, rgba(19, 245, 132, 0) 71.78%, rgba(19, 245, 132, 0.2) 124.92%)"
        }}
      >
        <div className="flex items-center justify-between">
          <div className="modal-description-enhanced text-base">
            <span className="text-white font-semibold">{Object.keys(customMapping).length}</span> of{" "}
            <span className="text-white font-semibold">{aiMappingData.all_excel_columns.length}</span> columns mapped
          </div>
        </div>
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
