"use client";

import React, { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import {
  Play,
  Loader2,
  CheckCircle,
  AlertCircle,
  FileSpreadsheet,
  Database,
  Settings,
} from "lucide-react";
import { useExcelToDB } from "@/lib/hooks/use-excel-to-db";
import { toast } from "sonner";

interface ExcelStep4ConfirmProps {
  selectedFile: File | null;
  selectedTable: string;
  mappingData: any;
  userId: string;
  onComplete: () => void;
  onBack: () => void;
}

export function ExcelStep4Confirm({
  selectedFile,
  selectedTable,
  mappingData,
  userId,
  onComplete,
  onBack,
}: ExcelStep4ConfirmProps) {
  const [isImporting, setIsImporting] = useState(false);
  const [importProgress, setImportProgress] = useState(0);
  const [importError, setImportError] = useState<string | null>(null);

  const { pushDataToDatabase, uploadProgress } = useExcelToDB();

  // Helper function to ensure table name includes schema
  const ensureTableFullName = useCallback((tableName: string): string => {
    if (!tableName.includes(".")) {
      return `dbo.${tableName}`;
    }
    return tableName;
  }, []);

  // Handle data import
  const handleImportData = async () => {
    if (!selectedFile || !selectedTable || !userId || !mappingData) {
      toast.error("Please complete all previous steps first");
      return;
    }

    setIsImporting(true);
    setImportError(null);
    setImportProgress(0);

    try {
      // Extract custom mapping from mapping data
      const customMapping: Record<string, string> = {};
      if (mappingData.mapping_details) {
        mappingData.mapping_details.forEach((detail: any) => {
          if (detail.is_mapped && detail.excel_column && detail.table_column) {
            const isIdentityColumn =
              detail.is_identity ||
              detail.table_column.toLowerCase().includes("id");

            if (!isIdentityColumn) {
              customMapping[detail.excel_column] = detail.table_column;
            }
          }
        });
      }

      const response = await pushDataToDatabase({
        user_id: userId,
        table_full_name: ensureTableFullName(selectedTable),
        column_mapping: customMapping,
        skip_first_row: true, // Default to skip headers
        excel_file: selectedFile,
      });

      if (response) {
        setImportProgress(100);
        toast.success(`Successfully imported ${response.rows_inserted} rows`);

        // Wait a moment then complete
        setTimeout(() => {
          onComplete();
        }, 1500);
      }
    } catch (err) {
      console.error("Error importing data:", err);
      const errorMessage =
        err instanceof Error
          ? err.message
          : "Failed to import data to database";
      setImportError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setIsImporting(false);
    }
  };

  // Get table name from full name
  const getTableName = (fullName: string) => {
    return fullName.split(".").pop() || fullName;
  };

  return (
    <div className="space-y-6">
      {/* Import Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-4 bg-slate-700/30 rounded-xl border border-slate-600/50">
          <div className="flex items-center gap-3 mb-2">
            <FileSpreadsheet className="h-5 w-5 text-green-400" />
            <Label className="text-slate-400 text-sm">File</Label>
          </div>
          <p className="text-white font-semibold">{selectedFile?.name}</p>
          <p className="text-slate-400 text-sm">
            {(selectedFile?.size ? selectedFile.size / 1024 / 1024 : 0).toFixed(
              2
            )}{" "}
            MB
          </p>
        </div>

        <div className="p-4 bg-slate-700/30 rounded-xl border border-slate-600/50">
          <div className="flex items-center gap-3 mb-2">
            <Database className="h-5 w-5 text-green-400" />
            <Label className="text-slate-400 text-sm">Target Table</Label>
          </div>
          <p className="text-white font-semibold">
            {getTableName(selectedTable)}
          </p>
          <p className="text-slate-400 text-sm">{selectedTable}</p>
        </div>
      </div>

      {/* Import Progress */}
      {isImporting && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-slate-400">Import Progress</span>
            <span className="text-white font-semibold">
              {uploadProgress || importProgress}%
            </span>
          </div>
          <Progress
            value={uploadProgress || importProgress}
            className="w-full h-3"
          />
          <div className="text-center text-sm text-slate-400">
            Processing your Excel file and importing data...
          </div>
        </div>
      )}

      {/* Error Display */}
      {importError && (
        <Alert
          variant="destructive"
          className="border-red-500/50 bg-red-500/10"
        >
          <AlertCircle className="h-4 w-4" />
          <AlertDescription className="text-red-300">
            {importError}
          </AlertDescription>
        </Alert>
      )}

      {/* Mapping Preview */}
      {mappingData && (
        <div className="space-y-4">
          <h3 className="font-semibold text-white">Column Mapping Preview</h3>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {mappingData.mapping_details?.map((detail: any, index: number) => {
              if (!detail.is_mapped || detail.is_identity) return null;

              return (
                <div
                  key={index}
                  className="flex items-center gap-3 p-3 bg-slate-700/30 rounded-lg"
                >
                  <div className="flex-1 text-sm">
                    <span className="text-green-400">
                      {detail.excel_column}
                    </span>
                    <span className="text-slate-400 mx-2">→</span>
                    <span className="text-green-400">
                      {detail.table_column}
                    </span>
                  </div>
                  <Badge
                    className={`${
                      detail.mapping_status === "MAPPED"
                        ? "bg-green-500/20 text-green-400 border-green-500/30"
                        : "bg-yellow-500/20 text-yellow-400 border-yellow-500/30"
                    }`}
                  >
                    {detail.mapping_status}
                  </Badge>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="modal-footer-enhanced">
        <Button
          onClick={onBack}
          variant="outline"
          disabled={isImporting}
          className="modal-button-secondary disabled:opacity-50 w-full sm:w-auto"
        >
          Back
        </Button>

        <Button
          onClick={handleImportData}
          disabled={isImporting || !mappingData}
          className="modal-button-primary disabled:opacity-50 disabled:cursor-not-allowed w-full sm:w-auto"
        >
          {isImporting ? (
            <>
              <Loader2 className="h-5 w-5 animate-spin mr-2" />
              Importing...
            </>
          ) : (
            <>Import Data to Database</>
          )}
        </Button>
      </div>
    </div>
  );
}
