"use client";

import React, { useState, useCallback, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Label } from "@/components/ui/label";
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
  Upload,
  FileSpreadsheet,
  Database,
  CheckCircle,
  AlertCircle,
  Zap,
  ArrowRight,
  RefreshCw,
  Trash2,
  Sparkles,
  FileText,
  Table2,
  Settings,
  Play,
  Check,
  X,
} from "lucide-react";
import { useDropzone } from "react-dropzone";
import { useExcelToDB } from "@/lib/hooks/use-excel-to-db";
import { useNewTable } from "@/lib/hooks/use-new-table";
import { toast } from "sonner";

interface ExcelToDBManagerProps {
  userId: string;
  availableTables?: Array<{
    table_name: string;
    full_name: string;
    columns: Array<{
      column_name: string;
      data_type: string;
      is_nullable: boolean;
    }>;
  }>;
  onViewTableData?: (tableName: string) => void;
}

// Import types from the API types file
import type { UserTable, UserTablesResponse, ExcelToDBGetAIMappingResponse } from "@/types/api";

// Interface for AI mapping response - using the imported type
interface AIMappingResponse extends ExcelToDBGetAIMappingResponse {}

export function ExcelToDBManager({
  userId,
  availableTables = [],
  onViewTableData,
}: ExcelToDBManagerProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedTable, setSelectedTable] = useState<string>("");
  const [skipFirstRow, setSkipFirstRow] = useState(true);
  const [aiMappingData, setAIMappingData] = useState<AIMappingResponse | null>(null);
  const [customMapping, setCustomMapping] = useState<Record<string, string>>({});
  const [step, setStep] = useState<"upload" | "mapping" | "confirm" | "complete">("upload");

  // State for user tables
  const [userTables, setUserTables] = useState<UserTable[]>([]);
  const [isLoadingTables, setIsLoadingTables] = useState(false);
  const [lastTablesUpdate, setLastTablesUpdate] = useState<Date | null>(null);
  const [currentDbInfo, setCurrentDbInfo] = useState<{ db_id: number; business_rule?: string } | null>(null);
  const [tableLoadError, setTableLoadError] = useState<string | null>(null);

  const {
    getAIMapping,
    pushDataToDatabase,
    isLoading,
    error,
    uploadProgress,
    clearError,
  } = useExcelToDB();

  const { getUserTables } = useNewTable();

  // Helper function to ensure table name includes schema
  const ensureTableFullName = useCallback((tableName: string): string => {
    if (!tableName.includes('.')) {
      return `dbo.${tableName}`;
    }
    return tableName;
  }, []);

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
        
        // Extract database info
        if (response.current_db_id) {
          setCurrentDbInfo({
            db_id: response.current_db_id,
            business_rule: response.business_rule
          });
        }
        
        if (response.tables.length > 0) {
          toast.success(`Loaded ${response.tables.length} user table(s)`);
          
          // Log table data for debugging
          console.log('User Tables Loaded:', {
            count: response.tables.length,
            tables: response.tables.map(t => ({
              name: t.table_name,
              fullName: t.table_full_name,
              columns: t.table_schema?.columns?.length || 0
            })),
            dbInfo: {
              dbId: response.current_db_id,
              businessRule: response.business_rule
            }
          });
        } else {
          toast.info('No user tables found');
        }
      } else {
        console.warn('Invalid response structure:', response);
        toast.error('Invalid response structure from server');
      }
    } catch (error) {
      console.error('Failed to fetch user tables:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch user tables';
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
  const transformedTables = userTables.map(table => ({
    table_name: table.table_name,
    full_name: table.table_full_name,
    columns: table.table_schema.columns.map(col => ({
      column_name: col.name,
      data_type: col.type,
      is_nullable: !col.is_required
    }))
  }));

  // Use transformed tables if available, otherwise fall back to availableTables prop
  const displayTables = transformedTables.length > 0 ? transformedTables : availableTables;

  // File drop zone configuration
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      clearError();
      setStep("upload");
      setAIMappingData(null);
      setCustomMapping({});
    }
  }, [clearError]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/vnd.ms-excel": [".xls"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
    },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024, // 50MB
  });

  // Get AI mapping suggestions
  const handleGetAIMapping = async () => {
    if (!selectedFile || !selectedTable || !userId) {
      toast.error("Please select a file and table first");
      return;
    }

    try {
      const response = await getAIMapping({
        user_id: userId,
        table_full_name: ensureTableFullName(selectedTable),
        excel_file: selectedFile,
      });

      if (response && response.all_table_columns && response.all_excel_columns && response.mapping_details) {
        setAIMappingData(response);

        // Initialize custom mapping with AI suggestions, excluding identity columns
        const initialMapping: Record<string, string> = {};
        response.mapping_details.forEach((detail) => {
          if (detail.is_mapped && detail.excel_column && detail.table_column) {
            // Check if this is an identity column
            const isIdentityColumn = detail.is_identity || 
                                   detail.table_column === selectedTableData?.columns[0]?.column_name ||
                                   detail.table_column.toLowerCase().includes('id');
            
            // Only add to mapping if it's not an identity column
            if (!isIdentityColumn) {
              initialMapping[detail.excel_column] = detail.table_column;
            }
          }
        });

        setCustomMapping(initialMapping);
        setStep("mapping");
        toast.success("AI mapping suggestions generated successfully");
        
        // Log mapping details for debugging
        console.log('AI Mapping Response:', {
          tableColumns: response.all_table_columns,
          excelColumns: response.all_excel_columns,
          mappingDetails: response.mapping_details,
          identityColumns: response.identity_columns
        });
      } else {
        console.warn('Invalid AI mapping response structure:', response);
        toast.error("Invalid response from AI mapping service");
      }
    } catch (err) {
      console.error("Error getting AI mapping:", err);
      toast.error("Failed to get AI mapping suggestions");
    }
  };

  // Push data to database
  const handlePushData = async () => {
    if (!selectedFile || !selectedTable || !userId || Object.keys(customMapping).length === 0) {
      toast.error("Please complete the mapping configuration");
      return;
    }

    try {
      const response = await pushDataToDatabase({
        user_id: userId,
        table_full_name: ensureTableFullName(selectedTable),
        column_mapping: customMapping,
        skip_first_row: skipFirstRow,
        excel_file: selectedFile,
      });

      if (response) {
        setStep("complete");
        toast.success(`Successfully imported ${response.rows_inserted} rows`);
        
        // Log additional information
        console.log('Import completed:', {
          rowsProcessed: response.rows_processed,
          rowsInserted: response.rows_inserted,
          errors: response.errors
        });
      }
    } catch (err) {
      console.error("Error pushing data:", err);
      toast.error("Failed to import data to database");
    }
  };

  // Reset the form
  const handleReset = () => {
    setSelectedFile(null);
    setSelectedTable("");
    setAIMappingData(null);
    setCustomMapping({});
    setStep("upload");
    clearError();
  };

  // Handle table selection change
  const handleTableChange = (tableFullName: string) => {
    setSelectedTable(tableFullName);
    setAIMappingData(null);
    setCustomMapping({});
    setStep("upload");
    clearError();
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

  const selectedTableData = displayTables.find((table) => table.full_name === selectedTable);

  return (
    <div className="space-y-8">
      {/* Header Section */}
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center gap-3">
          <div className="p-3 bg-gradient-to-br from-green-500/20 to-blue-500/20 rounded-2xl">
            <FileSpreadsheet className="h-8 w-8 text-green-400" />
          </div>
          <div>
            <h1 className="text-4xl font-bold font-barlow bg-gradient-to-r from-white to-slate-300 bg-clip-text text-transparent">
              Excel to Database
            </h1>
            <p className="text-slate-400 text-lg">
              Transform your Excel data into structured database records
            </p>
          </div>
        </div>
        
        {displayTables.length > 0 && (
          <div className="flex items-center justify-center gap-2">
            <Badge variant="outline" className="bg-green-500/10 text-green-400 border-green-500/30">
              <Database className="h-3 w-3 mr-1" />
              {displayTables.length} table(s) available
            </Badge>
          </div>
        )}
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive" className="border-red-500/50 bg-red-500/10">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription className="text-red-300">{error}</AlertDescription>
        </Alert>
      )}

      {/* Step Indicator */}
      <div className="flex items-center justify-center">
        <div className="flex items-center space-x-4">
          {[
            { key: "upload", label: "Upload", icon: Upload, active: step === "upload" },
            { key: "mapping", label: "Mapping", icon: Settings, active: step === "mapping" },
            { key: "confirm", label: "Confirm", icon: Check, active: step === "confirm" },
            { key: "complete", label: "Complete", icon: CheckCircle, active: step === "complete" },
          ].map((stepInfo, index) => (
            <div key={stepInfo.key} className="flex items-center">
              <div className={`flex items-center justify-center w-12 h-12 rounded-full border-2 transition-all duration-300 ${
                stepInfo.active 
                  ? "border-green-500 bg-green-500/20 text-green-400" 
                  : "border-slate-600 text-slate-500"
              }`}>
                <stepInfo.icon className="h-5 w-5" />
              </div>
              <span className={`ml-2 text-sm font-medium ${
                stepInfo.active ? "text-white" : "text-slate-500"
              }`}>
                {stepInfo.label}
              </span>
              {index < 3 && (
                <div className={`ml-4 w-8 h-0.5 ${
                  stepInfo.active ? "bg-green-500" : "bg-slate-600"
                }`} />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Step 1: File Upload */}
      <Card className="bg-slate-800/50 border-slate-700/50 backdrop-blur-sm">
        <CardHeader className="text-center">
          <CardTitle className="flex items-center justify-center gap-3 text-white text-2xl">
            <div className="p-2 bg-blue-500/20 rounded-lg">
              <Upload className="h-6 w-6 text-blue-400" />
            </div>
            Step 1: Upload Excel File
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Database Context */}
          {isLoadingTables ? (
            <div className="p-4 bg-slate-700/30 rounded-xl border border-slate-600/50">
              <div className="flex items-center gap-3">
                <Loader2 className="h-5 w-5 animate-spin text-blue-400" />
                <span className="text-slate-300">Loading database information...</span>
              </div>
            </div>
          ) : currentDbInfo ? (
            <div className="p-4 bg-gradient-to-r from-slate-700/30 to-slate-600/30 rounded-xl border border-slate-600/50">
              <div className="flex items-center gap-3 mb-3">
                <Database className="h-5 w-5 text-green-400" />
                <span className="text-lg font-medium text-white">Current Database</span>
                <Badge variant="outline" className="bg-green-500/20 text-green-400 border-green-500/50">
                  ID: {currentDbInfo.db_id}
                </Badge>
              </div>
              {currentDbInfo.business_rule && (
                <div className="text-sm text-slate-300">
                  <span className="text-slate-400">Business Rule:</span>{" "}
                  <span className="text-white font-medium">{currentDbInfo.business_rule}</span>
                </div>
              )}
            </div>
          ) : null}

          {/* File Drop Zone */}
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300 ${
              isDragActive
                ? "border-green-400 bg-green-400/10 scale-105"
                : "border-slate-600 hover:border-slate-500 hover:bg-slate-700/20"
            }`}
          >
            <input {...getInputProps()} />
            <div className="space-y-4">
              <div className="flex justify-center">
                <div className={`p-4 rounded-2xl transition-all duration-300 ${
                  isDragActive 
                    ? "bg-green-500/20 text-green-400" 
                    : "bg-slate-700/50 text-slate-400"
                }`}>
                  <FileSpreadsheet className="h-16 w-16" />
                </div>
              </div>
              
              {isDragActive ? (
                <div className="space-y-2">
                  <p className="text-xl font-semibold text-green-400">Drop the Excel file here...</p>
                  <p className="text-green-300">Release to upload</p>
                </div>
              ) : (
                <div className="space-y-3">
                  <p className="text-xl font-semibold text-white">
                    Drag & drop an Excel file here, or click to select
                  </p>
                  <p className="text-slate-400">
                    Supports .xlsx and .xls files up to 50MB
                  </p>
                  <div className="flex items-center justify-center gap-2 text-sm text-slate-500">
                    <FileText className="h-4 w-4" />
                    <span>Excel files only</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Selected File Info */}
          {selectedFile && (
            <div className="flex items-center gap-4 p-4 bg-gradient-to-r from-green-500/10 to-blue-500/10 rounded-xl border border-green-500/30">
              <div className="p-2 bg-green-500/20 rounded-lg">
                <FileSpreadsheet className="h-6 w-6 text-green-400" />
              </div>
              <div className="flex-1">
                <p className="text-white font-semibold text-lg">{selectedFile.name}</p>
                <p className="text-slate-400">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              <Button
                onClick={() => setSelectedFile(null)}
                variant="ghost"
                size="sm"
                className="text-red-400 hover:text-red-300 hover:bg-red-500/10"
              >
                <Trash2 className="h-5 w-5" />
              </Button>
            </div>
          )}

          {/* Table Selection */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Label className="text-lg font-semibold text-white">Select Target Table</Label>
                {displayTables.length > 0 && (
                  <Badge variant="outline" className="bg-blue-500/20 text-blue-400 border-blue-500/50">
                    {displayTables.length} available
                  </Badge>
                )}
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
            
            {isLoadingTables ? (
              <div className="flex items-center gap-3 p-6 bg-slate-700/30 rounded-xl">
                <Loader2 className="h-6 w-6 animate-spin text-blue-400" />
                <span className="text-slate-300 text-lg">Loading user tables...</span>
              </div>
            ) : !lastTablesUpdate && !isLoadingTables ? (
              <div className="p-8 bg-slate-700/30 rounded-xl text-center">
                <Database className="h-16 w-16 text-slate-500 mx-auto mb-4" />
                <p className="text-slate-300 text-lg mb-2">Click refresh to load tables</p>
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
                <Database className="h-16 w-16 text-slate-500 mx-auto mb-4" />
                <p className="text-slate-300 text-lg mb-2">No tables found</p>
                <p className="text-slate-500 mb-6">You need to create tables first before importing Excel data</p>
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
                    <Database className="h-5 w-5 mr-2" />
                    Create Table
                  </Button>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                <Select value={selectedTable} onValueChange={handleTableChange}>
                  <SelectTrigger className="h-14 text-lg border-slate-600 bg-slate-700/50">
                    <SelectValue placeholder="Choose a database table" />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-slate-600">
                    {displayTables.map((table) => {
                      const userTable = userTables.find(ut => ut.table_full_name === table.full_name);
                      return (
                        <SelectItem key={table.full_name} value={table.full_name}>
                          <div className="flex items-center gap-3 p-2">
                            <div className="p-2 bg-blue-500/20 rounded-lg">
                              <Database className="h-4 w-4 text-blue-400" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="font-semibold text-white">{table.full_name}</div>
                                                           <div className="text-sm text-slate-400">
                               {/* Removed date and column count text */}
                             </div>
                            </div>
                            <Badge variant="secondary" className="bg-blue-500/20 text-blue-400 border-blue-500/30">
                              {table.columns.length} cols
                            </Badge>
                          </div>
                        </SelectItem>
                      );
                    })}
                  </SelectContent>
                </Select>

                
              </div>
            )}
            
            {/* Error display */}
            {tableLoadError && (
              <Alert variant="destructive" className="border-red-500/50 bg-red-500/10">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription className="text-red-300">
                  Failed to load tables: {tableLoadError}
                </AlertDescription>
              </Alert>
            )}

            {/* Last update timestamp */}
            {lastTablesUpdate && (
              <div className="text-center text-sm text-slate-500">
                Last updated: {lastTablesUpdate.toLocaleTimeString()}
              </div>
            )}
          </div>

          {/* Options */}
          <div className="flex items-center space-x-3 p-4 bg-slate-700/30 rounded-xl">
            <input
              type="checkbox"
              id="skipFirstRow"
              checked={skipFirstRow}
              onChange={(e) => setSkipFirstRow(e.target.checked)}
              className="w-5 h-5 rounded border-slate-600 bg-slate-700 text-green-500 focus:ring-green-500 focus:ring-offset-slate-800"
            />
            <Label htmlFor="skipFirstRow" className="text-white text-lg cursor-pointer">
              Skip first row (headers)
            </Label>
          </div>

          {/* Get AI Mapping Button */}
          <Button
            onClick={handleGetAIMapping}
            disabled={!selectedFile || !selectedTable || isLoading}
            className="w-full h-14 text-lg font-semibold bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700 text-white rounded-xl transition-all duration-300 transform hover:scale-105"
          >
            {isLoading ? (
              <Loader2 className="h-6 w-6 animate-spin mr-3" />
            ) : (
              <Sparkles className="h-6 w-6 mr-3" />
            )}
            Get AI Mapping Suggestions
          </Button>
        </CardContent>
      </Card>

      {/* Step 2: Column Mapping */}
      {step === "mapping" && aiMappingData && (
        <Card className="bg-slate-800/50 border-slate-700/50 backdrop-blur-sm">
          <CardHeader className="text-center">
            <CardTitle className="flex items-center justify-center gap-3 text-white text-2xl">
              <div className="p-2 bg-purple-500/20 rounded-lg">
                <Settings className="h-6 w-6 text-purple-400" />
              </div>
              Step 2: Configure Column Mapping
            </CardTitle>
            <div className="flex items-center justify-center gap-4 text-sm">
              <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/30">
                {aiMappingData.all_excel_columns.length} Excel columns
              </Badge>
              <Badge className="bg-green-500/20 text-green-400 border-blue-500/30">
                {aiMappingData.all_table_columns.length} DB columns
              </Badge>
              <Badge className="bg-purple-500/20 text-purple-400 border-purple-500/30">
                {aiMappingData.mapping_details.filter((d) => d.is_mapped).length} mapped
              </Badge>
            </div>
            
            {/* Identity column notice */}
            <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
              <div className="flex items-center gap-2 text-sm text-red-400">
                <AlertCircle className="h-4 w-4" />
                <span>
                  <strong>Note:</strong> Identity columns (auto-generated IDs) cannot be mapped from Excel data and will be handled automatically by the database.
                </span>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Mapping Table */}
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-6 text-sm font-semibold text-slate-400 pb-3 border-b border-slate-600">
                <div className="flex items-center gap-2">
                  <FileSpreadsheet className="h-4 w-4 text-blue-400" />
                  Excel Column
                </div>
                <div className="flex items-center gap-2">
                  <Database className="h-4 w-4 text-green-400" />
                  Database Column
                </div>
                <div className="flex items-center gap-2">
                  <Settings className="h-4 w-4 text-purple-400" />
                  Actions
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
                    className="grid grid-cols-3 gap-6 items-center p-4 bg-slate-700/30 rounded-xl border border-slate-600/50"
                  >
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-blue-500/20 rounded-lg">
                        <FileSpreadsheet className="h-4 w-4 text-blue-400" />
                      </div>
                      <span className="text-white font-semibold text-lg">{excelColumn}</span>
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
                                                     {selectedTableData?.columns.map((col) => {
                             // Check if this is an identity column (first column or has 'id' in name)
                             const isIdentityColumn = col.column_name === selectedTableData.columns[0]?.column_name || 
                                                    col.column_name.toLowerCase().includes('id');
                             
                             return (
                               <SelectItem 
                                 key={col.column_name} 
                                 value={col.column_name}
                                 disabled={isIdentityColumn}
                                 className={isIdentityColumn ? "opacity-50 cursor-not-allowed" : ""}
                               >
                                 <div className="flex items-center gap-2">
                                   <Database className="h-4 w-4 text-green-400" />
                                   <span className={isIdentityColumn ? "text-slate-500" : "text-white"}>
                                     {col.column_name}
                                   </span>
                                   <Badge variant="outline" className="text-xs bg-slate-700/50 border-slate-600">
                                     {col.data_type}
                                   </Badge>
                                   {isIdentityColumn && (
                                     <Badge variant="outline" className="text-xs bg-red-500/20 text-red-400 border-red-500/30">
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
                                ? "bg-blue-500/20 text-blue-400 border-blue-500/30"
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

                    <div className="flex justify-center">
                      {currentMapping && (
                        <Button
                          onClick={() => removeMapping(excelColumn)}
                          variant="ghost"
                          size="sm"
                          className="text-red-400 hover:text-red-300 hover:bg-red-500/10"
                        >
                          <Trash2 className="h-5 w-5" />
                        </Button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            <Separator className="bg-slate-600" />

            {/* Mapping Summary */}
            <div className="flex items-center justify-between p-4 bg-slate-700/30 rounded-xl">
              <div className="text-lg text-slate-300">
                <span className="text-white font-semibold">{Object.keys(customMapping).length}</span> of{" "}
                <span className="text-white font-semibold">{aiMappingData.all_excel_columns.length}</span> columns mapped
              </div>
              <Button
                onClick={() => setStep("confirm")}
                disabled={Object.keys(customMapping).length === 0}
                className="h-12 px-8 text-lg font-semibold bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700 text-white rounded-xl transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
              >
                Continue to Import
                <ArrowRight className="h-5 w-5 ml-2" />
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 3: Confirmation */}
      {step === "confirm" && (
        <Card className="bg-slate-800/50 border-slate-700/50 backdrop-blur-sm">
          <CardHeader className="text-center">
            <CardTitle className="flex items-center justify-center gap-3 text-white text-2xl">
              <div className="p-2 bg-green-500/20 rounded-lg">
                <Check className="h-6 w-6 text-green-400" />
              </div>
              Step 3: Confirm Import
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="p-4 bg-slate-700/30 rounded-xl border border-slate-600/50">
                <Label className="text-slate-400 text-sm">File</Label>
                <p className="text-white font-semibold text-lg">{selectedFile?.name}</p>
              </div>
              <div className="p-4 bg-slate-700/30 rounded-xl border border-slate-600/50">
                <Label className="text-slate-400 text-sm">Target Table</Label>
                <p className="text-white font-semibold text-lg">{selectedTableData?.table_name}</p>
              </div>
              <div className="p-4 bg-slate-700/30 rounded-xl border border-slate-600/50">
                <Label className="text-slate-400 text-sm">Mapped Columns</Label>
                <p className="text-white font-semibold text-lg">{Object.keys(customMapping).length}</p>
              </div>
              <div className="p-4 bg-slate-700/30 rounded-xl border border-slate-600/50">
                <Label className="text-slate-400 text-sm">Skip Headers</Label>
                <p className="text-white font-semibold text-lg">{skipFirstRow ? "Yes" : "No"}</p>
              </div>
            </div>

            {isLoading && (
              <div className="space-y-3">
                <div className="flex items-center justify-between text-lg">
                  <span className="text-slate-400">Upload Progress</span>
                  <span className="text-white font-semibold">{uploadProgress}%</span>
                </div>
                <Progress value={uploadProgress} className="w-full h-3" />
              </div>
            )}

            <div className="flex gap-4">
              <Button
                onClick={() => setStep("mapping")}
                variant="outline"
                size="lg"
                disabled={isLoading}
                className="flex-1 h-14 text-lg font-semibold border-slate-600 hover:bg-slate-700/50"
              >
                Back to Mapping
              </Button>
              <Button
                onClick={handlePushData}
                disabled={isLoading}
                className="flex-1 h-14 text-lg font-semibold bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700 text-white rounded-xl transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
              >
                {isLoading ? (
                  <Loader2 className="h-6 w-6 animate-spin mr-3" />
                ) : (
                  <Play className="h-6 w-6 mr-3" />
                )}
                Import Data to Database
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 4: Complete */}
      {step === "complete" && (
        <Card className="bg-slate-800/50 border-slate-700/50 backdrop-blur-sm">
          <CardContent className="pt-12 pb-12 text-center">
            <div className="flex justify-center mb-6">
              <div className="p-6 bg-green-500/20 rounded-full">
                <CheckCircle className="h-20 w-20 text-green-400" />
              </div>
            </div>
            <h3 className="text-3xl font-bold text-white mb-4">
              Import Completed Successfully!
            </h3>
            <p className="text-slate-400 text-lg mb-8 max-w-2xl mx-auto">
              Your Excel data has been successfully imported to the database. 
              The system has processed and mapped all columns according to your configuration.
            </p>
            <div className="flex gap-4 justify-center">
              <Button 
                onClick={handleReset} 
                variant="outline" 
                size="lg"
                className="h-14 px-8 text-lg font-semibold border-slate-600 hover:bg-slate-700/50"
              >
                Import Another File
              </Button>
              <Button
                onClick={() => onViewTableData?.(selectedTable)}
                className="h-14 px-8 text-lg font-semibold bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700 text-white rounded-xl transition-all duration-300 transform hover:scale-105"
              >
                <Database className="h-6 w-6 mr-3" />
                View Table Data
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Start Over Button */}
      {step !== "upload" && (
        <div className="text-center">
          <Button 
            onClick={handleReset} 
            variant="outline" 
            size="lg"
            className="border-slate-600 hover:bg-slate-700/50 text-slate-300"
          >
            <RefreshCw className="h-5 w-5 mr-2" />
            Start Over
          </Button>
        </div>
      )}
    </div>
  );
}
