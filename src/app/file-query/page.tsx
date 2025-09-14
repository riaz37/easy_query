"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useQueryStore } from "@/store/query-store";
import {
  useAuthContext,
  useDatabaseContext,
  useBusinessRulesContext,
} from "@/components/providers";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  FileText,
  CheckCircle,
  File,
  History,
  AlertCircle,
  X,
  Database,
  Brain,
} from "lucide-react";
import { Spinner } from "@/components/ui/loading";
import {
  FileUpload,
  FileResults,
  FileQueryForm,
  QueryHistoryPanel,
  TableSelector,
} from "@/components/data-query";
import { PageLayout, PageHeader } from "@/components/layout/PageLayout";
import { fileService } from "@/lib/api/services/file-service";
import type {
  UploadedFile,
  FileQueryResult,
  QueryOptions,
} from "@/components/data-query";

export default function FileQueryPage() {
  // Query state
  const [query, setQuery] = useState("");
  const [isExecuting, setIsExecuting] = useState(false);
  const [queryResults, setQueryResults] = useState<FileQueryResult[]>([]);
  const [queryError, setQueryError] = useState<string | null>(null);

  // File upload state
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);

  // Table selection state
  const [selectedTable, setSelectedTable] = useState<string | null>(null);
  const [useTable, setUseTable] = useState(true); // Track table usage from FileUpload

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);

  // Store and context
  const { fileQueryHistory, loadQueryHistory, saveQuery } = useQueryStore();

  const { user, isLoading: userLoading, isAuthenticated } = useAuthContext();
  const { currentDatabaseId, currentDatabaseName } = useDatabaseContext();
  const { businessRules, validateQuery } = useBusinessRulesContext();

  // Load query history on mount
  useEffect(() => {
    if (isAuthenticated && user?.user_id) {
      loadQueryHistory(user.user_id, "file");
    }
  }, [isAuthenticated, user?.user_id, loadQueryHistory]);

  // Handle file upload status changes
  const handleUploadStatusChange = useCallback((files: UploadedFile[]) => {
    setUploadedFiles(files);
  }, []);

  // Handle files uploaded (get file IDs for querying)
  const handleFilesUploaded = useCallback((fileIds: string[]) => {
    console.log("Files uploaded with IDs:", fileIds);
    // These IDs can be used when querying specific files
  }, []);

  // Handle table usage change from FileUpload
  const handleTableUsageChange = useCallback((useTable: boolean) => {
    console.log("FileQueryPage: handleTableUsageChange called with:", useTable);
    setUseTable(useTable);
    // Clear selected table if tables are disabled
    if (!useTable) {
      setSelectedTable(null);
      toast.info("Table selection cleared - tables are disabled");
    }
  }, []);

  // Execute file query
  const handleQuerySubmit = useCallback(
    async (queryText: string, options: QueryOptions) => {
      if (!queryText.trim()) {
        toast.error("Please enter a query");
        return;
      }

      if (!isAuthenticated) {
        toast.error("Please log in to execute queries");
        return;
      }

      // Validate query against business rules if database is selected
      if (currentDatabaseId) {
        const validationResult = validateQuery(queryText, currentDatabaseId);
        if (!validationResult.isValid) {
          toast.error(
            `Query validation failed: ${validationResult.errors.join(", ")}`
          );
          return;
        }
      }

      setIsExecuting(true);
      setQueryError(null);
      setQueryResults([]);
      setQuery(queryText);

      const startTime = Date.now();

      try {
        // Get file IDs from completed uploads
        const completedFileIds = uploadedFiles
          .filter((file) => file.status === "completed")
          .map((file) => file.id);

        // Execute file search
        const response = await fileService.searchFiles({
          query: queryText,
          user_id: user?.user_id,
          answer_style: options.answerStyle,
          table_specific: !!selectedTable, // Make query table-specific if table is selected
          tables: selectedTable ? [selectedTable] : undefined, // Include selected table
          file_ids: completedFileIds.length > 0 ? completedFileIds : undefined,
        });

        if (response.success && response.data) {
          const searchResponse = response.data;
          console.log("File search response:", searchResponse);

          // Extract results from the answer sources or create structured result
          let results: FileQueryResult[] = [];

          if (searchResponse.answer) {
            // Create the main result with the AI-generated answer
            const mainResult: FileQueryResult = {
              id: "main-answer",
              answer: searchResponse.answer.answer, // This is the actual AI response
              confidence: searchResponse.answer.confidence,
              sources_used: searchResponse.answer.sources_used,
              query: searchResponse.query,
              context_length: searchResponse.answer.context_length,
              prompt_length: searchResponse.answer.prompt_length,
              // Add source information
              sources: searchResponse.answer.sources || [],
            };

            results.push(mainResult);

            // If there are individual sources with content, add them as separate results
            if (
              searchResponse.answer.sources &&
              Array.isArray(searchResponse.answer.sources)
            ) {
              searchResponse.answer.sources.forEach((source, index) => {
                if (source.content || source.text) {
                  results.push({
                    id: `source-${index}`,
                    answer: source.content || source.text,
                    confidence: searchResponse.answer.confidence,
                    sources_used: 1,
                    query: searchResponse.query,
                    source_file: source.file_name,
                    source_title: source.title,
                    page_range: source.page_range,
                    document_number: source.document_number,
                    is_source: true, // Mark this as a source result
                  });
                }
              });
            }
          }

          setQueryResults(results);

          // Save to history
          if (user?.user_id) {
            saveQuery({
              id: Math.random().toString(36).substr(2, 9),
              type: "file",
              query: queryText,
              userId: user.user_id,
              timestamp: new Date(),
              results: results,
              metadata: {
                resultCount: results.length,
                fileIds: completedFileIds,
              },
            });
          }

          toast.success(
            `Query executed successfully! Found ${results.length} results.`
          );
        } else {
          throw new Error(response.error || "Query execution failed");
        }
      } catch (error) {
        console.error("File query execution error:", error);
        const errorMessage =
          error instanceof Error
            ? error.message
            : "An unexpected error occurred";
        setQueryError(errorMessage);
        toast.error(`Query failed: ${errorMessage}`);
      } finally {
        setIsExecuting(false);
      }
    },
    [
      isAuthenticated,
      user?.user_id,
      currentDatabaseId,
      validateQuery,
      uploadedFiles,
      saveQuery,
      selectedTable,
    ]
  );

  // Handle query save
  const handleQuerySave = useCallback(
    (queryText: string) => {
      if (!queryText.trim() || !user?.user_id) {
        toast.error("Cannot save empty query or user not authenticated");
        return;
      }

      try {
        saveQuery({
          id: Date.now().toString(),
          type: "file",
          query: queryText.trim(),
          userId: user.user_id,
          timestamp: new Date(),
        });
        toast.success("Query saved successfully!");
      } catch (error) {
        console.error("Failed to save query:", error);
        toast.error("Failed to save query");
      }
    },
    [user?.user_id, saveQuery]
  );

  // Handle query clear
  const handleQueryClear = useCallback(() => {
    setQuery("");
    setQueryResults([]);
    setQueryError(null);
    setSelectedTable(null); // Also clear selected table
  }, []);

  // Handle loading query from history
  const handleHistorySelect = useCallback((historyItem: any) => {
    setQuery(historyItem.query);
    setQueryResults(historyItem.results || []);
    // Note: Table selection would need to be stored in history metadata to restore it
    toast.success("Query loaded from history");
  }, []);

  // Loading state
  if (userLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Spinner size="lg" variant="accent-blue" className="mx-auto mb-4" />
          <p className="text-gray-600">Loading user data...</p>
        </div>
      </div>
    );
  }

  // Authentication required
  if (!isAuthenticated) {
    return (
      <div className="container mx-auto p-6">
        <Card className="max-w-md mx-auto">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-yellow-600 dark:text-yellow-400">
              <AlertCircle className="w-5 h-5" />
              Authentication Required
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              Please log in to use the file query feature.
            </p>
            <Button
              variant="outline"
              onClick={() => (window.location.href = "/auth")}
              className="w-full"
            >
              Go to Login
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <PageLayout background={["frame", "gridframe"]} maxWidth="7xl">
      {/* Welcome Header */}
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-green-400 mb-2">
          Hi there, {user?.username || ""}
        </h1>
        <p className="text-xl text-white">What would you like to know?</p>
      </div>

      {/* Top Controls Bar */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <span className="text-white">Use Table</span>
          <button
            onClick={() => setUseTable(!useTable)}
            className={`relative inline-flex h-6 w-11 items-center transition-colors ${
              useTable ? "bg-green-600" : "bg-gray-600"
            }`}
          >
            <span
              className={`inline-block h-4 w-4 transform bg-white transition-transform ${
                useTable ? "translate-x-6" : "translate-x-1"
              }`}
            />
          </button>
        </div>
        <Button
          variant="outline"
          className="border-green-400 text-green-400 hover:bg-green-400/10"
          onClick={() => {
            // Handle history button click
            console.log("History clicked");
          }}
        >
          <History className="w-4 h-4 mr-2" />
          History
        </Button>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Column - File Query */}
        <div className="space-y-6">
                {/* Query Form */}
          <div className="card-enhanced">
            <div className="card-content-enhanced">
              <div className="card-header-enhanced">
                <div className="card-title-enhanced flex items-center gap-2">
                  <Brain className="w-5 h-5 text-green-400" />
                  File Query
                </div>
              </div>
                    <div className="flex-1">
                      <FileQueryForm
                        onSubmit={handleQuerySubmit}
                        onSave={handleQuerySave}
                        onClear={handleQueryClear}
                        isLoading={isExecuting}
                        disabled={!isAuthenticated}
                      />
                      <div className="mt-4 flex justify-end">
                        <Button
                          onClick={() => setIsUploadModalOpen(true)}
                          className="bg-green-600 hover:bg-green-700 text-white"
                        >
                          <FileText className="w-4 h-4 mr-2" />
                          Upload Files
                        </Button>
                      </div>
                    </div>
            </div>
          </div>

          {/* Query Results */}
          {queryResults.length > 0 && (
            <div className="card-enhanced">
              <div className="card-content-enhanced">
                <div className="card-header-enhanced">
                  <div className="card-title-enhanced flex items-center gap-2">
                    <FileText className="w-5 h-5 text-green-400" />
                    Query Results
                  </div>
                </div>
                <div className="flex-1">
                  <FileResults
                    results={queryResults}
                    query={query}
                    isLoading={isExecuting}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Query Error */}
          {queryError && (
            <Card className="bg-red-900/20 border-red-500/30">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-red-400">
                  <AlertCircle className="w-5 h-5" />
                  Query Error
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="p-4 bg-red-900/30 border border-red-500/30">
                  <p className="text-red-300">{queryError}</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right Column - Connect Table */}
        <div className="space-y-6">
          {/* Connect Table Card */}
          <div className="card-enhanced">
            <div className="card-content-enhanced">
              <div className="card-header-enhanced">
                <div className="card-title-enhanced flex items-center gap-2">
                  <Database className="w-5 h-5 text-green-400" />
                  Connect Table
                </div>
              </div>
              <div className="space-y-4">
                {/* Table Selector - for specifying which table to query */}
                {useTable && (
                  <div className="space-y-3">
                    <TableSelector
                      databaseId={currentDatabaseId}
                      onTableSelect={(tableName) => {
                        setSelectedTable(tableName);
                        toast.success(`Selected table: ${tableName}`);
                      }}
                    />

                    {/* Selected Table Indicator */}
                    {selectedTable && (
                      <div className="flex items-center gap-2 text-green-400 p-2 bg-green-500/10">
                        <CheckCircle className="h-4 w-4" />
                        <span className="text-sm font-medium">
                          {selectedTable}
                        </span>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setSelectedTable(null)}
                          className="h-6 w-6 p-0 ml-auto text-green-400 hover:text-green-300"
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    )}
                  </div>
                )}

                {/* Table Usage Status */}
                <div className="text-center">
                  <div className="flex items-center justify-center gap-2 text-green-400 mb-2">
                    <Database className="h-4 w-4" />
                    <span className="text-sm font-medium">
                      {useTable ? "Table Mode Enabled" : "Table Mode Disabled"}
                    </span>
                  </div>
                  <p className="text-gray-400 text-xs">
                    {useTable
                      ? "Files will be processed with table names for structured queries"
                      : "Files will be processed without table names for general content search"}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Commands Section */}
      <div className="mt-12">
        <h3 className="text-xl font-semibold text-white mb-6">
          Recent Commands
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Sample recent commands - replace with actual data */}
          {[1, 2, 3, 4].map((index) => (
            <div key={index} className="card-enhanced">
              <div className="card-content-enhanced">
                <div className="space-y-2">
                  <ul className="text-sm text-slate-400 space-y-1">
                    <li>• last week</li>
                    <li>• this month</li>
                    <li>• yesterday</li>
                  </ul>
                  <div className="flex justify-center">
                    <div className="w-6 h-6 bg-green-400 flex items-center justify-center">
                      <span className="text-white text-xs">?</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Upload File Modal */}
      {isUploadModalOpen && (
        <div 
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setIsUploadModalOpen(false)}
        >
          <div 
            className="bg-slate-800 border border-slate-600 max-w-md w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="p-6 border-b border-slate-600">
              <h2 className="text-2xl font-bold text-white mb-2">Upload File</h2>
              <p className="text-slate-400 text-sm">
                Add User refund processes with configurable policy enforcement.
              </p>
            </div>

            {/* Modal Content */}
            <div className="p-6">
              {/* File Upload Component */}
              <FileUpload
                onFilesUploaded={handleFilesUploaded}
                onUploadStatusChange={handleUploadStatusChange}
                onTableUsageChange={handleTableUsageChange}
                disabled={!isAuthenticated}
              />
            </div>

            {/* Modal Footer */}
            <div className="p-6 border-t border-slate-600 flex gap-3 justify-end">
              <Button
                variant="outline"
                onClick={() => setIsUploadModalOpen(false)}
                className="border-slate-600 text-slate-300 hover:bg-slate-700"
              >
                Cancel
              </Button>
              <Button
                onClick={() => {
                  // Handle upload action
                  setIsUploadModalOpen(false);
                  toast.success("Files uploaded successfully!");
                }}
                className="bg-green-600 hover:bg-green-700 text-white"
              >
                Upload
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  setUploadedFiles([]);
                  toast.info("Files cleared");
                }}
                className="border-red-500 text-red-400 hover:bg-red-500/10"
              >
                Clear
              </Button>
            </div>
          </div>
        </div>
      )}
    </PageLayout>
  );
}
