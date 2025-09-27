"use client";

import React, { useState, useEffect, useCallback } from "react";
import Image from "next/image";
import { useQueryStore } from "@/store/query-store";
import { useTaskCreator } from "@/components/task-manager/TaskManagerProvider";
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
  QueryHistoryPanel,
} from "@/components/data-query";
import {
  FileQueryCard,
  FileQueryPageHeader,
  QuickSuggestions,
  TableSection,
  UseTableToggle,
} from "@/components/file-query";
import { EnhancedFileUploadModal } from "@/components/file-query/EnhancedFileUploadModal";
import { PageLayout, PageHeader } from "@/components/layout/PageLayout";
import { ContentWrapper } from "@/components/layout/ContentWrapper";
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
  const { createQueryTask, executeTask } = useTaskCreator();

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

      // Create a task for file query execution
      const taskId = createQueryTask(
        "File Query Execution",
        `Executing file query: "${queryText}"`,
        {
          query: queryText,
          mode: "file_query",
          selectedTable,
          useTable,
          fileIds: uploadedFiles
            .filter((file) => file.status === "completed")
            .map((file) => file.id),
        }
      );

      // Execute the task in background
      executeTask(
        taskId,
        async () => {
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
              
              return results;
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
            throw error;
          }
        }
      ).catch((error) => {
        console.error("Failed to execute file query:", error);
      }).finally(() => {
        setIsExecuting(false);
      });
    },
    [
      isAuthenticated,
      user?.user_id,
      currentDatabaseId,
      validateQuery,
      uploadedFiles,
      saveQuery,
      selectedTable,
      createQueryTask,
      executeTask,
      useTable,
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
    <PageLayout
      background={["frame", "gridframe"]}
      maxWidth="7xl"
      className="file-query-page min-h-screen flex flex-col justify-center py-6"
    >
      <div className="mt-2">
        <style
          dangerouslySetInnerHTML={{
            __html: `
            .file-query-page textarea {
              background: var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04)) !important;
            }
          `,
          }}
        />

      {/* Page Header - Only show when no query results */}
      {queryResults.length === 0 && !queryError && (
        <ContentWrapper className="mb-12">
            <div className="flex items-center justify-between">
              <div>
                <h1
                  className="text-4xl font-bold mb-2 block"
                  style={{
                    background:
                      "radial-gradient(70.83% 118.23% at 55.46% 50%, #0DAC5C 0%, #FFFFFF 84.18%)",
                    WebkitBackgroundClip: "text",
                    WebkitTextFillColor: "transparent",
                    backgroundClip: "text",
                    color: "transparent",
                    display: "block",
                    backgroundSize: "100% 100%",
                    backgroundRepeat: "no-repeat",
                  }}
                >
                  Hi there, {user?.username || ""}
                </h1>
                <p
                  className="text-xl block"
                  style={{
                    background:
                      "radial-gradient(70.83% 118.23% at 55.46% 50%, #0DAC5C 0%, #FFFFFF 84.18%)",
                    WebkitBackgroundClip: "text",
                    WebkitTextFillColor: "transparent",
                    backgroundClip: "text",
                    color: "transparent",
                    display: "block",
                    backgroundSize: "100% 100%",
                    backgroundRepeat: "no-repeat",
                  }}
                >
                  What would you like to know?
                </p>
              </div>
              <Button
                variant="outline"
                className="text-white flex items-center gap-2"
                style={{
                  background:
                    "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
                  border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
                  height: "48px",
                  minWidth: "64px",
                  borderRadius: "99px",
                }}
                onClick={() => {
                  // Handle history button click
                  console.log("History clicked");
                }}
              >
                <Image
                  src="/file-query/history.svg"
                  alt="History"
                  width={16}
                  height={16}
                  className="h-4 w-4"
                />
                History
              </Button>
            </div>
          </ContentWrapper>
        )}

        {/* Query Results - Now at the top */}
        {queryResults.length > 0 && (
          <ContentWrapper className="mb-12">
            <div className="query-content-gradient">
              <div className="p-4">
                <div className="flex items-center gap-2 mb-3">
                  <h3 className="text-white font-semibold text-lg">
                    Query: "{query}"
                  </h3>
                </div>
                
                {/* Fixed Top Separator */}
                <div
                  className="w-full border-bottom mb-4"
                  style={{
                    borderBottom: "1px solid var(--white-4, rgba(255, 255, 255, 0.04))",
                  }}
                ></div>
                
                {/* Scrollable Content Area */}
                <div className="max-h-[300px] overflow-y-auto">
                  <FileResults
                    results={queryResults}
                    query={query}
                    isLoading={isExecuting}
                  />
                </div>
                
                {/* Fixed Bottom Separator */}
                <div
                  className="w-full border-bottom mt-4"
                  style={{
                    borderBottom: "1px solid var(--white-4, rgba(255, 255, 255, 0.04))",
                  }}
                ></div>
                
                {/* Copy Button - After Separator */}
                <div className="flex justify-start mt-4">
                  <button
                    onClick={async () => {
                      try {
                        const allResultsText = queryResults
                          .map((result) => {
                            if (result.answer && result.answer.trim()) return result.answer;
                            if (result.content && result.content.trim()) return result.content;
                            if (result.text && result.text.trim()) return result.text;
                            return "No content available";
                          })
                          .join("\n\n");
                        
                        await navigator.clipboard.writeText(allResultsText);
                        toast.success("Results copied to clipboard!");
                      } catch (error) {
                        console.error("Failed to copy to clipboard:", error);
                        toast.error("Failed to copy to clipboard");
                      }
                    }}
                    className="flex items-center gap-2 text-green-400 hover:text-green-300 transition-colors cursor-pointer"
                    title="Copy all results to clipboard"
                  >
                    <Image
                      src="/file-query/copy.svg"
                      alt="Copy"
                      width={16}
                      height={16}
                      className="w-4 h-4"
                    />
                  </button>
                </div>
              </div>
            </div>
          </ContentWrapper>
        )}

      {/* Query Error - Also at the top */}
      {queryError && (
        <ContentWrapper className="mb-12">
            <div className="query-content-gradient max-h-[200px] overflow-y-auto">
              <div className="p-4">
                <div className="flex items-center gap-2 mb-3">
                  <h3 className="text-red-400 font-semibold text-lg">
                    Query Error
                  </h3>
                </div>
                <div className="p-4 bg-red-900/30 border border-red-500/30 rounded-lg">
                  <p className="text-red-300">{queryError}</p>
                </div>
              </div>
            </div>
          </ContentWrapper>
        )}

      {/* Table Toggle - Above Query Form */}
      <ContentWrapper className="mb-8">
        <UseTableToggle useTable={useTable} onToggle={setUseTable} />
      </ContentWrapper>

      {/* Main Content */}
      <ContentWrapper className="mb-16">
      <div className={`flex flex-col ${useTable ? 'lg:flex-row' : ''} mb-12 ${useTable ? 'lg:gap-4' : ''}`}>
          {/* Left Column - File Query */}
          <div className={`${useTable ? 'lg:flex-1' : 'w-full'}`}>
            <FileQueryCard
              query={query}
              setQuery={setQuery}
              isExecuting={isExecuting}
              onUploadClick={() => setIsUploadModalOpen(true)}
              onExecuteClick={() =>
                handleQuerySubmit(query, { answerStyle: "detailed" })
              }
            />
          </div>

          {/* Right Column - Connect Table */}
          {useTable && (
            <div className={`${useTable ? 'lg:w-80' : 'w-full'} ${useTable ? 'mt-4 lg:mt-0' : ''}`}>
                <TableSection
                  selectedTable={selectedTable}
                  onTableSelect={(tableName) => {
                    setSelectedTable(tableName);
                    toast.success(`Selected table: ${tableName}`);
                  }}
                  currentDatabaseId={currentDatabaseId}
                />
              </div>
            )}
          </div>
        </ContentWrapper>

      {/* Quick Suggestions Section - Only show when no query results */}
      {queryResults.length === 0 && !queryError && (
        <ContentWrapper className="mb-8">
            <QuickSuggestions onQuerySelect={setQuery} />
          </ContentWrapper>
        )}

        {/* Enhanced File Upload Modal */}
        <EnhancedFileUploadModal
          open={isUploadModalOpen}
          onOpenChange={setIsUploadModalOpen}
          onFilesUploaded={handleFilesUploaded}
          onUploadStatusChange={handleUploadStatusChange}
          disabled={!isAuthenticated}
        />
      </div>
    </PageLayout>
  );
}