import React, { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  File,
  Download,
  Copy,
  ChevronLeft,
  ChevronRight,
  FileText,
  Brain,
} from "lucide-react";
import { Spinner } from "@/components/ui/loading";
import { toast } from "sonner";
import Image from "next/image";

export interface FileQueryResult {
  id: string;
  answer?: string;
  confidence?: string | number;
  sources_used?: number;
  query?: string;
  content?: string;
  text?: string;
  source?: string;
  filename?: string;
  source_file?: string;
  source_title?: string;
  page_range?: string;
  document_number?: number;
  is_source?: boolean;
  sources?: any[];
  context_length?: number;
  prompt_length?: number;
  [key: string]: any; // Allow for additional properties
}

interface FileResultsProps {
  results: FileQueryResult[];
  query: string;
  isLoading?: boolean;
  className?: string;
}

export function FileResults({
  results,
  query,
  isLoading = false,
  className = "",
}: FileResultsProps) {
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);

  // Pagination calculations
  const totalPages = Math.ceil(results.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentResults = results.slice(startIndex, endIndex);

  // Handle page change
  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  // Handle items per page change
  const handleItemsPerPageChange = (items: number) => {
    setItemsPerPage(items);
    setCurrentPage(1); // Reset to first page
  };

  // Copy result to clipboard
  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast.success("Copied to clipboard");
    } catch (error) {
      toast.error("Failed to copy to clipboard");
    }
  };

  // Get all results text for copying
  const getAllResultsText = () => {
    return currentResults
      .map((result) => getResultContent(result))
      .join("\n\n");
  };

  // Export results
  const exportResults = () => {
    const csvContent = [
      ["Query", "Answer", "Confidence", "Sources Used"],
      ...results.map((result) => [
        query,
        result.answer || "",
        result.confidence || "",
        result.sources_used || 0,
      ]),
    ]
      .map((row) => row.map((field) => `"${field}"`).join(","))
      .join("\n");

    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `file-query-results-${
      new Date().toISOString().split("T")[0]
    }.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
    toast.success("Results exported successfully");
  };

  // Get result display content - prioritize answer over source content
  const getResultContent = (result: FileQueryResult) => {
    // First priority: actual answer from AI
    if (result.answer && result.answer.trim()) {
      return result.answer;
    }

    // Second priority: content field
    if (result.content && result.content.trim()) {
      return result.content;
    }

    // Third priority: text field
    if (result.text && result.text.trim()) {
      return result.text;
    }

    // Last resort: try other string fields
    const contentKeys = Object.keys(result).filter(
      (key) =>
        key !== "id" &&
        key !== "confidence" &&
        key !== "sources_used" &&
        key !== "query" &&
        key !== "filename" &&
        key !== "source" &&
        key !== "source_file" &&
        key !== "source_title" &&
        key !== "page_range" &&
        key !== "document_number" &&
        key !== "is_source" &&
        key !== "sources" &&
        key !== "context_length" &&
        key !== "prompt_length" &&
        typeof result[key] === "string" &&
        result[key] &&
        result[key].trim().length > 0
    );

    if (contentKeys.length > 0) {
      return result[contentKeys[0]];
    }

    return "No content available";
  };

  // Get result type and styling
  const getResultType = (result: FileQueryResult) => {
    if (result.is_source) {
      return {
        type: "source",
        icon: <FileText className="w-4 h-4 text-emerald-400" />,
        label: `Source ${result.document_number || "Document"}`,
        bgColor: "bg-emerald-900/20",
        borderColor: "border-emerald-400/30",
        textColor: "text-emerald-400",
      };
    }

    return {
      type: "answer",
      icon: <Brain className="w-4 h-4 text-emerald-400" />,
      label: "AI Answer",
      bgColor: "bg-emerald-900/20",
      borderColor: "border-emerald-400/30",
      textColor: "text-emerald-400",
    };
  };

  // Get confidence level styling
  const getConfidenceStyle = (confidence: string | number | undefined) => {
    if (!confidence) return "border-gray-400/30 text-gray-400";

    const conf =
      typeof confidence === "string" ? confidence.toLowerCase() : confidence;

    if (conf === "high" || (typeof conf === "number" && conf > 0.8)) {
      return "border-green-400/30 text-green-400";
    } else if (conf === "medium" || (typeof conf === "number" && conf > 0.5)) {
      return "border-yellow-400/30 text-yellow-400";
    } else {
      return "border-red-400/30 text-red-400";
    }
  };


  if (isLoading) {
    return (
      <div className={`${className} text-center py-8`}>
        <Spinner size="lg" variant="accent-green" className="mx-auto mb-4" />
        <p className="text-white font-medium">Processing your query...</p>
        <p className="text-gray-400 text-sm">This may take a few moments</p>
      </div>
    );
  }

  if (!results || results.length === 0) {
    return (
      <div className={`${className} text-center py-8`}>
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-800/50 flex items-center justify-center border border-gray-600/30">
          <AlertCircle className="w-8 h-8 text-yellow-400" />
        </div>
        <p className="text-white font-medium mb-2">No results found</p>
        <p className="text-gray-400 text-sm max-w-md mx-auto">
          No results found for your query. Try rephrasing your question or
          uploading different files.
        </p>
      </div>
    );
  }

  return (
    <div className={className}>
      {/* Response Content */}
      <div className="space-y-4">
        {currentResults.map(
          (result, index) => {
            const resultId = result.id || `result-${index}`;
            const content = getResultContent(result);
            const displayContent = content;

            return (
              <div key={resultId} className="text-white">
                <div className="text-white leading-relaxed whitespace-pre-wrap">
                  {displayContent}
                </div>

                {/* Source information if available */}
                {(result.source_file || result.source_title) && (
                  <div className="mt-2 text-sm text-green-400">
                    {result.source_file && (
                      <span className="underline cursor-pointer hover:text-green-300">
                        {result.source_file}
                      </span>
                    )}
                    {result.source_title && (
                      <span className="underline cursor-pointer hover:text-green-300 ml-2">
                        {result.source_title}
                      </span>
                    )}
                  </div>
                )}
              </div>
            );
          }
        )}
      </div>

    </div>
  );
}
