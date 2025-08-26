import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { File, Download, Copy, ChevronLeft, ChevronRight, FileText, Brain } from 'lucide-react';
import { toast } from 'sonner';

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

export function FileResults({ results, query, isLoading = false, className = "" }: FileResultsProps) {
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());

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
      toast.success('Copied to clipboard');
    } catch (error) {
      toast.error('Failed to copy to clipboard');
    }
  };

  // Export results
  const exportResults = () => {
    const csvContent = [
      ['Query', 'Answer', 'Confidence', 'Sources Used'],
      ...results.map(result => [
        query,
        result.answer || '',
        result.confidence || '',
        result.sources_used || 0
      ])
    ].map(row => row.map(field => `"${field}"`).join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `file-query-results-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
    toast.success('Results exported successfully');
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
    const contentKeys = Object.keys(result).filter(key => 
      key !== 'id' && 
      key !== 'confidence' && 
      key !== 'sources_used' && 
      key !== 'query' &&
      key !== 'filename' &&
      key !== 'source' &&
      key !== 'source_file' &&
      key !== 'source_title' &&
      key !== 'page_range' &&
      key !== 'document_number' &&
      key !== 'is_source' &&
      key !== 'sources' &&
      key !== 'context_length' &&
      key !== 'prompt_length' &&
      typeof result[key] === 'string' &&
      result[key] && 
      result[key].trim().length > 0
    );
    
    if (contentKeys.length > 0) {
      return result[contentKeys[0]];
    }
    
    return 'No content available';
  };

  // Get result type and styling
  const getResultType = (result: FileQueryResult) => {
    if (result.is_source) {
      return {
        type: 'source',
        icon: <FileText className="w-4 h-4 text-blue-400" />,
        label: `Source ${result.document_number || 'Document'}`,
        bgColor: 'bg-blue-900/20',
        borderColor: 'border-blue-400/30',
        textColor: 'text-blue-400'
      };
    }
    
    return {
      type: 'answer',
      icon: <Brain className="w-4 h-4 text-purple-400" />,
      label: 'AI Answer',
      bgColor: 'bg-purple-900/20',
      borderColor: 'border-purple-400/30',
      textColor: 'text-purple-400'
    };
  };

  // Get confidence level styling
  const getConfidenceStyle = (confidence: string | number | undefined) => {
    if (!confidence) return 'border-gray-400/30 text-gray-400';
    
    const conf = typeof confidence === 'string' ? confidence.toLowerCase() : confidence;
    
    if (conf === 'high' || (typeof conf === 'number' && conf > 0.8)) {
      return 'border-green-400/30 text-green-400';
    } else if (conf === 'medium' || (typeof conf === 'number' && conf > 0.5)) {
      return 'border-yellow-400/30 text-yellow-400';
    } else {
      return 'border-red-400/30 text-red-400';
    }
  };

  // Toggle expanded state
  const toggleExpanded = (id: string) => {
    setExpandedItems(prev => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      return newSet;
    });
  };

  if (isLoading) {
    return (
      <div className={`${className} text-center py-8`}>
        <Loader2 className="h-8 w-8 mx-auto text-purple-400 animate-spin mb-4" />
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
          No results found for your query. Try rephrasing your question or uploading different files.
        </p>
      </div>
    );
  }

  return (
    <div className={className}>
        {/* Results List */}
        <div className="space-y-4 mb-6">
          {currentResults.map((result, index) => {
            const resultId = result.id || `result-${index}`;
            const isExpanded = expandedItems.has(resultId);
            const content = getResultContent(result);
            const shouldTruncate = content.length > 300;
            const displayContent = isExpanded || !shouldTruncate 
              ? content 
              : content.substring(0, 300) + '...';
            
            const resultType = getResultType(result);

            return (
              <div
                key={resultId}
                className={`group ${resultType.bgColor} border ${resultType.borderColor} rounded-lg hover:bg-gray-700/30 hover:border-purple-400/30 transition-all duration-200`}
              >
                <div className="p-4">
                  {/* Result Header */}
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      {resultType.icon}
                      <span className={`font-medium ${resultType.textColor}`}>
                        {resultType.label}
                      </span>
                      {result.source_file && (
                        <Badge variant="outline" className="border-gray-400/30 text-gray-400 text-xs">
                          📄 {result.source_file}
                        </Badge>
                      )}
                      {result.source_title && (
                        <Badge variant="outline" className="border-green-400/30 text-green-400 text-xs">
                          📋 {result.source_title}
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      {result.confidence && (
                        <Badge variant="outline" className={`text-xs ${getConfidenceStyle(result.confidence)}`}>
                          {typeof result.confidence === 'number' 
                            ? `${Math.round(result.confidence * 100)}%` 
                            : result.confidence}
                        </Badge>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => copyToClipboard(content)}
                        className="opacity-0 group-hover:opacity-100 transition-opacity text-purple-400 hover:bg-purple-400/10"
                      >
                        <Copy className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                  
                  {/* Result Content */}
                  <div className="bg-gray-900/50 border border-gray-600/30 rounded-lg p-4 mb-3">
                    <div className="text-white leading-relaxed whitespace-pre-wrap">
                      {displayContent}
                    </div>
                    {shouldTruncate && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => toggleExpanded(resultId)}
                        className="mt-2 text-purple-400 hover:bg-purple-400/10 p-0 h-auto"
                      >
                        {isExpanded ? 'Show less' : 'Show more'}
                      </Button>
                    )}
                  </div>
                  
                  {/* Metadata */}
                  <div className="flex items-center gap-4 text-xs text-gray-400">
                    {result.sources_used !== undefined && (
                      <span className="flex items-center gap-1">
                        📚 {result.sources_used} sources
                      </span>
                    )}
                    {result.page_range && (
                      <span className="flex items-center gap-1">
                        📄 Page {result.page_range}
                      </span>
                    )}
                    {result.context_length && (
                      <span className="flex items-center gap-1">
                        📊 {result.context_length.toLocaleString()} chars context
                      </span>
                    )}
                    <span className="flex items-center gap-1">
                      📝 {content.length} characters
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700">
            <div className="text-sm text-gray-700 dark:text-gray-300">
              Showing {startIndex + 1} to {Math.min(endIndex, results.length)} of {results.length} results
            </div>
            
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => handlePageChange(currentPage - 1)}
                disabled={currentPage === 1}
              >
                <ChevronLeft className="w-4 h-4" />
                Previous
              </Button>
              
              <span className="text-sm text-gray-700 dark:text-gray-300">
                Page {currentPage} of {totalPages}
              </span>
              
              <Button
                variant="outline"
                size="sm"
                onClick={() => handlePageChange(currentPage + 1)}
                disabled={currentPage === totalPages}
              >
                Next
                <ChevronRight className="w-4 h-4" />
              </Button>
            </div>
          </div>
        )}

        {/* Items Per Page */}
        <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <span>Show:</span>
          <select
            value={itemsPerPage}
            onChange={(e) => handleItemsPerPageChange(Number(e.target.value))}
            className="border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-800"
          >
            <option value={5}>5</option>
            <option value={10}>10</option>
            <option value={20}>20</option>
            <option value={50}>50</option>
          </select>
          <span>per page</span>
        </div>
      </div>
    );
} 