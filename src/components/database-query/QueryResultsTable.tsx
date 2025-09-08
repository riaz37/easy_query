"use client";

import React, { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ChevronLeft, ChevronRight, Search, Filter, Download } from "lucide-react";

interface QueryResultsTableProps {
  data: any[];
  columns: string[];
}

export function QueryResultsTable({ data, columns }: QueryResultsTableProps) {
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [searchTerm, setSearchTerm] = useState("");
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");

  // Filter and search data
  const filteredData = useMemo(() => {
    let filtered = data;
    
    if (searchTerm) {
      filtered = filtered.filter((row) =>
        Object.values(row).some((value) =>
          String(value).toLowerCase().includes(searchTerm.toLowerCase())
        )
      );
    }
    
    return filtered;
  }, [data, searchTerm]);

  // Sort data
  const sortedData = useMemo(() => {
    if (!sortColumn) return filteredData;
    
    return [...filteredData].sort((a, b) => {
      const aValue = a[sortColumn];
      const bValue = b[sortColumn];
      
      if (aValue === null || aValue === undefined) return 1;
      if (bValue === null || bValue === undefined) return -1;
      
      if (typeof aValue === "number" && typeof bValue === "number") {
        return sortDirection === "asc" ? aValue - bValue : bValue - aValue;
      }
      
      const aString = String(aValue).toLowerCase();
      const bString = String(bValue).toLowerCase();
      
      if (sortDirection === "asc") {
        return aString.localeCompare(bString);
      } else {
        return bString.localeCompare(aString);
      }
    });
  }, [filteredData, sortColumn, sortDirection]);

  // Pagination
  const totalPages = Math.ceil(sortedData.length / pageSize);
  const startIndex = (currentPage - 1) * pageSize;
  const endIndex = startIndex + pageSize;
  const currentData = sortedData.slice(startIndex, endIndex);

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortColumn(column);
      setSortDirection("asc");
    }
    setCurrentPage(1);
  };

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  const handlePageSizeChange = (size: number) => {
    setPageSize(size);
    setCurrentPage(1);
  };

  const clearSearch = () => {
    setSearchTerm("");
    setCurrentPage(1);
  };

  if (!data || data.length === 0) {
    return (
      <Card className="bg-gray-900/50 border-emerald-400/30">
        <CardContent className="pt-12 pb-12 text-center">
          <div className="text-gray-400">
            No data to display
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-gray-900/50 border-emerald-400/30">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-emerald-400 flex items-center gap-2">
            <Filter className="w-5 h-5" />
            Query Results
          </CardTitle>
          <Badge variant="outline" className="border-emerald-400/30 text-emerald-400">
            {data.length} results
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Search and Controls */}
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2 flex-1 max-w-md">
            <Search className="w-4 h-4 text-gray-400" />
            <Input
              placeholder="Search in results..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentPage(1);
              }}
              className="bg-gray-800/50 border-emerald-400/30 text-white placeholder:text-gray-400"
            />
            {searchTerm && (
              <Button
                variant="ghost"
                size="sm"
                onClick={clearSearch}
                className="text-gray-400 hover:text-white"
              >
                Clear
              </Button>
            )}
          </div>
          
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">Page size:</span>
            <select
              value={pageSize}
              onChange={(e) => handlePageSizeChange(Number(e.target.value))}
              className="bg-gray-800/50 border border-emerald-400/30 text-white rounded px-2 py-1 text-sm"
            >
              <option value={5}>5</option>
              <option value={10}>10</option>
              <option value={20}>20</option>
              <option value={50}>50</option>
            </select>
          </div>
        </div>

        {/* Results Table */}
        <div className="rounded-lg border border-emerald-400/30 overflow-hidden">
          <Table>
            <TableHeader className="bg-gray-800/50">
              <TableRow className="hover:bg-transparent">
                {columns.map((column) => (
                  <TableHead
                    key={column}
                    className="text-emerald-400 font-medium cursor-pointer hover:bg-gray-700/50 transition-colors"
                    onClick={() => handleSort(column)}
                  >
                    <div className="flex items-center gap-2">
                      {column}
                      {sortColumn === column && (
                        <Badge variant="outline" size="sm" className="text-xs">
                          {sortDirection === "asc" ? "↑" : "↓"}
                        </Badge>
                      )}
                    </div>
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {currentData.map((row, rowIndex) => (
                <TableRow
                  key={rowIndex}
                  className="hover:bg-gray-700/30 border-b border-gray-700/50"
                >
                  {columns.map((column) => (
                    <TableCell key={column} className="text-gray-300">
                      <div className="max-w-xs truncate" title={String(row[column] || "")}>
                        {row[column] !== null && row[column] !== undefined
                          ? String(row[column])
                          : <span className="text-gray-500">-</span>
                        }
                      </div>
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>

        {/* Pagination */}
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-400">
            Showing {startIndex + 1}-{Math.min(endIndex, sortedData.length)} of {sortedData.length} results
          </div>
          
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => handlePageChange(currentPage - 1)}
              disabled={currentPage === 1}
              className="border-emerald-400/30 text-emerald-400 hover:bg-emerald-400/10"
            >
              <ChevronLeft className="w-4 h-4 mr-1" />
              Previous
            </Button>
            
            <div className="flex items-center gap-1">
              {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                let pageNum;
                if (totalPages <= 5) {
                  pageNum = i + 1;
                } else if (currentPage <= 3) {
                  pageNum = i + 1;
                } else if (currentPage >= totalPages - 2) {
                  pageNum = totalPages - 4 + i;
                } else {
                  pageNum = currentPage - 2 + i;
                }
                
                return (
                  <Button
                    key={pageNum}
                    variant={currentPage === pageNum ? "default" : "outline"}
                    size="sm"
                    onClick={() => handlePageChange(pageNum)}
                    className={
                      currentPage === pageNum
                        ? "bg-emerald-600 text-white"
                        : "border-emerald-400/30 text-emerald-400 hover:bg-emerald-400/10"
                    }
                  >
                    {pageNum}
                  </Button>
                );
              })}
            </div>
            
            <Button
              variant="outline"
              size="sm"
              onClick={() => handlePageChange(currentPage + 1)}
              disabled={currentPage === totalPages}
              className="border-emerald-400/30 text-emerald-400 hover:bg-emerald-400/10"
            >
              Next
              <ChevronRight className="w-4 h-4 ml-1" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 