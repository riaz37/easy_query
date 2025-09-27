"use client";

import React, { useState, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { 
  MoreVertical, 
  ChevronDown, 
  ChevronLeft, 
  ChevronRight, 
  ChevronsLeft, 
  ChevronsRight,
  Search
} from "lucide-react";
import { cn } from "@/lib/utils";

interface QueryResultsTableProps {
  data: any[];
  columns: string[];
}

export function QueryResultsTable({ data, columns }: QueryResultsTableProps) {
  const [selectedRows, setSelectedRows] = useState<Set<number>>(new Set());
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  const [currentPage, setCurrentPage] = useState(1);
  const [rowsPerPage, setRowsPerPage] = useState(5);
  const [searchTerm, setSearchTerm] = useState("");

  // Helper function to format dates
  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString);
      if (isNaN(date.getTime())) return dateString; // Return original if not a valid date
      
      const options: Intl.DateTimeFormatOptions = {
        day: 'numeric',
        month: 'long',
        year: 'numeric'
      };
      return date.toLocaleDateString('en-US', options);
    } catch (error) {
      return dateString; // Return original if formatting fails
    }
  };

  // Helper function to check if a value looks like a date
  const isDateString = (value: any) => {
    if (typeof value !== 'string') return false;
    const date = new Date(value);
    return !isNaN(date.getTime()) && value.match(/\d{1,2}[-\/]\d{1,2}[-\/]\d{2,4}/);
  };

  // Process data for display
  const processedData = data.map((row, index) => {
    const processedRow: any = { id: index, originalRow: row };
    
    // Format dates in the row data
    Object.keys(row).forEach(key => {
      const value = row[key];
      if (isDateString(value)) {
        processedRow[key] = formatDate(value);
      } else {
        processedRow[key] = value;
      }
    });
    
    return processedRow;
  });

  // Filter and search data
  const filteredData = useMemo(() => {
    let filtered = processedData;
    
    if (searchTerm) {
      filtered = filtered.filter((row) =>
        Object.values(row).some((value) =>
          String(value).toLowerCase().includes(searchTerm.toLowerCase())
        )
      );
    }
    
    return filtered;
  }, [processedData, searchTerm]);

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
  const totalPages = Math.ceil(sortedData.length / rowsPerPage);
  const startIndex = (currentPage - 1) * rowsPerPage;
  const endIndex = startIndex + rowsPerPage;
  const currentData = sortedData.slice(startIndex, endIndex);

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedRows(new Set(currentData.map(row => row.id)));
    } else {
      setSelectedRows(new Set());
    }
  };

  const handleSelectRow = (rowId: number, checked: boolean) => {
    const newSelected = new Set(selectedRows);
    if (checked) {
      newSelected.add(rowId);
    } else {
      newSelected.delete(rowId);
    }
    setSelectedRows(newSelected);
  };

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortColumn(column);
      setSortDirection("asc");
    }
  };

  const getSortIcon = (column: string) => {
    if (sortColumn !== column) {
      return <ChevronDown className="h-4 w-4 text-gray-400" />;
    }
    return sortDirection === "asc" 
      ? <ChevronDown className="h-4 w-4 text-white" />
      : <ChevronDown className="h-4 w-4 text-white rotate-180" />;
  };

  if (!data || data.length === 0) {
    return (
      <div className="text-center py-8">
        <div className="text-white">
          No data to display
        </div>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <div className="rounded-t-xl overflow-hidden">
        <table className="w-full">
          <thead>
            <tr 
              style={{
                background: "var(--components-Table-Head-filled, rgba(145, 158, 171, 0.08))",
                borderRadius: "12px 12px 0 0"
              }}
            >
              <th className="px-6 py-4 text-left rounded-tl-xl">
                <Checkbox
                  checked={selectedRows.size === currentData.length && currentData.length > 0}
                  onCheckedChange={handleSelectAll}
                  className="border-white/40 data-[state=checked]:bg-emerald-500 data-[state=checked]:border-emerald-500"
                />
              </th>
              {columns.map((column) => (
                <th 
                  key={column}
                  className="px-6 py-4 text-left cursor-pointer hover:bg-white/5 transition-colors"
                  onClick={() => handleSort(column)}
                >
                  <div className="flex items-center gap-2 text-white font-medium text-sm">
                    {column}
                  </div>
                </th>
              ))}
              <th className="px-6 py-4 text-right text-white font-medium text-sm rounded-tr-xl">
                Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {currentData.length === 0 ? (
              <tr>
                <td colSpan={columns.length + 2} className="px-6 py-8 text-center text-white">
                  No data found
                </td>
              </tr>
            ) : (
              currentData.map((row) => (
                <tr 
                  key={row.id} 
                  className="border-b border-white/10 hover:bg-white/5 transition-colors"
                >
                  <td className="px-6 py-4">
                    <Checkbox
                      checked={selectedRows.has(row.id)}
                      onCheckedChange={(checked) => handleSelectRow(row.id, checked as boolean)}
                      className="border-white/40 data-[state=checked]:bg-emerald-500 data-[state=checked]:border-emerald-500"
                    />
                  </td>
                  {columns.map((column) => (
                    <td key={column} className="px-6 py-4 text-white">
                      <div className="max-w-xs truncate" title={String(row[column] || "")}>
                        {row[column] !== null && row[column] !== undefined
                          ? String(row[column])
                          : <span className="text-gray-500">-</span>
                        }
                      </div>
                    </td>
                  ))}
                  <td className="px-6 py-4 text-right">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-white hover:bg-white/10"
                    >
                      <MoreVertical className="h-4 w-4" />
                    </Button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination and Selection Status */}
      <div className="px-6 py-4 flex items-center justify-between">
        {/* Left: Rows Selected */}
        <div className="flex items-center">
          <p className="text-white text-sm">
            {selectedRows.size} of {sortedData.length} Row(s) Selected
          </p>
        </div>
        
        {/* Right side: Rows per page and Page Info */}
        <div className="flex items-center gap-6">
          {/* Rows per page */}
          <div className="flex items-center gap-2">
            <span className="text-white text-sm">Rows per page:</span>
            <Select value={rowsPerPage.toString()} onValueChange={(value) => setRowsPerPage(Number(value))}>
              <SelectTrigger 
                className="bg-white/10 text-white px-2 py-1 text-sm focus:outline-none focus:ring-0"
                style={{ 
                  outline: 'none',
                  borderRadius: "99px",
                  width: "auto",
                  minWidth: "60px",
                  border: "1px solid var(--components-button-outlined, rgba(145, 158, 171, 0.32))"
                }}
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="modal-select-content-enhanced">
                <SelectItem value="5" className="dropdown-item">5</SelectItem>
                <SelectItem value="10" className="dropdown-item">10</SelectItem>
                <SelectItem value="25" className="dropdown-item">25</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          {/* Page Info and Controls */}
          <div className="flex items-center gap-2">
            <span className="text-white text-sm">
              Page {currentPage} of {totalPages}
            </span>
            <div className="flex gap-1">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setCurrentPage(1)}
                disabled={currentPage === 1}
                className="text-white disabled:opacity-50"
                style={{
                  borderRadius: "999px"
                }}
                onMouseEnter={(e) => {
                  if (!e.currentTarget.disabled) {
                    e.currentTarget.style.background = "rgba(255, 255, 255, 0.04)";
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "transparent";
                }}
              >
                <ChevronsLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setCurrentPage(currentPage - 1)}
                disabled={currentPage === 1}
                className="text-white disabled:opacity-50"
                style={{
                  borderRadius: "999px"
                }}
                onMouseEnter={(e) => {
                  if (!e.currentTarget.disabled) {
                    e.currentTarget.style.background = "rgba(255, 255, 255, 0.04)";
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "transparent";
                }}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setCurrentPage(currentPage + 1)}
                disabled={currentPage === totalPages}
                className="text-white disabled:opacity-50"
                style={{
                  borderRadius: "999px"
                }}
                onMouseEnter={(e) => {
                  if (!e.currentTarget.disabled) {
                    e.currentTarget.style.background = "rgba(255, 255, 255, 0.04)";
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "transparent";
                }}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setCurrentPage(totalPages)}
                disabled={currentPage === totalPages}
                className="text-white disabled:opacity-50"
                style={{
                  borderRadius: "999px"
                }}
                onMouseEnter={(e) => {
                  if (!e.currentTarget.disabled) {
                    e.currentTarget.style.background = "rgba(255, 255, 255, 0.04)";
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "transparent";
                }}
              >
                <ChevronsRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 