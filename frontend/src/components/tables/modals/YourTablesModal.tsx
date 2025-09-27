"use client";

import React, { useState, useCallback, useEffect, useMemo } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Checkbox } from "@/components/ui/checkbox";
import {
  XIcon,
  Columns,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Plus,
  MoreVertical,
  Eye,
  Edit,
  Trash2,
  Table as TableIcon,
} from "lucide-react";
import { UserTablesResponse, UserTable } from "@/types/api";
import { useNewTable } from "@/lib/hooks/use-new-table";
import { toast } from "sonner";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface YourTablesModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  userId?: string;
  onRefresh: () => void;
  onCreateTable: () => void;
}

type SortableColumn = "table_name" | "full_name" | "schema_name" | "columns";

export function YourTablesModal({
  open,
  onOpenChange,
  userId,
  onRefresh,
  onCreateTable,
}: YourTablesModalProps) {
  const [userTables, setUserTables] = useState<UserTable[]>([]);
  const [selectedTable, setSelectedTable] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [tableLoadError, setTableLoadError] = useState<string | null>(null);
  const [lastTablesUpdate, setLastTablesUpdate] = useState<Date | null>(null);
  const [selectedRows, setSelectedRows] = useState<Set<string>>(new Set());
  const [sortColumn, setSortColumn] = useState<SortableColumn | null>(null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  const [currentPage, setCurrentPage] = useState(1);
  const [rowsPerPage, setRowsPerPage] = useState(5);

  const { getUserTables } = useNewTable();

  // Transform user table data with memoization
  const transformedTables = useMemo(() => {
    return userTables.map(table => ({
      table_name: table.table_name,
      full_name: table.table_full_name,
      schema_name: table.schema_name || 'dbo',
      columns: table.table_schema?.columns?.map(col => ({
        column_name: col.name,
        data_type: col.type,
        is_nullable: !col.is_required,
        is_primary: col.is_primary || false,
        is_foreign: col.is_foreign || false,
        max_length: col.max_length || null,
      })) || []
    }));
  }, [userTables]);

  // Sort tables with memoization
  const sortedTables = useMemo(() => {
    if (!sortColumn) return transformedTables;

    return [...transformedTables].sort((a, b) => {
      let aValue: string | number;
      let bValue: string | number;

      switch (sortColumn) {
        case "columns":
          aValue = a.columns.length;
          bValue = b.columns.length;
          break;
        default:
          aValue = a[sortColumn] || "";
          bValue = b[sortColumn] || "";
      }

      if (typeof aValue === "string" && typeof bValue === "string") {
        aValue = aValue.toLowerCase();
        bValue = bValue.toLowerCase();
      }

      if (sortDirection === "asc") {
        return aValue < bValue ? -1 : aValue > bValue ? 1 : 0;
      } else {
        return aValue > bValue ? -1 : aValue < bValue ? 1 : 0;
      }
    });
  }, [transformedTables, sortColumn, sortDirection]);

  // Pagination with memoization
  const paginatedData = useMemo(() => {
    const totalPages = Math.ceil(sortedTables.length / rowsPerPage);
    const startIndex = (currentPage - 1) * rowsPerPage;
    const endIndex = startIndex + rowsPerPage;
    const currentTables = sortedTables.slice(startIndex, endIndex);

    return { totalPages, currentTables };
  }, [sortedTables, currentPage, rowsPerPage]);

  // Fetch user tables from API
  const fetchUserTables = useCallback(async () => {
    if (!userId) return;
    
    setLoading(true);
    setTableLoadError(null);
    try {
      console.log("Loading user tables for userId:", userId);
      const response = await getUserTables(userId);
      console.log("User tables response:", response);
      
      if (response && response.tables && Array.isArray(response.tables)) {
        setUserTables(response.tables);
        setLastTablesUpdate(new Date());
        
        if (response.tables.length > 0) {
          toast.success(`Loaded ${response.tables.length} user table(s)`);
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
      setLoading(false);
    }
  }, [userId, getUserTables]);

  // Fetch user tables on component mount
  useEffect(() => {
    if (open && userId) {
      fetchUserTables();
    }
  }, [open, userId, fetchUserTables]);

  // Reset pagination when data changes
  useEffect(() => {
    setCurrentPage(1);
    setSelectedRows(new Set());
  }, [sortedTables.length, rowsPerPage]);

  const handleSelectAll = useCallback((checked: boolean) => {
    if (checked) {
      setSelectedRows(new Set(paginatedData.currentTables.map(table => table.table_name)));
    } else {
      setSelectedRows(new Set());
    }
  }, [paginatedData.currentTables]);

  const handleSelectRow = useCallback((tableName: string, checked: boolean) => {
    setSelectedRows(prev => {
      const newSelected = new Set(prev);
      if (checked) {
        newSelected.add(tableName);
      } else {
        newSelected.delete(tableName);
      }
      return newSelected;
    });
  }, []);

  const handleSort = useCallback((column: SortableColumn) => {
    if (sortColumn === column) {
      setSortDirection(prev => prev === "asc" ? "desc" : "asc");
    } else {
      setSortColumn(column);
      setSortDirection("asc");
    }
  }, [sortColumn]);

  const getSortIcon = useCallback((column: SortableColumn) => {
    if (sortColumn !== column) {
      return <ChevronDown className="h-4 w-4 text-gray-400" />;
    }
    return sortDirection === "asc" 
      ? <ChevronDown className="h-4 w-4 text-white" />
      : <ChevronDown className="h-4 w-4 text-white rotate-180" />;
  }, [sortColumn, sortDirection]);

  const handlePageChange = useCallback((page: number) => {
    setCurrentPage(page);
  }, []);

  const handleRowsPerPageChange = useCallback((value: number) => {
    setRowsPerPage(value);
    setCurrentPage(1);
  }, []);

  const handleCreateTableClick = useCallback(() => {
    onOpenChange(false);
    onCreateTable();
  }, [onOpenChange, onCreateTable]);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent 
        className="p-0 border-0 bg-transparent" 
        showCloseButton={false}
        style={{
          width: '1000px',
          maxWidth: '1000px',
          maxHeight: '90vh',
        }}
      >
        <div className="modal-enhanced">
          <div className="modal-content-enhanced">
            <DialogHeader className="modal-header-enhanced">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <DialogTitle className="modal-title-enhanced">
                    Your Database Tables
                  </DialogTitle>
                  <p className="modal-description-enhanced">
                    View and manage all your created database tables
                  </p>
                </div>
                <div className="flex items-center gap-2">
                <button
                  onClick={() => onOpenChange(false)}
                  className="modal-close-button cursor-pointer"
                >
                  <XIcon className="h-5 w-5" />
                </button>
                </div>
              </div>
            </DialogHeader>

            <div className="modal-form-content overflow-y-auto max-h-[calc(90vh-200px)]">
              {/* Error Display */}
                {tableLoadError && (
                <Alert className="mb-6 border-red-500/50 bg-red-900/20">
                    <AlertDescription className="text-red-300">
                      {tableLoadError}
                    </AlertDescription>
                  </Alert>
                )}

              {/* Loading State */}
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-400 mx-auto mb-4" />
                    <p className="text-slate-400">Loading tables...</p>
                  </div>
                </div>
              ) : transformedTables.length === 0 ? (
                <div className="text-center py-12">
                  <TableIcon className="h-12 w-12 text-slate-500 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-slate-300 mb-2">
                    No tables created yet
                  </h3>
                  <p className="text-slate-400 mb-6">
                    Create your first table to get started
                  </p>
                  <Button
                    onClick={handleCreateTableClick}
                    className="modal-button-primary cursor-pointer"
                  >
                      <Plus className="h-4 w-4 mr-2" />
                    Create Table
                    </Button>
                </div>
              ) : (
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
                              checked={selectedRows.size === paginatedData.currentTables.length && paginatedData.currentTables.length > 0}
                              onCheckedChange={handleSelectAll}
                              className="border-white/40 data-[state=checked]:bg-emerald-500 data-[state=checked]:border-emerald-500"
                            />
                          </th>
                          <th 
                            className="px-6 py-4 text-left cursor-pointer hover:bg-white/5 transition-colors"
                            onClick={() => handleSort("table_name")}
                          >
                            <div className="flex items-center gap-2 text-white font-medium text-sm">
                              Table Name
                            </div>
                          </th>
                          <th 
                            className="px-6 py-4 text-left cursor-pointer hover:bg-white/5 transition-colors"
                            onClick={() => handleSort("full_name")}
                          >
                            <div className="flex items-center gap-2 text-white font-medium text-sm">
                              Full Name
                            </div>
                          </th>
                          <th 
                            className="px-6 py-4 text-left cursor-pointer hover:bg-white/5 transition-colors"
                            onClick={() => handleSort("schema_name")}
                          >
                            <div className="flex items-center gap-2 text-white font-medium text-sm">
                              Schema
                          </div>
                          </th>
                          <th 
                            className="px-6 py-4 text-left cursor-pointer hover:bg-white/5 transition-colors"
                            onClick={() => handleSort("columns")}
                          >
                            <div className="flex items-center gap-2 text-white font-medium text-sm">
                              Columns
                        </div>
                          </th>
                          <th className="px-6 py-4 text-right text-white font-medium text-sm rounded-tr-xl">
                            Actions
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {paginatedData.currentTables.map((table) => (
                          <tr 
                            key={table.table_name} 
                            className="border-b border-white/10 hover:bg-white/5 transition-colors"
                          >
                            <td className="px-6 py-4">
                              <Checkbox
                                checked={selectedRows.has(table.table_name)}
                                onCheckedChange={(checked) => handleSelectRow(table.table_name, checked as boolean)}
                                className="border-white/40 data-[state=checked]:bg-emerald-500 data-[state=checked]:border-emerald-500"
                              />
                            </td>
                            <td className="px-6 py-4 text-white font-medium">
                              {table.table_name}
                            </td>
                            <td className="px-6 py-4 text-white">
                              <span className="font-mono text-sm">{table.full_name}</span>
                            </td>
                            <td className="px-6 py-4 text-white">
                          <Badge variant="outline" className="border-slate-500 text-slate-300">
                                {table.schema_name}
                          </Badge>
                            </td>
                            <td className="px-6 py-4 text-white">
                            <Badge variant="outline" className="border-green-500 text-green-400">
                                {table.columns.length} columns
                            </Badge>
                            </td>
                            <td className="px-6 py-4 text-right">
                              <DropdownMenu>
                                <DropdownMenuTrigger asChild>
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    className="text-white hover:bg-white/10 transition-colors duration-200"
                                  >
                                    <MoreVertical className="h-4 w-4" />
                                  </Button>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent 
                                  align="end" 
                                  className="w-48 modal-select-content-enhanced border-emerald-500/20"
                                >
                                  <DropdownMenuItem 
                                    onClick={() => {
                                      setSelectedTable(table.table_name);
                                      toast.info(`Viewing table: ${table.table_name}`);
                                    }}
                                    className="dropdown-item text-white hover:bg-emerald-500/10 cursor-pointer transition-colors duration-200"
                                  >
                                    <Eye className="h-4 w-4 mr-2" />
                                    View Table
                                  </DropdownMenuItem>
                                  <DropdownMenuItem 
                                    onClick={() => {
                                      toast.info(`Editing table: ${table.table_name}`);
                                    }}
                                    className="dropdown-item text-white hover:bg-emerald-500/10 cursor-pointer transition-colors duration-200"
                                  >
                                    <Edit className="h-4 w-4 mr-2" />
                                    Edit Table
                                  </DropdownMenuItem>
                                  <DropdownMenuSeparator className="bg-emerald-500/20" />
                                  <DropdownMenuItem 
                                    onClick={() => {
                                      toast.info(`Deleting table: ${table.table_name}`);
                                    }}
                                    className="dropdown-item text-red-400 hover:bg-red-500/10 cursor-pointer transition-colors duration-200"
                                  >
                                    <Trash2 className="h-4 w-4 mr-2" />
                                    Delete Table
                                  </DropdownMenuItem>
                                </DropdownMenuContent>
                              </DropdownMenu>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Pagination and Selection Status */}
                  <div className="px-6 py-4 flex items-center justify-between">
                    {/* Left: Rows Selected */}
                    <div className="flex items-center">
                      <p className="text-white text-sm">
                        {selectedRows.size} of {transformedTables.length} Row(s) Selected
                      </p>
                        </div>

                    {/* Right side: Pagination controls */}
                    <div className="flex items-center gap-6">
                      {/* Rows per page */}
                                  <div className="flex items-center gap-2">
                        <span className="text-white text-sm">Rows per page:</span>
                        <select
                          value={rowsPerPage}
                          onChange={(e) => handleRowsPerPageChange(Number(e.target.value))}
                          className="bg-white/10 text-white px-2 py-1 text-sm"
                          style={{
                            border: "1px solid var(--components-button-outlined, rgba(145, 158, 171, 0.32))",
                            borderRadius: "99px"
                          }}
                        >
                          <option value={5}>5</option>
                          <option value={10}>10</option>
                          <option value={25}>25</option>
                        </select>
                                  </div>
                      
                      {/* Page Info and Controls */}
                      <div className="flex items-center gap-4">
                        <span className="text-white text-sm">
                          Page {currentPage} of {paginatedData.totalPages}
                                  </span>
                        <div className="flex items-center gap-1">
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => handlePageChange(currentPage - 1)}
                            disabled={currentPage === 1}
                            className="h-8 w-8 text-white hover:bg-white/10 disabled:opacity-50"
                          >
                            <ChevronLeft className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => handlePageChange(currentPage + 1)}
                            disabled={currentPage === paginatedData.totalPages}
                            className="h-8 w-8 text-white hover:bg-white/10 disabled:opacity-50"
                          >
                            <ChevronRight className="h-4 w-4" />
                          </Button>
                            </div>
                          </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}