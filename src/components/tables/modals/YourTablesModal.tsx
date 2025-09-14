"use client";

import React, { useState, useCallback, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table as TableIcon,
  Search,
  RefreshCw,
  Eye,
  Edit3,
  Trash2,
  Plus,
  XIcon,
  Database,
  Calendar,
  User,
  Columns,
  Key,
  CheckCircle,
  XCircle,
} from "lucide-react";
import { UserTablesResponse, UserTable } from "@/types/api";
import { useNewTable } from "@/lib/hooks/use-new-table";
import { toast } from "sonner";

interface YourTablesModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  userId?: string;
  onRefresh: () => void;
  onCreateTable: () => void;
}

export function YourTablesModal({
  open,
  onOpenChange,
  userId,
  onRefresh,
  onCreateTable,
}: YourTablesModalProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [userTables, setUserTables] = useState<UserTable[]>([]);
  const [selectedTable, setSelectedTable] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [tableLoadError, setTableLoadError] = useState<string | null>(null);
  const [lastTablesUpdate, setLastTablesUpdate] = useState<Date | null>(null);

  const { getUserTables } = useNewTable();

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
          
          // Log table data for debugging
          console.log('User Tables Loaded:', {
            count: response.tables.length,
            tables: response.tables.map(t => ({
              name: t.table_name,
              fullName: t.table_full_name,
              columns: t.table_schema?.columns?.length || 0
            }))
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
      setLoading(false);
    }
  }, [userId, getUserTables]);

  // Fetch user tables on component mount
  useEffect(() => {
    if (open && userId) {
      fetchUserTables();
    }
  }, [open, userId, fetchUserTables]);

  // Transform user table data to match the expected format (same as ExcelToDBManager)
  const transformedTables = userTables.map(table => ({
    table_name: table.table_name,
    full_name: table.table_full_name,
    columns: table.table_schema?.columns?.map(col => ({
      column_name: col.name,
      data_type: col.type,
      is_nullable: !col.is_required
    })) || []
  }));

  const filteredTables = transformedTables.filter((table) =>
    table.table_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    table.full_name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Get the selected table details
  const selectedTableData = userTables.find(table => table.table_name === selectedTable);

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
      <DialogContent className="max-w-7xl max-h-[90vh] p-0 border-0 bg-transparent" showCloseButton={false}>
        <div className="modal-enhanced">
          <div className="modal-content-enhanced">
            <DialogHeader className="modal-header-enhanced">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <DialogTitle className="modal-title-enhanced flex items-center gap-2">
                    <TableIcon className="h-5 w-5 text-green-400" />
                    Your Database Tables
                  </DialogTitle>
                  <p className="modal-description-enhanced">
                    View and manage all your created database tables
                  </p>
                </div>
                <button
                  onClick={() => onOpenChange(false)}
                  className="modal-close-button"
                >
                  <XIcon className="h-5 w-5" />
                </button>
              </div>
            </DialogHeader>

            <div className="modal-form-content overflow-y-auto max-h-[calc(90vh-200px)]">
              {/* Search and Actions */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-4 flex-1">
                  <div className="relative flex-1 max-w-md">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 h-4 w-4" />
                    <Input
                      className="modal-input-enhanced pl-10"
                      placeholder="Search tables..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                    />
                  </div>
                  <Button
                    onClick={fetchUserTables}
                    variant="outline"
                    size="sm"
                    className="border-slate-600 text-slate-300 hover:bg-slate-700"
                    disabled={loading}
                  >
                    <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                    Refresh
                  </Button>
                </div>
                <Button
                  onClick={onCreateTable}
                  className="modal-button-primary"
                >
                  <Plus className="h-4 w-4 mr-2" />
                  Create New Table
                </Button>
              </div>

              {/* Table Selection */}
              <div className="mb-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <label className="text-lg font-semibold text-white">Select Table to View Columns</label>
                    {filteredTables.length > 0 && (
                      <Badge variant="outline" className="bg-blue-500/20 text-blue-400 border-blue-500/50">
                        {filteredTables.length} available
                      </Badge>
                    )}
                  </div>
                  <Button
                    onClick={fetchUserTables}
                    variant="outline"
                    size="sm"
                    disabled={loading}
                    className="border-slate-600 hover:bg-slate-700/50"
                  >
                    {loading ? (
                      <RefreshCw className="h-4 w-4 animate-spin mr-2" />
                    ) : (
                      <RefreshCw className="h-4 w-4 mr-2" />
                    )}
                    Refresh
                  </Button>
                </div>
                
                {loading ? (
                  <div className="flex items-center gap-3 p-6 bg-slate-700/30 rounded-xl">
                    <RefreshCw className="h-6 w-6 animate-spin text-blue-400" />
                    <span className="text-slate-300 text-lg">Loading user tables...</span>
                  </div>
                ) : !lastTablesUpdate && !loading ? (
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
                ) : filteredTables.length === 0 ? (
                  <div className="p-8 bg-slate-700/30 rounded-xl text-center">
                    <Database className="h-16 w-16 text-slate-500 mx-auto mb-4" />
                    <p className="text-slate-300 text-lg mb-2">No tables found</p>
                    <p className="text-slate-500 mb-6">You need to create tables first</p>
                    <Button
                      onClick={onCreateTable}
                      variant="outline"
                      size="lg"
                      className="text-green-400 border-green-400/50 hover:bg-green-400/10"
                    >
                      <Database className="h-5 w-5 mr-2" />
                      Create Table
                    </Button>
                  </div>
                ) : (
                  <Select value={selectedTable} onValueChange={setSelectedTable}>
                    <SelectTrigger className="h-14 text-lg border-slate-600 bg-slate-700/50">
                      <SelectValue placeholder="Choose a database table" />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-slate-600">
                      {filteredTables.map((table) => (
                        <SelectItem 
                          key={table.table_name} 
                          value={table.table_name}
                          className="text-slate-200 hover:bg-slate-700"
                        >
                          <div className="flex items-center gap-3 p-2">
                            <div className="p-2 bg-blue-500/20 rounded-lg">
                              <Database className="h-4 w-4 text-blue-400" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="font-semibold text-white">{table.full_name}</div>
                              <div className="text-sm text-slate-400">
                                {table.table_name}
                              </div>
                            </div>
                            <Badge variant="secondary" className="bg-blue-500/20 text-blue-400 border-blue-500/30">
                              {table.columns.length} cols
                            </Badge>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              </div>

              {/* Tables Count and Error Display */}
              <div className="mb-6">
                <Card className="bg-slate-800/50 border-slate-600">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Database className="h-5 w-5 text-blue-400" />
                        <span className="text-slate-300">
                          {userTables.length} tables found
                        </span>
                        {lastTablesUpdate && (
                          <span className="text-slate-400 text-sm">
                            (Updated: {lastTablesUpdate.toLocaleTimeString()})
                          </span>
                        )}
                      </div>
                      {searchTerm && (
                        <Badge variant="outline" className="border-slate-500 text-slate-300">
                          {filteredTables.length} matching
                        </Badge>
                      )}
                    </div>
                  </CardContent>
                </Card>
                
                {tableLoadError && (
                  <Alert className="mt-4 border-red-500/50 bg-red-900/20">
                    <AlertDescription className="text-red-300">
                      {tableLoadError}
                    </AlertDescription>
                  </Alert>
                )}
              </div>

              {/* Loading State */}
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-400 mx-auto mb-4" />
                    <p className="text-slate-400">Loading tables...</p>
                  </div>
                </div>
              ) : userTables.length === 0 ? (
                <div className="text-center py-12">
                  <TableIcon className="h-12 w-12 text-slate-500 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-slate-300 mb-2">
                    {searchTerm ? 'No tables found' : 'No tables created yet'}
                  </h3>
                  <p className="text-slate-400 mb-6">
                    {searchTerm 
                      ? 'Try adjusting your search terms' 
                      : 'Create your first table to get started'
                    }
                  </p>
                  {!searchTerm && (
                    <Button onClick={onCreateTable} className="modal-button-primary">
                      <Plus className="h-4 w-4 mr-2" />
                      Create Your First Table
                    </Button>
                  )}
                </div>
              ) : !selectedTable ? (
                <div className="text-center py-12">
                  <Columns className="h-12 w-12 text-slate-500 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-slate-300 mb-2">
                    Select a Table
                  </h3>
                  <p className="text-slate-400">
                    Choose a table from the dropdown above to view its column structure
                  </p>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* Selected Table Info */}
                  <div className="border border-slate-600/50 rounded-xl overflow-hidden bg-gradient-to-br from-slate-800/60 to-slate-900/80 backdrop-blur-sm shadow-2xl">
                    <div className="bg-gradient-to-r from-slate-700/80 to-slate-800/80 px-6 py-4 border-b border-slate-600/50">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <TableIcon className="h-5 w-5 text-emerald-400" />
                          <div>
                            <h4 className="text-slate-100 font-semibold text-lg">
                              {selectedTableData?.table_name}
                            </h4>
                            <p className="text-slate-400 text-sm font-mono">
                              {selectedTableData?.table_full_name}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <Badge variant="outline" className="border-slate-500 text-slate-300">
                            Schema: {selectedTableData?.schema_name || 'dbo'}
                          </Badge>
                            <Badge variant="outline" className="border-green-500 text-green-400">
                            {selectedTableData?.table_schema?.columns?.length || 0} columns
                            </Badge>
                        </div>
                          </div>
                          </div>
                  </div>

                  {/* Columns Visualization */}
                  {selectedTableData?.table_schema?.columns && selectedTableData.table_schema.columns.length > 0 ? (
                    <div className="border border-slate-600/50 rounded-xl overflow-hidden bg-gradient-to-br from-slate-800/60 to-slate-900/80 backdrop-blur-sm shadow-2xl">
                      <div className="bg-gradient-to-r from-slate-700/80 to-slate-800/80 px-6 py-4 border-b border-slate-600/50">
                        <div className="flex items-center gap-3">
                          <Columns className="h-5 w-5 text-blue-400" />
                          <h4 className="text-slate-100 font-semibold text-lg">
                            Table Columns
                          </h4>
                          <span className="text-slate-400 text-sm font-mono">
                            ({selectedTableData.table_schema.columns.length} columns)
                          </span>
                          </div>
                        </div>

                      <div className="overflow-x-auto">
                        <Table>
                          <TableHeader>
                            <TableRow className="border-slate-600/50 hover:bg-slate-700/30 bg-slate-800/40">
                              <TableHead className="text-slate-200 font-semibold text-sm py-4 px-6">Column Name</TableHead>
                              <TableHead className="text-slate-200 font-semibold text-sm py-4 px-6">Data Type</TableHead>
                              <TableHead className="text-slate-200 font-semibold text-sm py-4 px-6 text-center">Max Length</TableHead>
                              <TableHead className="text-slate-200 font-semibold text-sm py-4 px-6 text-center">Required</TableHead>
                              <TableHead className="text-slate-200 font-semibold text-sm py-4 px-6 text-center">Primary Key</TableHead>
                              <TableHead className="text-slate-200 font-semibold text-sm py-4 px-6 text-center">Foreign Key</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {selectedTableData.table_schema.columns.map((column, index) => (
                              <TableRow key={index} className="border-slate-600/30 hover:bg-slate-700/20 transition-colors">
                                <TableCell className="text-slate-100 py-3 px-6">
                                  <div className="flex items-center gap-2">
                                    <span className="font-medium text-slate-200">{column.name}</span>
                                    {column.is_primary && (
                                      <Key className="h-4 w-4 text-yellow-400" title="Primary Key" />
                                    )}
                                  </div>
                                </TableCell>
                                <TableCell className="text-slate-300 py-3 px-6">
                                  <Badge variant="outline" className="border-blue-500 text-blue-400">
                                    {column.type}
                                  </Badge>
                                </TableCell>
                                <TableCell className="text-slate-300 py-3 px-6 text-center">
                                  <span className="text-sm font-mono">
                                    {column.max_length || '-'}
                                  </span>
                                </TableCell>
                                <TableCell className="text-slate-300 py-3 px-6 text-center">
                                  {column.is_required ? (
                                    <CheckCircle className="h-4 w-4 text-red-400 mx-auto" title="Required" />
                                  ) : (
                                    <XCircle className="h-4 w-4 text-green-400 mx-auto" title="Optional" />
                                  )}
                                </TableCell>
                                <TableCell className="text-slate-300 py-3 px-6 text-center">
                                  {column.is_primary ? (
                                    <CheckCircle className="h-4 w-4 text-yellow-400 mx-auto" title="Primary Key" />
                                  ) : (
                                    <XCircle className="h-4 w-4 text-slate-500 mx-auto" title="Not Primary Key" />
                                  )}
                                </TableCell>
                                <TableCell className="text-slate-300 py-3 px-6 text-center">
                                  {column.is_foreign ? (
                                    <CheckCircle className="h-4 w-4 text-purple-400 mx-auto" title="Foreign Key" />
                                  ) : (
                                    <XCircle className="h-4 w-4 text-slate-500 mx-auto" title="Not Foreign Key" />
                                  )}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                            </div>
                          </div>
                  ) : (
                    <div className="text-center py-8">
                      <Columns className="h-8 w-8 text-slate-500 mx-auto mb-2" />
                      <p className="text-slate-400">No columns found for this table</p>
                    </div>
                        )}
                </div>
              )}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
