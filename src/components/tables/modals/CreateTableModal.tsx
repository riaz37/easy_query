"use client";

import React, { useState, useCallback } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Plus,
  Trash2,
  Save,
  Database,
  XIcon,
} from "lucide-react";
import { TableColumn } from "@/types/api";

interface CreateTableModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  tableName: string;
  setTableName: (name: string) => void;
  schema: string;
  setSchema: (schema: string) => void;
  columns: TableColumn[];
  dataTypes: any;
  loading: boolean;
  onAddColumn: () => void;
  onUpdateColumn: (index: number, field: keyof TableColumn, value: any) => void;
  onRemoveColumn: (index: number) => void;
  onGetDataTypeOptions: () => React.ReactNode;
  onSubmit: (modalData: {
    tableName: string;
    schema: string;
    columns: TableColumn[];
  }) => void;
}

export function CreateTableModal({
  open,
  onOpenChange,
  tableName,
  setTableName,
  schema,
  setSchema,
  columns,
  dataTypes,
  loading,
  onAddColumn,
  onUpdateColumn,
  onRemoveColumn,
  onGetDataTypeOptions,
  onSubmit,
}: CreateTableModalProps) {
  const [localError, setLocalError] = useState<string | null>(null);

  const handleSubmit = () => {
    setLocalError(null);
    onSubmit({ tableName, schema, columns });
  };

  const validateColumnName = (name: string): string | null => {
    if (!name.trim()) {
      return "Column name is required";
    }
    
    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name)) {
      return "Column name must start with a letter or underscore and contain only letters, numbers, and underscores";
    }
    
    return null;
  };

  const validateColumnData = (column: TableColumn): string | null => {
    if (
      column.is_identity &&
      ![
        "INT",
        "INTEGER",
        "BIGINT",
        "SMALLINT",
        "TINYINT",
        "DECIMAL",
        "FLOAT",
        "DOUBLE",
      ].includes(column.data_type)
    ) {
      return `Identity columns can only be used with numeric data types (INT, BIGINT, etc.), not with ${column.data_type}`;
    }
    
    return null;
  };

  const handleColumnValidation = (index: number, field: keyof TableColumn, value: any) => {
    onUpdateColumn(index, field, value);
    
    if (field === 'name') {
      const nameError = validateColumnName(value);
      if (nameError) {
        setLocalError(`Column ${index + 1}: ${nameError}`);
      } else {
        setLocalError(null);
      }
    } else if (field === 'data_type' || field === 'is_identity') {
      const column = { ...columns[index], [field]: value };
      const dataError = validateColumnData(column);
      if (dataError) {
        setLocalError(`Column ${index + 1}: ${dataError}`);
      } else {
        setLocalError(null);
      }
    }
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
                    <Database className="h-5 w-5 text-green-400" />
                    Create New Table
                  </DialogTitle>
                  <p className="modal-description-enhanced">
                    Define your table structure with columns and data types
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
              {/* Error Alert */}
              {localError && (
                <Alert variant="destructive" className="mb-6">
                  <AlertDescription>{localError}</AlertDescription>
                </Alert>
              )}

              {/* Table Basic Info */}
              <div className="modal-form-group">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label className="modal-form-label">Table Name</Label>
                    <Input
                      className="modal-input-enhanced"
                      value={tableName}
                      onChange={(e) => setTableName(e.target.value)}
                      placeholder="Enter table name"
                    />
                  </div>
                  <div>
                    <Label className="modal-form-label">Schema</Label>
                    <Input
                      className="modal-input-enhanced"
                      value={schema}
                      onChange={(e) => setSchema(e.target.value)}
                      placeholder="Enter schema name"
                    />
                  </div>
                </div>
              </div>

              {/* Interactive Table Preview with Column Management */}
              <div className="modal-form-group">
                <div className="flex items-center justify-between mb-4">
                  <Label className="modal-form-label text-lg">Table Columns</Label>
                  <Button
                    onClick={onAddColumn}
                    variant="outline"
                    size="sm"
                    className="border-green-200 text-green-300 hover:bg-green-800/20"
                  >
                    <Plus className="h-4 w-4 mr-2" />
                    Add Column
                  </Button>
                </div>
                
                <div className="border border-slate-600/50 rounded-xl overflow-hidden bg-gradient-to-br from-slate-800/60 to-slate-900/80 backdrop-blur-sm shadow-2xl">
                  <div className="bg-gradient-to-r from-slate-700/80 to-slate-800/80 px-6 py-4 border-b border-slate-600/50">
                    <div className="flex items-center gap-3">
                      <Database className="h-5 w-5 text-emerald-400" />
                      <h4 className="text-slate-100 font-semibold text-lg">
                        {tableName || "table_name"}
                      </h4>
                      <span className="text-slate-400 text-sm font-mono">
                        ({schema || "dbo"})
                      </span>
                    </div>
                  </div>
                  
                  <div className="overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow className="border-slate-600/50 hover:bg-slate-700/30 bg-slate-800/40">
                          <TableHead className="text-slate-200 font-semibold text-sm py-4 px-6">Column Name</TableHead>
                          <TableHead className="text-slate-200 font-semibold text-sm py-4 px-6">Data Type</TableHead>
                          <TableHead className="text-slate-200 font-semibold text-sm py-4 px-6 text-center">Nullable</TableHead>
                          <TableHead className="text-slate-200 font-semibold text-sm py-4 px-6 text-center">Primary Key</TableHead>
                          <TableHead className="text-slate-200 font-semibold text-sm py-4 px-6 text-center">Identity</TableHead>
                          <TableHead className="text-slate-200 font-semibold text-sm py-4 px-6 text-center">Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {columns.map((column, index) => (
                          <TableRow key={index} className="border-slate-600/30 hover:bg-slate-700/20 transition-colors">
                            <TableCell className="text-slate-100 py-3 px-6">
                              <Input
                                className="modal-input-enhanced h-9 text-sm bg-slate-700/50 border-slate-600/50 focus:border-emerald-400/50 focus:ring-emerald-400/20"
                                value={column.name}
                                onChange={(e) => handleColumnValidation(index, 'name', e.target.value)}
                                placeholder="column_name"
                              />
                            </TableCell>
                            <TableCell className="text-slate-100 py-3 px-6">
                              <Select
                                value={column.data_type}
                                onValueChange={(value) => handleColumnValidation(index, 'data_type', value)}
                              >
                                <SelectTrigger className="modal-input-enhanced h-9 text-sm bg-slate-700/50 border-slate-600/50 focus:border-emerald-400/50 focus:ring-emerald-400/20">
                                  <SelectValue placeholder="Select data type" />
                                </SelectTrigger>
                                <SelectContent className="bg-slate-800 border-slate-600">
                                  <SelectItem value="VARCHAR" className="text-slate-200 hover:bg-slate-700">VARCHAR</SelectItem>
                                  <SelectItem value="NVARCHAR" className="text-slate-200 hover:bg-slate-700">NVARCHAR</SelectItem>
                                  <SelectItem value="CHAR" className="text-slate-200 hover:bg-slate-700">CHAR</SelectItem>
                                  <SelectItem value="NCHAR" className="text-slate-200 hover:bg-slate-700">NCHAR</SelectItem>
                                  <SelectItem value="TEXT" className="text-slate-200 hover:bg-slate-700">TEXT</SelectItem>
                                  <SelectItem value="NTEXT" className="text-slate-200 hover:bg-slate-700">NTEXT</SelectItem>
                                  <SelectItem value="INT" className="text-slate-200 hover:bg-slate-700">INT</SelectItem>
                                  <SelectItem value="INTEGER" className="text-slate-200 hover:bg-slate-700">INTEGER</SelectItem>
                                  <SelectItem value="BIGINT" className="text-slate-200 hover:bg-slate-700">BIGINT</SelectItem>
                                  <SelectItem value="SMALLINT" className="text-slate-200 hover:bg-slate-700">SMALLINT</SelectItem>
                                  <SelectItem value="TINYINT" className="text-slate-200 hover:bg-slate-700">TINYINT</SelectItem>
                                  <SelectItem value="DECIMAL" className="text-slate-200 hover:bg-slate-700">DECIMAL</SelectItem>
                                  <SelectItem value="NUMERIC" className="text-slate-200 hover:bg-slate-700">NUMERIC</SelectItem>
                                  <SelectItem value="FLOAT" className="text-slate-200 hover:bg-slate-700">FLOAT</SelectItem>
                                  <SelectItem value="REAL" className="text-slate-200 hover:bg-slate-700">REAL</SelectItem>
                                  <SelectItem value="DOUBLE" className="text-slate-200 hover:bg-slate-700">DOUBLE</SelectItem>
                                  <SelectItem value="BIT" className="text-slate-200 hover:bg-slate-700">BIT</SelectItem>
                                  <SelectItem value="DATE" className="text-slate-200 hover:bg-slate-700">DATE</SelectItem>
                                  <SelectItem value="TIME" className="text-slate-200 hover:bg-slate-700">TIME</SelectItem>
                                  <SelectItem value="DATETIME" className="text-slate-200 hover:bg-slate-700">DATETIME</SelectItem>
                                  <SelectItem value="DATETIME2" className="text-slate-200 hover:bg-slate-700">DATETIME2</SelectItem>
                                  <SelectItem value="SMALLDATETIME" className="text-slate-200 hover:bg-slate-700">SMALLDATETIME</SelectItem>
                                  <SelectItem value="TIMESTAMP" className="text-slate-200 hover:bg-slate-700">TIMESTAMP</SelectItem>
                                  <SelectItem value="UNIQUEIDENTIFIER" className="text-slate-200 hover:bg-slate-700">UNIQUEIDENTIFIER</SelectItem>
                                  <SelectItem value="BINARY" className="text-slate-200 hover:bg-slate-700">BINARY</SelectItem>
                                  <SelectItem value="VARBINARY" className="text-slate-200 hover:bg-slate-700">VARBINARY</SelectItem>
                                  <SelectItem value="IMAGE" className="text-slate-200 hover:bg-slate-700">IMAGE</SelectItem>
                                  <SelectItem value="XML" className="text-slate-200 hover:bg-slate-700">XML</SelectItem>
                                  <SelectItem value="JSON" className="text-slate-200 hover:bg-slate-700">JSON</SelectItem>
                                </SelectContent>
                              </Select>
                            </TableCell>
                            <TableCell className="text-slate-100 py-3 px-6 text-center">
                              <div className="flex items-center justify-center space-x-2">
                                <Checkbox
                                  id={`nullable-${index}`}
                                  checked={column.nullable}
                                  onCheckedChange={(checked) => onUpdateColumn(index, 'nullable', checked)}
                                  className="border-slate-500 data-[state=checked]:bg-emerald-500 data-[state=checked]:border-emerald-500"
                                />
                                <Label htmlFor={`nullable-${index}`} className="text-sm font-medium">
                                  {column.nullable ? (
                                    <span className="text-emerald-400 font-semibold">✓ Yes</span>
                                  ) : (
                                    <span className="text-red-400 font-semibold">✗ No</span>
                                  )}
                                </Label>
                              </div>
                            </TableCell>
                            <TableCell className="text-slate-100 py-3 px-6 text-center">
                              <div className="flex items-center justify-center space-x-2">
                                <Checkbox
                                  id={`primary-${index}`}
                                  checked={column.is_primary}
                                  onCheckedChange={(checked) => onUpdateColumn(index, 'is_primary', checked)}
                                  className="border-slate-500 data-[state=checked]:bg-blue-500 data-[state=checked]:border-blue-500"
                                />
                                <Label htmlFor={`primary-${index}`} className="text-sm font-medium">
                                  {column.is_primary ? (
                                    <span className="text-blue-400 font-semibold">✓ Yes</span>
                                  ) : (
                                    <span className="text-slate-500 font-semibold">✗ No</span>
                                  )}
                                </Label>
                              </div>
                            </TableCell>
                            <TableCell className="text-slate-100 py-3 px-6 text-center">
                              <div className="flex items-center justify-center space-x-2">
                                <Checkbox
                                  id={`identity-${index}`}
                                  checked={column.is_identity}
                                  onCheckedChange={(checked) => handleColumnValidation(index, 'is_identity', checked)}
                                  className="border-slate-500 data-[state=checked]:bg-purple-500 data-[state=checked]:border-purple-500"
                                />
                                <Label htmlFor={`identity-${index}`} className="text-sm font-medium">
                                  {column.is_identity ? (
                                    <span className="text-purple-400 font-semibold">✓ Yes</span>
                                  ) : (
                                    <span className="text-slate-500 font-semibold">✗ No</span>
                                  )}
                                </Label>
                              </div>
                            </TableCell>
                            <TableCell className="text-slate-100 py-3 px-6 text-center">
                              <Button
                                onClick={() => onRemoveColumn(index)}
                                variant="outline"
                                size="sm"
                                disabled={columns.length === 1}
                                className="border-red-400/50 text-red-400 hover:bg-red-500/20 hover:border-red-400 h-9 w-9 p-0 disabled:opacity-50 disabled:cursor-not-allowed"
                              >
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              </div>

              {/* Submit Button */}
              <div className="modal-footer-enhanced">
                <Button
                  onClick={handleSubmit}
                  disabled={loading || !tableName.trim()}
                  className="modal-button-primary"
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                      Creating Table...
                    </>
                  ) : (
                    <>
                      <Save className="h-4 w-4 mr-2" />
                      Create Table
                    </>
                  )}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}