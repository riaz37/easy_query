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
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Loader2,
  Plus,
  Trash2,
  Save,
  Table as TableIcon,
  X,
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
  onSubmit: (data: { tableName: string; schema: string; columns: TableColumn[] }) => void;
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
  // Local state for better performance
  const [localTableName, setLocalTableName] = useState(tableName);
  const [localSchema, setLocalSchema] = useState(schema);
  const [localColumns, setLocalColumns] = useState<TableColumn[]>(columns);

  // Update local state when props change
  React.useEffect(() => {
    setLocalTableName(tableName);
    setLocalSchema(schema);
    setLocalColumns(columns);
  }, [tableName, schema, columns]);

  // Local column update function
  const updateLocalColumn = useCallback((index: number, field: keyof TableColumn, value: any) => {
    setLocalColumns(prev => {
      const newColumns = [...prev];
      newColumns[index] = { ...newColumns[index], [field]: value };
      return newColumns;
    });
  }, []);

  // Add column locally
  const addLocalColumn = useCallback(() => {
    setLocalColumns(prev => [
      ...prev,
      {
        name: "",
        data_type: "VARCHAR",
        nullable: true,
        is_primary: false,
        is_identity: false,
      },
    ]);
  }, []);

  // Remove column locally
  const removeLocalColumn = useCallback((index: number) => {
    if (localColumns.length > 1) {
      setLocalColumns(prev => prev.filter((_, i) => i !== index));
    }
  }, [localColumns.length]);

  // Add validation for reserved column names
  const isReservedColumnName = (name: string): boolean => {
    const reservedNames = [
      'id', 'ID', 'Id',
      'name', 'Name', 'NAME',
      'type', 'Type', 'TYPE',
      'key', 'Key', 'KEY',
      'value', 'Value', 'VALUE',
      'data', 'Data', 'DATA',
      'user', 'User', 'USER',
      'table', 'Table', 'TABLE',
      'column', 'Column', 'COLUMN',
      'schema', 'Schema', 'SCHEMA',
      'database', 'Database', 'DATABASE'
    ];
    return reservedNames.includes(name);
  };

  const validateColumnName = (name: string): string | null => {
    if (!name.trim()) {
      return null; // Don't show error for empty names while typing
    }
    
    if (isReservedColumnName(name)) {
      return `"${name}" is a reserved name`;
    }
    
    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name)) {
      return "Must start with letter/underscore, only letters/numbers/underscores allowed";
    }
    
    return null;
  };

  const validateColumnData = (column: TableColumn): string | null => {
    // Check if identity is set on non-numeric types
    if (column.is_identity && !['INT', 'INTEGER', 'BIGINT', 'SMALLINT', 'TINYINT', 'DECIMAL', 'FLOAT', 'DOUBLE'].includes(column.data_type)) {
      return `Identity columns can only be used with numeric data types (INT, BIGINT, etc.), not with ${column.data_type}`;
    }
    
    return null;
  };

  // Handle form submission
  const handleSubmit = useCallback(() => {
    // Pass modal data to parent for validation and submission
    onSubmit({
      tableName: localTableName,
      schema: localSchema,
      columns: localColumns,
    });
  }, [localTableName, localSchema, localColumns, onSubmit]);

  // Reset form when modal closes
  const handleClose = useCallback(() => {
    setLocalTableName(tableName);
    setLocalSchema(schema);
    setLocalColumns(columns);
    onOpenChange(false);
  }, [tableName, schema, columns, onOpenChange]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-6xl max-h-[95vh] bg-slate-900 border-slate-600">
        <DialogHeader className="pb-6">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-blue-500/20 rounded-xl flex items-center justify-center">
              <TableIcon className="h-7 w-7 text-blue-400" />
            </div>
            <div>
              <DialogTitle className="text-2xl font-bold text-white">Create New Table</DialogTitle>
              <DialogDescription className="text-slate-300 text-base mt-1">
                Define your table structure and columns with proper spacing
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <div className="space-y-8 overflow-y-auto max-h-[calc(95vh-250px)] px-1">
          {/* Basic Table Info */}
          <div className="space-y-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-3 h-3 bg-blue-400 rounded-full"></div>
              <h3 className="text-xl font-semibold text-white">Table Information</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="space-y-3">
                <Label htmlFor="tableName" className="text-slate-200 text-base font-medium">
                  Table Name <span className="text-red-400">*</span>
                </Label>
                <Input
                  id="tableName"
                  value={localTableName}
                  onChange={(e) => setLocalTableName(e.target.value)}
                  placeholder="e.g., users, products, orders"
                  className="bg-slate-800 border-slate-600 text-white h-12 text-base"
                  autoComplete="off"
                />
                <p className="text-sm text-slate-400">
                  Start with a letter, use only letters, numbers, and underscores
                </p>
              </div>
              
              <div className="space-y-3">
                <Label htmlFor="schema" className="text-slate-200 text-base font-medium">
                  Schema
                </Label>
                <Input
                  id="schema"
                  value={localSchema}
                  onChange={(e) => setLocalSchema(e.target.value)}
                  placeholder="dbo"
                  className="bg-slate-800 border-slate-600 text-white h-12 text-base"
                  autoComplete="off"
                />
                <p className="text-sm text-slate-400">
                  Database schema for organizing tables
                </p>
              </div>
            </div>
          </div>

          {/* Columns Section */}
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="w-3 h-3 bg-green-400 rounded-full"></div>
                <h3 className="text-xl font-semibold text-white">Table Columns</h3>
                <Badge variant="secondary" className="bg-slate-700/50 text-slate-300">
                  {localColumns.length} column{localColumns.length !== 1 ? 's' : ''}
                </Badge>
              </div>
              <Button
                onClick={addLocalColumn}
                size="default"
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 h-12"
              >
                <Plus className="h-5 w-5 mr-2" />
                Add Column
              </Button>
            </div>

            {/* Column List */}
            <div className="space-y-6">
              {localColumns.map((column, index) => (
                <div
                  key={index}
                  className="bg-slate-800/50 rounded-xl p-6 border border-slate-600/30 hover:border-slate-500/50 transition-all duration-200"
                >
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-medium text-white">
                      Column {index + 1}
                    </h4>
                    <Button
                      onClick={() => removeLocalColumn(index)}
                      size="sm"
                      variant="outline"
                      className="border-red-500/50 text-red-400 hover:bg-red-500/20 h-10 w-10 p-0"
                      disabled={localColumns.length === 1}
                    >
                      <X className="h-5 w-5" />
                    </Button>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Column Name */}
                    <div className="space-y-3">
                      <Label className="text-slate-200 text-base font-medium flex items-center gap-2">
                        <span className="text-red-400">*</span>
                        Column Name
                      </Label>
                      <Input
                        value={column.name}
                        onChange={(e) => updateLocalColumn(index, 'name', e.target.value)}
                        placeholder="e.g., user_id, product_name"
                        className="bg-slate-700 border-slate-600 text-white h-11 text-base"
                        autoComplete="off"
                      />
                      {column.name && validateColumnName(column.name) && (
                        <p className="text-sm text-red-400">
                          {validateColumnName(column.name)}
                        </p>
                      )}
                      {column.name && validateColumnData(column) && (
                        <p className="text-sm text-red-400">
                          {validateColumnData(column)}
                        </p>
                      )}
                    </div>

                    {/* Data Type */}
                    <div className="space-y-3">
                      <Label className="text-slate-200 text-base font-medium">
                        Data Type
                      </Label>
                      <Select
                        value={column.data_type}
                        onValueChange={(value) => updateLocalColumn(index, 'data_type', value)}
                      >
                        <SelectTrigger className="bg-slate-700 border-slate-600 text-white h-11 text-base">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-slate-800 border-slate-600 max-h-60">
                          {onGetDataTypeOptions()}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  {/* Column Properties */}
                  <div className="mt-6">
                    <Label className="text-slate-200 text-base font-medium mb-4 block">
                      Column Properties
                    </Label>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                      {/* Nullable */}
                      <div className="flex items-center space-x-3 p-4 bg-slate-700/30 rounded-lg border border-slate-600/30">
                        <Checkbox
                          id={`nullable-${index}`}
                          checked={column.nullable}
                          onCheckedChange={(checked) => updateLocalColumn(index, 'nullable', checked)}
                          className="border-slate-500 data-[state=checked]:bg-blue-500 w-5 h-5"
                        />
                        <div className="space-y-1">
                          <Label htmlFor={`nullable-${index}`} className="text-sm font-medium text-slate-200 cursor-pointer">
                            Nullable
                          </Label>
                          <p className="text-xs text-slate-400">
                            Column can contain NULL values
                          </p>
                        </div>
                      </div>

                      {/* Primary Key */}
                      <div className="flex items-center space-x-3 p-4 bg-slate-700/30 rounded-lg border border-slate-600/30">
                        <Checkbox
                          id={`primary-${index}`}
                          checked={column.is_primary}
                          onCheckedChange={(checked) => updateLocalColumn(index, 'is_primary', checked)}
                          className="border-slate-500 data-[state=checked]:bg-green-500 w-5 h-5"
                        />
                        <div className="space-y-1">
                          <Label htmlFor={`primary-${index}`} className="text-sm font-medium text-slate-200 cursor-pointer">
                            Primary Key
                          </Label>
                          <p className="text-xs text-slate-400">
                            Unique identifier for the table
                          </p>
                        </div>
                      </div>

                      {/* Identity */}
                      <div className="flex items-center space-x-3 p-4 bg-slate-700/30 rounded-lg border border-slate-600/30">
                        <Checkbox
                          id={`identity-${index}`}
                          checked={column.is_identity}
                          onCheckedChange={(checked) => updateLocalColumn(index, 'is_identity', checked)}
                          className="border-slate-500 data-[state=checked]:bg-purple-500 w-5 h-5"
                        />
                        <div className="space-y-1">
                          <Label htmlFor={`identity-${index}`} className="text-sm font-medium text-slate-200 cursor-pointer">
                            Identity
                          </Label>
                          <p className="text-xs text-slate-400">
                            Auto-incrementing column
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Column Preview */}
                  {column.name && (
                    <div className="mt-6 p-4 bg-slate-700/20 rounded-lg border border-slate-600/20">
                      <div className="flex items-center gap-2 text-sm text-slate-300">
                        <span className="font-mono text-blue-400">{column.name}</span>
                        <span className="text-slate-500">•</span>
                        <span className="font-mono text-green-400">{column.data_type}</span>
                        {column.is_primary && (
                          <>
                            <span className="text-slate-500">•</span>
                            <span className="text-green-400">PRIMARY KEY</span>
                          </>
                        )}
                        {column.is_identity && (
                          <>
                            <span className="text-slate-500">•</span>
                            <span className="text-purple-400">IDENTITY</span>
                          </>
                        )}
                        {column.nullable ? (
                          <span className="text-blue-400">NULL</span>
                        ) : (
                          <span className="text-red-400">NOT NULL</span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Help Text */}
            <div className="bg-slate-800/30 rounded-lg p-5 border border-slate-600/20">
              <div className="flex items-start gap-4">
                <div className="w-6 h-6 bg-blue-500/20 rounded-full flex items-center justify-center mt-0.5">
                  <span className="text-blue-400 text-sm">ℹ</span>
                </div>
                <div className="text-sm text-slate-300 space-y-2">
                  <p><strong>Column Guidelines:</strong></p>
                  <ul className="list-disc list-inside space-y-1 text-slate-400">
                    <li>Column names must start with a letter and contain only letters, numbers, and underscores</li>
                    <li>Primary key columns should typically be non-nullable</li>
                    <li>Identity columns are auto-incrementing and should be numeric types</li>
                    <li>Consider adding indexes for frequently queried columns</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex justify-between items-center pt-6 border-t border-slate-600/30">
          <div className="text-sm text-slate-400">
            <span className="text-red-400">*</span> Required fields
          </div>
          <div className="flex gap-4">
            <Button
              onClick={handleClose}
              variant="outline"
              className="border-slate-600 text-slate-300 px-6 py-3 h-12"
            >
              Cancel
            </Button>
            <Button
              onClick={handleSubmit}
              disabled={
                !localTableName.trim() || 
                localColumns.some(col => !col.name.trim()) || 
                localColumns.some(col => col.name && validateColumnName(col.name)) ||
                localColumns.some(col => col.name && validateColumnData(col)) ||
                loading
              }
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 h-12"
            >
              {loading ? (
                <>
                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                  Creating...
                </>
              ) : (
                <>
                  <Save className="h-5 w-5 mr-2" />
                  Create Table
                </>
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
} 