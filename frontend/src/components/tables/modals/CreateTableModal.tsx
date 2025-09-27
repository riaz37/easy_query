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
import { Plus, Trash2, Save, Database, XIcon } from "lucide-react";
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

  const handleColumnValidation = (
    index: number,
    field: keyof TableColumn,
    value: any
  ) => {
    onUpdateColumn(index, field, value);

    if (field === "name") {
      const nameError = validateColumnName(value);
      if (nameError) {
        setLocalError(`Column ${index + 1}: ${nameError}`);
      } else {
        setLocalError(null);
      }
    } else if (field === "data_type" || field === "is_identity") {
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
      <DialogContent
        className="p-0 border-0 bg-transparent"
        showCloseButton={false}
        style={{
          width: "900px",
          maxWidth: "900px",
          maxHeight: "90vh",
        }}
      >
        <div className="modal-enhanced">
          <div className="modal-content-enhanced">
            <DialogHeader className="modal-header-enhanced">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <DialogTitle className="modal-title-enhanced flex items-center gap-2">
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

              {/* Column Management Section */}
              <div className="mb-6">
                <div className="mb-4">
                  <h3 className="modal-title-enhanced">Column List</h3>
                </div>

                <div className="overflow-x-auto">
                  <div className="rounded-t-xl overflow-hidden">
                    <Table className="w-full">
                      <TableHeader>
                        <TableRow
                          style={{
                            background:
                              "var(--components-Table-Head-filled, rgba(145, 158, 171, 0.08))",
                            borderRadius: "12px 12px 0 0",
                          }}
                        >
                          <TableHead className="px-4 py-4 text-left rounded-tl-xl w-[30%] min-w-[200px]">
                            <span className="text-white font-medium text-sm">
                              Column Name
                            </span>
                          </TableHead>
                          <TableHead className="px-4 py-4 text-left w-[20%] min-w-[150px]">
                            <span className="text-white font-medium text-sm">
                              Data Type
                            </span>
                          </TableHead>
                          <TableHead className="px-3 py-4 text-center w-[12%] min-w-[100px]">
                            <span className="text-white font-medium text-sm">
                              Nullable
                            </span>
                          </TableHead>
                          <TableHead className="px-3 py-4 text-center w-[12%] min-w-[100px]">
                            <span className="text-white font-medium text-sm">
                              Primary Key
                            </span>
                          </TableHead>
                          <TableHead className="px-3 py-4 text-center w-[12%] min-w-[100px]">
                            <span className="text-white font-medium text-sm">
                              Identity
                            </span>
                          </TableHead>
                          <TableHead className="px-3 py-4 text-center rounded-tr-xl w-[14%] min-w-[120px]">
                            <span className="text-white font-medium text-sm">
                              Actions
                            </span>
                          </TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {columns.map((column, index) => (
                          <TableRow
                            key={index}
                            className="border-slate-600/30"
                          >
                            <TableCell className="text-white py-3 px-4 w-[30%] min-w-[200px]">
                              <Input
                                className="text-sm text-white placeholder-slate-400 focus:border-green-400/50 focus:ring-green-400/20 rounded-lg w-full"
                                style={{
                                  background:
                                    "var(--components-Table-Head-filled, rgba(145, 158, 171, 0.08))",
                                  border: "none",
                                  height: "40px",
                                }}
                                value={column.name}
                                onChange={(e) =>
                                  handleColumnValidation(
                                    index,
                                    "name",
                                    e.target.value
                                  )
                                }
                                placeholder="Column Name"
                              />
                            </TableCell>
                            <TableCell className="text-white py-3 px-4 w-[20%] min-w-[150px]">
                              <Select
                                value={column.data_type}
                                onValueChange={(value) =>
                                  handleColumnValidation(
                                    index,
                                    "data_type",
                                    value
                                  )
                                }
                              >
                                <SelectTrigger
                                  className="modal-select-enhanced text-sm text-white w-full"
                                  style={{
                                    height: "40px",
                                  }}
                                >
                                  <SelectValue placeholder="Select data type" />
                                </SelectTrigger>
                                <SelectContent className="modal-select-content-enhanced">
                                  <SelectItem
                                    value="VARCHAR"
                                    className="dropdown-item"
                                  >
                                    VARCHAR
                                  </SelectItem>
                                  <SelectItem
                                    value="NVARCHAR"
                                    className="dropdown-item"
                                  >
                                    NVARCHAR
                                  </SelectItem>
                                  <SelectItem
                                    value="CHAR"
                                    className="dropdown-item"
                                  >
                                    CHAR
                                  </SelectItem>
                                  <SelectItem
                                    value="NCHAR"
                                    className="dropdown-item"
                                  >
                                    NCHAR
                                  </SelectItem>
                                  <SelectItem
                                    value="TEXT"
                                    className="dropdown-item"
                                  >
                                    TEXT
                                  </SelectItem>
                                  <SelectItem
                                    value="NTEXT"
                                    className="dropdown-item"
                                  >
                                    NTEXT
                                  </SelectItem>
                                  <SelectItem
                                    value="INT"
                                    className="dropdown-item"
                                  >
                                    INT
                                  </SelectItem>
                                  <SelectItem
                                    value="INTEGER"
                                    className="dropdown-item"
                                  >
                                    INTEGER
                                  </SelectItem>
                                  <SelectItem
                                    value="BIGINT"
                                    className="dropdown-item"
                                  >
                                    BIGINT
                                  </SelectItem>
                                  <SelectItem
                                    value="SMALLINT"
                                    className="dropdown-item"
                                  >
                                    SMALLINT
                                  </SelectItem>
                                  <SelectItem
                                    value="TINYINT"
                                    className="dropdown-item"
                                  >
                                    TINYINT
                                  </SelectItem>
                                  <SelectItem
                                    value="DECIMAL"
                                    className="dropdown-item"
                                  >
                                    DECIMAL
                                  </SelectItem>
                                  <SelectItem
                                    value="NUMERIC"
                                    className="dropdown-item"
                                  >
                                    NUMERIC
                                  </SelectItem>
                                  <SelectItem
                                    value="FLOAT"
                                    className="dropdown-item"
                                  >
                                    FLOAT
                                  </SelectItem>
                                  <SelectItem
                                    value="REAL"
                                    className="dropdown-item"
                                  >
                                    REAL
                                  </SelectItem>
                                  <SelectItem
                                    value="DOUBLE"
                                    className="dropdown-item"
                                  >
                                    DOUBLE
                                  </SelectItem>
                                  <SelectItem
                                    value="BIT"
                                    className="dropdown-item"
                                  >
                                    BIT
                                  </SelectItem>
                                  <SelectItem
                                    value="DATE"
                                    className="dropdown-item"
                                  >
                                    DATE
                                  </SelectItem>
                                  <SelectItem
                                    value="TIME"
                                    className="dropdown-item"
                                  >
                                    TIME
                                  </SelectItem>
                                  <SelectItem
                                    value="DATETIME"
                                    className="dropdown-item"
                                  >
                                    DATETIME
                                  </SelectItem>
                                  <SelectItem
                                    value="DATETIME2"
                                    className="dropdown-item"
                                  >
                                    DATETIME2
                                  </SelectItem>
                                  <SelectItem
                                    value="SMALLDATETIME"
                                    className="dropdown-item"
                                  >
                                    SMALLDATETIME
                                  </SelectItem>
                                  <SelectItem
                                    value="TIMESTAMP"
                                    className="dropdown-item"
                                  >
                                    TIMESTAMP
                                  </SelectItem>
                                  <SelectItem
                                    value="UNIQUEIDENTIFIER"
                                    className="dropdown-item"
                                  >
                                    UNIQUEIDENTIFIER
                                  </SelectItem>
                                  <SelectItem
                                    value="BINARY"
                                    className="dropdown-item"
                                  >
                                    BINARY
                                  </SelectItem>
                                  <SelectItem
                                    value="VARBINARY"
                                    className="dropdown-item"
                                  >
                                    VARBINARY
                                  </SelectItem>
                                  <SelectItem
                                    value="IMAGE"
                                    className="dropdown-item"
                                  >
                                    IMAGE
                                  </SelectItem>
                                  <SelectItem
                                    value="XML"
                                    className="dropdown-item"
                                  >
                                    XML
                                  </SelectItem>
                                  <SelectItem
                                    value="JSON"
                                    className="dropdown-item"
                                  >
                                    JSON
                                  </SelectItem>
                                </SelectContent>
                              </Select>
                            </TableCell>
                            <TableCell className="text-white py-3 px-3 text-center w-[12%] min-w-[100px]">
                              <div className="flex items-center justify-center space-x-1">
                                <Checkbox
                                  id={`nullable-${index}`}
                                  checked={column.nullable}
                                  onCheckedChange={(checked) =>
                                    onUpdateColumn(index, "nullable", checked)
                                  }
                                  style={{
                                    backgroundColor: column.nullable
                                      ? "var(--primary-main, rgba(19, 245, 132, 1))"
                                      : "transparent",
                                    borderColor:
                                      "var(--action-active, rgba(145, 158, 171, 1))",
                                    borderWidth: "1px",
                                  }}
                                />
                                <Label
                                  htmlFor={`nullable-${index}`}
                                  className="text-xs font-medium text-white cursor-pointer"
                                >
                                  {column.nullable ? "Yes" : "No"}
                                </Label>
                              </div>
                            </TableCell>
                            <TableCell className="text-white py-3 px-3 text-center w-[12%] min-w-[100px]">
                              <div className="flex items-center justify-center space-x-1">
                                <Checkbox
                                  id={`primary-${index}`}
                                  checked={column.is_primary}
                                  onCheckedChange={(checked) =>
                                    onUpdateColumn(index, "is_primary", checked)
                                  }
                                  style={{
                                    backgroundColor: column.is_primary
                                      ? "var(--primary-main, rgba(19, 245, 132, 1))"
                                      : "transparent",
                                    borderColor:
                                      "var(--action-active, rgba(145, 158, 171, 1))",
                                    borderWidth: "1px",
                                  }}
                                />
                                <Label
                                  htmlFor={`primary-${index}`}
                                  className="text-xs font-medium text-white cursor-pointer"
                                >
                                  {column.is_primary ? "Yes" : "No"}
                                </Label>
                              </div>
                            </TableCell>
                            <TableCell className="text-white py-3 px-3 text-center w-[12%] min-w-[100px]">
                              <div className="flex items-center justify-center space-x-1">
                                <Checkbox
                                  id={`identity-${index}`}
                                  checked={column.is_identity}
                                  onCheckedChange={(checked) =>
                                    handleColumnValidation(
                                      index,
                                      "is_identity",
                                      checked
                                    )
                                  }
                                  style={{
                                    backgroundColor: column.is_identity
                                      ? "var(--primary-main, rgba(19, 245, 132, 1))"
                                      : "transparent",
                                    borderColor:
                                      "var(--action-active, rgba(145, 158, 171, 1))",
                                    borderWidth: "1px",
                                  }}
                                />
                                <Label
                                  htmlFor={`identity-${index}`}
                                  className="text-xs font-medium text-white cursor-pointer"
                                >
                                  {column.is_identity ? "Yes" : "No"}
                                </Label>
                              </div>
                            </TableCell>
                            <TableCell className="text-white py-3 px-3 text-center w-[14%] min-w-[120px]">
                              <Button
                                onClick={() => onRemoveColumn(index)}
                                size="sm"
                                disabled={columns.length === 1}
                                className="modal-button-primary h-8 px-3 text-xs disabled:opacity-50 disabled:cursor-not-allowed"
                              >
                                Remove
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </div>

                {/* Add Column Button */}
                <div className="mt-4 flex justify-start">
                  <Button
                    onClick={onAddColumn}
                    className="modal-button-primary"
                  >
                    <Plus className="h-4 w-4 mr-2" />
                    Add Column
                  </Button>
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
                    <>Create Table</>
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
