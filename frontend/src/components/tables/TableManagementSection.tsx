"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { TableFlowVisualization } from "./TableFlowVisualization";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Loader2,
  Plus,
  Trash2,
  Save,
  Database,
  FileText,
  Settings,
  Eye,
  Edit3,
  CheckCircle,
  AlertCircle,
  RefreshCw,
  Table as TableIcon,
  Search,
} from "lucide-react";
import { useNewTable } from "@/lib/hooks/use-new-table";
import {
  CreateTableRequest,
  TableColumn,
  UserTablesResponse,
} from "@/types/api";
import UserTableList from "./UserTableList";

// Reusable Components
import { CreateTableModal } from "./modals/CreateTableModal";
import { BusinessRulesModal } from "./modals/BusinessRulesModal";
import { QuickActionsGrid } from "./sections/QuickActionsGrid";
import { UserTablesSection } from "./sections/UserTablesSection";
import { TableVisualizationSection } from "./sections/TableVisualizationSection";
import { QuickStatsSection } from "./sections/QuickStatsSection";
import { SectionHeader } from "./ui/SectionHeader";

interface TableManagementSectionProps {
  userId?: string;
  databaseId?: number;
  onTableCreated?: () => void;
}

export function TableManagementSection({
  userId: propUserId,
  databaseId = 1,
  onTableCreated,
}: TableManagementSectionProps) {
  const {
    createTable,
    getUserTables,
    getTablesByDatabase,
    getDataTypes,
    updateUserBusinessRule,
    getUserBusinessRule,
    loading,
    error,
    success,
    clearError,
    clearSuccess,
  } = useNewTable();

  const [tableName, setTableName] = useState("");
  const [schema, setSchema] = useState("dbo");
  const [columns, setColumns] = useState<TableColumn[]>([
    {
      name: "column_1",
      data_type: "INT",
      nullable: false,
      is_primary: true,
      is_identity: true,
    },
  ]);
  const [businessRule, setBusinessRule] = useState("");
  const [currentBusinessRule, setCurrentBusinessRule] = useState<string>("");
  const [successMessage, setSuccessMessage] = useState<string>("");

  const [dataTypes, setDataTypes] = useState<any>(null);
  const [userTables, setUserTables] = useState<
    UserTablesResponse["data"] | null
  >(null);
  const [databaseTables, setDatabaseTables] = useState<any>(null);

  const [showCreateTableDialog, setShowCreateTableDialog] = useState(false);
  const [showBusinessRuleDialog, setShowBusinessRuleDialog] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");

  // Error and success state management
  const [localError, setLocalError] = useState<string | null>(null);
  const [localSuccess, setLocalSuccess] = useState<string | null>(null);

  useEffect(() => {
    loadDataTypes();
  }, []);

  useEffect(() => {
    if (propUserId) {
      loadUserTables();
      loadUserBusinessRule();
    }
  }, [propUserId]);

  useEffect(() => {
    if (databaseId) {
      loadDatabaseTables();
    }
  }, [databaseId]);

  const loadDataTypes = async () => {
    try {
      console.log("Loading data types...");
      const types = await getDataTypes();
      console.log("Data types response:", types);
      if (types) {
        setDataTypes(types);
        console.log("Data types set to state:", types);
      } else {
        console.log("No data types returned from API");
      }
    } catch (error) {
      console.error("Failed to load data types:", error);
    }
  };

  const loadUserTables = async () => {
    if (!propUserId?.trim()) return;

    try {
      const tables = await getUserTables(propUserId);
      if (tables) {
        setUserTables(tables);
      }
    } catch (error) {
      console.error("Failed to load user tables:", error);
    }
  };

  const loadDatabaseTables = async () => {
    try {
      const tables = await getTablesByDatabase(databaseId);
      if (tables) {
        setDatabaseTables(tables);
      }
    } catch (error) {
      console.error("Failed to load database tables:", error);
    }
  };

  const loadUserBusinessRule = async () => {
    if (!propUserId?.trim()) return;

    try {
      const rule = await getUserBusinessRule(propUserId);
      if (rule) {
        setCurrentBusinessRule(rule.business_rule || "");
        setBusinessRule(rule.business_rule || "");
      }
      setSuccessMessage("");
    } catch (error) {
      console.error("Failed to load business rule:", error);
    }
  };

  const handleCreateTable = async (modalData?: {
    tableName: string;
    schema: string;
    columns: TableColumn[];
  }) => {
    console.log("handleCreateTable called with:", {
      modalData,
      tableName,
      schema,
      columns,
    });

    if (!propUserId?.trim()) {
      return;
    }

    // Use modal data if provided, otherwise use local state
    const finalTableName = modalData?.tableName || tableName;
    const finalSchema = modalData?.schema || schema;
    const finalColumns = modalData?.columns || columns;

    console.log("Final values:", { finalTableName, finalSchema, finalColumns });

    if (!finalTableName.trim()) {
      setLocalError("Table name is required");
      return;
    }

    // Validate each column
    for (let i = 0; i < finalColumns.length; i++) {
      const column = finalColumns[i];
      const nameError = validateColumnName(column.name);
      if (nameError) {
        setLocalError(`Column ${i + 1}: ${nameError}`);
        return;
      }

      const dataError = validateColumnData(column);
      if (dataError) {
        setLocalError(`Column ${i + 1}: ${dataError}`);
        return;
      }
    }

    if (finalColumns.some((col) => !col.name.trim())) {
      setLocalError("All columns must have names");
      return;
    }

    const request: CreateTableRequest = {
      user_id: propUserId,
      table_name: finalTableName,
      schema: finalSchema,
      columns: finalColumns,
    };

    console.log("Creating table with request:", request);

    try {
      const result = await createTable(request);
      if (result) {
        setSuccessMessage(`Table "${finalTableName}" created successfully!`);
        setShowCreateTableDialog(false);
        resetTableForm();
        onTableCreated?.();
        loadUserTables();
      }
    } catch (error) {
      console.error("Failed to create table:", error);
      setLocalError("Failed to create table. Please try again.");
    }
  };

  const handleUpdateBusinessRule = async () => {
    if (!propUserId?.trim()) return;

    try {
      const result = await updateUserBusinessRule(propUserId, {
        business_rule: businessRule,
      });

      if (result) {
        setSuccessMessage("Business rule updated successfully!");
        setCurrentBusinessRule(businessRule);
        setShowBusinessRuleDialog(false);
      }
    } catch (error) {
      console.error("Failed to update business rule:", error);
      setLocalError("Failed to update business rule. Please try again.");
    }
  };

  const resetTableForm = () => {
    setTableName("");
    setSchema("dbo");
    setColumns([
      {
        name: "column_1",
        data_type: "INT",
        nullable: false,
        is_primary: true,
        is_identity: true,
      },
    ]);
  };

  const addColumn = () => {
    setColumns([
      ...columns,
      {
        name: "",
        data_type: "VARCHAR",
        nullable: true,
        is_primary: false,
        is_identity: false,
      },
    ]);
  };

  const updateColumn = useCallback(
    (index: number, field: keyof TableColumn, value: any) => {
      setColumns((prevColumns) => {
        const newColumns = [...prevColumns];
        newColumns[index] = { ...newColumns[index], [field]: value };
        return newColumns;
      });
    },
    []
  );

  const removeColumn = (index: number) => {
    if (columns.length > 1) {
      setColumns(columns.filter((_, i) => i !== index));
    }
  };

  const getDataTypeOptions = () => {
    console.log("getDataTypeOptions called, dataTypes:", dataTypes);

    if (!dataTypes) {
      console.log("No data types available, returning fallback types");
      return [];
    }

    // Handle different data type structures
    let allTypes: string[] = [];

    if (dataTypes.data_types && Array.isArray(dataTypes.data_types)) {
      allTypes = dataTypes.data_types;
      console.log("Using dataTypes.data_types:", allTypes);
    } else if (dataTypes.types && Array.isArray(dataTypes.types)) {
      allTypes = dataTypes.types.map((t: any) => t.name || t);
      console.log("Using dataTypes.types:", allTypes);
    } else if (dataTypes.numeric && dataTypes.string && dataTypes.date_time) {
      allTypes = [
        ...dataTypes.numeric,
        ...dataTypes.string,
        ...dataTypes.date_time,
        ...(dataTypes.binary || []),
        ...(dataTypes.other || []),
      ];
      console.log("Using categorized data types:", allTypes);
    }

    // Fallback to common SQL types if no data types loaded
    if (allTypes.length === 0) {
      console.log("No data types found, using fallback types");
      allTypes = [
        "INT",
        "INTEGER",
        "BIGINT",
        "SMALLINT",
        "TINYINT",
        "VARCHAR",
        "CHAR",
        "TEXT",
        "LONGTEXT",
        "DECIMAL",
        "FLOAT",
        "DOUBLE",
        "DATE",
        "DATETIME",
        "TIMESTAMP",
        "TIME",
        "BOOLEAN",
        "BOOL",
        "JSON",
        "BLOB",
      ];
    }

    console.log("Final allTypes array:", allTypes);

    // Filter out any types with parameters (like NVARCHAR(50))
    allTypes = allTypes.filter((type) => {
      const typeString = String(type);
      return !typeString.includes("(") && !typeString.includes(")");
    });

    return allTypes.map((type, index) => {
      const typeString = typeof type === "string" ? type : String(type);
      return (
        <SelectItem key={`type-${index}-${typeString}`} value={typeString}>
          {typeString}
        </SelectItem>
      );
    });
  };

  // Add validation for reserved column names
  const isReservedColumnName = (name: string): boolean => {
    const reservedNames = [
      "id",
      "ID",
      "Id",
      "name",
      "Name",
      "NAME",
      "type",
      "Type",
      "TYPE",
      "key",
      "Key",
      "KEY",
      "value",
      "Value",
      "VALUE",
      "data",
      "Data",
      "DATA",
      "user",
      "User",
      "USER",
      "table",
      "Table",
      "TABLE",
      "column",
      "Column",
      "COLUMN",
      "schema",
      "Schema",
      "SCHEMA",
      "database",
      "Database",
      "DATABASE",
    ];
    return reservedNames.includes(name);
  };

  const validateColumnName = (name: string): string | null => {
    if (!name.trim()) {
      return "Column name is required";
    }

    if (isReservedColumnName(name)) {
      return `Column name "${name}" is reserved and cannot be used`;
    }

    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name)) {
      return "Column name must start with a letter or underscore and contain only letters, numbers, and underscores";
    }

    return null;
  };

  const validateColumnData = (column: TableColumn): string | null => {
    // Check if identity is set on non-numeric types
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

  const transformUserTablesForFlow = () => {
    if (!userTables?.tables) return [];

    return userTables.tables.map((table) => ({
      name: table.table_name,
      full_name: table.full_name,
      columns: table.columns || [],
      relationships: table.relationships || [],
    }));
  };

  const clearMessages = () => {
    setLocalError(null);
    setLocalSuccess(null);
    setSuccessMessage("");
    clearError();
    clearSuccess();
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <SectionHeader
        title="Table Management"
        description="Create tables, manage business rules, and configure database settings"
        actionButton={
          <Button
            onClick={clearMessages}
            variant="outline"
            size="sm"
            className="border-slate-600 text-slate-300 hover:bg-slate-700"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Clear Alerts
          </Button>
        }
      />

      {/* Error Alert */}
      {(error || localError) && (
        <Alert variant="destructive">
          <AlertDescription>{error || localError}</AlertDescription>
        </Alert>
      )}

      {/* Quick Actions Grid */}
      <QuickActionsGrid
        onCreateTable={() => setShowCreateTableDialog(true)}
        onManageRules={() => setShowBusinessRuleDialog(true)}
      />

      {/* Main Content Sections */}
      <div className="space-y-8">
        {/* User Tables Section */}
        <UserTablesSection
          userTables={userTables}
          loading={loading}
          onRefresh={loadUserTables}
          onCreateTable={() => setShowCreateTableDialog(true)}
        />
      </div>

      {/* Quick Stats */}
      {(userTables || currentBusinessRule) && (
        <QuickStatsSection
          tableCount={userTables?.tables?.length || 0}
          businessRuleCount={currentBusinessRule ? 1 : 0}
        />
      )}

      {/* Create Table Modal */}
      <CreateTableModal
        open={showCreateTableDialog}
        onOpenChange={setShowCreateTableDialog}
        tableName={tableName}
        setTableName={setTableName}
        schema={schema}
        setSchema={setSchema}
        columns={columns}
        dataTypes={dataTypes}
        loading={loading}
        onAddColumn={addColumn}
        onUpdateColumn={updateColumn}
        onRemoveColumn={removeColumn}
        onGetDataTypeOptions={getDataTypeOptions}
        onSubmit={(modalData) => handleCreateTable(modalData)}
      />

      {/* Business Rules Modal */}
      <BusinessRulesModal
        open={showBusinessRuleDialog}
        onOpenChange={setShowBusinessRuleDialog}
        businessRule={businessRule}
        setBusinessRule={setBusinessRule}
        loading={loading}
        onSubmit={handleUpdateBusinessRule}
      />
    </div>
  );
}
