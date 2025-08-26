"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Loader2,
  Database,
  Search,
  RefreshCw,
  Settings,
  FileSpreadsheet,
  Table,
  Eye,
} from "lucide-react";
import { TableFlowVisualization } from "./TableFlowVisualization";
import { ExcelToDBManager } from "./ExcelToDBManager";
import { TableManagementSection } from "./TableManagementSection";
import { ServiceRegistry } from "@/lib/api/services/service-registry";
import { UserCurrentDBTableData } from "@/types/api";
import { useAuthContext } from "@/components/providers/AuthContextProvider";

export function TablesManager() {
  const { user, isLoading: authLoading } = useAuthContext();
  
  const [tableData, setTableData] = useState<UserCurrentDBTableData | null>(
    null,
  );
  const [loading, setLoading] = useState(false);
  const [settingDB, setSettingDB] = useState(false);
  const [generatingTables, setGeneratingTables] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [dbId, setDbId] = useState<number>(1); // Default database ID
  const [searchTerm, setSearchTerm] = useState("");
  const [activeTab, setActiveTab] = useState("visualization");
  const [selectedTableForViewing, setSelectedTableForViewing] =
    useState<string>("");

  const setCurrentDatabase = async () => {
    if (!user?.user_id) {
      setError("Please log in to set database");
      return;
    }

    if (!dbId || dbId <= 0) {
      setError("Please enter a valid database ID (must be greater than 0)");
      return;
    }

    setSettingDB(true);
    setError(null);
    setSuccess(null);

    try {
      await ServiceRegistry.userCurrentDB.setUserCurrentDB({ db_id: dbId }, user.user_id);
      setSuccess(`Successfully set database ID ${dbId} for user ${user.user_id}`);
      // Auto-fetch table data after setting the database
      setTimeout(() => {
        fetchTableData();
      }, 1000);
    } catch (err) {
      console.error("Error setting current database:", err);
      setError(
        "Failed to set current database. Please check the database ID.",
      );
    } finally {
      setSettingDB(false);
    }
  };

  const fetchTableData = async () => {
    if (!user?.user_id) {
      setError("Please log in to view tables");
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await ServiceRegistry.userCurrentDB.getUserCurrentDB(user.user_id);

      // The API client already extracts the data portion, so we need to access response.data
      const responseData = response.data || response;

      // Check if db_schema has the table data (primary structure)
      if (responseData.db_schema && responseData.db_schema.matched_tables_details && Array.isArray(responseData.db_schema.matched_tables_details)) {
        // Transform the matched_tables_details to the expected format
        const transformedTableInfo = {
          tables: responseData.db_schema.matched_tables_details.map((table: any) => ({
            table_name: table.table_name || table.name || "Unknown",
            full_name: table.full_name || `dbo.${table.table_name || table.name || "unknown"}`,
            schema: table.schema || "dbo",
            columns: table.columns || [],
            relationships: table.relationships || [],
            primary_keys: table.primary_keys || [],
            sample_data: table.sample_data || [],
            row_count_sample: table.row_count_sample || 0,
          })),
          metadata: {
            total_tables: responseData.db_schema.metadata?.total_schema_tables || responseData.db_schema.schema_tables?.length || 0,
            processed_tables: responseData.db_schema.matched_tables_details.length,
            failed_tables: 0,
            extraction_date: new Date().toISOString(),
            sample_row_count: 0,
            database_url: responseData.db_url || "",
          },
          unmatched_business_rules: responseData.db_schema.unmatched_business_rules || []
        };

        const structuredData: UserCurrentDBTableData = {
          ...responseData,
          table_info: transformedTableInfo,
          // Also add the db_schema for the visualization parser
          db_schema: responseData.db_schema
        };
        
        setTableData(structuredData);
      } 
      // Fallback: Check if db_schema has schema_tables (just table names)
      else if (responseData.db_schema && responseData.db_schema.schema_tables && Array.isArray(responseData.db_schema.schema_tables)) {
        // Transform the schema_tables to the expected format with minimal data
        const transformedTableInfo = {
          tables: responseData.db_schema.schema_tables.map((tableName: string) => ({
            table_name: tableName,
            full_name: `dbo.${tableName}`,
            schema: "dbo",
            columns: [],
            relationships: [],
            primary_keys: [],
            sample_data: [],
            row_count_sample: 0,
          })),
          metadata: {
            total_tables: responseData.db_schema.schema_tables.length,
            processed_tables: responseData.db_schema.matched_tables?.length || 0,
            failed_tables: 0,
            extraction_date: new Date().toISOString(),
            sample_row_count: 0,
            database_url: responseData.db_url || "",
          },
          unmatched_business_rules: responseData.db_schema.unmatched_business_rules || []
        };

        const structuredData: UserCurrentDBTableData = {
          ...responseData,
          table_info: transformedTableInfo,
          db_schema: responseData.db_schema
        };
        
        setTableData(structuredData);
      } 
      else {
        setError(
          "Table information is not available. Please generate table info first.",
        );
        setTableData(null);
      }

      // Update the dbId state with the current database ID
      setDbId(responseData.db_id);
    } catch (err) {
      console.error("Error fetching table data:", err);
      setError(
        "Failed to fetch table data. Please check the user ID and try again.",
      );
    } finally {
      setLoading(false);
    }
  };

  const generateTableInfo = async () => {
    if (!user?.user_id) {
      setError("Please log in to generate table info");
      return;
    }

    setGeneratingTables(true);
    setError(null);
    setSuccess(null);

    try {
      // For now, let's just reload the database to refresh table info
      const response = await ServiceRegistry.database.reloadDatabase();
      setSuccess(
        `Database reloaded successfully. Please try loading tables again.`,
      );

      // Auto-fetch table data after reloading
      setTimeout(() => {
        fetchTableData();
      }, 2000);
    } catch (err) {
      console.error("Error reloading database:", err);
      setError("Failed to reload database. Please try again.");
    } finally {
      setGeneratingTables(false);
    }
  };

  useEffect(() => {
    if (user?.user_id) {
      fetchTableData();
    }
  }, [user?.user_id]);

  const filteredTables =
    tableData?.table_info?.tables?.filter(
      (table) =>
        table.full_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        table.table_name.toLowerCase().includes(searchTerm.toLowerCase()),
    ) || [];

  // Prepare available tables for Excel to DB
  const availableTables =
    tableData?.table_info?.tables?.map((table) => ({
      table_name: table.table_name,
      full_name: table.full_name,
      columns: (table.columns || []).map((column) => ({
        column_name: column.name,
        data_type: column.type,
        is_nullable: !column.is_required,
      })),
    })) || [];

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Database Tables</h1>
          <p className="text-slate-400 mt-2">
            Manage table relationships and import data
          </p>
        </div>
      </div>

      {/* Authentication Check */}
      {authLoading ? (
        <Alert>
          <AlertDescription className="flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            Checking authentication...
          </AlertDescription>
        </Alert>
      ) : !user?.user_id ? (
        <Alert>
          <AlertDescription>
            Please log in to access database table management features.
          </AlertDescription>
        </Alert>
      ) : null}

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Success Alert */}
      {success && (
        <Alert className="border-emerald-500/50 bg-emerald-500/10">
          <AlertDescription className="text-emerald-400">
            {success}
          </AlertDescription>
        </Alert>
      )}

      {/* Database Configuration */}
      {user?.user_id && (
        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-white">
              <Settings className="h-5 w-5" />
              Database Configuration
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <label className="text-sm text-slate-400 whitespace-nowrap">
                  User ID:
                </label>
                <div className="w-40 px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white text-sm">
                  {user?.user_id || "Not authenticated"}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <label className="text-sm text-slate-400 whitespace-nowrap">
                  Database ID:
                </label>
                <Input
                  type="number"
                  placeholder="Enter DB ID"
                  value={dbId || ""}
                  onChange={(e) => {
                    const value = parseInt(e.target.value);
                    if (value > 0) {
                      setDbId(value);
                    } else {
                      setDbId(0); // Set to 0 to indicate invalid state
                    }
                  }}
                  className={`w-32 ${dbId <= 0 ? 'border-red-500 focus:border-red-500' : ''}`}
                  min="1"
                  required
                />
                {dbId <= 0 && (
                  <span className="text-red-400 text-xs">Invalid ID</span>
                )}
              </div>
              <Button
                onClick={setCurrentDatabase}
                disabled={settingDB || !dbId || dbId <= 0 || !user?.user_id}
                variant="outline"
              >
                {settingDB ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Settings className="h-4 w-4" />
                )}
                Set Database
              </Button>
              <Button onClick={fetchTableData} disabled={loading || !user?.user_id}>
                {loading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <RefreshCw className="h-4 w-4" />
                )}
                Load Tables
              </Button>
              <Button
                onClick={generateTableInfo}
                disabled={generatingTables || !user?.user_id}
                variant="secondary"
              >
                {generatingTables ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Database className="h-4 w-4" />
                )}
                Reload Database
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Database Info */}
      {user?.user_id && tableData && (
        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-white">
              <Database className="h-5 w-5" />
              Database Information
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <p className="text-sm text-slate-400">Database ID</p>
                <p className="text-white font-medium">{tableData.db_id}</p>
              </div>
              <div>
                <p className="text-sm text-slate-400">Total Tables</p>
                <p className="text-white font-medium">
                  {tableData.table_info?.metadata?.total_tables || 0}
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-400">Processed Tables</p>
                <p className="text-white font-medium">
                  {tableData.table_info?.metadata?.processed_tables || 0}
                </p>
              </div>
            </div>

            {tableData.table_info?.unmatched_business_rules?.length > 0 && (
              <div>
                <p className="text-sm text-slate-400 mb-2">
                  Unmatched Business Rules
                </p>
                <div className="flex flex-wrap gap-2">
                  {tableData.table_info.unmatched_business_rules.map(
                    (rule, index) => (
                      <Badge
                        key={`rule-${index}-${rule.substring(0, 20)}`}
                        variant="secondary"
                        className="bg-yellow-900/20 text-yellow-400"
                      >
                        {rule}
                      </Badge>
                    ),
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Main Content Tabs */}
      {user?.user_id && (
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                  <TabsList className="grid w-full grid-cols-3 bg-slate-800/50">
          <TabsTrigger
            value="visualization"
            className="flex items-center gap-2 data-[state=active]:bg-slate-700"
          >
            <Table className="h-4 w-4" />
            Table Visualization
          </TabsTrigger>
          <TabsTrigger
            value="table-management"
            className="flex items-center gap-2 data-[state=active]:bg-slate-700"
          >
            <Settings className="h-4 w-4" />
            Table Management
          </TabsTrigger>
          <TabsTrigger
            value="excel-import"
            className="flex items-center gap-2 data-[state=active]:bg-slate-700"
          >
            <FileSpreadsheet className="h-4 w-4" />
            Excel Import
          </TabsTrigger>
        </TabsList>

          {/* Table Visualization Tab */}
          <TabsContent value="visualization" className="space-y-6 mt-6">
            {/* Search */}
            {tableData && (
              <Card className="bg-slate-800/50 border-slate-700">
                <CardContent className="pt-6">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-400" />
                    <Input
                      placeholder="Search tables..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Table Flow Visualization */}
            {tableData && (
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white">
                    Table Relationships
                  </CardTitle>
                  <p className="text-slate-400 text-sm">
                    Interactive visualization of table relationships and structure
                  </p>
                </CardHeader>
                <CardContent>
                  <div className="h-[600px] w-full">
                    <TableFlowVisualization rawData={tableData} />
                  </div>
                </CardContent>
              </Card>
            )}

            {/* No Data State */}
            {!loading && !tableData && (
              <Card className="bg-slate-800/50 border-slate-700">
                <CardContent className="text-center py-12">
                  <Database className="h-12 w-12 text-slate-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-white mb-2">
                    No Table Data
                  </h3>
                  <p className="text-slate-400 mb-4">
                    Enter a user ID and click Load to fetch table information
                  </p>
                </CardContent>
              </Card>
            )}

            {/* No Results State */}
            {tableData && filteredTables.length === 0 && searchTerm && (
              <Card className="bg-slate-800/50 border-slate-700">
                <CardContent className="text-center py-12">
                  <Search className="h-12 w-12 text-slate-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-white mb-2">
                    No Tables Found
                  </h3>
                  <p className="text-slate-400">
                    No tables match your search criteria: "{searchTerm}"
                  </p>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Table Management Tab */}
          <TabsContent value="table-management" className="mt-6">
            <TableManagementSection
              userId={user?.user_id}
              databaseId={dbId}
              onTableCreated={() => {
                // Refresh table data when a new table is created
                fetchTableData();
              }}
            />
          </TabsContent>

          {/* Excel Import Tab */}
          <TabsContent value="excel-import" className="mt-6">
            <ExcelToDBManager
              userId={user?.user_id || ""}
              availableTables={availableTables}
              onViewTableData={(tableName) => {
                setSelectedTableForViewing(tableName);
                setActiveTab("visualization");
              }}
            />
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}
