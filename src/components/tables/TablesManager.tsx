"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  // Loader2,
  Database,
  Search,
  RefreshCw,
  Settings,
  FileSpreadsheet,
  Table,
  Eye,
} from "lucide-react";
import { Spinner } from "@/components/ui/loading";
import { TableFlowVisualization } from "./TableFlowVisualization";
import { ExcelToDBManager } from "./ExcelToDBManager";
import { TableManagementSection } from "./TableManagementSection";
import { ServiceRegistry } from "@/lib/api/services/service-registry";
import { UserCurrentDBTableData } from "@/types/api";
import { useAuthContext } from "@/components/providers/AuthContextProvider";
import { useTheme } from "@/store/theme-store";
import { TableStatsCards, TablesManagerHeader } from "./components";

export function TablesManager() {
  const { user, isLoading: authLoading } = useAuthContext();
  const theme = useTheme();
  const isDark = theme === "dark";

  const [tableData, setTableData] = useState<UserCurrentDBTableData | null>(
    null
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
      await ServiceRegistry.userCurrentDB.setUserCurrentDB(
        { db_id: dbId },
        user.user_id
      );
      setSuccess(
        `Successfully set database ID ${dbId} for user ${user.user_id}`
      );
      // Auto-fetch table data after setting the database
      setTimeout(() => {
        fetchTableData();
      }, 1000);
    } catch (err) {
      console.error("Error setting current database:", err);
      setError("Failed to set current database. Please check the database ID.");
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
      const response = await ServiceRegistry.userCurrentDB.getUserCurrentDB(
        user.user_id
      );

      // The API client already extracts the data portion, so we need to access response.data
      const responseData = response.data || response;

      // Check if db_schema has the table data (primary structure)
      if (
        responseData.db_schema &&
        responseData.db_schema.matched_tables_details &&
        Array.isArray(responseData.db_schema.matched_tables_details)
      ) {
        // Transform the matched_tables_details to the expected format
        const transformedTableInfo = {
          tables: responseData.db_schema.matched_tables_details.map(
            (table: any) => ({
              table_name: table.table_name || table.name || "Unknown",
              full_name:
                table.full_name ||
                `dbo.${table.table_name || table.name || "unknown"}`,
              schema: table.schema || "dbo",
              columns: table.columns || [],
              relationships: table.relationships || [],
              primary_keys: table.primary_keys || [],
              sample_data: table.sample_data || [],
              row_count_sample: table.row_count_sample || 0,
            })
          ),
          metadata: {
            total_tables:
              responseData.db_schema.metadata?.total_schema_tables ||
              responseData.db_schema.schema_tables?.length ||
              0,
            processed_tables:
              responseData.db_schema.matched_tables_details.length,
            failed_tables: 0,
            extraction_date: new Date().toISOString(),
            sample_row_count: 0,
            database_url: responseData.db_url || "",
          },
          unmatched_business_rules:
            responseData.db_schema.unmatched_business_rules || [],
        };

        const structuredData: UserCurrentDBTableData = {
          ...responseData,
          table_info: transformedTableInfo,
          // Also add the db_schema for the visualization parser
          db_schema: responseData.db_schema,
        };

        setTableData(structuredData);
      }
      // Fallback: Check if db_schema has schema_tables (just table names)
      else if (
        responseData.db_schema &&
        responseData.db_schema.schema_tables &&
        Array.isArray(responseData.db_schema.schema_tables)
      ) {
        // Transform the schema_tables to the expected format with minimal data
        const transformedTableInfo = {
          tables: responseData.db_schema.schema_tables.map(
            (tableName: string) => ({
              table_name: tableName,
              full_name: `dbo.${tableName}`,
              schema: "dbo",
              columns: [],
              relationships: [],
              primary_keys: [],
              sample_data: [],
              row_count_sample: 0,
            })
          ),
          metadata: {
            total_tables: responseData.db_schema.schema_tables.length,
            processed_tables:
              responseData.db_schema.matched_tables?.length || 0,
            failed_tables: 0,
            extraction_date: new Date().toISOString(),
            sample_row_count: 0,
            database_url: responseData.db_url || "",
          },
          unmatched_business_rules:
            responseData.db_schema.unmatched_business_rules || [],
        };

        const structuredData: UserCurrentDBTableData = {
          ...responseData,
          table_info: transformedTableInfo,
          db_schema: responseData.db_schema,
        };

        setTableData(structuredData);
      } else {
        setError(
          "Table information is not available. Please generate table info first."
        );
        setTableData(null);
      }

      // Update the dbId state with the current database ID
      setDbId(responseData.db_id);
    } catch (err) {
      console.error("Error fetching table data:", err);
      setError(
        "Failed to fetch table data. Please check the user ID and try again."
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
        `Database reloaded successfully. Please try loading tables again.`
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
        table.table_name.toLowerCase().includes(searchTerm.toLowerCase())
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
    <div className="space-y-6">
      {/* Header */}
      <TablesManagerHeader isDark={isDark} />

      {/* Stats Cards */}
      {user?.user_id && tableData && (
        <TableStatsCards tableData={tableData} isDark={isDark} />
      )}

      {/* Authentication Check */}
      {authLoading ? (
        <div className="card-enhanced">
          <div className="card-content-enhanced">
            <div className="flex items-center gap-2 text-emerald-400">
              <Spinner size="sm" variant="accent-blue" />
              Checking authentication...
            </div>
          </div>
        </div>
      ) : !user?.user_id ? (
        <div className="card-enhanced">
          <div className="card-content-enhanced">
            <div className="text-gray-300">
              Please log in to access database table management features.
            </div>
          </div>
        </div>
      ) : null}

      {/* Error Alert */}
      {error && (
        <div className="card-enhanced">
          <div className="card-content-enhanced">
            <div className="text-red-400">{error}</div>
          </div>
        </div>
      )}

      {/* Success Alert */}
      {success && (
        <div className="card-enhanced">
          <div className="card-content-enhanced">
            <div className="text-emerald-400">{success}</div>
          </div>
        </div>
      )}

      {/* Database Configuration */}
      {user?.user_id && (
        <div className="card-enhanced">
          <div className="card-content-enhanced">
            <div className="card-header-enhanced">
              <div className="card-title-enhanced flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Database Configuration
              </div>
            </div>
            <div className="mt-4">
              <div className="flex items-center gap-4">
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
                    className={`w-32 ${
                      dbId <= 0 ? "border-red-500 focus:border-red-500" : ""
                    }`}
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
                  className="card-button-enhanced"
                >
                  {settingDB ? (
                    <Spinner size="sm" variant="accent-blue" />
                  ) : (
                    <Settings className="h-4 w-4" />
                  )}
                  Set Database
                </Button>
                <Button
                  onClick={fetchTableData}
                  disabled={loading || !user?.user_id}
                  className="card-button-enhanced"
                >
                  {loading ? (
                    <Spinner size="sm" variant="accent-blue" />
                  ) : (
                    <RefreshCw className="h-4 w-4" />
                  )}
                  Load Tables
                </Button>
                <Button
                  onClick={generateTableInfo}
                  disabled={generatingTables || !user?.user_id}
                  className="card-button-enhanced"
                >
                  {generatingTables ? (
                    <Spinner size="sm" variant="accent-blue" />
                  ) : (
                    <Database className="h-4 w-4" />
                  )}
                  Reload Database
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Content Tabs */}
      {user?.user_id && (
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3 bg-slate-800/50 border-slate-600">
            <TabsTrigger
              value="visualization"
              className="flex items-center gap-2 data-[state=active]:bg-slate-700 data-[state=active]:text-white"
            >
              <Table className="h-4 w-4" />
              Table Visualization
            </TabsTrigger>
            <TabsTrigger
              value="table-management"
              className="flex items-center gap-2 data-[state=active]:bg-slate-700 data-[state=active]:text-white"
            >
              <Settings className="h-4 w-4" />
              Table Management
            </TabsTrigger>
            <TabsTrigger
              value="excel-import"
              className="flex items-center gap-2 data-[state=active]:bg-slate-700 data-[state=active]:text-white"
            >
              <FileSpreadsheet className="h-4 w-4" />
              Excel Import
            </TabsTrigger>
          </TabsList>

          {/* Table Visualization Tab */}
          <TabsContent value="visualization" className="space-y-6 mt-6">
            {/* Table Flow Visualization */}
            {tableData && (
              <div className="card-enhanced">
                <div className="card-content-enhanced">
                  <div className="card-header-enhanced">
                    <div className="card-title-enhanced">
                      Table Relationships
                    </div>
                    <p className="card-description-enhanced">
                      Interactive visualization of table relationships and
                      structure
                    </p>
                  </div>
                  <div className="mt-4">
                    <div className="h-[600px] w-full">
                      <TableFlowVisualization rawData={tableData} />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* No Data State */}
            {!loading && !tableData && (
              <div className="card-enhanced">
                <div className="card-content-enhanced">
                  <div className="text-center py-12">
                    <Database className="h-12 w-12 text-slate-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-white mb-2">
                      No Table Data
                    </h3>
                    <p className="text-slate-400 mb-4">
                      Enter a user ID and click Load to fetch table information
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* No Results State */}
            {tableData && filteredTables.length === 0 && searchTerm && (
              <div className="card-enhanced">
                <div className="card-content-enhanced">
                  <div className="text-center py-12">
                    <Search className="h-12 w-12 text-slate-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-white mb-2">
                      No Tables Found
                    </h3>
                    <p className="text-slate-400">
                      No tables match your search criteria: "{searchTerm}"
                    </p>
                  </div>
                </div>
              </div>
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
