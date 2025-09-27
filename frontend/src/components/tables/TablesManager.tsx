"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { XIcon, Table as TableIcon } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Database,
  Search,
  RefreshCw,
  Settings,
  FileSpreadsheet,
  Table,
  Eye,
  Plus,
  Upload,
  BarChart3,
  Users,
  FileText,
  Zap,
} from "lucide-react";
import { Spinner } from "@/components/ui/loading";
import { TableFlowVisualization } from "./TableFlowVisualization";
import { useReactFlow } from "reactflow";
import { ExcelToDBManager } from "./ExcelToDBManager";
import { TableManagementSection } from "./TableManagementSection";
import {
  CreateTableModal,
  BusinessRulesModal,
  YourTablesModal,
  ExcelImportModal,
  AnalyticsModal,
} from "./modals";
import { ServiceRegistry } from "@/lib/api/services/service-registry";
import { UserCurrentDBTableData } from "@/types/api";
import { useAuthContext } from "@/components/providers/AuthContextProvider";
import { useTheme } from "@/store/theme-store";
import { TableStatsCards, TablesManagerHeader } from "./components";


export function TablesManager() {
  const { user, isLoading: authLoading } = useAuthContext();
  const theme = useTheme();
  const isDark = theme === "dark";

  // React Flow instance for zoom functionality
  const [reactFlowInstance, setReactFlowInstance] = React.useState<any>(null);

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
  const [openModal, setOpenModal] = useState<string | null>(null);
  
  // Separate modal states
  const [showCreateTableModal, setShowCreateTableModal] = useState(false);
  const [showBusinessRulesModal, setShowBusinessRulesModal] = useState(false);
  const [showYourTablesModal, setShowYourTablesModal] = useState(false);
  const [showExcelImportModal, setShowExcelImportModal] = useState(false);
  const [showAnalyticsModal, setShowAnalyticsModal] = useState(false);
  
  // Table creation state
  const [tableName, setTableName] = useState("");
  const [schema, setSchema] = useState("dbo");
  const [columns, setColumns] = useState([
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
    <div className="h-screen w-full relative overflow-hidden">
      {/* Main Content Area - PageLayout handles navbar spacing */}
      <div className="flex h-full">
        {/* ReactFlow Container - Full width */}
        <div className="flex-1 relative w-full h-full">
          {user?.user_id && tableData ? (
            <TableFlowVisualization 
              rawData={tableData} 
              onZoomIn={() => {
                // This will be called by TableFlowVisualization to set up zoom in
              }}
              onZoomOut={() => {
                // This will be called by TableFlowVisualization to set up zoom out
              }}
              onReactFlowInstance={(instance) => {
                setReactFlowInstance(instance);
              }}
            />
          ) : !loading && !tableData ? (
            <div className="h-full flex items-center justify-center">
              <div className="bg-white/10 dark:bg-gray-800/20 backdrop-blur-sm border border-emerald-200/20 dark:border-emerald-800/20 rounded-xl p-8">
                <div className="text-center">
                  <Database className="h-12 w-12 text-emerald-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                    No Table Data
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300 mb-4">
                    Enter a user ID and click Load to fetch table information
                  </p>
                </div>
              </div>
            </div>
          ) : tableData && filteredTables.length === 0 && searchTerm ? (
            <div className="h-full flex items-center justify-center">
              <div className="bg-white/10 dark:bg-gray-800/20 backdrop-blur-sm border border-emerald-200/20 dark:border-emerald-800/20 rounded-xl p-8">
                <div className="text-center">
                  <Search className="h-12 w-12 text-emerald-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                    No Tables Found
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300">
                    No tables match your search criteria: "{searchTerm}"
                  </p>
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </div>

      {/* Error/Success Messages */}
      {(error || success) && (
        <div className="fixed top-4 right-4 z-50 max-w-md">
          <Alert variant={error ? "destructive" : "default"} className="bg-slate-800/90 border-slate-600">
            <AlertDescription className="text-white">
              {error || success}
            </AlertDescription>
          </Alert>
        </div>
      )}

      {/* Bottom Icon Controls */}
      {user?.user_id && (
        <div className="fixed bottom-4 left-1/2 transform -translate-x-1/2 z-50">
          <div className="backdrop-blur-sm p-4 shadow-2xl" 
               style={{
                 background: "rgba(255, 255, 255, 0.03)",
                 borderRadius: "32px",
                 border: "1.5px solid",
                 borderImageSource: "linear-gradient(158.39deg, rgba(255, 255, 255, 0.06) 14.19%, rgba(255, 255, 255, 0) 50.59%, rgba(255, 255, 255, 0) 68.79%, rgba(255, 255, 255, 0.015) 105.18%)"
               }}>
            <TooltipProvider>
            <div className="flex items-center gap-4">
              {/* Zoom In */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    onClick={() => {
                      if (reactFlowInstance) {
                        reactFlowInstance.zoomIn();
                      }
                    }}
                    className="w-12 h-12 rounded-full text-white shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-110 cursor-pointer"
                    size="icon"
                    style={{
                      background: "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
                      border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))"
                    }}
                  >
                    <img src="/tables/zoomin.svg" alt="Zoom In" className="h-6 w-6" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Zoom In</p>
                </TooltipContent>
              </Tooltip>

              {/* Zoom Out */}
              <Tooltip>
                <TooltipTrigger asChild>
                      <Button
                    onClick={() => {
                      if (reactFlowInstance) {
                        reactFlowInstance.zoomOut();
                      }
                    }}
                    className="w-12 h-12 rounded-full text-white shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-110 cursor-pointer"
                    size="icon"
                    style={{
                      background: "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
                      border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))"
                    }}
                  >
                    <img src="/tables/zoomout.svg" alt="Zoom Out" className="h-6 w-6" />
                      </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Zoom Out</p>
                </TooltipContent>
              </Tooltip>

              {/* Create Table Modal */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    onClick={() => setShowCreateTableModal(true)}
                    className="w-12 h-12 rounded-full text-white shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-110 cursor-pointer"
                    size="icon"
                    style={{
                      background: "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
                      border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))"
                    }}
                  >
                    <img src="/tables/Table.svg" alt="Add Table" className="h-6 w-6" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Add Table</p>
                </TooltipContent>
              </Tooltip>

              {/* Excel Import Modal */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    onClick={() => setShowExcelImportModal(true)}
                    className="w-12 h-12 rounded-full text-white shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-110 cursor-pointer"
                    size="icon"
                    style={{
                      background: "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
                      border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))"
                    }}
                  >
                    <img src="/tables/excel.svg" alt="Excel Import" className="h-6 w-6" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Excel Import</p>
                </TooltipContent>
              </Tooltip>

              {/* Business Rules Modal */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    onClick={() => setShowBusinessRulesModal(true)}
                    className="w-12 h-12 rounded-full text-white shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-110 cursor-pointer"
                    size="icon"
                    style={{
                      background: "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
                      border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))"
                    }}
                  >
                    <img src="/tables/business.svg" alt="Analytics" className="h-6 w-6" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Analytics</p>
                </TooltipContent>
              </Tooltip>

              {/* Settings - Your Tables Modal */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    onClick={() => setShowYourTablesModal(true)}
                    className="w-12 h-12 rounded-full text-white shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-110 cursor-pointer"
                    size="icon"
                    style={{
                      background: "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
                      border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))"
                    }}
                  >
                    <img src="/tables/Setting.svg" alt="Your Tables" className="h-6 w-6" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Your Tables</p>
                </TooltipContent>
              </Tooltip>

                  </div>
            </TooltipProvider>
          </div>
        </div>
      )}

      {/* Separate Modals */}
      <CreateTableModal
        open={showCreateTableModal}
        onOpenChange={setShowCreateTableModal}
        tableName={tableName}
        setTableName={setTableName}
        schema={schema}
        setSchema={setSchema}
        columns={columns}
        dataTypes={null}
        loading={loading}
        onAddColumn={() => {
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
        }}
        onUpdateColumn={(index, field, value) => {
          setColumns((prevColumns) => {
            const newColumns = [...prevColumns];
            newColumns[index] = { ...newColumns[index], [field]: value };
            return newColumns;
          });
        }}
        onRemoveColumn={(index) => {
          if (columns.length > 1) {
            setColumns(columns.filter((_, i) => i !== index));
          }
        }}
        onGetDataTypeOptions={() => []}
        onSubmit={async (modalData) => {
          try {
            setLoading(true);
            setError(null);
            
            // Validate input
            if (!modalData.tableName.trim()) {
              setError("Table name is required");
              return;
            }
            
            if (!modalData.schema.trim()) {
              setError("Schema is required");
              return;
            }
            
            if (modalData.columns.some(col => !col.name.trim())) {
              setError("All columns must have names");
              return;
            }
            
            // Create table request
            const request = {
              user_id: user?.user_id || "",
              table_name: modalData.tableName,
              schema: modalData.schema,
              columns: modalData.columns,
            };
            
            console.log("Creating table with request:", request);
            
            // Call the create table API using ServiceRegistry
            const result = await ServiceRegistry.newTable.createTable(request);
            
            if (result.success) {
              setSuccess("Table created successfully!");
              setShowCreateTableModal(false);
              // Reset form
              setTableName("");
              setSchema("dbo");
              setColumns([{
                name: "column_1",
                data_type: "INT",
                nullable: false,
                is_primary: true,
                is_identity: true,
              }]);
              // Refresh table data
              await setCurrentDatabase();
            } else {
              setError(result.error || "Failed to create table");
            }
          } catch (error) {
            console.error("Failed to create table:", error);
            setError("Failed to create table. Please try again.");
          } finally {
            setLoading(false);
          }
        }}
      />

      <BusinessRulesModal
        open={showBusinessRulesModal}
        onOpenChange={setShowBusinessRulesModal}
        businessRule={businessRule}
        setBusinessRule={setBusinessRule}
        loading={false}
        onSubmit={() => {
          console.log("Update business rule:", businessRule);
          setShowBusinessRulesModal(false);
        }}
      />


      <ExcelImportModal
        open={showExcelImportModal}
        onOpenChange={setShowExcelImportModal}
        userId={user?.user_id || ""}
        availableTables={availableTables}
        onViewTableData={(tableName) => {
          setSelectedTableForViewing(tableName);
          setShowExcelImportModal(false);
        }}
      />

      <AnalyticsModal
        open={showAnalyticsModal}
        onOpenChange={setShowAnalyticsModal}
        tableData={tableData}
      />

      <YourTablesModal
        open={showYourTablesModal}
        onOpenChange={setShowYourTablesModal}
        userId={user?.user_id}
        onRefresh={() => {
          console.log("Refresh tables");
        }}
        onCreateTable={() => {
          setShowYourTablesModal(false);
          setShowCreateTableModal(true);
        }}
      />

      {/* Authentication Check */}
      {authLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/10 dark:bg-gray-800/20 backdrop-blur-sm z-40">
          <div className="bg-white/20 dark:bg-gray-800/40 backdrop-blur-sm border border-emerald-200/20 dark:border-emerald-800/20 rounded-xl p-6">
            <div className="flex items-center gap-2 text-emerald-400">
              <Spinner size="sm" variant="accent-blue" />
              Checking authentication...
            </div>
          </div>
        </div>
      )}

      {!user?.user_id && !authLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/10 dark:bg-gray-800/20 backdrop-blur-sm z-40">
          <div className="bg-white/20 dark:bg-gray-800/40 backdrop-blur-sm border border-emerald-200/20 dark:border-emerald-800/20 rounded-xl p-6">
            <div className="text-gray-700 dark:text-gray-300">
              Please log in to access database table management features.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
