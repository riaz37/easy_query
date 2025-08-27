"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Database,
  BarChart3,
  FileText,
  ArrowLeft,
  Clock,
  User,
  Code,
  CheckCircle,
  Loader2,
} from "lucide-react";
import { toast } from "sonner";
import { QueryResultsTable } from "@/components/database-query/QueryResultsTable";
import { QueryCharts } from "@/components/database-query/QueryCharts";

export default function DatabaseQueryResultsPage() {
  const [currentQuery, setCurrentQuery] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<"table" | "charts">("table");
  const router = useRouter();

  // Load query result from sessionStorage on component mount
  useEffect(() => {
    const storedResult = sessionStorage.getItem("databaseQueryResult");
    if (storedResult) {
      try {
        const queryData = JSON.parse(storedResult);
        setCurrentQuery(queryData);

        // Keep the data in sessionStorage so user can refresh or navigate back
        // sessionStorage.removeItem("databaseQueryResult");
      } catch (error) {
        console.error("Error parsing query result:", error);
        toast.error("Failed to load query results");
      }
    } else {
      // No results found, redirect back to query page
      router.push("/database-query");
    }
  }, [router]);

  const handleBackToQuery = () => {
    router.push("/database-query");
  };

  const handleBackToDashboard = () => {
    router.push("/");
  };

  if (!currentQuery) {
    return (
      <div className="w-full min-h-screen relative bg-gradient-to-br from-slate-900 via-slate-800 to-emerald-900">
        <div className="pt-24 pb-8">
          <div className="container mx-auto px-4">
            <div className="max-w-6xl mx-auto text-center">
              <Loader2 className="w-12 h-12 text-blue-400 animate-spin mx-auto mb-4" />
              <p className="text-gray-400 text-lg">Loading query results...</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Extract data for display
  const queryData = currentQuery.result?.payload?.data || [];
  const columns = queryData.length > 0 ? Object.keys(queryData[0]) : [];
  const sqlQuery = currentQuery.result?.payload?.sql || "";

  return (
    <div className="w-full min-h-screen relative bg-gradient-to-br from-slate-900 via-slate-800 to-emerald-900">
      {/* Add top padding to account for fixed navbar */}
      <div className="pt-24 pb-8">
        <div className="container mx-auto px-4">
          <div className="max-w-7xl mx-auto">
            {/* Header with Back Buttons */}
            <div className="mb-8">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-gradient-to-br from-blue-500/30 to-blue-600/20 rounded-xl flex items-center justify-center border border-blue-500/40">
                    <Database className="w-6 h-6 text-blue-400" />
                  </div>
                  <div>
                    <h1 className="text-3xl font-bold text-white">
                      Query Results
                    </h1>
                    <p className="text-gray-400">
                      Your natural language query results with interactive visualization
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Button
                    onClick={handleBackToQuery}
                    variant="outline"
                    className="border-blue-400/30 text-blue-400 hover:bg-blue-400/10"
                  >
                    <ArrowLeft className="w-4 h-4 mr-2" />
                    Back to Query
                  </Button>
                  <Button
                    onClick={handleBackToDashboard}
                    variant="outline"
                    className="border-gray-400/30 text-gray-400 hover:bg-gray-400/10"
                  >
                    Dashboard
                  </Button>
                  <Button
                    onClick={() => {
                      sessionStorage.removeItem("databaseQueryResult");
                      toast.success("Results cleared");
                      router.push("/database-query");
                    }}
                    variant="outline"
                    className="border-red-400/30 text-red-400 hover:bg-red-400/10"
                  >
                    Clear Results
                  </Button>
                </div>
              </div>
            </div>

            {/* Query Information */}
            <div className="mb-8">
              <Card className="bg-gray-900/50 border-blue-400/30">
                <CardHeader>
                  <CardTitle className="text-blue-400 flex items-center gap-2">
                    <FileText className="w-5 h-5" />
                    Query Information
                  </CardTitle>
                  <CardDescription className="text-gray-400">
                    Details about your natural language query and generated SQL
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Query Info */}
                  <div className="flex items-center gap-4 text-sm">
                    <div className="flex items-center gap-2">
                      <User className="w-4 h-4 text-blue-400" />
                      <span className="text-blue-400">User:</span>
                      <span className="text-white">
                        {currentQuery.userId}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Clock className="w-4 h-4 text-purple-400" />
                      <span className="text-purple-400">Time:</span>
                      <span className="text-white">
                        {new Date(currentQuery.timestamp).toLocaleString()}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-green-400" />
                      <span className="text-green-400">Status:</span>
                      <Badge variant="outline" className="border-green-400/30 text-green-400">
                        Success
                      </Badge>
                    </div>
                  </div>

                  {/* Natural Language Query */}
                  <div className="p-3 bg-blue-900/20 border border-blue-400/30 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <FileText className="w-4 h-4 text-blue-400" />
                      <span className="text-blue-400 font-medium">
                        Natural Language Query:
                      </span>
                    </div>
                    <p className="text-white">{currentQuery.query}</p>
                  </div>

                  {/* Generated SQL */}
                  {sqlQuery && (
                    <div className="p-3 bg-green-900/20 border border-green-400/30 rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <Code className="w-4 h-4 text-green-400" />
                        <span className="text-green-400 font-medium">
                          Generated SQL:
                        </span>
                      </div>
                      <code className="text-white text-sm bg-gray-800/50 p-2 rounded block overflow-x-auto">
                        {sqlQuery}
                      </code>
                    </div>
                  )}

                  {/* Results Summary */}
                  <div className="p-3 bg-purple-900/20 border border-purple-400/30 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Database className="w-4 h-4 text-purple-400" />
                      <span className="text-purple-400 font-medium">
                        Results Summary:
                      </span>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-purple-400">
                          {queryData.length}
                        </div>
                        <div className="text-xs text-gray-400">Total Rows</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-purple-400">
                          {columns.length}
                        </div>
                        <div className="text-xs text-gray-400">Columns</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-purple-400">
                          {currentQuery.result?.payload?.status_code || "N/A"}
                        </div>
                        <div className="text-xs text-gray-400">Status Code</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-purple-400">
                          ✓
                        </div>
                        <div className="text-xs text-gray-400">Success</div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Results Tabs */}
            <div className="mb-8">
              <div className="flex items-center gap-2 mb-4">
                <Button
                  variant={activeTab === "table" ? "default" : "outline"}
                  onClick={() => setActiveTab("table")}
                  className={
                    activeTab === "table"
                      ? "bg-blue-600 text-white"
                      : "border-blue-400/30 text-blue-400 hover:bg-blue-400/10"
                  }
                >
                  <Database className="w-4 h-4 mr-2" />
                  Table View
                </Button>
                <Button
                  variant={activeTab === "charts" ? "default" : "outline"}
                  onClick={() => setActiveTab("charts")}
                  className={
                    activeTab === "charts"
                      ? "bg-green-600 text-white"
                      : "border-green-400/30 text-green-400 hover:bg-green-400/10"
                  }
                >
                  <BarChart3 className="w-4 h-4 mr-2" />
                  Charts & Visualization
                </Button>
              </div>

              {/* Tab Content */}
              {activeTab === "table" ? (
                <QueryResultsTable 
                  data={queryData}
                  columns={columns}
                />
              ) : (
                <QueryCharts 
                  data={queryData}
                  columns={columns}
                />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 