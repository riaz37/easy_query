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
  CheckCircle,
  // Loader2,
} from "lucide-react";
import { Spinner } from "@/components/ui/loading";
import { toast } from "sonner";
import { QueryResultsTable } from "@/components/database-query/QueryResultsTable";
import { QueryCharts } from "@/components/database-query/QueryCharts";
import { PageLayout, PageHeader } from "@/components/layout/PageLayout";

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
      <PageLayout background="gradient" maxWidth="6xl">
        <div className="text-center">
          <Spinner size="lg" variant="accent-blue" className="mx-auto mb-4" />
          <p className="text-gray-400 text-lg">Loading query results...</p>
        </div>
      </PageLayout>
    );
  }

  // Extract data for display
  const queryData = currentQuery.result?.payload?.data || [];
  const columns = queryData.length > 0 ? Object.keys(queryData[0]) : [];

  return (
    <PageLayout background="gradient" maxWidth="7xl">
      <PageHeader
        title="Query Results"
        description="Your natural language query results with interactive visualization"
        icon={<Database className="w-6 h-6 text-blue-400" />}
        actions={
          <>
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
          </>
        }
      />

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
                <span className="text-white">{currentQuery.userId}</span>
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
                <Badge
                  variant="outline"
                  className="border-green-400/30 text-green-400"
                >
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
          <QueryResultsTable data={queryData} columns={columns} />
        ) : (
          <QueryCharts data={queryData} columns={columns} />
        )}
      </div>
    </PageLayout>
  );
}
