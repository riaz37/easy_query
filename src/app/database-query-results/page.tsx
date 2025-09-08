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
      <PageLayout background="enhanced" maxWidth="6xl">
        <div className="text-center">
          <Spinner size="lg" variant="accent-emerald" className="mx-auto mb-4" />
          <p className="text-gray-400 text-lg">Loading query results...</p>
        </div>
      </PageLayout>
    );
  }

  // Extract data for display
  const queryData = currentQuery.result?.payload?.data || [];
  const columns = queryData.length > 0 ? Object.keys(queryData[0]) : [];

  return (
    <PageLayout background="enhanced" maxWidth="7xl">
      <PageHeader
        title="Query Results"
        description="Your natural language query results with interactive visualization"
        icon={<Database className="w-6 h-6 text-emerald-400" />}
        actions={
          <>
            <Button
              onClick={handleBackToQuery}
              variant="outline"
              className="border-emerald-400/30 text-emerald-400 hover:bg-emerald-400/10"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Query
            </Button>
            <Button
              onClick={handleBackToDashboard}
              variant="outline"
              className="border-emerald-400/30 text-emerald-400 hover:bg-emerald-400/10"
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
              className="border-emerald-400/30 text-emerald-400 hover:bg-emerald-400/10"
            >
              Clear Results
            </Button>
          </>
        }
      />

      {/* Query Information */}
      <div className="mb-8">
        <div className="card-enhanced">
          <div className="card-content-enhanced">
            <div className="card-header-enhanced">
              <h3 className="card-title-enhanced flex items-center gap-2">
                <FileText className="w-5 h-5" />
                Query Information
              </h3>
              <p className="card-description-enhanced">
                Details about your natural language query and generated SQL
              </p>
            </div>
            <div className="space-y-4">
              {/* Query Info */}
              <div className="flex items-center gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <User className="w-4 h-4 text-emerald-400" />
                  <span className="text-emerald-400">User:</span>
                  <span className="text-white">{currentQuery.userId}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4 text-emerald-400" />
                  <span className="text-emerald-400">Time:</span>
                  <span className="text-white">
                    {new Date(currentQuery.timestamp).toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-emerald-400" />
                  <span className="text-emerald-400">Status:</span>
                  <Badge
                    variant="outline"
                    className="border-emerald-400/30 text-emerald-400"
                  >
                    Success
                  </Badge>
                </div>
              </div>

              {/* Natural Language Query */}
              <div className="p-3 bg-emerald-500/10 border border-emerald-400/30 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <FileText className="w-4 h-4 text-emerald-400" />
                  <span className="text-emerald-400 font-medium">
                    Natural Language Query:
                  </span>
                </div>
                <p className="text-white">{currentQuery.query}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Results Tabs */}
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-4">
          <Button
            variant={activeTab === "table" ? "default" : "outline"}
            onClick={() => setActiveTab("table")}
            className={
              activeTab === "table"
                ? "bg-emerald-600 text-white"
                : "border-emerald-400/30 text-emerald-400 hover:bg-emerald-400/10"
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
                ? "bg-emerald-600 text-white"
                : "border-emerald-400/30 text-emerald-400 hover:bg-emerald-400/10"
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
