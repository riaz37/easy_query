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
import { Input } from "@/components/ui/input";
import {
  Database,
  BarChart3,
  FileText,
  Clock,
  User,
  CheckCircle,
  Search,
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


  if (!currentQuery) {
    return (
      <PageLayout background={["frame", "gridframe"]} maxWidth="6xl">
        <div className="text-center">
          <Spinner
            size="lg"
            variant="accent-emerald"
            className="mx-auto mb-4"
          />
          <p className="text-gray-400 text-lg">Loading query results...</p>
        </div>
      </PageLayout>
    );
  }

  // Extract data for display
  const queryData = currentQuery.result?.payload?.data || [];
  const columns = queryData.length > 0 ? Object.keys(queryData[0]) : [];

  return (
    <PageLayout background={["frame", "gridframe"]} maxWidth="7xl">
      <div className="modal-enhanced">
        <div 
          className="modal-content-enhanced overflow-hidden"
          style={{
            background: `linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)),
linear-gradient(230.27deg, rgba(19, 245, 132, 0) 71.59%, rgba(19, 245, 132, 0.2) 98.91%),
linear-gradient(67.9deg, rgba(19, 245, 132, 0) 66.65%, rgba(19, 245, 132, 0.2) 100%)`,
            backdropFilter: "blur(30px)"
          }}
        >
          {/* Header Section */}
          <div className="p-6">
            {/* Title and Query Info */}
            <div className="mb-6">
              <h1 className="modal-title-enhanced text-3xl font-bold mb-2">
                Query
              </h1>
              <p className="text-white text-sm">{currentQuery.query}</p>
            </div>

            {/* Action Bar */}
            <div className="flex items-center justify-between">
              {/* Tabs */}
              <div className="flex gap-8">
                <button
                  onClick={() => setActiveTab("table")}
                  className={`text-sm font-medium pb-2 border-b-2 transition-colors cursor-pointer ${
                    activeTab === "table"
                      ? ""
                      : "text-gray-400 border-transparent hover:text-white"
                  }`}
                  style={activeTab === "table" ? { 
                    color: "var(--primary-main, rgba(19, 245, 132, 1))",
                    borderBottomColor: "var(--primary-main, rgba(19, 245, 132, 1))"
                  } : {}}
                >
                  Table View
                </button>
                <button
                  onClick={() => setActiveTab("charts")}
                  className={`text-sm font-medium pb-2 border-b-2 transition-colors cursor-pointer ${
                    activeTab === "charts"
                      ? ""
                      : "text-gray-400 border-transparent hover:text-white"
                  }`}
                  style={activeTab === "charts" ? { 
                    color: "var(--primary-main, rgba(19, 245, 132, 1))",
                    borderBottomColor: "var(--primary-main, rgba(19, 245, 132, 1))"
                  } : {}}
                >
                  Charts & Visualization
                </button>
              </div>

              {/* Search Input and Filter */}
              <div className="flex items-center gap-3">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <Input
                    type="text"
                    placeholder="Search results..."
                    className="pl-10 text-white placeholder:text-gray-400 rounded-full border-0 focus:ring-0 focus:outline-none"
                    style={{
                      background: "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
                      borderRadius: "999px",
                      height: "50px",
                      width: "300px"
                    }}
                  />
                </div>
                <Button
                  variant="outline"
                  size="icon"
                  className="border-0 text-white hover:bg-white/10 cursor-pointer"
                  style={{
                    background: "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
                    borderRadius: "118.8px",
                    width: "48px",
                    height: "48px"
                  }}
                >
                  <img src="/tables/filter.svg" alt="Filter" className="w-6 h-6" />
                </Button>
          </div>
        </div>
      </div>

          {/* Content Section */}
          <div className="px-6 pb-6">
        {activeTab === "table" ? (
          <QueryResultsTable data={queryData} columns={columns} />
        ) : (
          <QueryCharts data={queryData} columns={columns} />
        )}
          </div>
        </div>
      </div>
    </PageLayout>
  );
}
