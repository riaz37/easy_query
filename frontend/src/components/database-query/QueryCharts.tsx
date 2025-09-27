"use client";

import React, { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
} from "recharts";
import { BarChart3, PieChart as PieChartIcon, TrendingUp, BarChart2 } from "lucide-react";

interface QueryChartsProps {
  data: any[];
  columns: string[];
}

type ChartType = "bar" | "line" | "pie" | "area";

const COLORS = [
  "rgba(13, 172, 92, 0.6)", "rgba(13, 172, 92, 0.5)", "rgba(13, 172, 92, 0.4)", "rgba(13, 172, 92, 0.3)", 
  "rgba(139, 92, 246, 0.6)", "rgba(6, 182, 212, 0.6)", "rgba(132, 204, 22, 0.6)", "rgba(249, 115, 22, 0.6)"
];

export function QueryCharts({ data, columns }: QueryChartsProps) {
  const [chartType, setChartType] = useState<ChartType>("bar");
  const [xAxis, setXAxis] = useState<string>("");
  const [yAxis, setYAxis] = useState<string>("");
  const [aggregation, setAggregation] = useState<"sum" | "count" | "average">("sum");

  // Auto-detect suitable columns for charts
  const chartableColumns = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    return columns.filter(column => {
      const sampleValues = data.slice(0, 10).map(row => row[column]);
      const hasNumericValues = sampleValues.some(val => 
        typeof val === "number" && !isNaN(val)
      );
      const hasStringValues = sampleValues.some(val => 
        typeof val === "string" && val.length < 50
      );
      
      return hasNumericValues || hasStringValues;
    });
  }, [data, columns]);

  // Auto-select default axes
  useMemo(() => {
    if (chartableColumns.length >= 2 && !xAxis && !yAxis) {
      // Find a good string column for X-axis
      const stringColumn = chartableColumns.find(col => {
        const sampleValues = data.slice(0, 10).map(row => row[col]);
        return sampleValues.some(val => typeof val === "string" && val.length < 50);
      });
      
      // Find a good numeric column for Y-axis
      const numericColumn = chartableColumns.find(col => {
        const sampleValues = data.slice(0, 10).map(row => row[col]);
        return sampleValues.some(val => typeof val === "number" && !isNaN(val));
      });
      
      if (stringColumn) setXAxis(stringColumn);
      if (numericColumn) setYAxis(numericColumn);
    }
  }, [chartableColumns, data, xAxis, yAxis]);

  // Helper function to format dates
  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString);
      if (isNaN(date.getTime())) return dateString; // Return original if not a valid date
      
      const options: Intl.DateTimeFormatOptions = {
        day: 'numeric',
        month: 'long',
        year: 'numeric'
      };
      return date.toLocaleDateString('en-US', options);
    } catch (error) {
      return dateString; // Return original if formatting fails
    }
  };

  // Helper function to check if a value looks like a date
  const isDateString = (value: any) => {
    if (typeof value !== 'string') return false;
    const date = new Date(value);
    return !isNaN(date.getTime()) && value.match(/\d{1,2}[-\/]\d{1,2}[-\/]\d{2,4}/);
  };

  // Process data for charts
  const chartData = useMemo(() => {
    if (!xAxis || !yAxis || !data || data.length === 0) return [];
    
    if (chartType === "pie") {
      // For pie charts, group by X-axis and aggregate Y-axis
      const grouped = data.reduce((acc: any, row) => {
        let key = String(row[xAxis] || "Unknown");
        // Format date if it looks like a date
        if (isDateString(key)) {
          key = formatDate(key);
        }
        const value = Number(row[yAxis]) || 0;
        
        if (!acc[key]) acc[key] = 0;
        acc[key] += value;
        return acc;
      }, {});
      
      return Object.entries(grouped).map(([name, value]) => ({
        name: name.length > 20 ? name.substring(0, 20) + "..." : name,
        value,
        fullName: name,
      }));
    } else {
      // For other charts, group by X-axis and aggregate Y-axis
      const grouped = data.reduce((acc: any, row) => {
        let key = String(row[xAxis] || "Unknown");
        // Format date if it looks like a date
        if (isDateString(key)) {
          key = formatDate(key);
        }
        const value = Number(row[yAxis]) || 0;
        
        if (!acc[key]) acc[key] = { count: 0, sum: 0, values: [] };
        acc[key].count += 1;
        acc[key].sum += value;
        acc[key].values.push(value);
        return acc;
      }, {});
      
      return Object.entries(grouped).map(([name, stats]: [string, any]) => {
        let aggregatedValue;
        switch (aggregation) {
          case "sum":
            aggregatedValue = stats.sum;
            break;
          case "count":
            aggregatedValue = stats.count;
            break;
          case "average":
            aggregatedValue = stats.sum / stats.count;
            break;
          default:
            aggregatedValue = stats.sum;
        }
        
        return {
          name: name.length > 20 ? name.substring(0, 20) + "..." : name,
          value: aggregatedValue,
          fullName: name,
          count: stats.count,
        };
      }).sort((a, b) => b.value - a.value);
    }
  }, [data, xAxis, yAxis, chartType, aggregation]);

  // Render chart based on type
  const renderChart = () => {
    if (!xAxis || !yAxis || chartData.length === 0) {
      return (
        <div className="h-64 flex items-center justify-center text-gray-400">
          <div className="text-center">
            <BarChart3 className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>Select columns to display chart</p>
          </div>
        </div>
      );
    }

    const commonProps = {
      data: chartData,
      margin: { top: 20, right: 30, left: 20, bottom: 80 },
    };

    switch (chartType) {
      case "bar":
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart {...commonProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="name" 
                stroke="#9CA3AF"
                angle={-45}
                textAnchor="end"
                height={100}
                interval={0}
                tick={{ fontSize: 12 }}
              />
              <YAxis stroke="#9CA3AF" />
              <Legend />
              <Bar 
                dataKey="value" 
                fill="rgba(13, 172, 92, 0.6)" 
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        );

      case "line":
        return (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart {...commonProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="name" 
                stroke="#9CA3AF"
                angle={-45}
                textAnchor="end"
                height={100}
                interval={0}
                tick={{ fontSize: 12 }}
              />
              <YAxis stroke="#9CA3AF" />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="rgba(13, 172, 92, 0.6)" 
                strokeWidth={3}
                dot={{ fill: "rgba(13, 172, 92, 0.6)", strokeWidth: 2, r: 4 }}
                activeDot={{ r: 6, stroke: "rgba(13, 172, 92, 0.6)", strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        );

      case "pie":
        return (
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        );

      case "area":
        return (
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart {...commonProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="name" 
                stroke="#9CA3AF"
                angle={-45}
                textAnchor="end"
                height={100}
                interval={0}
                tick={{ fontSize: 12 }}
              />
              <YAxis stroke="#9CA3AF" />
              <Legend />
              <Area 
                type="monotone" 
                dataKey="value" 
                stroke="rgba(13, 172, 92, 0.6)" 
                fill="rgba(13, 172, 92, 0.6)" 
                fillOpacity={0.3}
              />
            </AreaChart>
          </ResponsiveContainer>
        );

      default:
        return null;
    }
  };

  if (!data || data.length === 0) {
    return (
      <div className="text-center py-8">
        <div className="text-white">
          No data available for charts
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
        {/* Chart Controls */}
        <div className="grid grid-cols-4 gap-4">
          <div className="space-y-2">
            <label className="text-sm text-white font-medium">Type</label>
            <Select value={chartType} onValueChange={(value: ChartType) => setChartType(value)}>
              <SelectTrigger 
                className="bg-white/10 text-white px-3 py-2 text-sm focus:outline-none focus:ring-0"
                style={{ 
                  outline: 'none',
                  borderRadius: "99px",
                  border: "1px solid var(--components-button-outlined, rgba(145, 158, 171, 0.32))"
                }}
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="modal-select-content-enhanced">
                <SelectItem value="bar" className="dropdown-item">
                  <div className="flex items-center gap-2">
                    <BarChart2 className="w-4 h-4" />
                    Bar
                  </div>
                </SelectItem>
                <SelectItem value="line" className="dropdown-item">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-4 h-4" />
                    Line
                  </div>
                </SelectItem>
                <SelectItem value="pie" className="dropdown-item">
                  <div className="flex items-center gap-2">
                    <PieChartIcon className="w-4 h-4" />
                    Pie
                  </div>
                </SelectItem>
                <SelectItem value="area" className="dropdown-item">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="w-4 h-4" />
                    Area
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-sm text-white font-medium">Aggregation</label>
            <Select value={aggregation} onValueChange={(value: "sum" | "count" | "average") => setAggregation(value)}>
              <SelectTrigger 
                className="bg-white/10 text-white px-3 py-2 text-sm focus:outline-none focus:ring-0"
                style={{ 
                  outline: 'none',
                  borderRadius: "99px",
                  border: "1px solid var(--components-button-outlined, rgba(145, 158, 171, 0.32))"
                }}
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="modal-select-content-enhanced">
                <SelectItem value="sum" className="dropdown-item">Sum</SelectItem>
                <SelectItem value="count" className="dropdown-item">Count</SelectItem>
                <SelectItem value="average" className="dropdown-item">Average</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-sm text-white font-medium">X-axis</label>
            <Select value={xAxis} onValueChange={setXAxis}>
              <SelectTrigger 
                className="bg-white/10 text-white px-3 py-2 text-sm focus:outline-none focus:ring-0"
                style={{ 
                  outline: 'none',
                  borderRadius: "99px",
                  border: "1px solid var(--components-button-outlined, rgba(145, 158, 171, 0.32))"
                }}
              >
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent className="modal-select-content-enhanced">
                {chartableColumns.map((column) => (
                  <SelectItem key={column} value={column} className="dropdown-item">
                    {column}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-sm text-white font-medium">Y-axis</label>
            <Select value={yAxis} onValueChange={setYAxis}>
              <SelectTrigger 
                className="bg-white/10 text-white px-3 py-2 text-sm focus:outline-none focus:ring-0"
                style={{ 
                  outline: 'none',
                  borderRadius: "99px",
                  border: "1px solid var(--components-button-outlined, rgba(145, 158, 171, 0.32))"
                }}
              >
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent className="modal-select-content-enhanced">
                {chartableColumns.map((column) => (
                  <SelectItem key={column} value={column} className="dropdown-item">
                    {column}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Chart Display */}
        <div 
          className="mt-6 p-4"
          style={{
            background: "var(--primary-8, rgba(19, 245, 132, 0.08))",
            border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
            borderRadius: "16px"
          }}
        >
          {renderChart()}
        </div>
    </div>
  );
} 