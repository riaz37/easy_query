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
  Tooltip,
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
  "#3B82F6", "#10B981", "#F59E0B", "#EF4444", 
  "#8B5CF6", "#06B6D4", "#84CC16", "#F97316"
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

  // Process data for charts
  const chartData = useMemo(() => {
    if (!xAxis || !yAxis || !data || data.length === 0) return [];
    
    if (chartType === "pie") {
      // For pie charts, group by X-axis and aggregate Y-axis
      const grouped = data.reduce((acc: any, row) => {
        const key = String(row[xAxis] || "Unknown");
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
        const key = String(row[xAxis] || "Unknown");
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
      margin: { top: 5, right: 30, left: 20, bottom: 5 },
    };

    switch (chartType) {
      case "bar":
        return (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart {...commonProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="name" 
                stroke="#9CA3AF"
                angle={-45}
                textAnchor="end"
                height={80}
                interval={0}
              />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{
                  backgroundColor: "#1F2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#F9FAFB",
                }}
                labelStyle={{ color: "#9CA3AF" }}
              />
              <Legend />
              <Bar 
                dataKey="value" 
                fill="#3B82F6" 
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        );

      case "line":
        return (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart {...commonProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="name" 
                stroke="#9CA3AF"
                angle={-45}
                textAnchor="end"
                height={80}
                interval={0}
              />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{
                  backgroundColor: "#1F2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#F9FAFB",
                }}
                labelStyle={{ color: "#9CA3AF" }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#3B82F6" 
                strokeWidth={3}
                dot={{ fill: "#3B82F6", strokeWidth: 2, r: 4 }}
                activeDot={{ r: 6, stroke: "#3B82F6", strokeWidth: 2 }}
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
              <Tooltip 
                contentStyle={{
                  backgroundColor: "#1F2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#F9FAFB",
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        );

      case "area":
        return (
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart {...commonProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="name" 
                stroke="#9CA3AF"
                angle={-45}
                textAnchor="end"
                height={80}
                interval={0}
              />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{
                  backgroundColor: "#1F2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#F9FAFB",
                }}
                labelStyle={{ color: "#9CA3AF" }}
              />
              <Legend />
              <Area 
                type="monotone" 
                dataKey="value" 
                stroke="#3B82F6" 
                fill="#3B82F6" 
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
      <Card className="bg-gray-900/50 border-emerald-400/30">
        <CardContent className="pt-12 pb-12 text-center">
          <div className="text-gray-400">
            No data available for charts
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-gray-900/50 border-emerald-400/30">
      <CardHeader>
        <CardTitle className="text-green-400 flex items-center gap-2">
          <BarChart3 className="w-5 h-5" />
          Data Visualization
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Chart Controls */}
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-2">
            <label className="text-sm text-gray-400">Chart Type</label>
            <Select value={chartType} onValueChange={(value: ChartType) => setChartType(value)}>
              <SelectTrigger className="bg-gray-800/50 border-emerald-400/30 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-gray-800 border-emerald-400/30">
                <SelectItem value="bar">
                  <div className="flex items-center gap-2">
                    <BarChart2 className="w-4 h-4" />
                    Bar Chart
                  </div>
                </SelectItem>
                <SelectItem value="line">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-4 h-4" />
                    Line Chart
                  </div>
                </SelectItem>
                <SelectItem value="pie">
                  <div className="flex items-center gap-2">
                    <PieChartIcon className="w-4 h-4" />
                    Pie Chart
                  </div>
                </SelectItem>
                <SelectItem value="area">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="w-4 h-4" />
                    Area Chart
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-sm text-gray-400">Aggregation</label>
            <Select value={aggregation} onValueChange={(value: "sum" | "count" | "average") => setAggregation(value)}>
              <SelectTrigger className="bg-gray-800/50 border-emerald-400/30 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-gray-800 border-emerald-400/30">
                <SelectItem value="sum">Sum</SelectItem>
                <SelectItem value="count">Count</SelectItem>
                <SelectItem value="average">Average</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-2">
            <label className="text-sm text-gray-400">X-Axis (Categories)</label>
            <Select value={xAxis} onValueChange={setXAxis}>
              <SelectTrigger className="bg-gray-800/50 border-emerald-400/30 text-white">
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent className="bg-gray-800 border-emerald-400/30">
                {chartableColumns.map((column) => (
                  <SelectItem key={column} value={column}>
                    {column}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-sm text-gray-400">Y-Axis (Values)</label>
            <Select value={yAxis} onValueChange={setYAxis}>
              <SelectTrigger className="bg-gray-800/50 border-emerald-400/30 text-white">
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent className="bg-gray-800 border-emerald-400/30">
                {chartableColumns.map((column) => (
                  <SelectItem key={column} value={column}>
                    {column}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Chart Display */}
        <div className="mt-6">
          {renderChart()}
        </div>

        {/* Chart Info */}
        {chartData.length > 0 && (
          <div className="p-3 bg-green-900/20 border border-green-400/30 rounded-lg">
            <div className="text-sm text-green-300">
              <strong>Chart Info:</strong>
              <div className="mt-1 text-green-200">
                • {chartData.length} data points
                • X-axis: {xAxis}
                • Y-axis: {yAxis} ({aggregation})
                • Chart type: {chartType.charAt(0).toUpperCase() + chartType.slice(1)}
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
} 