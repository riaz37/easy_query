"use client";

import React, { useState, useEffect } from "react";
import {
  BarChart3,
  TrendingUp,
  PieChart,
  Activity,
  Target,
  Layers,
  Map,
  Globe,
  Zap,
  Circle,
} from "lucide-react";
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
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
  ScatterChart,
  Scatter as RechartsScatter,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Treemap,
  FunnelChart,
  Funnel,
} from "recharts";

interface DynamicGraphProps {
  graphData: any;
  tableData: any[];
  columns: string[];
}

export function DynamicGraph({
  graphData,
  tableData,
  columns,
}: DynamicGraphProps) {
  const [processedData, setProcessedData] = useState<any[]>([]);

  useEffect(() => {
    if (!graphData || !tableData || !columns) {
      console.warn("DynamicGraph: Missing required props:", {
        graphData,
        tableData,
        columns,
      });
      return;
    }

    // Log the actual graphData structure for debugging
    console.log("DynamicGraph: Received graphData:", graphData);
    console.log("DynamicGraph: Available properties:", Object.keys(graphData));

    // Try to find column mapping in different possible locations
    let columnMapping = null;

    if (graphData.column_mapping) {
      columnMapping = graphData.column_mapping;
    } else if (graphData.analysis && graphData.analysis.columns_count) {
      // Generate default column mapping based on available data
      columnMapping = {
        x: columns[0] || "column1",
        y: columns[1] || "column2",
        color: columns[2] || "column3",
        size: columns[3] || "column4",
      };
      console.log(
        "DynamicGraph: Generated default column mapping:",
        columnMapping
      );
    } else if (columns.length >= 2) {
      // Use first two columns as default
      columnMapping = {
        x: columns[0],
        y: columns[1],
        color: columns[2] || columns[0],
        size: columns[3] || columns[1],
      };
      console.log(
        "DynamicGraph: Using first two columns as mapping:",
        columnMapping
      );
    }

    if (!columnMapping) {
      console.warn(
        "DynamicGraph: No column mapping available, cannot render chart"
      );
      setProcessedData([]);
      return;
    }

    const { graph_type } = graphData;

    // Process data based on column mapping and graph type
    const data = processDataForChart(tableData, columnMapping, graph_type);
    if (data) {
      setProcessedData(data);
    } else {
      console.warn("DynamicGraph: Failed to process chart data");
      setProcessedData([]);
    }
  }, [graphData, tableData, columns]);

  const processDataForChart = (data: any[], mapping: any, type: string) => {
    try {
      // Check if mapping exists and has required properties
      if (!mapping || typeof mapping !== "object") {
        console.warn("DynamicGraph: Invalid mapping object:", mapping);
        return null;
      }

      const { x, y, color, size } = mapping;

      if (!x || !y) {
        console.warn(
          "DynamicGraph: Missing required x or y mapping properties:",
          { x, y, mapping }
        );
        return null;
      }

      // Process data based on chart type
      switch (type) {
        case "scatter":
        case "bubble":
        case "3d_scatter":
          // For scatter plots, preserve individual data points
          return data.map((row: any) => ({
            name: String(row[x] || "Unknown"),
            value: parseFloat(row[y]) || 0,
            size: size ? parseFloat(row[size]) || 1 : 1,
            color: color ? row[color] : undefined,
          }));

        case "histogram":
        case "density":
          // For histograms, group data into bins
          const values = data
            .map((row) => parseFloat(row[y]) || 0)
            .filter((v) => !isNaN(v));
          const min = Math.min(...values);
          const max = Math.max(...values);
          const binCount = Math.min(10, Math.ceil(Math.sqrt(values.length)));
          const binSize = (max - min) / binCount;

          const bins: { [key: string]: number } = {};
          for (let i = 0; i < binCount; i++) {
            const binStart = min + i * binSize;
            const binEnd = min + (i + 1) * binSize;
            const binLabel = `${binStart.toFixed(1)}-${binEnd.toFixed(1)}`;
            bins[binLabel] = 0;
          }

          values.forEach((value) => {
            const binIndex = Math.floor((value - min) / binSize);
            const binStart = min + binIndex * binSize;
            const binEnd = min + (binIndex + 1) * binSize;
            const binLabel = `${binStart.toFixed(1)}-${binEnd.toFixed(1)}`;
            bins[binLabel]++;
          });

          return Object.entries(bins).map(([name, value]) => ({
            name,
            value,
          }));

        case "pie":
        case "donut":
        case "treemap":
        case "sunburst":
          // For pie charts, group data by x-axis values
          const groupedData = data.reduce((acc: any, row: any) => {
            const xValue = row[x] || "Unknown";
            const yValue = parseFloat(row[y]) || 0;

            if (acc[xValue]) {
              acc[xValue] += yValue;
            } else {
              acc[xValue] = yValue;
            }
            return acc;
          }, {});

          return Object.entries(groupedData).map(([name, value]) => ({
            name: String(name),
            value: Number(value),
          }));

        case "grouped_bar":
        case "stacked_bar":
          // For grouped/stacked bars, preserve category structure
          const categoryData = data.reduce((acc: any, row: any) => {
            const xValue = row[x] || "Unknown";
            const yValue = parseFloat(row[y]) || 0;
            const colorValue = color ? row[color] : "Default";

            if (!acc[xValue]) {
              acc[xValue] = {};
            }
            if (!acc[xValue][colorValue]) {
              acc[xValue][colorValue] = 0;
            }
            acc[xValue][colorValue] += yValue;
            return acc;
          }, {});

          return Object.entries(categoryData).map(
            ([name, categories]: [string, any]) => ({
              name,
              ...categories,
            })
          );

        case "heatmap":
        case "correlation_matrix":
          // For heatmaps, create matrix structure
          const uniqueX = [...new Set(data.map((row) => row[x]))];
          const uniqueY = [...new Set(data.map((row) => row[color || y]))];

          return uniqueX.map((xVal) => {
            const row: any = { name: xVal };
            uniqueY.forEach((yVal) => {
              const matchingRow = data.find(
                (r) => r[x] === xVal && r[color || y] === yVal
              );
              row[yVal] = matchingRow ? parseFloat(matchingRow[y]) || 0 : 0;
            });
            return row;
          });

        case "radar":
        case "polar":
          // For radar charts, use multiple metrics
          const metrics = columns.slice(1, 6); // Use up to 5 metrics
          return data.slice(0, 10).map((row, index) => {
            const point: any = { name: row[x] || `Point ${index + 1}` };
            metrics.forEach((metric) => {
              point[metric] = parseFloat(row[metric]) || 0;
            });
            return point;
          });

        default:
          // Default processing for bar, line, area charts
          const defaultGroupedData = data.reduce((acc: any, row: any) => {
            const xValue = row[x] || "Unknown";
            const yValue = parseFloat(row[y]) || 0;

            if (acc[xValue]) {
              acc[xValue] += yValue;
            } else {
              acc[xValue] = yValue;
            }
            return acc;
          }, {});

          return Object.entries(defaultGroupedData).map(([name, value]) => ({
            name: String(name),
            value: Number(value),
          }));
      }
    } catch (error) {
      console.error("Error processing chart data:", error);
      return null;
    }
  };

  if (!processedData || processedData.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-800/30 rounded-lg border border-gray-700/50">
        <div className="text-center">
          <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-2" />
          <p className="text-gray-400 mb-2">No chart data available</p>
          <p className="text-xs text-gray-500">
            {!graphData?.column_mapping
              ? "Missing column mapping"
              : !graphData?.graph_type
              ? "Missing graph type"
              : "No data to display"}
          </p>
        </div>
      </div>
    );
  }

  const colors = [
    "#3b82f6",
    "#10b981",
    "#f59e0b",
    "#ef4444",
    "#8b5cf6",
    "#06b6d4",
    "#84cc16",
    "#f97316",
    "#ec4899",
    "#06b6d4",
    "#8b5cf6",
    "#f59e0b",
  ];

  const getChartIcon = (graphType: string) => {
    switch (graphType) {
      case "bar":
      case "column":
      case "grouped_bar":
      case "stacked_bar":
        return <BarChart3 className="w-5 h-5 text-blue-400" />;
      case "line":
      case "multi_line":
      case "step":
        return <TrendingUp className="w-5 h-5 text-green-400" />;
      case "pie":
      case "donut":
      case "treemap":
      case "sunburst":
        return <PieChart className="w-5 h-5 text-purple-400" />;
      case "area":
      case "stacked_area":
        return <Activity className="w-5 h-5 text-orange-400" />;
      case "scatter":
      case "bubble":
      case "3d_scatter":
        return <Circle className="w-5 h-5 text-red-400" />;
      case "histogram":
      case "box":
      case "violin":
      case "density":
        return <Target className="w-5 h-5 text-yellow-400" />;
      case "heatmap":
      case "correlation_matrix":
        return <Layers className="w-5 h-5 text-indigo-400" />;
      case "radar":
      case "polar":
        return <Zap className="w-5 h-5 text-pink-400" />;
      case "choropleth":
      case "scatter_map":
      case "bubble_map":
        return <Map className="w-5 h-5 text-emerald-400" />;
      default:
        return <BarChart3 className="w-5 h-5 text-gray-400" />;
    }
  };

  const renderChart = () => {
    const { graph_type } = graphData;

    switch (graph_type) {
      case "column":
      case "bar":
        return (
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={processedData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="name"
                tick={{ fill: "#9ca3af" }}
                axisLine={{ stroke: "#374151" }}
              />
              <YAxis
                tick={{ fill: "#9ca3af" }}
                axisLine={{ stroke: "#374151" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#f9fafb",
                }}
              />
              <Legend />
              <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        );

      case "grouped_bar":
        return (
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={processedData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="name"
                tick={{ fill: "#9ca3af" }}
                axisLine={{ stroke: "#374151" }}
              />
              <YAxis
                tick={{ fill: "#9ca3af" }}
                axisLine={{ stroke: "#374151" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#f9fafb",
                }}
              />
              <Legend />
              {Object.keys(processedData[0] || {})
                .filter((key) => key !== "name")
                .map((key, index) => (
                  <Bar
                    key={key}
                    dataKey={key}
                    fill={colors[index % colors.length]}
                    radius={[4, 4, 0, 0]}
                  />
                ))}
            </BarChart>
          </ResponsiveContainer>
        );

      case "stacked_bar":
        return (
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={processedData} stackOffset="expand">
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="name"
                tick={{ fill: "#9ca3af" }}
                axisLine={{ stroke: "#374151" }}
              />
              <YAxis
                tick={{ fill: "#9ca3af" }}
                axisLine={{ stroke: "#374151" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#f9fafb",
                }}
              />
              <Legend />
              {Object.keys(processedData[0] || {})
                .filter((key) => key !== "name")
                .map((key, index) => (
                  <Bar
                    key={key}
                    dataKey={key}
                    fill={colors[index % colors.length]}
                    stackId="stack"
                    radius={[4, 4, 0, 0]}
                  />
                ))}
            </BarChart>
          </ResponsiveContainer>
        );

      case "line":
      case "multi_line":
      case "step":
        return (
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={processedData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="name"
                tick={{ fill: "#9ca3af" }}
                axisLine={{ stroke: "#374151" }}
              />
              <YAxis
                tick={{ fill: "#9ca3af" }}
                axisLine={{ stroke: "#374151" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#f9fafb",
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#10b981"
                strokeWidth={3}
                dot={{ fill: "#10b981", strokeWidth: 2, r: 5 }}
              />
            </LineChart>
          </ResponsiveContainer>
        );

      case "pie":
      case "donut":
        return (
          <ResponsiveContainer width="100%" height={350}>
            <RechartsPieChart>
              <Pie
                data={processedData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) =>
                  `${name} (${(percent * 100).toFixed(0)}%)`
                }
                outerRadius={graph_type === "donut" ? 120 : 120}
                innerRadius={graph_type === "donut" ? 60 : 0}
                fill="#8884d8"
                dataKey="value"
              >
                {processedData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={colors[index % colors.length]}
                  />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#f9fafb",
                }}
              />
            </RechartsPieChart>
          </ResponsiveContainer>
        );

      case "area":
        return (
          <ResponsiveContainer width="100%" height={350}>
            <AreaChart data={processedData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="name"
                tick={{ fill: "#9ca3af" }}
                axisLine={{ stroke: "#374151" }}
              />
              <YAxis
                tick={{ fill: "#9ca3af" }}
                axisLine={{ stroke: "#374151" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#f9fafb",
                }}
              />
              <Legend />
              <Area
                type="monotone"
                dataKey="value"
                stroke="#f59e0b"
                fill="#f59e0b"
                fillOpacity={0.6}
              />
            </AreaChart>
          </ResponsiveContainer>
        );

      case "scatter":
      case "bubble":
        return (
          <ResponsiveContainer width="100%" height={350}>
            <ScatterChart data={processedData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="name"
                tick={{ fill: "#9ca3af" }}
                axisLine={{ stroke: "#374151" }}
              />
              <YAxis
                tick={{ fill: "#9ca3af" }}
                axisLine={{ stroke: "#374151" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#f9fafb",
                }}
              />
              <Legend />
              <RechartsScatter
                dataKey="value"
                fill="#ef4444"
                shape="circle"
                r={graph_type === "bubble" ? 8 : 5}
              />
            </ScatterChart>
          </ResponsiveContainer>
        );

      case "histogram":
      case "density":
        return (
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={processedData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="name"
                tick={{ fill: "#9ca3af" }}
                axisLine={{ stroke: "#374151" }}
              />
              <YAxis
                tick={{ fill: "#9ca3af" }}
                axisLine={{ stroke: "#374151" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#f9fafb",
                }}
              />
              <Legend />
              <Bar dataKey="value" fill="#f59e0b" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        );

      case "radar":
        return (
          <ResponsiveContainer width="100%" height={350}>
            <RadarChart data={processedData}>
              <PolarGrid stroke="#374151" />
              <PolarAngleAxis dataKey="name" tick={{ fill: "#9ca3af" }} />
              <PolarRadiusAxis tick={{ fill: "#9ca3af" }} stroke="#374151" />
              <Radar
                dataKey="value"
                stroke="#8b5cf6"
                fill="#8b5cf6"
                fillOpacity={0.3}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#f9fafb",
                }}
              />
            </RadarChart>
          </ResponsiveContainer>
        );

              case "treemap":
          return (
            <ResponsiveContainer width="100%" height={350}>
              <Treemap
                data={processedData}
                dataKey="value"
                aspectRatio={4 / 3}
                stroke="#374151"
                fill="#3b82f6"
              >
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#1f2937",
                    border: "1px solid #374151",
                    borderRadius: "8px",
                    color: "#f9fafb",
                  }}
                />
              </Treemap>
            </ResponsiveContainer>
          );

      case "funnel":
        return (
          <ResponsiveContainer width="100%" height={350}>
            <FunnelChart data={processedData}>
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#f9fafb",
                }}
              />
              <Funnel dataKey="value" fill="#06b6d4" stroke="#374151" />
            </FunnelChart>
          </ResponsiveContainer>
        );

      default:
        // Default to bar chart
        return (
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={processedData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="name"
                tick={{ fill: "#9ca3af" }}
                axisLine={{ stroke: "#374151" }}
              />
              <YAxis
                tick={{ fill: "#9ca3af" }}
                axisLine={{ stroke: "#374151" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#f9fafb",
                }}
              />
              <Legend />
              <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        );
    }
  };

  const getChartTypeName = (graphType: string) => {
    const typeNames: { [key: string]: string } = {
      bar: "Bar Chart",
      column: "Column Chart",
      line: "Line Chart",
      area: "Area Chart",
      pie: "Pie Chart",
      donut: "Donut Chart",
      scatter: "Scatter Plot",
      bubble: "Bubble Chart",
      histogram: "Histogram",
      density: "Density Plot",
      grouped_bar: "Grouped Bar Chart",
      stacked_bar: "Stacked Bar Chart",
      radar: "Radar Chart",
      treemap: "Treemap",
      funnel: "Funnel Chart",
      heatmap: "Heatmap",
      correlation_matrix: "Correlation Matrix",
      "3d_scatter": "3D Scatter Plot",
      surface: "3D Surface Plot",
      contour: "Contour Plot",
      choropleth: "Choropleth Map",
      scatter_map: "Scatter Map",
      bubble_map: "Bubble Map",
    };

    return typeNames[graphType] || `${graphType} Chart`;
  };

  return (
    <div className="bg-gray-800/30 rounded-lg border border-gray-700/50 p-4">
      <div className="flex items-center gap-2 mb-4">
        {getChartIcon(graphData.graph_type)}
        <h4 className="text-lg font-semibold text-white">
          {getChartTypeName(graphData.graph_type)}
        </h4>
      </div>

      <div className="bg-gray-900/50 rounded-lg p-4">{renderChart()}</div>

      <div className="mt-4 text-sm text-gray-400">
        <p>
          <strong>X-Axis:</strong>{" "}
          {graphData.column_mapping?.x || "Auto-generated"}
        </p>
        <p>
          <strong>Y-Axis:</strong>{" "}
          {graphData.column_mapping?.y || "Auto-generated"}
        </p>
        {graphData.column_mapping?.color && (
          <p>
            <strong>Color:</strong> {graphData.column_mapping.color}
          </p>
        )}
        {graphData.column_mapping?.size && (
          <p>
            <strong>Size:</strong> {graphData.column_mapping.size}
          </p>
        )}
        {!graphData.column_mapping && (
          <p className="text-yellow-400">
            <strong>Note:</strong> Column mapping was auto-generated from
            available data
          </p>
        )}
      </div>
    </div>
  );
}
