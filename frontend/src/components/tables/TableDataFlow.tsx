"use client";

import React, { useState, useEffect, useCallback, useMemo } from "react";
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  MiniMap,
  Background,
  BackgroundVariant,
  Connection,
  NodeTypes,
  Handle,
  Position,
} from "reactflow";
import "reactflow/dist/style.css";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Database,
  Maximize2,
  Minimize2,
  RefreshCw,
  Settings,
  Eye,
  Hash,
  Calendar,
  DollarSign,
  FileText,
} from "lucide-react";

interface TableDataFlowProps {
  tableData: {
    columns: string[];
    rows: any[][];
    total_count: number;
  } | null;
  tableName: string;
}

// Custom node component for table rows
const TableRowNode = ({ data }: { data: any }) => {
  const { row, columns, index, isHeader } = data;

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 min-w-[300px] ${
        isHeader
          ? "bg-green-600/20 border-green-400/50 text-green-300"
          : "bg-slate-800/90 border-slate-600/50 text-white hover:border-blue-400/50 transition-colors"
      }`}
    >
      <Handle type="target" position={Position.Top} className="w-2 h-2" />
      
      <div className="space-y-2">
        {isHeader ? (
          <div className="text-center">
            <div className="flex items-center justify-center gap-2 mb-2">
              <Database className="h-4 w-4" />
              <span className="font-bold">Table Columns</span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {columns.map((col: string, idx: number) => (
                <Badge key={idx} variant="secondary" className="text-xs">
                  {col}
                </Badge>
              ))}
            </div>
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between mb-2">
              <Badge className="bg-blue-500/20 text-blue-400 text-xs">
                Row {index + 1}
              </Badge>
              <Hash className="h-3 w-3 text-slate-400" />
            </div>
            <div className="space-y-1">
              {columns.slice(0, 4).map((col: string, idx: number) => {
                const value = row[idx];
                return (
                  <div key={idx} className="flex justify-between items-center text-sm">
                    <span className="text-slate-400 truncate max-w-[100px]">
                      {col}:
                    </span>
                    <span className="text-white font-medium truncate max-w-[150px]">
                      {formatValue(value)}
                    </span>
                  </div>
                );
              })}
              {columns.length > 4 && (
                <div className="text-xs text-slate-500 text-center">
                  +{columns.length - 4} more columns
                </div>
              )}
            </div>
          </>
        )}
      </div>
      
      <Handle type="source" position={Position.Bottom} className="w-2 h-2" />
    </div>
  );
};

// Format values for display
const formatValue = (value: any): string => {
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return value.toLocaleString();
  if (typeof value === "string") {
    if (value.length > 20) return value.substring(0, 20) + "...";
    return value;
  }
  return String(value);
};

// Custom node component for column headers
const ColumnHeaderNode = ({ data }: { data: any }) => {
  const { column, dataType, sampleValues } = data;

  const getColumnIcon = (column: string, dataType: string) => {
    const colLower = column.toLowerCase();
    if (colLower.includes("id")) return <Hash className="h-4 w-4 text-blue-400" />;
    if (colLower.includes("date") || colLower.includes("time")) return <Calendar className="h-4 w-4 text-purple-400" />;
    if (colLower.includes("price") || colLower.includes("amount") || colLower.includes("cost")) return <DollarSign className="h-4 w-4 text-green-400" />;
    return <FileText className="h-4 w-4 text-slate-400" />;
  };

  return (
    <div className="px-4 py-3 rounded-lg border-2 bg-slate-700/90 border-slate-500/50 text-white min-w-[200px]">
      <Handle type="target" position={Position.Top} className="w-2 h-2" />
      
      <div className="text-center space-y-2">
        <div className="flex items-center justify-center gap-2">
          {getColumnIcon(column, dataType)}
          <span className="font-bold text-sm">{column}</span>
        </div>
        
        <Badge variant="outline" className="text-xs">
          {dataType || "unknown"}
        </Badge>
        
        {sampleValues && sampleValues.length > 0 && (
          <div className="space-y-1">
            <div className="text-xs text-slate-400">Sample values:</div>
            {sampleValues.slice(0, 3).map((value: any, idx: number) => (
              <div key={idx} className="text-xs text-slate-300 truncate">
                {formatValue(value)}
              </div>
            ))}
          </div>
        )}
      </div>
      
      <Handle type="source" position={Position.Bottom} className="w-2 h-2" />
    </div>
  );
};

const nodeTypes: NodeTypes = {
  tableRow: TableRowNode,
  columnHeader: ColumnHeaderNode,
};

export function TableDataFlow({ tableData, tableName }: TableDataFlowProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [viewMode, setViewMode] = useState<"rows" | "columns">("rows");
  const [maxRows, setMaxRows] = useState(20);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Generate nodes and edges based on view mode
  const generateFlowData = useCallback(() => {
    if (!tableData || !tableData.rows.length) {
      setNodes([]);
      setEdges([]);
      return;
    }

    const newNodes: Node[] = [];
    const newEdges: Edge[] = [];

    if (viewMode === "rows") {
      // Row-based view: Each row is a node
      const rowsToShow = Math.min(maxRows, tableData.rows.length);
      
      // Add header node
      newNodes.push({
        id: "header",
        type: "tableRow",
        position: { x: 400, y: 0 },
        data: {
          columns: tableData.columns,
          isHeader: true,
        },
      });

      // Add row nodes
      for (let i = 0; i < rowsToShow; i++) {
        const row = tableData.rows[i];
        const nodeId = `row-${i}`;
        
        // Calculate position in a grid layout
        const cols = Math.ceil(Math.sqrt(rowsToShow));
        const col = i % cols;
        const rowIdx = Math.floor(i / cols);
        
        newNodes.push({
          id: nodeId,
          type: "tableRow",
          position: {
            x: col * 350 + (col * 50),
            y: (rowIdx + 1) * 200 + 100,
          },
          data: {
            row,
            columns: tableData.columns,
            index: i,
            isHeader: false,
          },
        });

        // Connect header to first few rows
        if (i < 5) {
          newEdges.push({
            id: `header-${nodeId}`,
            source: "header",
            target: nodeId,
            type: "smoothstep",
            style: { stroke: "#10b981", strokeWidth: 2 },
            animated: true,
          });
        }
      }
    } else {
      // Column-based view: Each column is a node
      tableData.columns.forEach((column, index) => {
        // Get sample values for this column
        const sampleValues = tableData.rows
          .slice(0, 10)
          .map(row => row[index])
          .filter(val => val !== null && val !== undefined);

        // Calculate position in a circular layout
        const angle = (index / tableData.columns.length) * 2 * Math.PI;
        const radius = 300;
        const centerX = 400;
        const centerY = 300;
        
        newNodes.push({
          id: `column-${index}`,
          type: "columnHeader",
          position: {
            x: centerX + Math.cos(angle) * radius - 100,
            y: centerY + Math.sin(angle) * radius - 50,
          },
          data: {
            column,
            dataType: "string", // You could infer this from the data
            sampleValues: sampleValues.slice(0, 3),
          },
        });

        // Connect columns that might be related (e.g., ID columns to other columns)
        if (column.toLowerCase().includes("id") && index < tableData.columns.length - 1) {
          const nextIndex = (index + 1) % tableData.columns.length;
          newEdges.push({
            id: `column-${index}-${nextIndex}`,
            source: `column-${index}`,
            target: `column-${nextIndex}`,
            type: "smoothstep",
            style: { stroke: "#6366f1", strokeWidth: 1, strokeDasharray: "5,5" },
          });
        }
      });
    }

    setNodes(newNodes);
    setEdges(newEdges);
  }, [tableData, viewMode, maxRows]);

  // Regenerate flow when data or settings change
  useEffect(() => {
    generateFlowData();
  }, [generateFlowData]);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  if (!tableData) {
    return (
      <Card className="bg-slate-800/50 border-slate-700">
        <CardContent className="text-center py-12">
          <Database className="h-12 w-12 text-slate-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-white mb-2">No Data Available</h3>
          <p className="text-slate-400">Load table data to see the flow visualization</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Controls */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardContent className="pt-6">
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-sm text-slate-400">View Mode:</span>
              <Select value={viewMode} onValueChange={(value: "rows" | "columns") => setViewMode(value)}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="rows">Rows</SelectItem>
                  <SelectItem value="columns">Columns</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {viewMode === "rows" && (
              <div className="flex items-center gap-2">
                <span className="text-sm text-slate-400">Max Rows:</span>
                <Select value={maxRows.toString()} onValueChange={(value) => setMaxRows(parseInt(value))}>
                  <SelectTrigger className="w-20">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="10">10</SelectItem>
                    <SelectItem value="20">20</SelectItem>
                    <SelectItem value="50">50</SelectItem>
                    <SelectItem value="100">100</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            )}

            <div className="flex items-center gap-2">
              <Badge className="bg-blue-500/20 text-blue-400">
                {tableName}
              </Badge>
              <Badge variant="secondary">
                {tableData.total_count.toLocaleString()} total rows
              </Badge>
              <Badge variant="secondary">
                {tableData.columns.length} columns
              </Badge>
            </div>

            <div className="flex gap-2 ml-auto">
              <Button
                onClick={generateFlowData}
                variant="outline"
                size="sm"
              >
                <RefreshCw className="h-4 w-4" />
              </Button>
              <Button
                onClick={() => setIsFullscreen(!isFullscreen)}
                variant="outline"
                size="sm"
              >
                {isFullscreen ? (
                  <Minimize2 className="h-4 w-4" />
                ) : (
                  <Maximize2 className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* React Flow */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-white">
            <Eye className="h-5 w-5" />
            {viewMode === "rows" ? "Table Rows Flow" : "Table Columns Flow"}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className={`${isFullscreen ? "fixed inset-0 z-50 bg-slate-900" : "h-[600px]"} w-full`}>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              nodeTypes={nodeTypes}
              fitView
              attributionPosition="bottom-left"
              className="bg-slate-900"
            >
              <Controls className="bg-slate-800 border-slate-600" />
              <MiniMap
                className="bg-slate-800 border-slate-600"
                nodeColor={(node) => {
                  if (node.type === "columnHeader") return "#6366f1";
                  if (node.data?.isHeader) return "#10b981";
                  return "#475569";
                }}
              />
              <Background 
                variant={BackgroundVariant.Dots} 
                gap={20} 
                size={1} 
                color="#374151"
              />
            </ReactFlow>
          </div>
        </CardContent>
      </Card>

      {/* Legend */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardContent className="pt-6">
          <div className="flex flex-wrap gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-green-600/20 border border-green-400/50 rounded"></div>
              <span className="text-slate-400">Header/Column Info</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-slate-800 border border-slate-600 rounded"></div>
              <span className="text-slate-400">Data Rows</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-slate-700 border border-slate-500 rounded"></div>
              <span className="text-slate-400">Column Headers</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-0.5 bg-green-500"></div>
              <span className="text-slate-400">Data Flow</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-0.5 bg-blue-500 border-dashed"></div>
              <span className="text-slate-400">Column Relations</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}