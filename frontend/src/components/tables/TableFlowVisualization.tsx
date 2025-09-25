"use client";

import React, { useCallback, useMemo } from "react";
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Connection,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  BackgroundVariant,
} from "reactflow";
import "reactflow/dist/style.css";
import { TableNode } from "./TableNode";
import {
  parseApiResponse,
  filterImportantTables,
  ParsedTable,
} from "@/lib/utils/sql-parser";

interface TableFlowVisualizationProps {
  tables?: ParsedTable[];
  rawData?: any; // For handling raw API response
  maxTables?: number; // Maximum number of tables to display
  onZoomIn?: () => void;
  onZoomOut?: () => void;
  onReactFlowInstance?: (instance: any) => void;
}

const nodeTypes = {
  tableNode: TableNode,
};

export function TableFlowVisualization({
  tables = [],
  rawData,
  maxTables = 15,
  onZoomIn,
  onZoomOut,
  onReactFlowInstance,
}: TableFlowVisualizationProps) {
  // Parse raw API data if provided, otherwise use tables prop
  const processedTables = useMemo(() => {
    if (rawData) {
      const parsed = parseApiResponse(rawData);
      return filterImportantTables(parsed.tables, maxTables);
    }
    return tables;
  }, [rawData, tables, maxTables]);

  // Generate nodes with intelligent layout and enhanced edges
  const { initialNodes, initialEdges } = useMemo(() => {
    const nodes: Node[] = [];
    const edges: Edge[] = [];

    if (processedTables.length === 0) {
      return { initialNodes: nodes, initialEdges: edges };
    }

    // Separate tables by relationship count for better layout
    const tablesWithRelationships = processedTables.filter(
      (t) => t.relationships && t.relationships.length > 0
    );
    const isolatedTables = processedTables.filter(
      (t) => !t.relationships || t.relationships.length === 0
    );

    // Create a more intelligent layout with increased spacing for full page
    const createLayout = () => {
      let currentX = 150;
      let currentY = 150;
      const nodeSpacing = 400;
      const rowHeight = 300;
      const maxCols = Math.ceil(Math.sqrt(processedTables.length));

      // Place connected tables first in a more organized way
      tablesWithRelationships.forEach((table, index) => {
        const row = Math.floor(index / maxCols);
        const col = index % maxCols;

        nodes.push({
          id: table.full_name,
          type: "tableNode",
          position: {
            x: col * nodeSpacing + currentX,
            y: row * rowHeight + currentY,
          },
          data: {
            table,
            label: table.name,
          },
        });
      });

      // Place isolated tables in a separate area
      const isolatedStartY =
        currentY +
        Math.ceil(tablesWithRelationships.length / maxCols) * rowHeight +
        150;
      isolatedTables.forEach((table, index) => {
        const row = Math.floor(index / maxCols);
        const col = index % maxCols;

        nodes.push({
          id: table.full_name,
          type: "tableNode",
          position: {
            x: col * nodeSpacing + currentX,
            y: row * rowHeight + isolatedStartY,
          },
          data: {
            table,
            label: table.name,
          },
        });
      });
    };

    createLayout();

    // Enhanced edges with better styling and colors
    processedTables.forEach((table) => {
      if (table.relationships) {
        table.relationships.forEach((relationship, index) => {
          const targetExists = processedTables.some(
            (t) => t.full_name === relationship.related_table
          );

          if (targetExists) {
            // Unified green color for all relationships
            const getRelationshipColor = (type: string) => {
              return { stroke: "#10b981", fill: "#10b981" }; // emerald green
            };

            const colors = getRelationshipColor(
              relationship.type || "foreign_key"
            );

            edges.push({
              id: `${table.full_name}-${relationship.related_table}-${index}`,
              source: table.full_name,
              target: relationship.related_table,
              type: "smoothstep",
              style: {
                stroke: colors.stroke,
                strokeWidth: 2,
              },
              markerEnd: {
                type: "arrowclosed",
                color: colors.stroke,
                width: 20,
                height: 20,
              },
              animated: true,
            });
          }
        });
      }
    });

    return { initialNodes: nodes, initialEdges: edges };
  }, [processedTables]);

  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selectedNodeId, setSelectedNodeId] = React.useState<string | null>(
    null
  );

  // Ref to store React Flow instance
  const reactFlowRef = React.useRef<any>(null);

  // Set up zoom functions when component mounts
  React.useEffect(() => {
    if (onZoomIn && onZoomOut) {
      // Call the parent's zoom functions to notify that setup is complete
      onZoomIn();
      onZoomOut();
    }
  }, [onZoomIn, onZoomOut]);

  // Pass React Flow instance to parent when it's available
  React.useEffect(() => {
    if (reactFlowRef.current && onReactFlowInstance) {
      onReactFlowInstance(reactFlowRef.current);
    }
  }, [onReactFlowInstance]);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  // Handle node selection
  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNodeId((prevId) => (prevId === node.id ? null : node.id));
  }, []);

  // Memoize the node and edge styles
  const nodeStyle = useCallback(
    (node: Node) => ({
      opacity:
        !selectedNodeId ||
        node.id === selectedNodeId ||
        edges.some(
          (e) =>
            (e.source === selectedNodeId && e.target === node.id) ||
            (e.target === selectedNodeId && e.source === node.id)
        )
          ? 1
          : 0.3,
      transition: "opacity 0.2s ease-in-out",
    }),
    [selectedNodeId, edges]
  );

  const edgeStyle = useCallback(
    (edge: Edge) => ({
      ...edge.style,
      opacity:
        !selectedNodeId ||
        edge.source === selectedNodeId ||
        edge.target === selectedNodeId
          ? 1
          : 0.2,
      strokeWidth:
        edge.source === selectedNodeId || edge.target === selectedNodeId
          ? 3
          : 2,
      transition: "all 0.2s ease-in-out",
    }),
    [selectedNodeId]
  );

  return (
    <div className="w-full h-full relative">
      <ReactFlow
        ref={reactFlowRef}
        nodes={nodes.map((node) => ({
          ...node,
          style: {
            ...node.style,
            ...nodeStyle(node),
          },
        }))}
        edges={edges.map((edge) => ({
          ...edge,
          style: edgeStyle(edge),
        }))}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onPaneClick={() => setSelectedNodeId(null)}
        onInit={(instance) => {
          reactFlowRef.current = instance;
          if (onReactFlowInstance) {
            onReactFlowInstance(instance);
          }
        }}
        nodeTypes={nodeTypes}
        fitView
        className="bg-transparent"
        proOptions={{ hideAttribution: true }}
        defaultViewport={{ x: 0, y: 0, zoom: 0.6 }}
        minZoom={0.2}
        maxZoom={3}
        nodesDraggable={true}
      >
        <Controls
          className="bg-gray-900/90 border-gray-700 backdrop-blur-sm"
          showInteractive={false}
          position="bottom-left"
        />
      </ReactFlow>
    </div>
  );
}
