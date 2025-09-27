"use client";

import React, { useCallback, useMemo, useState, useEffect } from "react";
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Connection,
  useNodesState,
  useEdgesState,
  Controls,
  MarkerType,
  useReactFlow,
  ReactFlowProvider,
} from "reactflow";
import "reactflow/dist/style.css";
import { CompanyCard } from "./ui/CompanyCard";
import { EmptyState } from "./ui/EmptyState";
import { CompanyTreeSidebar } from "./ui/CompanyTreeSidebar";
import { CompanyCreationModal } from "./CompanyCreationModal";
import { CompanyFormData } from "./types";
import { useParentCompanies } from "@/lib/hooks/use-parent-companies";
import { useSubCompanies } from "@/lib/hooks/use-sub-companies";
import { ParentCompanyData, SubCompanyData } from "@/types/api";
import { toast } from "sonner";
import { Company, CompanyTreeViewProps } from "./types";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

// Custom Node Component using the new CompanyCard
const CompanyNode = React.memo(({ data, selected }: { data: any; selected: boolean }) => {
  return (
    <CompanyCard
      company={data.company}
      onAddSubCompany={data.onAddSubCompany}
      isSelected={selected}
      onSelect={() => data.onSelect?.(data.company.id)}
      level={data.level}
    />
  );
});

// Empty State Node Component using the new EmptyState
const EmptyStateNode = React.memo(({ data }: { data: any }) => {
  return <EmptyState onAddParentCompany={data.onAddParentCompany} />;
});

// Loading State Node Component
const LoadingStateNode = React.memo(() => {
  return (
    <div className="flex items-center justify-center min-h-[400px]">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-400 mx-auto mb-4" />
        <p className="text-white text-lg font-medium">Loading companies...</p>
        <p className="text-gray-400 text-sm mt-2">Please wait while we fetch your company data</p>
      </div>
    </div>
  );
});

// Define nodeTypes outside component to prevent recreation on every render
const nodeTypes = {
  company: CompanyNode,
  emptyState: EmptyStateNode,
  loadingState: LoadingStateNode,
};

// Define defaultEdgeOptions outside component to prevent recreation on every render
const defaultEdgeOptions = {
  animated: true,
  style: {
    stroke: "#10b981",
    strokeWidth: 2,
  },
  markerEnd: {
    type: MarkerType.ArrowClosed,
    color: "#10b981",
    width: 12,
    height: 12,
  },
};

function CompanyTreeViewContent({ onCompanyCreated }: CompanyTreeViewProps) {
  // Use hooks for consistent API calls and state management
  const { getParentCompanies, createParentCompany } = useParentCompanies();

  const { getSubCompanies, createSubCompany } = useSubCompanies();

  // React Flow instance for viewport control
  const { fitView, zoomIn, zoomOut } = useReactFlow();

  // State management
  const [companies, setCompanies] = useState<Company[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedCompany, setSelectedCompany] = useState<string | null>(null);
  const [selectedParentForFlow, setSelectedParentForFlow] = useState<
    string | null
  >(null);
  // Company creation state
  const [modalOpen, setModalOpen] = useState(false);
  const [modalType, setModalType] = useState<"parent" | "sub">("parent");
  const [parentCompanyId, setParentCompanyId] = useState<number | null>(null);
  // Sidebar visibility state
  const [showSidebar, setShowSidebar] = useState(false);


  // Load companies on mount
  useEffect(() => {
    loadCompanies();
  }, []);


  const loadCompanies = async () => {
    try {
      setIsLoading(true);
      // Use hooks for consistent API calls - no more response structure guessing!
      const [parentCompanies, subCompanies] = await Promise.all([
        getParentCompanies(),
        getSubCompanies(),
      ]);

      // Handle null responses gracefully
      const safeParentCompanies = parentCompanies || [];
      const safeSubCompanies = subCompanies || [];

      // Transform API data to our Company interface
      const transformedCompanies: Company[] = safeParentCompanies.map(
        (parent: ParentCompanyData) => ({
          id: `parent-${parent.parent_company_id}`,
          name: parent.company_name,
          description: parent.description,
          address: parent.address,
          contactEmail: parent.contact_email,
          dbId: parent.db_id,
          children: safeSubCompanies
            .filter(
              (sub: SubCompanyData) =>
                sub.parent_company_id === parent.parent_company_id
            )
            .map((sub: SubCompanyData) => ({
              id: `sub-${sub.sub_company_id}`,
              name: sub.company_name,
              description: sub.description,
              address: sub.address,
              contactEmail: sub.contact_email,
              dbId: sub.db_id,
              parentId: `parent-${sub.parent_company_id}`,
            })),
        })
      );

      setCompanies(transformedCompanies);

      // Show success message if we have data
      if (transformedCompanies.length > 0) {
        toast.success(
          `Loaded ${transformedCompanies.length} companies successfully`
        );
      }
    } catch (error) {
      console.error("Error loading companies:", error);
      toast.error("Failed to load companies");
      setCompanies([]); // Set empty array as fallback
    } finally {
      setIsLoading(false);
    }
  };

  // Create nodes and edges for React Flow - Simplified and precise
  const { nodes, edges } = useMemo(() => {
    // Loading state
    if (isLoading) {
      return {
        nodes: [
          {
            id: "loading-state",
            type: "loadingState",
            position: { x: 0, y: 0 },
            data: {},
            draggable: false,
          },
        ],
        edges: [],
      };
    }

    // Empty state (only show after loading is complete)
    if (companies.length === 0) {
      return {
        nodes: [
          {
            id: "empty-state",
            type: "emptyState",
            position: { x: 0, y: 0 },
            data: {
              onAddParentCompany: () => {
                setModalType("parent");
                setParentCompanyId(null);
                setModalOpen(true);
              },
            },
            draggable: false,
          },
        ],
        edges: [],
      };
    }

    const flowNodes: Node[] = [];
    const flowEdges: Edge[] = [];

    // Helper function to create a company node
    const createCompanyNode = (
      company: Company,
      position: { x: number; y: number },
      level: number
    ) => {
      const node = {
        id: company.id,
        type: "company",
        position,
        data: {
          company,
          level,
          onSelect: (companyId: string) => {
            setSelectedCompany(
              companyId === selectedCompany ? null : companyId
            );
          },
          onAddSubCompany: (parentId: string) => {
            setModalType("sub");
            // Extract the numeric ID from the prefixed ID (e.g., "parent-1" -> 1)
            const numericId = parseInt(parentId.replace("parent-", ""));
            setParentCompanyId(numericId);
            setModalOpen(true);
          },
        },
        selected: selectedCompany === company.id,
        draggable: true,
      };

      return node;
    };

    // Helper function to create an edge with proper handle IDs
    const createEdge = (sourceId: string, targetId: string) => {
      const edge = {
        id: `${sourceId}-${targetId}`,
        source: sourceId,
        sourceHandle: "bottom",
        target: targetId,
        targetHandle: "top",
        type: "smoothstep",
        style: {
          stroke: "#10b981",
          strokeWidth: 1.5,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: "#10b981",
          width: 12,
          height: 12,
        },
        animated: true,
      };

      return edge;
    };

    // Get the company to display (selected parent or first company)
    const displayCompany = selectedParentForFlow
      ? companies.find((c) => c.id === selectedParentForFlow)
      : companies[0];

    if (!displayCompany) return { nodes: [], edges: [] };

    // Calculate centered positioning
    const containerWidth = 1600; // Increased container width for wider cards
    const containerHeight = 1000; // Increased container height

    // Add parent company node - center it horizontally
    const parentPosition = {
      x: containerWidth / 2 - 250, // Offset by half card width to center (500px/2)
      y: 150, // Increased from top for better centering
    };
    flowNodes.push(createCompanyNode(displayCompany, parentPosition, 0));

    // Add child nodes and edges if they exist
    if (displayCompany.children && displayCompany.children.length > 0) {
      const childY = 600; // Increased vertical spacing between parent and children
      const childCount = displayCompany.children.length;
      const childCardWidth = 480; // Updated sub-company card width
      const parentCenterX = parentPosition.x + 250; // Parent card center (500px/2)

      // For consistent positioning, use fixed spacing based on child count
      let childPositions: { x: number; y: number }[] = [];

      if (childCount === 1) {
        // Single child: center directly under parent
        childPositions = [{ x: parentCenterX - childCardWidth / 2, y: childY }];
      } else if (childCount === 2) {
        // Two children: ensure adequate spacing to prevent overlap
        const minSpacing = childCardWidth + 80; // Card width + 80px gap
        childPositions = [
          { x: parentCenterX - minSpacing / 2 - childCardWidth / 2, y: childY },
          { x: parentCenterX + minSpacing / 2 - childCardWidth / 2, y: childY },
        ];
      } else {
        // Multiple children: distribute evenly with minimum spacing
        const minSpacing = childCardWidth + 60; // Card width + 60px gap
        const totalWidth = (childCount - 1) * minSpacing;
        const startX = parentCenterX - totalWidth / 2 - childCardWidth / 2;

        childPositions = displayCompany.children.map((_, index) => ({
          x: startX + index * minSpacing,
          y: childY,
        }));
      }

      displayCompany.children.forEach((child, index) => {
        // Add child node with calculated position
        flowNodes.push(createCompanyNode(child, childPositions[index], 1));

        // Add edge from parent to child
        flowEdges.push(createEdge(displayCompany.id, child.id));
      });
    }

    return { nodes: flowNodes, edges: flowEdges };
  }, [companies, selectedCompany, selectedParentForFlow, isLoading]);

  const [flowNodes, setNodes, onNodesChange] = useNodesState([]);
  const [flowEdges, setEdges, onEdgesChange] = useEdgesState([]);

  // Update nodes and edges when they change
  useEffect(() => {
    setNodes(nodes);
    setEdges(edges);

    // Fit view to nodes after a short delay to ensure nodes are rendered
    const timer = setTimeout(() => {
      fitView({ 
        padding: 0.15, 
        duration: 800,
        includeHiddenNodes: false,
        minZoom: 0.3,
        maxZoom: 1.2
      });
    }, 100);

    return () => clearTimeout(timer);
  }, [nodes, edges, setNodes, setEdges, fitView]);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  // Handle company creation using hooks consistently
  const handleCompanySubmit = async (companyData: CompanyFormData) => {
    try {
      if (modalType === "parent") {
        const result = await createParentCompany({
          company_name: companyData.name,
          description: companyData.description,
          address: companyData.address,
          contact_email: companyData.contactEmail,
          db_id: companyData.dbId,
        });

        if (!result) {
          throw new Error("Failed to create parent company");
        }

        toast.success(
          `Parent company "${companyData.name}" created successfully`
        );
      } else {
        if (!parentCompanyId) {
          throw new Error(
            "Parent company ID is required for sub-company creation"
          );
        }

        const result = await createSubCompany({
          company_name: companyData.name,
          description: companyData.description,
          address: companyData.address,
          contact_email: companyData.contactEmail,
          db_id: companyData.dbId,
          parent_company_id: parentCompanyId,
        });

        if (!result) {
          throw new Error("Failed to create sub company");
        }

        toast.success(`Sub company "${companyData.name}" created successfully`);
      }

      // Reload companies after creation
      await loadCompanies();
      onCompanyCreated?.();
      setModalOpen(false); // Close modal on success
    } catch (error) {
      console.error("Error creating company:", error);
      toast.error(
        error instanceof Error ? error.message : "Failed to create company"
      );
      throw error; // Re-throw to let modal handle the error
    }
  };

  return (
    <div className="w-full h-full relative overflow-hidden" style={{ height: 'calc(100vh - 140px)' }}>
      {/* Main Content Area - Full height with proper spacing */}
      <div className="flex h-full">
        {/* ReactFlow Container - Full width with bottom padding for controls */}
        <div className="flex-1 relative w-full h-full pb-20">
          <ReactFlow
            nodes={flowNodes}
            edges={flowEdges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            fitView
            className="bg-transparent"
            proOptions={{ hideAttribution: true }}
            defaultViewport={{ x: 0, y: 0, zoom: 0.6 }}
            minZoom={0.2}
            maxZoom={3}
            defaultEdgeOptions={defaultEdgeOptions}
            nodesDraggable={true}
            nodesConnectable={false}
            elementsSelectable={true}
            selectNodesOnDrag={false}
            panOnDrag={true}
            panOnScroll={true}
            zoomOnScroll={true}
            zoomOnPinch={true}
            preventScrolling={false}
            deleteKeyCode={null}
            multiSelectionKeyCode={null}
          >
          </ReactFlow>
        </div>

        {/* Bottom Control Section */}
        <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 z-50">
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
                    onClick={() => zoomIn()}
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
                    onClick={() => zoomOut()}
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

              {/* File Icon - Toggle Sidebar */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    onClick={() => setShowSidebar(!showSidebar)}
                    className="w-12 h-12 rounded-full text-white shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-110 cursor-pointer"
                    size="icon"
                    style={{
                      background: "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
                      border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))"
                    }}
                  >
                    <img src="/filelogo.svg" alt="Company Tree" className="h-6 w-6" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Company Tree</p>
                </TooltipContent>
              </Tooltip>

              {/* Add Sub Company Icon */}
              {companies.length > 0 && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      onClick={() => {
                        const firstParent = companies.find(c => c.id.startsWith('parent-'));
                        if (firstParent) {
                          setModalType("sub");
                          const numericId = parseInt(firstParent.id.replace("parent-", ""));
                          setParentCompanyId(numericId);
                          setModalOpen(true);
                        }
                      }}
                      className="w-12 h-12 rounded-full text-white shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-110 cursor-pointer"
                      size="icon"
                      style={{
                        background: "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
                        border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))"
                      }}
                    >
                      <img src="/tables/adduser.svg" alt="Add Sub Company" className="h-6 w-6" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Add Sub Company</p>
                  </TooltipContent>
                </Tooltip>
              )}
            </div>
            </TooltipProvider>
          </div>
        </div>

        {/* Sidebar - Dropdown from bottom controls */}
        {showSidebar && (
          <CompanyTreeSidebar
            companies={companies}
            selectedParentForFlow={selectedParentForFlow}
            selectedCompany={selectedCompany}
            onSelectParentForFlow={setSelectedParentForFlow}
            onSelectCompany={setSelectedCompany}
            onClose={() => setShowSidebar(false)}
          />
        )}
      </div>

      {/* Company Creation Modal */}
      <CompanyCreationModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        onSubmit={handleCompanySubmit}
        type={modalType}
        parentCompanyId={parentCompanyId}
      />
    </div>
  );
}

// Wrapper component with ReactFlowProvider
export function CompanyTreeView({ onCompanyCreated }: CompanyTreeViewProps) {
  return (
    <ReactFlowProvider>
      <CompanyTreeViewContent onCompanyCreated={onCompanyCreated} />
    </ReactFlowProvider>
  );
}

// Export the main component as default
export default CompanyTreeView;
