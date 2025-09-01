"use client";

import { useCallback, useMemo, useState, useEffect } from "react";
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
import { AnimatedGridBackground } from "./AnimatedGridBackground";
import { Company, CompanyTreeViewProps } from "./types";

// Custom Node Component using the new CompanyCard
const CompanyNode = ({ data, selected }: { data: any; selected: boolean }) => {
  return (
    <CompanyCard
      company={data.company}
      onAddSubCompany={data.onAddSubCompany}
      isSelected={selected}
      onSelect={() => data.onSelect?.(data.company.id)}
      level={data.level}
    />
  );
};

// Empty State Node Component using the new EmptyState
const EmptyStateNode = ({ data }: { data: any }) => {
  return <EmptyState onAddParentCompany={data.onAddParentCompany} />;
};

// Define nodeTypes outside component to prevent recreation on every render
const nodeTypes = {
  company: CompanyNode,
  emptyState: EmptyStateNode,
};

// Define defaultEdgeOptions outside component to prevent recreation on every render
const defaultEdgeOptions = {
  animated: true,
  style: {
    stroke: "#10b981",
    strokeWidth: 3,
  },
  markerEnd: {
    type: MarkerType.ArrowClosed,
    color: "#10b981",
    width: 16,
    height: 16,
  },
};

function CompanyTreeViewContent({ onCompanyCreated }: CompanyTreeViewProps) {
  // Use hooks for consistent API calls and state management
  const { getParentCompanies, createParentCompany } = useParentCompanies();

  const { getSubCompanies, createSubCompany } = useSubCompanies();

  // React Flow instance for viewport control
  const { fitView } = useReactFlow();

  // State management
  const [companies, setCompanies] = useState<Company[]>([]);
  const [selectedCompany, setSelectedCompany] = useState<string | null>(null);
  const [selectedParentForFlow, setSelectedParentForFlow] = useState<
    string | null
  >(null);
  // Company creation state
  const [modalOpen, setModalOpen] = useState(false);
  const [modalType, setModalType] = useState<"parent" | "sub">("parent");
  const [parentCompanyId, setParentCompanyId] = useState<number | null>(null);

  // Load companies on mount
  useEffect(() => {
    loadCompanies();
  }, []);

  const loadCompanies = async () => {
    try {
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
    }
  };

  // Create nodes and edges for React Flow - Simplified and precise
  const { nodes, edges } = useMemo(() => {
    // Empty state
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
          strokeWidth: 2,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: "#10b981",
          width: 16,
          height: 16,
        },
      };

      return edge;
    };

    // Get the company to display (selected parent or first company)
    const displayCompany = selectedParentForFlow
      ? companies.find((c) => c.id === selectedParentForFlow)
      : companies[0];

    if (!displayCompany) return { nodes: [], edges: [] };

    // Calculate centered positioning
    const containerWidth = 1200; // Approximate container width
    const containerHeight = 800; // Approximate container height

    // Add parent company node - center it horizontally
    const parentPosition = {
      x: containerWidth / 2 - 200, // Offset by half card width to center
      y: 100,
    };
    flowNodes.push(createCompanyNode(displayCompany, parentPosition, 0));

    // Add child nodes and edges if they exist
    if (displayCompany.children && displayCompany.children.length > 0) {
      const childY = 350;
      const childCount = displayCompany.children.length;
      const childCardWidth = 320; // Actual sub-company card width (w-80)
      const parentCenterX = parentPosition.x + 200; // Parent card center (w-96 / 2)

      // For consistent positioning, use fixed spacing based on child count
      let childPositions: { x: number; y: number }[] = [];

      if (childCount === 1) {
        // Single child: center directly under parent
        childPositions = [{ x: parentCenterX - childCardWidth / 2, y: childY }];
      } else if (childCount === 2) {
        // Two children: ensure adequate spacing to prevent overlap
        const minSpacing = childCardWidth + 40; // Card width + 40px gap
        childPositions = [
          { x: parentCenterX - minSpacing / 2 - childCardWidth / 2, y: childY },
          { x: parentCenterX + minSpacing / 2 - childCardWidth / 2, y: childY },
        ];
      } else {
        // Multiple children: distribute evenly with minimum spacing
        const minSpacing = childCardWidth + 30; // Card width + 30px gap
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
  }, [companies, selectedCompany, selectedParentForFlow]);

  const [flowNodes, setNodes, onNodesChange] = useNodesState([]);
  const [flowEdges, setEdges, onEdgesChange] = useEdgesState([]);

  // Update nodes and edges when they change
  useEffect(() => {
    setNodes(nodes);
    setEdges(edges);

    // Fit view to nodes after a short delay to ensure nodes are rendered
    const timer = setTimeout(() => {
      fitView({ padding: 0.2, duration: 800 });
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
    <div className="min-h-screen w-full relative bg-gradient-to-br from-gray-50 via-emerald-50/30 to-gray-100 dark:from-gray-900 dark:via-emerald-950/20 dark:to-gray-800">
      {/* Animated Grid Background */}
      <AnimatedGridBackground />

      {/* Main Content Area - Account for navbar height */}
      <div className="flex h-screen pt-20">
        {/* ReactFlow Container - Full width */}
        <div className="flex-1 relative">
          <ReactFlow
            nodes={flowNodes}
            edges={flowEdges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{
              padding: 0.2,
              minZoom: 0.5,
              maxZoom: 1.5,
            }}
            defaultViewport={{ x: 0, y: 0, zoom: 0.8 }}
            minZoom={0.3}
            maxZoom={2}
            className="bg-transparent"
            proOptions={{ hideAttribution: true }}
            defaultEdgeOptions={defaultEdgeOptions}
          >
            <Controls className="opacity-70" />
          </ReactFlow>
        </div>

        {/* Sidebar - Fixed to right side */}
        <div className="fixed top-24 right-6 z-40 flex-shrink-0">
          <CompanyTreeSidebar
            companies={companies}
            selectedParentForFlow={selectedParentForFlow}
            selectedCompany={selectedCompany}
            onSelectParentForFlow={setSelectedParentForFlow}
            onSelectCompany={setSelectedCompany}
            onAddSubCompany={(parentId: string) => {
              setModalType("sub");
              // Extract the numeric ID from the prefixed ID (e.g., "parent-1" -> 1)
              const numericId = parseInt(parentId.replace("parent-", ""));
              setParentCompanyId(numericId);
              setModalOpen(true);
            }}
          />
        </div>
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
