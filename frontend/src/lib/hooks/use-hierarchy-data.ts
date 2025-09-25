import { useState, useEffect } from "react";
import { useParentCompanies, useSubCompanies } from "@/lib/hooks";
import { ParentCompanyData, SubCompanyData } from "@/types/api";
// Define HierarchyNode type locally since the component doesn't exist
interface HierarchyNode {
  id: string;
  name: string;
  description: string;
  type: "parent" | "sub" | "database";
  data: any;
  children?: HierarchyNode[];
}

interface UseHierarchyDataReturn {
  hierarchyData: HierarchyNode[];
  isLoading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export function useHierarchyData(): UseHierarchyDataReturn {
  const [hierarchyData, setHierarchyData] = useState<HierarchyNode[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const { getParentCompanies } = useParentCompanies();
  const { getSubCompanies } = useSubCompanies();

  const buildHierarchy = (
    parentCompanies: ParentCompanyData[],
    subCompanies: SubCompanyData[]
  ): HierarchyNode[] => {
    // If no parent companies exist, return empty array to show EmptyHierarchyState
    if (!parentCompanies || parentCompanies.length === 0) {
      return [];
    }

    // Build company-focused hierarchy: Parent Companies -> Sub Companies
    return parentCompanies.map((parent) => {
      // Get sub-companies for this parent
      const parentSubCompanies = subCompanies
        .filter((sub) => sub.parent_company_id === parent.parent_company_id)
        .map((sub) => ({
          id: `sub-${sub.sub_company_id}`,
          name: sub.company_name,
          description:
            sub.description || "Sub-company under " + parent.company_name,
          type: "sub" as const,
          data: sub,
        }));

      return {
        id: `parent-${parent.parent_company_id}`,
        name: parent.company_name,
        description:
          parent.description || "Parent company with business operations",
        type: "parent" as const,
        data: parent,
        children: parentSubCompanies,
      };
    });
  };

  const loadHierarchyData = async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Only fetch company data - no database configs needed for hierarchy display
      try {
        const [parentCompanies, subCompanies] = await Promise.all([
          getParentCompanies(),
          getSubCompanies(),
        ]);

        // Handle the case where data might be null or empty arrays
        const safeParentCompanies = parentCompanies || [];
        const safeSubCompanies = subCompanies || [];

        const hierarchy = buildHierarchy(safeParentCompanies, safeSubCompanies);
        setHierarchyData(hierarchy);
      } catch (apiError) {
        // If API fails, show empty state for now
        console.warn("API not available, showing empty state:", apiError);
        setHierarchyData([]);
      }
    } catch (err: any) {
      // For development, let's not show errors for empty data
      // Instead, just set empty hierarchy data
      console.warn("Could not load hierarchy data, showing empty state:", err);
      setHierarchyData([]);

      // Only set error for actual API failures, not empty data
      if (
        err?.message &&
        !err.message.includes("Failed to load hierarchy data")
      ) {
        const errorMessage = err?.message || "Error loading hierarchy data";
        setError(errorMessage);
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadHierarchyData();
  }, []);

  return {
    hierarchyData,
    isLoading,
    error,
    refetch: loadHierarchyData,
  };
}
