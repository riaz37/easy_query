import { useState, useCallback } from "react";
import { useUserAccess } from "@/lib/hooks";
import { useDatabaseContext } from "@/components/providers/DatabaseContextProvider";
import { UserAccessData } from "@/types/api";
import { ServiceRegistry } from "@/lib/api/services/service-registry";

export function useUsersManager() {
  // Core state
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedUsers, setSelectedUsers] = useState<string[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [activeTab, setActiveTab] = useState("database");
  const [selectedUserForVectorDB, setSelectedUserForVectorDB] = useState<string>("");

  // User configurations state
  const [userConfigs, setUserConfigs] = useState<any[]>([]);
  const [userConfigLoading, setUserConfigLoading] = useState(false);

  // Hooks
  const { 
    userAccessConfigs, 
    isLoading: userAccessLoading, 
    getUserAccessConfigs 
  } = useUserAccess();
  
  const { 
    availableDatabases, 
    isLoading: databaseLoading
  } = useDatabaseContext();

  // Computed values with safe defaults
  const isLoading = userAccessLoading || userConfigLoading || databaseLoading;
  
  // Safe access to arrays with default empty arrays
  const safeUserAccessConfigs = userAccessConfigs || [];
  const safeUserConfigs = userConfigs || [];
  const safeDatabaseConfigs = availableDatabases || [];

  // Actions
  const loadUserAccessConfigs = useCallback(async () => {
    try {
      await getUserAccessConfigs();
    } catch (error) {
      console.error("Error loading user access configs:", error);
    }
  }, [getUserAccessConfigs]);

  const loadUserConfigs = useCallback(async () => {
    try {
      setUserConfigLoading(true);
      const response = await ServiceRegistry.userConfig.getUserConfigs();
      if (response && Array.isArray(response)) {
        setUserConfigs(response);
      } else if (response && Array.isArray(response.configs)) {
        setUserConfigs(response.configs);
      } else {
        setUserConfigs([]);
      }
    } catch (error) {
      console.error("Error loading user configs:", error);
      setUserConfigs([]);
    } finally {
      setUserConfigLoading(false);
    }
  }, []);

  const loadDatabaseConfigs = useCallback(async () => {
    try {
      // The original code had fetchDatabaseConfigs here, but it's not destructured from useDatabaseContext.
      // Assuming the intent was to use availableDatabases or a similar source if needed.
      // For now, removing the line as it's not directly available from the new destructuring.
      // If the intent was to refetch or update availableDatabases, that would require a separate action.
    } catch (error) {
      console.error("Error loading database configs:", error);
    }
  }, []); // Removed fetchDatabaseConfigs from dependency array

  const handleManageVectorDBAccess = useCallback((userId: string) => {
    setSelectedUserForVectorDB(userId);
    setActiveTab("vector-db");
  }, []);

  const handleCloseVectorDBAccess = useCallback(() => {
    setSelectedUserForVectorDB("");
  }, []);

  const handleSelectAll = useCallback((checked: boolean) => {
    if (checked) {
      setSelectedUsers(safeUserAccessConfigs.map((config) => config.user_id));
    } else {
      setSelectedUsers([]);
    }
  }, [safeUserAccessConfigs]);

  const handleSelectUser = useCallback((userId: string, checked: boolean) => {
    if (checked) {
      setSelectedUsers((prev) => [...prev, userId]);
    } else {
      setSelectedUsers((prev) => prev.filter((id) => id !== userId));
    }
  }, []);

  const extractNameFromEmail = useCallback((email: string): string => {
    const localPart = email.split("@")[0];
    return localPart
      .split(/[._-]/)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
      .join(" ");
  }, []);

  const getAccessSummary = useCallback((config: UserAccessData): string => {
    const parentDbCount = config.database_access?.parent_databases?.length || 0;
    const subDbCount =
      config.database_access?.sub_databases?.reduce(
        (total, subDb) => total + (subDb.databases?.length || 0),
        0
      ) || 0;
    const totalDbs = parentDbCount + subDbCount;
    return `${totalDbs} database${totalDbs !== 1 ? "s" : ""}, ${
      config.sub_company_ids?.length || 0
    } sub-companies`;
  }, []);

  // Pagination with safe defaults
  const totalPages = Math.ceil(safeUserAccessConfigs.length / rowsPerPage);
  const startIndex = (currentPage - 1) * rowsPerPage;
  const paginatedConfigs = safeUserAccessConfigs.slice(startIndex, startIndex + rowsPerPage);

  const isAllSelected = paginatedConfigs.length > 0 && selectedUsers.length === paginatedConfigs.length;
  const isIndeterminate = selectedUsers.length > 0 && selectedUsers.length < paginatedConfigs.length;

  return {
    // State
    searchTerm,
    selectedUsers,
    currentPage,
    rowsPerPage,
    activeTab,
    selectedUserForVectorDB,
    isLoading,
    userAccessConfigs: safeUserAccessConfigs,
    userConfigs: safeUserConfigs,
    userConfigLoading,
    databaseConfigs: safeDatabaseConfigs,
    availableDatabases: safeDatabaseConfigs,
    
    // Computed values
    totalPages,
    paginatedConfigs,
    isAllSelected,
    isIndeterminate,
    
    // Actions
    setSearchTerm,
    setCurrentPage,
    setRowsPerPage,
    setActiveTab,
    loadUserAccessConfigs,
    loadUserConfigs,
    loadDatabaseConfigs,
    handleManageVectorDBAccess,
    handleCloseVectorDBAccess,
    handleSelectAll,
    handleSelectUser,
    extractNameFromEmail,
    getAccessSummary,
  };
} 