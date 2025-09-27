"use client";

import React, { useState, useMemo, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { TabsContent } from "@/components/ui/tabs";
import { PageLoader, UsersPageSkeleton } from "@/components/ui/loading";
import { useUsersManager } from "./hooks/useUsersManager";
import { CreateDatabaseAccessModal } from "./modals/CreateDatabaseAccessModal";
import { CreateVectorDBAccessModal } from "./modals/CreateVectorDBAccessModal";
import { useTheme } from "@/store/theme-store";
import {
  UsersManagerHeader,
  UsersTableSection,
  UserSearchInput,
  UserStatsCards,
  UserAccessTabs,
  MSSQLUsersList,
  VectorDBUsersList,
} from "./components";
import { UserStats } from "./types";

export function UsersManager() {
  // Theme
  const theme = useTheme();
  const isDark = theme === "dark";

  // Local state for modals
  const [activeTab, setActiveTab] = useState("mssql");
  const [isDatabaseModalOpen, setIsDatabaseModalOpen] = useState(false);
  const [isVectorDBModalOpen, setIsVectorDBModalOpen] = useState(false);
  const [selectedUser, setSelectedUser] = useState<string>("");
  const [editingUser, setEditingUser] = useState<string>("");

  const {
    searchTerm,
    setSearchTerm,
    userAccessConfigs,
    userConfigs,
    userConfigLoading,
    loadUserAccessConfigs,
    loadUserConfigs,
    extractNameFromEmail,
    availableDatabases,
  } = useUsersManager();

  // Load data on component mount
  useEffect(() => {
    loadUserAccessConfigs();
    loadUserConfigs();
  }, [loadUserAccessConfigs, loadUserConfigs]);

  // Filtered data based on search term
  const filteredUserAccess = useMemo(() => {
    if (!userAccessConfigs || !Array.isArray(userAccessConfigs)) return [];
    if (!searchTerm.trim()) return userAccessConfigs;

    return userAccessConfigs.filter(
      (config) =>
        config &&
        config.user_id &&
        (config.user_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
          extractNameFromEmail(config.user_id)
            .toLowerCase()
            .includes(searchTerm.toLowerCase()))
    );
  }, [userAccessConfigs, searchTerm, extractNameFromEmail]);

  // Filter users by access type
  const mssqlUsers = useMemo(() => {
    return filteredUserAccess.filter(
      (config) =>
        config.database_access?.parent_databases?.length > 0 ||
        config.database_access?.sub_databases?.some(
          (sub: any) => sub.databases?.length > 0
        )
    );
  }, [filteredUserAccess]);

  const vectorDBUsers = useMemo(() => {
    return userConfigs.filter(
      (config) =>
        config.db_id && config.table_names && config.table_names.length > 0
    );
  }, [userConfigs]);

  // Calculate stats
  const stats: UserStats = useMemo(
    () => ({
      totalUsers: filteredUserAccess.length,
      mssqlUsers: mssqlUsers.length,
      vectorDBUsers: vectorDBUsers.length,
      fullAccessUsers: filteredUserAccess.filter(
        (config) =>
          (config.database_access?.parent_databases?.length > 0 ||
            config.database_access?.sub_databases?.some(
              (sub: any) => sub.databases?.length > 0
            )) &&
          config.access_level >= 2
      ).length,
    }),
    [filteredUserAccess, mssqlUsers, vectorDBUsers]
  );

  // Handle modal operations
  const handleCreateMSSQLAccess = () => {
    setSelectedUser("");
    setEditingUser("");
    setIsDatabaseModalOpen(true);
  };

  const handleCreateVectorDBAccess = () => {
    setSelectedUser("");
    setEditingUser("");
    setIsVectorDBModalOpen(true);
  };

  const handleEditUser = (userId: string, type: "mssql" | "vector") => {
    setSelectedUser(userId);
    setEditingUser(userId);
    if (type === "mssql") {
      setIsDatabaseModalOpen(true);
    } else {
      setIsVectorDBModalOpen(true);
    }
  };

  const handleModalSuccess = () => {
    loadUserAccessConfigs();
    loadUserConfigs();
    setIsDatabaseModalOpen(false);
    setIsVectorDBModalOpen(false);
    setSelectedUser("");
    setEditingUser("");
  };

  const handleModalClose = () => {
    setIsDatabaseModalOpen(false);
    setIsVectorDBModalOpen(false);
    setSelectedUser("");
    setEditingUser("");
  };

  // Helper functions
  const getDatabaseName = (dbId: number) => {
    const database = availableDatabases?.find((db) => db.db_id === dbId);
    return database ? database.db_name : `DB ${dbId}`;
  };

  const formatTableNames = (tableNames: string[]) => {
    if (!tableNames || tableNames.length === 0) return "No tables";
    if (tableNames.length <= 3) return tableNames.join(", ");
    return `${tableNames.slice(0, 3).join(", ")} +${
      tableNames.length - 3
    } more`;
  };

  const getAccessLevelBadge = (config: any) => {
    const hasMSSQL =
      config.database_access?.parent_databases?.length > 0 ||
      config.database_access?.sub_databases?.some(
        (sub: any) => sub.databases?.length > 0
      );
    const hasVectorDB = config.access_level >= 2;

    if (hasMSSQL && hasVectorDB) {
      return (
        <Badge className="bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 text-white border-emerald-400/30 shadow-lg shadow-emerald-500/25">
          Full Access
        </Badge>
      );
    } else if (hasMSSQL) {
      return (
        <Badge className="bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 text-white border-emerald-400/30 shadow-lg shadow-emerald-500/25">
          MSSQL Access
        </Badge>
      );
    } else if (hasVectorDB) {
      return (
        <Badge className="bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 text-white border-emerald-400/30 shadow-lg shadow-emerald-500/25">
          Vector DB Access
        </Badge>
      );
    } else {
      return (
        <Badge className="bg-gradient-to-r from-gray-500 to-gray-600 hover:from-gray-600 hover:to-gray-700 text-white border-gray-400/30">
          No Access
        </Badge>
      );
    }
  };

  const getDatabaseCount = (config: any) => {
    const parentCount = config.database_access?.parent_databases?.length || 0;
    const subCount =
      config.database_access?.sub_databases?.reduce(
        (total: number, sub: any) => total + (sub.databases?.length || 0),
        0
      ) || 0;
    return parentCount + subCount;
  };

  if (userConfigLoading || !activeTab) {
    return (
      <UsersPageSkeleton
        size="lg"
        activeTab={activeTab}
        showTabs={true}
        showSearch={true}
        showActions={true}
        showPagination={true}
        rowCount={5}
      />
    );
  }

  return (
    <div>
      {/* Combined Table Section with Header and Data Table */}
      {activeTab === "mssql" ? (
        <UsersTableSection
          activeTab={activeTab}
          onTabChange={setActiveTab}
          searchTerm={searchTerm}
          onSearchChange={setSearchTerm}
          onCreateMSSQLAccess={handleCreateMSSQLAccess}
          onCreateVectorDBAccess={handleCreateVectorDBAccess}
          isDark={isDark}
          users={mssqlUsers}
          onEditUser={(userId) => handleEditUser(userId, "mssql")}
          extractNameFromEmail={extractNameFromEmail}
          getAccessLevelBadge={getAccessLevelBadge}
          getDatabaseCount={getDatabaseCount}
          type="mssql"
          isLoading={userConfigLoading}
        />
      ) : (
        <UsersTableSection
          activeTab={activeTab}
          onTabChange={setActiveTab}
          searchTerm={searchTerm}
          onSearchChange={setSearchTerm}
          onCreateMSSQLAccess={handleCreateMSSQLAccess}
          onCreateVectorDBAccess={handleCreateVectorDBAccess}
          isDark={isDark}
          users={vectorDBUsers}
          onEditUser={(userId) => handleEditUser(userId, "vector")}
          extractNameFromEmail={extractNameFromEmail}
          getAccessLevelBadge={getAccessLevelBadge}
          getDatabaseName={getDatabaseName}
          formatTableNames={formatTableNames}
          type="vector"
          isLoading={userConfigLoading}
        />
      )}

      {/* Modals */}
      <CreateDatabaseAccessModal
        isOpen={isDatabaseModalOpen}
        onClose={handleModalClose}
        onSuccess={handleModalSuccess}
        selectedUser={selectedUser}
        editingUser={editingUser}
      />

      <CreateVectorDBAccessModal
        isOpen={isVectorDBModalOpen}
        onClose={handleModalClose}
        onSuccess={handleModalSuccess}
        selectedUser={selectedUser}
        editingUser={editingUser}
      />
    </div>
  );
}
