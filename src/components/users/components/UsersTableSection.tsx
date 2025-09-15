"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { 
  MoreVertical, 
  ChevronDown, 
  ChevronLeft, 
  ChevronRight, 
  ChevronsLeft, 
  ChevronsRight,
  Search
} from "lucide-react";
import { useTheme } from "@/store/theme-store";
import { cn } from "@/lib/utils";
import { UserAccessData, UserConfig } from "../types";

interface UsersTableSectionProps {
  // Header props
  activeTab: string;
  onTabChange: (tab: string) => void;
  searchTerm: string;
  onSearchChange: (value: string) => void;
  onCreateMSSQLAccess: () => void;
  onCreateVectorDBAccess: () => void;
  isDark: boolean;
  // Table props
  users: UserAccessData[] | UserConfig[];
  onEditUser: (userId: string) => void;
  extractNameFromEmail: (email: string) => string;
  getAccessLevelBadge: (config: UserAccessData | UserConfig) => React.ReactNode;
  getDatabaseCount?: (config: UserAccessData) => number;
  getDatabaseName?: (dbId: number) => string;
  formatTableNames?: (tableNames: string[]) => string;
  type?: 'mssql' | 'vector';
}

export function UsersTableSection({
  // Header props
  activeTab,
  onTabChange,
  searchTerm,
  onSearchChange,
  onCreateMSSQLAccess,
  onCreateVectorDBAccess,
  isDark,
  // Table props
  users,
  onEditUser,
  extractNameFromEmail,
  getAccessLevelBadge,
  getDatabaseCount,
  getDatabaseName,
  formatTableNames,
  type = "mssql"
}: UsersTableSectionProps) {
  const theme = useTheme();
  const [selectedRows, setSelectedRows] = useState<Set<string>>(new Set());
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  const [currentPage, setCurrentPage] = useState(1);
  const [rowsPerPage, setRowsPerPage] = useState(5);

  const isMSSQL = type === "mssql";

  // Process actual user data based on the real data structures
  const processedUsers = users.map((user) => {
    if (isMSSQL) {
      const mssqlUser = user as any; // UserAccessData type
      return {
        id: mssqlUser.user_id,
        email: mssqlUser.user_id,
        parentCompany: mssqlUser.parent_company_name || "N/A",
        subCompany: mssqlUser.sub_company_ids?.length > 0 ? `${mssqlUser.sub_company_ids.length} companies` : "None",
        date: mssqlUser.created_at ? new Date(mssqlUser.created_at).toLocaleDateString() : "N/A",
        accessLevel: "Full Access", // MSSQL users typically have full access
        databaseCount: getDatabaseCount?.(mssqlUser) || 0,
        databaseName: "N/A", // Not applicable for MSSQL
        tableNames: Object.keys(mssqlUser.table_shows || {}).length > 0 ? 
          `${Object.keys(mssqlUser.table_shows || {}).length} databases` : "No databases",
        originalUser: mssqlUser
      };
    } else {
      const vectorUser = user as any; // UserConfigData type
      return {
        id: vectorUser.user_id,
        email: vectorUser.user_id,
        parentCompany: "N/A", // Vector DB users don't have company info
        subCompany: "N/A",
        date: vectorUser.created_at ? new Date(vectorUser.created_at).toLocaleDateString() : "N/A",
        accessLevel: `Level ${vectorUser.access_level}`,
        databaseCount: 0, // Not applicable for Vector DB
        databaseName: getDatabaseName?.(vectorUser.db_id) || `DB ${vectorUser.db_id}`,
        tableNames: vectorUser.table_names?.length > 0 ? 
          `${vectorUser.table_names.length} tables` : "No tables",
        originalUser: vectorUser
      };
    }
  });

  const totalPages = Math.ceil(processedUsers.length / rowsPerPage);
  const startIndex = (currentPage - 1) * rowsPerPage;
  const endIndex = startIndex + rowsPerPage;
  const currentUsers = processedUsers.slice(startIndex, endIndex);

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedRows(new Set(currentUsers.map(user => user.id)));
    } else {
      setSelectedRows(new Set());
    }
  };

  const handleSelectRow = (userId: string, checked: boolean) => {
    const newSelected = new Set(selectedRows);
    if (checked) {
      newSelected.add(userId);
    } else {
      newSelected.delete(userId);
    }
    setSelectedRows(newSelected);
  };

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortColumn(column);
      setSortDirection("asc");
    }
  };

  const getSortIcon = (column: string) => {
    if (sortColumn !== column) {
      return <ChevronDown className="h-4 w-4 text-gray-400" />;
    }
    return sortDirection === "asc" 
      ? <ChevronDown className="h-4 w-4 text-white" />
      : <ChevronDown className="h-4 w-4 text-white rotate-180" />;
  };

  return (
    <div className="modal-enhanced">
      <div 
        className="modal-content-enhanced overflow-hidden"
        style={{
          background: `linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)),
linear-gradient(230.27deg, rgba(19, 245, 132, 0) 71.59%, rgba(19, 245, 132, 0.2) 98.91%),
linear-gradient(67.9deg, rgba(19, 245, 132, 0) 66.65%, rgba(19, 245, 132, 0.2) 100%)`,
          backdropFilter: "blur(30px)"
        }}
      >
      {/* Header Section */}
      <div className="p-6">
        {/* Tabs */}
        <div className="flex gap-8 mb-6">
          <button
            onClick={() => onTabChange("mssql")}
            className={cn(
              "text-lg font-semibold pb-2 border-b-2 transition-colors cursor-pointer",
              activeTab === "mssql"
                ? ""
                : "text-gray-400 border-transparent hover:text-white"
            )}
            style={activeTab === "mssql" ? { 
              color: "var(--primary-main, rgba(19, 245, 132, 1))",
              borderBottomColor: "var(--primary-main, rgba(19, 245, 132, 1))"
            } : {}}
          >
            MSSQL DATABASE ACCESS
          </button>
          <button
            onClick={() => onTabChange("vector")}
            className={cn(
              "text-lg font-semibold pb-2 border-b-2 transition-colors cursor-pointer",
              activeTab === "vector"
                ? ""
                : "text-gray-400 border-transparent hover:text-white"
            )}
            style={activeTab === "vector" ? { 
              color: "var(--primary-main, rgba(19, 245, 132, 1))",
              borderBottomColor: "var(--primary-main, rgba(19, 245, 132, 1))"
            } : {}}
          >
            VECTOR DATABASE ACCESS
          </button>
        </div>

        {/* Action Bar */}
        <div className="flex items-center justify-between">
          {/* Search Input */}
          <div className="flex items-center gap-4 flex-1">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                type="text"
                placeholder="Search users..."
                value={searchTerm}
                onChange={(e) => onSearchChange(e.target.value)}
                className="pl-10 text-white placeholder:text-gray-400 rounded-full border-0 focus:ring-0 focus:outline-none"
                style={{
                  background: "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
                  borderRadius: "999px",
                  height: "50px"
                }}
              />
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              size="icon"
              className="border-0 text-white hover:bg-white/10 cursor-pointer"
              style={{
                background: "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
                borderRadius: "118.8px",
                width: "48px",
                height: "48px"
              }}
            >
              <img src="/tables/download.svg" alt="Download" className="w-6 h-6" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              className="border-0 text-white hover:bg-white/10 cursor-pointer"
              style={{
                background: "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
                borderRadius: "118.8px",
                width: "48px",
                height: "48px"
              }}
            >
              <img src="/tables/filter.svg" alt="Filter" className="w-6 h-6" />
            </Button>
            <Button
              onClick={
                activeTab === "mssql"
                  ? onCreateMSSQLAccess
                  : onCreateVectorDBAccess
              }
              variant="outline"
              size="icon"
              className="border-0 text-white hover:bg-white/10 cursor-pointer"
              style={{
                background: "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
                borderRadius: "118.8px",
                width: "48px",
                height: "48px"
              }}
            >
              <img src="/tables/adduser.svg" alt="Add User" className="w-6 h-6" />
            </Button>
          </div>
        </div>
      </div>

      {/* Table Section */}
      <div className="overflow-x-auto px-6 pb-6">
        <div className="rounded-t-xl overflow-hidden">
          <table className="w-full">
          <thead>
            <tr 
              style={{
                background: "var(--components-Table-Head-filled, rgba(145, 158, 171, 0.08))",
                borderRadius: "12px 12px 0 0"
              }}
            >
              <th className="px-6 py-4 text-left rounded-tl-xl">
                <Checkbox
                  checked={selectedRows.size === currentUsers.length && currentUsers.length > 0}
                  onCheckedChange={handleSelectAll}
                  className="border-white/40 data-[state=checked]:bg-emerald-500 data-[state=checked]:border-emerald-500"
                />
              </th>
              <th 
                className="px-6 py-4 text-left cursor-pointer hover:bg-white/5 transition-colors"
                onClick={() => handleSort("email")}
              >
                <div className="flex items-center gap-2 text-white font-semibold">
                  EMAIL
                  {getSortIcon("email")}
                </div>
              </th>
              {isMSSQL ? (
                <>
                  <th 
                    className="px-6 py-4 text-left cursor-pointer hover:bg-white/5 transition-colors"
                    onClick={() => handleSort("parentCompany")}
                  >
                    <div className="flex items-center gap-2 text-white font-semibold">
                      PARENT COMPANY
                      {getSortIcon("parentCompany")}
                    </div>
                  </th>
                  <th 
                    className="px-6 py-4 text-left cursor-pointer hover:bg-white/5 transition-colors"
                    onClick={() => handleSort("subCompany")}
                  >
                    <div className="flex items-center gap-2 text-white font-semibold">
                      SUB COMPANIES
                      {getSortIcon("subCompany")}
                    </div>
                  </th>
                  <th 
                    className="px-6 py-4 text-left cursor-pointer hover:bg-white/5 transition-colors"
                    onClick={() => handleSort("databaseCount")}
                  >
                    <div className="flex items-center gap-2 text-white font-semibold">
                      DATABASES
                      {getSortIcon("databaseCount")}
                    </div>
                  </th>
                </>
              ) : (
                <>
                  <th 
                    className="px-6 py-4 text-left cursor-pointer hover:bg-white/5 transition-colors"
                    onClick={() => handleSort("databaseName")}
                  >
                    <div className="flex items-center gap-2 text-white font-semibold">
                      DATABASE
                      {getSortIcon("databaseName")}
                    </div>
                  </th>
                  <th 
                    className="px-6 py-4 text-left cursor-pointer hover:bg-white/5 transition-colors"
                    onClick={() => handleSort("accessLevel")}
                  >
                    <div className="flex items-center gap-2 text-white font-semibold">
                      ACCESS LEVEL
                      {getSortIcon("accessLevel")}
                    </div>
                  </th>
                  <th 
                    className="px-6 py-4 text-left cursor-pointer hover:bg-white/5 transition-colors"
                    onClick={() => handleSort("tableNames")}
                  >
                    <div className="flex items-center gap-2 text-white font-semibold">
                      TABLES
                      {getSortIcon("tableNames")}
                    </div>
                  </th>
                </>
              )}
              <th 
                className="px-6 py-4 text-left cursor-pointer hover:bg-white/5 transition-colors"
                onClick={() => handleSort("date")}
              >
                <div className="flex items-center gap-2 text-white font-semibold">
                  DATE
                  {getSortIcon("date")}
                </div>
              </th>
              <th className="px-6 py-4 text-right text-white font-semibold rounded-tr-xl">
                ACTIONS
              </th>
            </tr>
          </thead>
          <tbody>
            {currentUsers.length === 0 ? (
              <tr>
                <td colSpan={isMSSQL ? 7 : 7} className="px-6 py-8 text-center text-white">
                  No users found
                </td>
              </tr>
            ) : (
              currentUsers.map((user) => (
              <tr 
                key={user.id} 
                className="border-b border-white/10 hover:bg-white/5 transition-colors"
              >
                <td className="px-6 py-4">
                  <Checkbox
                    checked={selectedRows.has(user.id)}
                    onCheckedChange={(checked) => handleSelectRow(user.id, checked as boolean)}
                    className="border-white/40 data-[state=checked]:bg-emerald-500 data-[state=checked]:border-emerald-500"
                  />
                </td>
                <td className="px-6 py-4 text-white font-medium">
                  {user.email}
                </td>
                {isMSSQL ? (
                  <>
                    <td className="px-6 py-4 text-white">
                      {user.parentCompany}
                    </td>
                    <td className="px-6 py-4 text-white">
                      {user.subCompany}
                    </td>
                    <td className="px-6 py-4 text-white">
                      {user.databaseCount}
                    </td>
                  </>
                ) : (
                  <>
                    <td className="px-6 py-4 text-white">
                      {user.databaseName}
                    </td>
                    <td className="px-6 py-4 text-white">
                      {user.accessLevel}
                    </td>
                    <td className="px-6 py-4 text-white">
                      {user.tableNames}
                    </td>
                  </>
                )}
                <td className="px-6 py-4 text-white">
                  {user.date}
                </td>
                <td className="px-6 py-4 text-right">
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => onEditUser(user.id)}
                    className="text-white hover:bg-white/10"
                  >
                    <MoreVertical className="h-4 w-4" />
                  </Button>
                </td>
              </tr>
              ))
            )}
          </tbody>
        </table>
        </div>
      </div>

      {/* Pagination and Selection Status */}
      <div className="px-6 py-4 flex items-center justify-between">
        {/* Left: Rows Selected */}
        <div className="flex items-center">
          <p className="text-white text-sm">
            {selectedRows.size} of {processedUsers.length} Row(s) Selected
          </p>
        </div>
        
        {/* Right side: Rows per page and Page Info */}
        <div className="flex items-center gap-6">
          {/* Rows per page */}
          <div className="flex items-center gap-2">
            <span className="text-white text-sm">Rows per page:</span>
            <select
              value={rowsPerPage}
              onChange={(e) => setRowsPerPage(Number(e.target.value))}
              className="bg-white/10 text-white px-2 py-1 text-sm"
              style={{
                border: "1px solid var(--components-button-outlined, rgba(145, 158, 171, 0.32))",
                borderRadius: "99px"
              }}
            >
              <option value={5}>5</option>
              <option value={10}>10</option>
              <option value={25}>25</option>
            </select>
          </div>
          
          {/* Page Info and Controls */}
          <div className="flex items-center gap-2">
            <span className="text-white text-sm">
              Page {currentPage} of {totalPages}
            </span>
            <div className="flex gap-1">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setCurrentPage(1)}
                disabled={currentPage === 1}
                className="text-white disabled:opacity-50"
                style={{
                  borderRadius: "999px"
                }}
                onMouseEnter={(e) => {
                  if (!e.currentTarget.disabled) {
                    e.currentTarget.style.background = "rgba(255, 255, 255, 0.04)";
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "transparent";
                }}
              >
                <ChevronsLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setCurrentPage(currentPage - 1)}
                disabled={currentPage === 1}
                className="text-white disabled:opacity-50"
                style={{
                  borderRadius: "999px"
                }}
                onMouseEnter={(e) => {
                  if (!e.currentTarget.disabled) {
                    e.currentTarget.style.background = "rgba(255, 255, 255, 0.04)";
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "transparent";
                }}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setCurrentPage(currentPage + 1)}
                disabled={currentPage === totalPages}
                className="text-white disabled:opacity-50"
                style={{
                  borderRadius: "999px"
                }}
                onMouseEnter={(e) => {
                  if (!e.currentTarget.disabled) {
                    e.currentTarget.style.background = "rgba(255, 255, 255, 0.04)";
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "transparent";
                }}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setCurrentPage(totalPages)}
                disabled={currentPage === totalPages}
                className="text-white disabled:opacity-50"
                style={{
                  borderRadius: "999px"
                }}
                onMouseEnter={(e) => {
                  if (!e.currentTarget.disabled) {
                    e.currentTarget.style.background = "rgba(255, 255, 255, 0.04)";
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "transparent";
                }}
              >
                <ChevronsRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>
      </div>
    </div>
  );
}
