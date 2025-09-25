"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Edit, UserCheck, Brain } from "lucide-react";
import { useTheme } from "@/store/theme-store";
import { cn } from "@/lib/utils";
import { UserCardProps } from "../types";

export function UserCard({
  user,
  type,
  onEdit,
  extractNameFromEmail,
  getAccessLevelBadge,
  getDatabaseCount,
  getDatabaseName,
  formatTableNames,
  isDark
}: UserCardProps) {
  const theme = useTheme();
  const isMSSQL = type === 'mssql';
  const userAccessData = user as any; // Type assertion for MSSQL users
  const userConfig = user as any; // Type assertion for Vector DB users

  const getIcon = () => {
    if (isMSSQL) {
      return <UserCheck className="w-5 h-5 text-white" />;
    }
    return <Brain className="w-5 h-5 text-white" />;
  };

  const getIconBg = () => {
    if (isMSSQL) {
      return "bg-gradient-to-br from-emerald-500 to-emerald-600";
    }
    return "bg-gradient-to-br from-emerald-500 to-emerald-600";
  };

  const renderUserDetails = () => {
    if (isMSSQL) {
      return (
        <div className={cn(
          "flex items-center gap-4 mt-2 text-xs font-medium",
          theme === "dark" ? "text-gray-400" : "text-gray-600"
        )}>
          <span>Databases: {getDatabaseCount?.(userAccessData) || 0}</span>
          <span>Sub-companies: {userAccessData.sub_company_ids?.length || 0}</span>
        </div>
      );
    } else {
      return (
        <>
          <div className={cn(
            "flex items-center gap-4 mt-2 text-xs font-medium",
            theme === "dark" ? "text-gray-400" : "text-gray-600"
          )}>
            <span>Access Level: {userConfig.access_level}</span>
            <span>Database: {getDatabaseName?.(userConfig.db_id) || `DB ${userConfig.db_id}`}</span>
            <span>Tables: {formatTableNames?.(userConfig.table_names) || 'No tables'}</span>
          </div>
          {userConfig.table_names && userConfig.table_names.length > 3 && (
            <div className={cn(
              "mt-2 text-xs font-medium",
              theme === "dark" ? "text-gray-400" : "text-gray-600"
            )}>
              <details className="cursor-pointer">
                <summary className={cn(
                  "hover:transition-colors font-medium",
                  theme === "dark" ? "hover:text-gray-300" : "hover:text-gray-700"
                )}>Show all tables</summary>
                <div className="mt-2 pl-4">
                  {userConfig.table_names.map((table: string, index: number) => (
                    <div key={index} className={cn(
                      "font-medium",
                      theme === "dark" ? "text-gray-400" : "text-gray-600"
                    )}>
                      • {table}
                    </div>
                  ))}
                </div>
              </details>
            </div>
          )}
        </>
      );
    }
  };

  return (
    <div className="user-card-enhanced">
      <div className="user-card-content-enhanced">
        <div className="flex items-center gap-4">
          <div className={`w-12 h-12 ${getIconBg()} rounded-full flex items-center justify-center shadow-lg`}>
            {getIcon()}
          </div>
          <div>
            <div className="flex items-center gap-3 mb-1">
              <h4 className={cn(
                "font-semibold",
                theme === "dark" ? "text-white" : "text-gray-800"
              )}>
                {extractNameFromEmail(user.user_id)}
              </h4>
              {getAccessLevelBadge(user)}
            </div>
            <p className={cn(
              "text-sm font-medium",
              theme === "dark" ? "text-gray-300" : "text-gray-700"
            )}>
              {user.user_id}
            </p>
            {renderUserDetails()}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            onClick={() => onEdit(user.user_id)}
            variant="outline"
            size="sm"
            className="card-button-enhanced"
          >
            <Edit className="w-4 h-4 mr-2" />
            Edit
          </Button>
        </div>
      </div>
    </div>
  );
}
