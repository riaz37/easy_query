"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Edit, UserCheck, Brain } from "lucide-react";
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
    return "bg-gradient-to-br from-teal-500 to-teal-600";
  };

  const renderUserDetails = () => {
    if (isMSSQL) {
      return (
        <div className="flex items-center gap-4 mt-2 text-xs text-gray-400">
          <span>Databases: {getDatabaseCount?.(userAccessData) || 0}</span>
          <span>Sub-companies: {userAccessData.sub_company_ids?.length || 0}</span>
        </div>
      );
    } else {
      return (
        <>
          <div className="flex items-center gap-4 mt-2 text-xs text-gray-400">
            <span>Access Level: {userConfig.access_level}</span>
            <span>Database: {getDatabaseName?.(userConfig.db_id) || `DB ${userConfig.db_id}`}</span>
            <span>Tables: {formatTableNames?.(userConfig.table_names) || 'No tables'}</span>
          </div>
          {userConfig.table_names && userConfig.table_names.length > 3 && (
            <div className="mt-2 text-xs text-gray-400">
              <details className="cursor-pointer">
                <summary className="hover:text-gray-300">Show all tables</summary>
                <div className="mt-2 pl-4">
                  {userConfig.table_names.map((table: string, index: number) => (
                    <div key={index} className="text-gray-400">
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
              <h4 className="font-medium text-white">
                {extractNameFromEmail(user.user_id)}
              </h4>
              {getAccessLevelBadge(user)}
            </div>
            <p className="text-sm text-gray-300">
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
