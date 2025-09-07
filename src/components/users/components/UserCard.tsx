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
      return "bg-gradient-to-br from-blue-500 to-blue-600";
    }
    return "bg-gradient-to-br from-purple-500 to-purple-600";
  };

  const getHoverStyles = () => {
    if (isMSSQL) {
      return isDark 
        ? 'hover:border-blue-500/50 hover:shadow-blue-500/10' 
        : 'hover:border-blue-300 hover:shadow-blue-500/10';
    }
    return isDark 
      ? 'hover:border-purple-500/50 hover:shadow-purple-500/10' 
      : 'hover:border-purple-300 hover:shadow-purple-500/10';
  };

  const getButtonStyles = () => {
    if (isMSSQL) {
      return isDark 
        ? "border-blue-500/50 text-blue-300 hover:bg-blue-500/20 hover:border-blue-400" 
        : "border-blue-300 text-blue-600 hover:bg-blue-50 hover:border-blue-400";
    }
    return isDark 
      ? "border-purple-500/50 text-purple-300 hover:bg-purple-500/20 hover:border-purple-400" 
      : "border-purple-300 text-purple-600 hover:bg-purple-50 hover:border-purple-400";
  };

  const renderUserDetails = () => {
    if (isMSSQL) {
      return (
        <div className={`flex items-center gap-4 mt-2 text-xs ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>
          <span>Databases: {getDatabaseCount?.(userAccessData) || 0}</span>
          <span>Sub-companies: {userAccessData.sub_company_ids?.length || 0}</span>
        </div>
      );
    } else {
      return (
        <>
          <div className={`flex items-center gap-4 mt-2 text-xs ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>
            <span>Access Level: {userConfig.access_level}</span>
            <span>Database: {getDatabaseName?.(userConfig.db_id) || `DB ${userConfig.db_id}`}</span>
            <span>Tables: {formatTableNames?.(userConfig.table_names) || 'No tables'}</span>
          </div>
          {userConfig.table_names && userConfig.table_names.length > 3 && (
            <div className={`mt-2 text-xs ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              <details className="cursor-pointer">
                <summary className={isDark ? 'hover:text-gray-300' : 'hover:text-gray-800'}>Show all tables</summary>
                <div className="mt-2 pl-4">
                  {userConfig.table_names.map((table: string, index: number) => (
                    <div key={index} className={isDark ? 'text-gray-400' : 'text-gray-600'}>
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
    <div
      className={`flex items-center justify-between p-4 rounded-lg border transition-all duration-200 hover:scale-[1.02] hover:shadow-lg ${
        isDark 
          ? 'bg-gradient-to-r from-slate-700/40 to-slate-600/30 border-slate-600' 
          : 'bg-gradient-to-r from-gray-50 to-white border-gray-200'
      } ${getHoverStyles()}`}
    >
      <div className="flex items-center gap-4">
        <div className={`w-10 h-10 ${getIconBg()} rounded-full flex items-center justify-center shadow-lg`}>
          {getIcon()}
        </div>
        <div>
          <div className="flex items-center gap-3 mb-1">
            <h4 className={`font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>
              {extractNameFromEmail(user.user_id)}
            </h4>
            {getAccessLevelBadge(user)}
          </div>
          <p className={`text-sm ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
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
          className={`transition-all duration-200 ${getButtonStyles()}`}
        >
          <Edit className="w-4 h-4 mr-2" />
          Edit
        </Button>
      </div>
    </div>
  );
}
