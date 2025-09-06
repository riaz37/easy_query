"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Database, Plus } from "lucide-react";
import { MSSQLUsersListProps } from "../types";
import { UserCard } from "./UserCard";
import { EmptyState } from "./EmptyState";

export function MSSQLUsersList({
  users,
  onEditUser,
  onCreateAccess,
  extractNameFromEmail,
  getAccessLevelBadge,
  getDatabaseCount,
  isDark
}: MSSQLUsersListProps) {
  return (
    <Card className={`transition-all duration-200 ${
      isDark 
        ? "bg-gradient-to-br from-slate-800/80 to-slate-700/60 border-slate-600 hover:border-slate-500" 
        : "bg-gradient-to-br from-white to-gray-50 border-gray-200 shadow-sm hover:shadow-md"
    }`}>
      <CardHeader>
        <CardTitle className={`${isDark ? 'text-white' : 'text-gray-900'} flex items-center gap-2`}>
          <Database className="h-5 w-5 text-blue-500" />
          MSSQL Database Access Users
        </CardTitle>
        <p className={`${isDark ? 'text-gray-300' : 'text-gray-600'} text-sm`}>
          Users with access to MSSQL databases for data operations
        </p>
      </CardHeader>
      <CardContent>
        {users.length === 0 ? (
          <EmptyState
            icon={<Database className="h-12 w-12" />}
            title="No MSSQL Access Users"
            description="No users have been granted access to MSSQL databases yet."
            actionLabel="Grant First Access"
            onAction={onCreateAccess}
            isDark={isDark}
          />
        ) : (
          <div className="space-y-4">
            {users.map((user) => (
              <UserCard
                key={user.user_id}
                user={user}
                type="mssql"
                onEdit={onEditUser}
                extractNameFromEmail={extractNameFromEmail}
                getAccessLevelBadge={getAccessLevelBadge}
                getDatabaseCount={getDatabaseCount}
                isDark={isDark}
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
