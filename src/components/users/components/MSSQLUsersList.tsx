"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Database, Plus } from "lucide-react";
import { useTheme } from "@/store/theme-store";
import { cn } from "@/lib/utils";
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
  const theme = useTheme();
  
  return (
    <div className="card-enhanced">
      <div className="card-content-enhanced">
        <div className="card-header-enhanced">
          <div className="card-title-enhanced flex items-center gap-2">
            <Database className={cn(
              "h-5 w-5",
              theme === "dark" ? "text-emerald-400" : "text-emerald-500"
            )} />
            MSSQL Database Access Users
          </div>
          <p className="card-description-enhanced">
            Users with access to MSSQL databases for data operations
          </p>
        </div>
        <div className="flex-1">
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
        </div>
      </div>
    </div>
  );
}
