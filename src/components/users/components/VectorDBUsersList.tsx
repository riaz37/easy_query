"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Brain, RefreshCw } from "lucide-react";
import { PageLoader, InlineLoader } from "@/components/ui/loading";
import { useTheme } from "@/store/theme-store";
import { cn } from "@/lib/utils";
import { VectorDBUsersListProps } from "../types";
import { UserCard } from "./UserCard";
import { EmptyState } from "./EmptyState";

export function VectorDBUsersList({
  users,
  onEditUser,
  onCreateAccess,
  onRefresh,
  extractNameFromEmail,
  getAccessLevelBadge,
  getDatabaseName,
  formatTableNames,
  isLoading,
  isDark
}: VectorDBUsersListProps) {
  const theme = useTheme();
  
  return (
    <div className="card-enhanced">
      <div className="card-content-enhanced">
        <div className="card-header-enhanced">
          <div className="card-title-enhanced flex items-center gap-2">
            <Brain className={cn(
              "h-5 w-5",
              theme === "dark" ? "text-emerald-400" : "text-emerald-500"
            )} />
            Vector Database Access Users
          </div>
          <p className="card-description-enhanced">
            Users with access to vector databases 
          </p>
          <div className="flex justify-end mt-2">
            <Button
              onClick={onRefresh}
              variant="outline"
              size="sm"
              className="card-button-enhanced"
              disabled={isLoading}
            >
              {isLoading ? (
                <InlineLoader size="sm" variant="primary" className="mr-2" />
              ) : (
                <RefreshCw className="h-4 w-4 mr-2" />
              )}
              Refresh
            </Button>
          </div>
        </div>
        <div className="flex-1">
          {isLoading ? (
            <div className="text-center py-12">
              <PageLoader
                size="lg"
                variant="primary"
                message="Loading Vector DB Access Users"
                description="Fetching user configurations from the server..."
                showProgress={false}
              />
            </div>
          ) : users.length === 0 ? (
            <EmptyState
              icon={<Brain className="h-12 w-12" />}
              title="No Vector DB Access Users"
              description="No users have been granted access to vector databases yet."
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
                  type="vector"
                  onEdit={onEditUser}
                  extractNameFromEmail={extractNameFromEmail}
                  getAccessLevelBadge={getAccessLevelBadge}
                  getDatabaseName={getDatabaseName}
                  formatTableNames={formatTableNames}
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
