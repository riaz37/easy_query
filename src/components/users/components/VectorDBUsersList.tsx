"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Brain, RefreshCw } from "lucide-react";
import { PageLoader, InlineLoader } from "@/components/ui/loading";
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
  return (
    <Card className={`transition-all duration-200 ${
      isDark 
        ? "bg-gradient-to-br from-slate-800/80 to-slate-700/60 border-slate-600 hover:border-slate-500" 
        : "bg-gradient-to-br from-white to-gray-50 border-gray-200 shadow-sm hover:shadow-md"
    }`}>
      <CardHeader>
        <CardTitle className={`${isDark ? 'text-white' : 'text-gray-900'} flex items-center gap-2`}>
          <Brain className="h-5 w-5 text-purple-500" />
          Vector Database Access Users
        </CardTitle>
        <p className={`${isDark ? 'text-gray-300' : 'text-gray-600'} text-sm`}>
          Users with access to vector databases for AI and ML operations
        </p>
        <div className="flex justify-end mt-2">
          <Button
            onClick={onRefresh}
            variant="outline"
            size="sm"
            className={`transition-all duration-200 ${
              isDark 
                ? "border-purple-500/50 text-purple-300 hover:bg-purple-500/20 hover:border-purple-400" 
                : "border-purple-300 text-purple-600 hover:bg-purple-50 hover:border-purple-400"
            }`}
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
      </CardHeader>
      <CardContent>
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
      </CardContent>
    </Card>
  );
}
