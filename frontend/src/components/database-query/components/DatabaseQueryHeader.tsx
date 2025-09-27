"use client";

import React from "react";
import { Database, User, History, BarChart3 } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface DatabaseQueryHeaderProps {
  user?: { user_id: string };
  currentDatabase?: { database_name: string };
  isDark?: boolean;
}

export function DatabaseQueryHeader({
  user,
  currentDatabase,
  isDark = true,
}: DatabaseQueryHeaderProps) {
  return (
    <div className="mb-8">
      <div className="flex items-center justify-center mb-6">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-white flex items-center justify-center gap-3">
            <div className="relative">
              <Database className="h-8 w-8 text-emerald-400" />
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-emerald-400 rounded-full animate-pulse" />
            </div>
            Database Query & AI Reports
          </h1>
          <p className="text-gray-300 mt-2">
            Run quick queries or generate comprehensive AI-powered reports
          </p>
        </div>
      </div>

      {/* Status Badges */}
      <div className="flex justify-center gap-4">
        {user?.user_id && (
          <Badge
            variant="outline"
            className="border-green-400/30 text-green-400"
          >
            <User className="w-4 h-4 mr-2" />
            User: {user.user_id}
          </Badge>
        )}
      </div>
    </div>
  );
}
