"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Users, Plus, Brain } from "lucide-react";
import { useTheme } from "@/store/theme-store";
import { cn } from "@/lib/utils";
import { UsersManagerHeaderProps } from "../types";

export function UsersManagerHeader({
  onCreateMSSQLAccess,
  onCreateVectorDBAccess,
  isDark
}: UsersManagerHeaderProps) {
  const theme = useTheme();
  
  return (
    <div className="mb-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className={cn(
            "text-3xl font-bold font-barlow flex items-center gap-3",
            theme === "dark" ? "text-white" : "text-gray-800"
          )}>
            <div className="relative">
              <Users className={cn(
                "h-8 w-8",
                theme === "dark" ? "text-emerald-400" : "text-emerald-600"
              )} />
              <div className={cn(
                "absolute -top-1 -right-1 w-3 h-3 rounded-full animate-pulse",
                theme === "dark" ? "bg-emerald-400" : "bg-emerald-600"
              )} />
            </div>
            User Access Management
          </h1>
          <p className={cn(
            "mt-2 font-public-sans font-medium",
            theme === "dark" ? "text-gray-300" : "text-gray-700"
          )}>
            Manage user access to MSSQL databases and vector databases
          </p>
        </div>
        <div className="flex gap-3">
          <Button
            onClick={onCreateMSSQLAccess}
            className="card-button-enhanced"
          >
            <Plus className="w-4 h-4 mr-2" />
            Add MSSQL Access
          </Button>
          <Button
            onClick={onCreateVectorDBAccess}
            className="card-button-enhanced"
          >
            <Brain className="w-4 h-4 mr-2" />
            Add Vector DB Access
          </Button>
        </div>
      </div>
    </div>
  );
}
