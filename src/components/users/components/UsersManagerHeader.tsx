"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Users, Plus, Brain } from "lucide-react";
import { UsersManagerHeaderProps } from "../types";

export function UsersManagerHeader({
  onCreateMSSQLAccess,
  onCreateVectorDBAccess,
  isDark
}: UsersManagerHeaderProps) {
  return (
    <div className="mb-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className={`text-3xl font-bold ${isDark ? 'text-white' : 'text-gray-900'} flex items-center gap-3`}>
            <Users className="h-8 w-8 text-emerald-400" />
            User Access Management
          </h1>
          <p className={`${isDark ? 'text-gray-300' : 'text-gray-600'} mt-2`}>
            Manage user access to MSSQL databases and vector databases
          </p>
        </div>
        <div className="flex gap-3">
          <Button
            onClick={onCreateMSSQLAccess}
            className="bg-blue-500 hover:bg-blue-600 text-white shadow-lg hover:shadow-blue-500/25 transition-all duration-200"
          >
            <Plus className="w-4 h-4 mr-2" />
            Add MSSQL Access
          </Button>
          <Button
            onClick={onCreateVectorDBAccess}
            className="bg-purple-500 hover:bg-purple-600 text-white shadow-lg hover:shadow-purple-500/25 transition-all duration-200"
          >
            <Brain className="w-4 h-4 mr-2" />
            Add Vector DB Access
          </Button>
        </div>
      </div>
    </div>
  );
}
