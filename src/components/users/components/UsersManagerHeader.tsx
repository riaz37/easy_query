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
          <h1 className="text-3xl font-bold font-barlow text-white flex items-center gap-3">
            <div className="relative">
              <Users className="h-8 w-8 text-emerald-400" />
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-emerald-400 rounded-full animate-pulse" />
            </div>
            User Access Management
          </h1>
          <p className="text-gray-300 mt-2 font-public-sans">
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
