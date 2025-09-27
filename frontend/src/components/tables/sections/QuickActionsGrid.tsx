import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Plus, FileText } from "lucide-react";

interface QuickActionsGridProps {
  onCreateTable: () => void;
  onManageRules: () => void;
}

export function QuickActionsGrid({ onCreateTable, onManageRules }: QuickActionsGridProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Create Table Card */}
      <Card className="bg-gradient-to-br from-blue-900/30 to-blue-800/20 border-blue-500/30 hover:border-blue-400/50 transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/20 group">
        <CardContent className="p-6">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500/20 to-blue-600/20 rounded-xl flex items-center justify-center border border-blue-500/30 group-hover:border-blue-400/50 transition-all duration-300">
              <Plus className="h-6 w-6 text-blue-400 group-hover:scale-110 transition-transform duration-300" />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-white">Create Table</h3>
              <p className="text-blue-200 text-sm">Design new table structures</p>
            </div>
          </div>
          <p className="text-slate-300 text-sm mb-4">
            Create custom tables with columns, data types, and constraints
          </p>
          <Button 
            onClick={onCreateTable}
            className="w-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white border-0 shadow-lg hover:shadow-blue-500/25 transition-all duration-200 group-hover:scale-[1.02]"
          >
            <Plus className="h-4 w-4 mr-2" />
            Create New Table
          </Button>
        </CardContent>
      </Card>

      {/* Business Rules Card */}
      <Card className="bg-gradient-to-br from-purple-900/30 to-purple-800/20 border-purple-500/30 hover:border-purple-400/50 transition-all duration-300 hover:shadow-lg hover:shadow-purple-500/20 group">
        <CardContent className="p-6">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-12 h-12 bg-gradient-to-br from-purple-500/20 to-purple-600/20 rounded-xl flex items-center justify-center border border-purple-500/30 group-hover:border-purple-400/50 transition-all duration-300">
              <FileText className="h-6 w-6 text-purple-400 group-hover:scale-110 transition-transform duration-300" />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-white">Business Rules</h3>
              <p className="text-purple-200 text-sm">Configure data policies</p>
            </div>
          </div>
          <p className="text-slate-300 text-sm mb-4">
            Set up business rules and data validation policies
          </p>
          <Button 
            onClick={onManageRules}
            className="w-full bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white border-0 shadow-lg hover:shadow-purple-500/25 transition-all duration-200 group-hover:scale-[1.02]"
          >
            <FileText className="h-4 w-4 mr-2" />
            Manage Rules
          </Button>
        </CardContent>
      </Card>
    </div>
  );
} 