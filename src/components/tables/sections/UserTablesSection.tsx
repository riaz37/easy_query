import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Loader2, RefreshCw, Plus, Database, Eye } from "lucide-react";
import { UserTablesResponse } from "@/types/api";
import UserTableList from "../UserTableList";

interface UserTablesSectionProps {
  userTables: UserTablesResponse["data"] | null;
  loading: boolean;
  onRefresh: () => void;
  onCreateTable: () => void;
}

export function UserTablesSection({ 
  userTables, 
  loading, 
  onRefresh, 
  onCreateTable 
}: UserTablesSectionProps) {
  const [searchTerm, setSearchTerm] = React.useState("");

  return (
    <Card className="bg-gradient-to-br from-slate-800/50 to-slate-700/30 border-slate-600/50 hover:border-slate-500/50 transition-all duration-300">
      <CardHeader className="pb-6 border-b border-slate-600/30">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 bg-gradient-to-br from-indigo-500/20 to-indigo-600/20 rounded-xl flex items-center justify-center border border-indigo-500/30">
              <Eye className="h-5 w-5 text-indigo-400" />
            </div>
            <div>
              <CardTitle className="text-2xl text-white flex items-center gap-3">
                Your Tables
              </CardTitle>
              <p className="text-slate-400 mt-2">
                Tables managed by the current user
              </p>
            </div>
          </div>
          <Button 
            onClick={onRefresh}
            variant="outline"
            size="sm"
            className="border-indigo-500/50 text-indigo-300 hover:bg-indigo-500/20 hover:border-indigo-400/50 transition-all duration-200"
            disabled={loading}
          >
            {loading ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4 mr-2" />
            )}
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {userTables && userTables.tables && userTables.tables.length > 0 ? (
          <UserTableList
            tables={userTables.tables}
            searchTerm={searchTerm}
            onSearchTermChange={setSearchTerm}
          />
        ) : (
          <div className="text-center py-16">
            <div className="w-20 h-20 bg-slate-700/30 rounded-full flex items-center justify-center mx-auto mb-6 border border-slate-600/30">
              <Database className="h-10 w-10 text-slate-400" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-3">
              No Tables Found
            </h3>
            <p className="text-slate-400 mb-6 max-w-md mx-auto">
              You haven't created any tables yet. Start by creating your first table to organize your data!
            </p>
            <Button 
              onClick={onCreateTable}
              className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white border-0 shadow-lg hover:shadow-blue-500/25 transition-all duration-200"
            >
              <Plus className="h-4 w-4 mr-2" />
              Create Your First Table
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
} 