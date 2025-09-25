"use client";

import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Search } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface UserTableListProps {
  tables: any[]; // Replace 'any' with your actual table type
  searchTerm: string;
  onSearchTermChange: (term: string) => void;
}

const UserTableList: React.FC<UserTableListProps> = ({
  tables,
  searchTerm,
  onSearchTermChange,
}) => {
  const filteredTables = tables.filter((table) => {
    const tableName = table?.table_name || '';
    const fullName = table?.full_name || '';
    
    return tableName.toLowerCase().includes(searchTerm.toLowerCase()) ||
           fullName.toLowerCase().includes(searchTerm.toLowerCase());
  });

  return (
    <div className="space-y-4">
      <Card className="bg-slate-800/50 border-slate-700">
        <CardContent className="pt-6">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-400" />
            <Input
              placeholder="Search tables..."
              value={searchTerm}
              onChange={(e) => onSearchTermChange(e.target.value)}
              className="pl-10"
            />
          </div>
        </CardContent>
      </Card>

      {filteredTables.length > 0 ? (
        <div className="space-y-2">
          {filteredTables.map((table, index) => (
            <Card
              key={index}
              className="p-3 bg-slate-700/50 rounded-lg border border-slate-600"
            >
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-white font-medium">{table?.table_name || 'Unnamed Table'}</h4>
                  <p className="text-slate-400 text-sm">{table?.full_name || 'No full name'}</p>
                </div>
                <Badge variant="secondary">{table?.columns?.length || 0} columns</Badge>
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="text-center py-8">
            <Search className="h-12 w-12 text-slate-400 mx-auto mb-4" />
            <p className="text-slate-400">
              No tables found matching your search criteria.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default UserTableList;
