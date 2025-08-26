import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Settings, Table as TableIcon, FileText } from "lucide-react";

interface QuickStatsSectionProps {
  tableCount: number;
  businessRuleCount: number;
}

export function QuickStatsSection({ tableCount, businessRuleCount }: QuickStatsSectionProps) {
  return (
    <Card className="bg-gradient-to-br from-slate-800/50 to-slate-700/30 border-slate-600/50 hover:border-slate-500/50 transition-all duration-300">
      <CardHeader className="pb-4 border-b border-slate-600/30">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-br from-slate-500/20 to-slate-600/20 rounded-lg flex items-center justify-center border border-slate-500/30">
            <Settings className="h-4 w-4 text-slate-400" />
          </div>
          <CardTitle className="text-lg text-white">Quick Stats</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="text-center p-4 bg-gradient-to-br from-blue-900/20 to-blue-800/10 rounded-xl border border-blue-500/20">
            <div className="w-16 h-16 bg-gradient-to-br from-blue-500/20 to-blue-600/20 rounded-full flex items-center justify-center mx-auto mb-3 border border-blue-500/30">
              <TableIcon className="h-8 w-8 text-blue-400" />
            </div>
            <p className="text-3xl font-bold text-blue-400 mb-1">
              {tableCount}
            </p>
            <p className="text-slate-300 text-sm font-medium">User Tables</p>
          </div>
          <div className="text-center p-4 bg-gradient-to-br from-green-900/20 to-green-800/10 rounded-xl border border-green-500/20">
            <div className="w-16 h-16 bg-gradient-to-br from-green-500/20 to-green-600/20 rounded-full flex items-center justify-center mx-auto mb-3 border border-green-500/30">
              <FileText className="h-8 w-8 text-green-400" />
            </div>
            <p className="text-3xl font-bold text-green-400 mb-1">
              {businessRuleCount}
            </p>
            <p className="text-slate-300 text-sm font-medium">Business Rules</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 