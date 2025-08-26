import React from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Loader2, Save, FileText } from "lucide-react";

interface BusinessRulesModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  businessRule: string;
  setBusinessRule: (rule: string) => void;
  loading: boolean;
  onSubmit: () => void;
}

export function BusinessRulesModal({
  open,
  onOpenChange,
  businessRule,
  setBusinessRule,
  loading,
  onSubmit,
}: BusinessRulesModalProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent 
        className="max-w-3xl bg-gradient-to-br from-slate-900/95 to-slate-800/95 border-slate-600/50 backdrop-blur-xl shadow-2xl"
        onOpenAutoFocus={(e) => e.preventDefault()}
        onCloseAutoFocus={(e) => e.preventDefault()}
      >
        <DialogHeader className="pb-6 border-b border-slate-600/30">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-gradient-to-br from-purple-500/20 to-purple-600/20 rounded-xl flex items-center justify-center border border-purple-500/30">
              <FileText className="h-6 w-6 text-purple-400" />
            </div>
            <div>
              <DialogTitle className="text-2xl font-bold text-white">Manage Business Rules</DialogTitle>
              <DialogDescription className="text-slate-300 mt-1">
                Configure business rules and data validation policies for your tables
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <div className="space-y-6 py-6">
          <div className="space-y-4">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
              <h3 className="text-lg font-semibold text-white">Business Rule Definition</h3>
            </div>
            
            <div className="space-y-3">
              <Label htmlFor="businessRule" className="text-slate-200 font-medium">
                Business Rule Description
              </Label>
              <div className="relative">
                <Textarea
                  id="businessRule"
                  value={businessRule}
                  onChange={(e) => setBusinessRule(e.target.value)}
                  placeholder="Describe your business rule (e.g., 'All user emails must be unique', 'Product prices must be greater than 0')"
                  className="bg-slate-800/50 border-slate-600/50 text-white placeholder-slate-400 focus:border-purple-500/50 focus:ring-purple-500/20 transition-all duration-200 min-h-[140px] resize-none text-base"
                  autoComplete="off"
                  spellCheck="false"
                />
                {businessRule && (
                  <div className="absolute bottom-3 right-3">
                    <Badge variant="secondary" className="bg-slate-700/50 text-slate-300 text-xs">
                      {businessRule.length} characters
                    </Badge>
                  </div>
                )}
              </div>
              <p className="text-xs text-slate-400">
                Define clear, specific rules that govern data integrity and business logic
              </p>
            </div>

            {/* Business Rule Examples */}
            <div className="bg-slate-800/30 rounded-lg p-4 border border-slate-600/20">
              <div className="flex items-start gap-3">
                <div className="w-5 h-5 bg-purple-500/20 rounded-full flex items-center justify-center mt-0.5">
                  <span className="text-purple-400 text-xs">💡</span>
                </div>
                <div className="text-sm text-slate-300 space-y-2">
                  <p><strong>Example Business Rules:</strong></p>
                  <div className="grid grid-cols-1 gap-2 text-slate-400">
                    <div className="bg-slate-700/30 rounded p-2">
                      <span className="text-purple-300">•</span> User accounts must have unique email addresses
                    </div>
                    <div className="bg-slate-700/30 rounded p-2">
                      <span className="text-purple-300">•</span> Product inventory cannot go below zero
                    </div>
                    <div className="bg-slate-700/30 rounded p-2">
                      <span className="text-purple-300">•</span> Order total must equal sum of line items
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex justify-between items-center pt-6 border-t border-slate-600/30">
          <div className="text-sm text-slate-400">
            Business rules help maintain data quality and enforce business logic
          </div>
          <div className="flex gap-3">
            <Button
              onClick={() => onOpenChange(false)}
              variant="outline"
              className="border-slate-600/50 text-slate-300 hover:bg-slate-700/50 hover:border-slate-500/50 transition-all duration-200 px-6"
            >
              Cancel
            </Button>
            <Button
              onClick={onSubmit}
              disabled={!businessRule.trim() || loading}
              className="bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white border-0 shadow-lg hover:shadow-purple-500/25 transition-all duration-200 px-8 h-11 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="h-4 w-4 mr-2" />
                  Save Rule
                </>
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
} 