"use client";

import React, { useState } from "react";
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
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FileText, Save, XIcon, CheckCircle, AlertCircle } from "lucide-react";

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
  const [localError, setLocalError] = useState<string | null>(null);
  const [localSuccess, setLocalSuccess] = useState<string | null>(null);

  const handleSubmit = () => {
    setLocalError(null);
    setLocalSuccess(null);

    if (!businessRule.trim()) {
      setLocalError("Business rule cannot be empty");
      return;
    }

    onSubmit();
  };

  const handleSuccess = () => {
    setLocalSuccess("Business rule updated successfully!");
    setTimeout(() => {
      setLocalSuccess(null);
      onOpenChange(false);
    }, 2000);
  };

  const handleError = (error: string) => {
    setLocalError(error);
  };

  // Call the parent's onSubmit and handle success/error
  React.useEffect(() => {
    if (localSuccess) {
      handleSuccess();
    }
  }, [localSuccess]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="max-w-3xl max-h-[90vh] p-0 border-0 bg-transparent"
        showCloseButton={false}
      >
        <div className="modal-enhanced">
          <div className="modal-content-enhanced">
            <DialogHeader className="modal-header-enhanced">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <DialogTitle className="modal-title-enhanced flex items-center gap-2">
                    Business Rules Management
                  </DialogTitle>
                  <p className="modal-description-enhanced">
                    Define and manage business rules for your database tables
                  </p>
                </div>
                <button
                  onClick={() => onOpenChange(false)}
                  className="modal-close-button cursor-pointer"
                >
                  <XIcon className="h-5 w-5" />
                </button>
              </div>
            </DialogHeader>

            <div className="modal-form-content overflow-y-auto max-h-[calc(90vh-200px)]">
              {/* Success Alert */}
              {localSuccess && (
                <Alert className="mb-6 border-green-200 bg-green-800/20 text-green-300">
                  <CheckCircle className="h-4 w-4" />
                  <AlertDescription>{localSuccess}</AlertDescription>
                </Alert>
              )}

              {/* Error Alert */}
              {localError && (
                <Alert variant="destructive" className="mb-6">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{localError}</AlertDescription>
                </Alert>
              )}

              {/* Business Rule Input */}
              <div className="modal-form-group">
                <Label className="modal-form-label">
                  Business Rule Definition
                </Label>
                <Textarea
                  className="modal-input-enhanced min-h-[200px] resize-none"
                  value={businessRule}
                  onChange={(e) => setBusinessRule(e.target.value)}
                  placeholder="Enter your business rules here...

Example:
- All user emails must be unique
- Product prices must be greater than 0
- Order dates cannot be in the future
- Customer age must be between 18 and 120
- All required fields must be non-null"
                />
                <p className="text-xs text-slate-400 mt-2">
                  Define clear, actionable business rules that will be enforced
                  across your database tables.
                </p>
              </div>

              {/* Submit Button */}
              <div className="modal-footer-enhanced">
                <Button
                  onClick={handleSubmit}
                  disabled={loading || !businessRule.trim()}
                  className="modal-button-primary cursor-pointer"
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                      Saving...
                    </>
                  ) : (
                    <>Save</>
                  )}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
