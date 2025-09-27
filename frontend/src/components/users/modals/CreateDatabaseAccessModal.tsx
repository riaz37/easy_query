"use client";

import React, { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Database, Building2, CheckCircle, AlertCircle, X } from "lucide-react";
import { useUserAccess } from "@/lib/hooks/use-user-access";
import { useDatabaseContext } from "@/components/providers/DatabaseContextProvider";
import { useParentCompanies } from "@/lib/hooks/use-parent-companies";
import { useSubCompanies } from "@/lib/hooks/use-sub-companies";
import { useAuthContext } from "@/components/providers/AuthContextProvider";
import { useTheme } from "@/store/theme-store";
import { cn } from "@/lib/utils";
import {
  UserAccessCreateRequest,
  ParentCompanyData,
  SubCompanyData,
} from "@/types/api";
import { toast } from "sonner";

interface CreateDatabaseAccessModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
  selectedUser?: string;
  editingUser?: string;
}

export function CreateDatabaseAccessModal({
  isOpen,
  onClose,
  onSuccess,
  selectedUser = "",
  editingUser = "",
}: CreateDatabaseAccessModalProps) {
  const { user } = useAuthContext();
  const theme = useTheme();

  // Form state
  const [selectedParentCompany, setSelectedParentCompany] =
    useState<string>("");
  const [selectedSubCompany, setSelectedSubCompany] = useState<string>("");
  const [selectedDatabase, setSelectedDatabase] = useState<string>("");
  const [selectedUserId, setSelectedUserId] = useState<string>(""); // New state for user ID

  // Data state
  // Removed local state for parent and sub companies - using hook state directly

  // Hooks
  const { createUserAccess, isLoading, error } = useUserAccess();
  const { availableDatabases } = useDatabaseContext();
  const {
    parentCompanies,
    getParentCompanies,
    isLoading: isLoadingParentCompanies,
  } = useParentCompanies();
  const {
    subCompanies,
    getSubCompanies,
    isLoading: isLoadingSubCompanies,
  } = useSubCompanies();

  // Load data when modal opens
  useEffect(() => {
    if (isOpen) {
      loadData();
    }
  }, [isOpen]);

  const loadData = async () => {
    try {
      // Load parent companies
      await getParentCompanies();

      // Load sub companies
      await getSubCompanies();

      // Note: availableDatabases is already loaded from context
      // No need to fetch separately
    } catch (error) {
      console.error("Error loading data:", error);
    }
  };

  // Update userId when selectedUser prop changes
  useEffect(() => {
    // This useEffect is no longer needed as user ID is now from auth context
  }, [selectedUser]);

  // Auto-populate database when company selection changes
  useEffect(() => {
    console.log("Company selection changed:", {
      selectedParentCompany,
      selectedSubCompany,
    });

    if (selectedParentCompany || selectedSubCompany) {
      let databaseId = "";

      if (selectedParentCompany) {
        // Get database ID from parent company
        const parentCompany = parentCompanies.find(
          (c) => c.parent_company_id === parseInt(selectedParentCompany)
        );
        if (parentCompany) {
          databaseId = parentCompany.db_id.toString();
          console.log("Setting database ID from parent company:", databaseId);
        }
      } else if (selectedSubCompany) {
        // Get database ID from sub company
        const subCompany = subCompanies.find(
          (c) => c.sub_company_id === parseInt(selectedSubCompany)
        );
        if (subCompany) {
          databaseId = subCompany.db_id.toString();
          console.log("Setting database ID from sub company:", databaseId);
        }
      }

      if (databaseId) {
        setSelectedDatabase(databaseId);
      }
    } else {
      setSelectedDatabase("");
    }
  }, [
    selectedParentCompany,
    selectedSubCompany,
    parentCompanies,
    subCompanies,
  ]);

  // Handle company selection changes
  const handleParentCompanyChange = (value: string) => {
    setSelectedParentCompany(value);
    setSelectedSubCompany(""); // Clear sub company when parent changes
    setSelectedDatabase(""); // Clear database selection
  };

  const handleSubCompanyChange = (value: string) => {
    setSelectedSubCompany(value);
    setSelectedDatabase(""); // Clear database selection
  };

  // Get available sub companies for selected parent company
  const getAvailableSubCompanies = () => {
    if (!selectedParentCompany) return [];
    return subCompanies.filter(
      (sub) => sub.parent_company_id === parseInt(selectedParentCompany)
    );
  };

  // Handle table selection
  const handleTableSelection = (tableName: string, checked: boolean) => {
    // This function is no longer needed
  };

  // Handle form submission
  const handleSubmit = async () => {
    console.log("Submit attempt:", {
      selectedUserId,
      selectedParentCompany,
      selectedSubCompany,
      selectedDatabase,
      parentCompanies: parentCompanies.length,
      subCompanies: subCompanies.length,
    });

    if (!selectedUserId) {
      toast.error("User ID is required");
      return;
    }

    if (!selectedParentCompany || !selectedDatabase) {
      toast.error("Parent company and database are required");
      return;
    }

    const request: UserAccessCreateRequest = {
      user_id: selectedUserId, // Keep as string - API expects string
      parent_company_id: parseInt(selectedParentCompany),
      sub_company_ids: selectedSubCompany ? [parseInt(selectedSubCompany)] : [],
      database_access: {
        parent_databases: [
          {
            db_id: parseInt(selectedDatabase),
            access_level: "full",
          },
        ],
        sub_databases: selectedSubCompany
          ? [
              {
                sub_company_id: parseInt(selectedSubCompany),
                databases: [
                  {
                    db_id: parseInt(selectedDatabase),
                    access_level: "full", // Use string enum instead of number
                  },
                ],
              },
            ]
          : [],
      },
      table_shows: {
        [selectedDatabase]: [], // Empty array for tables since we removed table selection
      },
    };

    try {
      console.log("Sending request:", request);
      const result = await createUserAccess(request);
      console.log("API Response:", result);

      if (result) {
        toast.success("Database access created successfully");
        onSuccess();
        handleClose();
        resetForm();
      } else {
        toast.error("Failed to create database access");
      }
    } catch (error) {
      console.error("Error creating user access:", error);
      toast.error(
        `Failed to create database access: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  };

  // Reset form
  const resetForm = () => {
    setSelectedParentCompany("");
    setSelectedSubCompany("");
    setSelectedDatabase("");
    setSelectedUserId(""); // Reset user ID
  };

  // Handle close
  const handleClose = () => {
    resetForm();
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent
        className="max-w-4xl max-h-[90vh] p-0 border-0 bg-transparent"
        showCloseButton={false}
      >
        <div className="modal-enhanced">
          <div className="modal-content-enhanced flex flex-col max-h-[90vh]">
            <DialogHeader className="modal-header-enhanced flex-shrink-0">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <DialogTitle className="modal-title-enhanced">
                    {editingUser
                      ? "Edit Database Access"
                      : "Create Database Access"}
                  </DialogTitle>
                  <p className="modal-description-enhanced">
                    {editingUser
                      ? "Update user access to MSSQL databases"
                      : "Grant user access to MSSQL databases for data operations"}
                  </p>
                </div>
                <button onClick={handleClose} className="modal-close-button">
                  <X className="h-5 w-5" />
                </button>
              </div>
            </DialogHeader>

            <div className="modal-form-content flex-1 overflow-y-auto px-6 pb-6">
              {/* User ID Input */}
              <div className="modal-form-group">
                <Label className="modal-label-enhanced">
                  User ID <span className="text-red-500">*</span>
                </Label>
                <Input
                  placeholder="Enter user ID"
                  value={selectedUserId}
                  onChange={(e) => setSelectedUserId(e.target.value)}
                  className="modal-input-enhanced"
                />
              </div>

              {/* Company Selection */}
              <div className="modal-form-group">
                <Label className="modal-label-enhanced">
                  Parent Company <span className="text-red-500">*</span>
                </Label>
                <Select
                  value={selectedParentCompany}
                  onValueChange={handleParentCompanyChange}
                >
                  <SelectTrigger className="modal-select-enhanced w-full">
                    <SelectValue
                      placeholder={
                        isLoadingParentCompanies
                          ? "Loading..."
                          : "Select parent company"
                      }
                    />
                  </SelectTrigger>
                  <SelectContent className="modal-select-content-enhanced">
                    {parentCompanies.map((company) => (
                      <SelectItem
                        key={company.parent_company_id}
                        value={company.parent_company_id.toString()}
                        className="dropdown-item"
                      >
                        <span>{company.company_name}</span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <div className="modal-form-description">
                  Select the parent company for this database access
                </div>
              </div>

              {/* Sub Company - Optional */}
              {selectedParentCompany && (
                <div className="modal-form-group">
                  <Label className="modal-label-enhanced">
                    Sub Company (Optional)
                  </Label>
                  <Select
                    value={selectedSubCompany}
                    onValueChange={handleSubCompanyChange}
                  >
                    <SelectTrigger className="modal-select-enhanced w-full">
                      <SelectValue placeholder="Select sub company (optional)" />
                    </SelectTrigger>
                    <SelectContent className="modal-select-content-enhanced">
                      <SelectItem value="none" className="dropdown-item">
                        <span>None</span>
                      </SelectItem>
                      {getAvailableSubCompanies().map((company) => (
                        <SelectItem
                          key={company.sub_company_id}
                          value={company.sub_company_id.toString()}
                          className="dropdown-item"
                        >
                          <span>{company.company_name}</span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <div className="modal-form-description">
                    Only sub companies belonging to{" "}
                    {
                      parentCompanies.find(
                        (c) =>
                          c.parent_company_id ===
                          parseInt(selectedParentCompany)
                      )?.company_name
                    }{" "}
                    are shown
                  </div>
                </div>
              )}

              {/* Debug Info - Show when no database is selected */}
              {(selectedParentCompany || selectedSubCompany) &&
                !selectedDatabase && (
                  <div className="space-y-3">
                    <Label
                      className={cn(
                        "font-semibold",
                        theme === "dark" ? "text-white" : "text-gray-800"
                      )}
                    >
                      Database Selection Issue
                    </Label>
                    <div className="p-3 bg-yellow-900/20 border border-yellow-500/30 rounded-lg">
                      <div className="text-yellow-400 text-sm">
                        No databases found for the selected company
                        configuration.
                      </div>
                      <div className="text-xs text-yellow-300 mt-2">
                        Available databases: {availableDatabases.length}
                        {selectedParentCompany && (
                          <div>Parent Company ID: {selectedParentCompany}</div>
                        )}
                        {selectedSubCompany && (
                          <div>Sub Company ID: {selectedSubCompany}</div>
                        )}
                        <div>Total DB Configs: {availableDatabases.length}</div>
                      </div>
                    </div>
                  </div>
                )}

              {/* Error Display */}
              {error && (
                <div className="modal-form-group">
                  <div className="flex items-center p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
                    <span className="text-red-400 text-sm">{error}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Action Buttons - Fixed Footer */}
            <div className="flex-shrink-0 px-6 py-6">
              <div className="flex justify-start gap-3">
                <Button
                  variant="outline"
                  onClick={handleClose}
                  className="modal-button-secondary"
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleSubmit}
                  disabled={
                    isLoading ||
                    !selectedUserId ||
                    !selectedParentCompany ||
                    !selectedDatabase
                  }
                  className="modal-button-primary"
                >
                  {isLoading ? "Creating..." : "Create Access"}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
