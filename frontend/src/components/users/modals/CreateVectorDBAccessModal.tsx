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
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Brain,
  Database,
  CheckCircle,
  AlertCircle,
  X,
  Plus,
  Search,
} from "lucide-react";
import { Spinner } from "@/components/ui/loading";
import { useAuthContext } from "@/components/providers";
import { useVectorDB } from "@/lib/hooks/use-vector-db";
import { useTheme } from "@/store/theme-store";
import { cn } from "@/lib/utils";
import { UserConfigCreateRequest } from "@/types/api";
import { VectorDBService } from "@/lib/api/services/vector-db-service";
import { ServiceRegistry } from "@/lib/api/services/service-registry";

interface CreateVectorDBAccessModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
  selectedUser?: string;
  editingUser?: string;
}

export function CreateVectorDBAccessModal({
  isOpen,
  onClose,
  onSuccess,
  selectedUser = "",
  editingUser = "",
}: CreateVectorDBAccessModalProps) {
  const theme = useTheme();

  // Form state
  const [selectedUserId, setSelectedUserId] = useState<string>(
    selectedUser || ""
  );
  const [selectedDatabase, setSelectedDatabase] = useState<string>("");
  const [selectedTables, setSelectedTables] = useState<string[]>([]);
  const [newTableName, setNewTableName] = useState("");
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string>("");
  const [submitSuccess, setSubmitSuccess] = useState(false);

  // User table fetching state
  const [existingUserTables, setExistingUserTables] = useState<string[]>([]);
  const [isFetchingUserTables, setIsFetchingUserTables] = useState(false);
  const [userTablesError, setUserTablesError] = useState<string>("");

  // Database loading state
  // Removed local availableDatabases state - using context directly

  // Hooks
  const { user } = useAuthContext();

  // Placeholder state - will be implemented with proper database context
  const isLoading = false;
  const error = null;

  const createUserConfig = async (data: UserConfigCreateRequest) => {
    try {
      const response = await ServiceRegistry.userConfig.createUserConfig(data);
      return response;
    } catch (error) {
      console.error("Error creating user config:", error);
      throw error;
    }
  };
  const {
    vectorDBConfigs,
    getVectorDBConfigs,
    isLoading: isLoadingDatabases,
  } = useVectorDB();

  // Load data when modal opens
  useEffect(() => {
    if (isOpen) {
      loadData();
    }
  }, [isOpen]);

  // Update local state when props change
  useEffect(() => {
    setSelectedUserId(selectedUser || "");
  }, [selectedUser]);

  const loadData = async () => {
    try {
      // Load vector DB configurations
      const result = await getVectorDBConfigs();
      console.log("Vector DB configs loaded:", result);
      console.log("vectorDBConfigs state:", vectorDBConfigs);
    } catch (error) {
      console.error("Error loading data:", error);
    }
  };

  // Load existing user tables when user is authenticated
  useEffect(() => {
    if (user?.user_id && isOpen) {
      fetchExistingUserTables(user.user_id);
    }
  }, [user?.user_id, isOpen]);

  // Fetch existing tables for the authenticated user
  const fetchExistingUserTables = async (userId: string) => {
    if (!userId.trim()) return;

    setIsFetchingUserTables(true);
    setUserTablesError("");

    try {
      // TODO: Implement actual API call to fetch user tables
      // For now, simulate with empty array
      setExistingUserTables([]);
    } catch (error) {
      console.error("Error fetching user tables:", error);
      setUserTablesError("Failed to fetch user tables");
    } finally {
      setIsFetchingUserTables(false);
    }
  };

  // Handle table selection
  const handleTableSelection = (tableName: string, checked: boolean) => {
    if (checked) {
      setSelectedTables((prev) => [...prev, tableName]);
    } else {
      setSelectedTables((prev) => prev.filter((t) => t !== tableName));
    }
  };

  // Add new table
  const handleAddNewTable = () => {
    if (newTableName.trim() && !selectedTables.includes(newTableName.trim())) {
      setSelectedTables((prev) => [...prev, newTableName.trim()]);
      setNewTableName("");
    }
  };

  // Remove table
  const handleRemoveTable = (tableName: string) => {
    setSelectedTables((prev) => prev.filter((t) => t !== tableName));
  };

  // Filter tables based on search
  const filteredTables = existingUserTables.filter((table) =>
    table.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Handle form submission
  const handleSubmit = async () => {
    // Clear previous errors
    setSubmitError("");
    setSubmitSuccess(false);

    // Validation
    if (!selectedUserId) {
      setSubmitError("User ID is required");
      return;
    }

    if (!selectedDatabase || selectedTables.length === 0) {
      setSubmitError("Please select a database and at least one table");
      return;
    }

    setIsSubmitting(true);

    try {
      const request: UserConfigCreateRequest = {
        user_id: selectedUserId,
        db_id: parseInt(selectedDatabase),
        access_level: 2, // Default value as per your spec
        accessible_tables: [], // Empty array as per your spec
        table_names: selectedTables,
      };

      console.log("Submitting vector DB access request:", request);

      const result = await createUserConfig(request);

      if (result && (result.status === "success" || result.config_id)) {
        setSubmitSuccess(true);
        console.log("Vector DB access created successfully:", result);

        // Wait a moment to show success message
        setTimeout(() => {
          onSuccess();
          onClose();
          resetForm();
        }, 1500);
      } else {
        setSubmitError("Failed to create vector DB access. Please try again.");
      }
    } catch (error: any) {
      console.error("Error creating vector DB access:", error);

      // Provide user-friendly error messages
      if (error.response?.status === 400) {
        setSubmitError("Invalid request data. Please check your input.");
      } else if (error.response?.status === 409) {
        setSubmitError("User already has access to this database.");
      } else if (error.response?.status === 404) {
        setSubmitError("Database not found. Please select a valid database.");
      } else if (error.message) {
        setSubmitError(`Error: ${error.message}`);
      } else {
        setSubmitError("An unexpected error occurred. Please try again.");
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  // Reset form
  const resetForm = () => {
    setSelectedUserId("");
    setSelectedDatabase("");
    setSelectedTables([]);
    setNewTableName("");
    setSearchTerm("");
    setSubmitError("");
    setSubmitSuccess(false);
    setIsSubmitting(false);
    setExistingUserTables([]);
    setIsFetchingUserTables(false);
    setUserTablesError("");
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
          <div className="modal-content-enhanced max-h-[90vh] overflow-y-auto">
            <DialogHeader className="modal-header-enhanced">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <DialogTitle className="modal-title-enhanced">
                    {editingUser
                      ? "Edit Vector DB Access"
                      : "Create Vector DB Access"}
                  </DialogTitle>
                  <p className="modal-description-enhanced">
                    {editingUser
                      ? "Update user access to vector databases"
                      : "Grant user access to vector databases for AI operations"}
                  </p>
                </div>
                <button onClick={handleClose} className="modal-close-button">
                  <X className="h-5 w-5" />
                </button>
              </div>
            </DialogHeader>

            <div className="modal-form-content">
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

              {/* Database Selection */}
              <div className="modal-form-group">
                <Label className="modal-label-enhanced">Database</Label>

                {isLoadingDatabases ? (
                  <div className="flex items-center space-x-2 p-3 modal-input-enhanced rounded-lg">
                    <Spinner size="sm" variant="accent-purple" />
                    <span
                      className={cn(
                        "text-sm font-medium",
                        theme === "dark" ? "text-gray-400" : "text-gray-600"
                      )}
                    >
                      Loading available databases...
                    </span>
                  </div>
                ) : vectorDBConfigs.length === 0 ? (
                  <div
                    className={cn(
                      "text-sm font-medium modal-input-enhanced p-3 rounded-lg",
                      theme === "dark" ? "text-gray-400" : "text-gray-600"
                    )}
                  >
                    No databases available. Please check your database
                    configuration.
                  </div>
                ) : (
                  <Select
                    value={selectedDatabase}
                    onValueChange={setSelectedDatabase}
                  >
                    <SelectTrigger className="modal-select-enhanced w-full">
                      <SelectValue placeholder="Select database" />
                    </SelectTrigger>
                    <SelectContent className="modal-select-content-enhanced">
                      {Array.isArray(vectorDBConfigs) &&
                        vectorDBConfigs.map((db) => (
                          <SelectItem
                            key={db.db_id}
                            value={db.db_id.toString()}
                            className="dropdown-item"
                          >
                            <div className="flex flex-col">
                              <span className="font-medium">
                                {db.db_config.DB_NAME}
                              </span>
                              <span
                                className={cn(
                                  "text-xs font-medium",
                                  theme === "dark"
                                    ? "text-gray-400"
                                    : "text-gray-600"
                                )}
                              >
                                {db.db_config.DB_HOST}:{db.db_config.DB_PORT}
                              </span>
                            </div>
                          </SelectItem>
                        ))}
                    </SelectContent>
                  </Select>
                )}
              </div>

              {/* Table Selection */}
              <div className="modal-form-group">
                <Label className="modal-label-enhanced">Tables</Label>

                {/* Add New Table */}
                <div className="flex gap-2">
                  <Input
                    value={newTableName}
                    onChange={(e) => setNewTableName(e.target.value)}
                    placeholder="Enter new table name"
                    className="modal-input-enhanced flex-1"
                    onKeyPress={(e) => {
                      if (e.key === "Enter") {
                        handleAddNewTable();
                      }
                    }}
                  />
                  <Button
                    onClick={handleAddNewTable}
                    disabled={!newTableName.trim()}
                    variant="outline"
                    size="sm"
                    className="modal-button-secondary"
                  >
                    Add Table
                  </Button>
                </div>

                {/* Search Existing Tables */}
                {existingUserTables.length > 0 && (
                  <div className="space-y-3">
                    <div className="relative">
                      <Search
                        className={cn(
                          "absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4",
                          theme === "dark" ? "text-gray-400" : "text-gray-500"
                        )}
                      />
                      <Input
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        placeholder="Search existing tables..."
                        className="pl-10 modal-input-enhanced"
                      />
                    </div>

                    {/* Existing Tables List */}
                    <div className="max-h-48 overflow-y-auto modal-input-enhanced rounded-lg p-3">
                      <div
                        className={cn(
                          "text-sm mb-2 font-medium",
                          theme === "dark" ? "text-gray-400" : "text-gray-600"
                        )}
                      >
                        Select from existing tables:
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                        {filteredTables.map((table) => (
                          <div
                            key={table}
                            className="flex items-center space-x-2"
                          >
                            <input
                              type="checkbox"
                              id={`existing-${table}`}
                              checked={selectedTables.includes(table)}
                              onChange={(e) =>
                                handleTableSelection(table, e.target.checked)
                              }
                              className="rounded border-slate-500 bg-slate-600 text-green-400 focus:ring-green-400"
                            />
                            <Label
                              htmlFor={`existing-${table}`}
                              className={cn(
                                "text-sm cursor-pointer flex-1 font-medium",
                                theme === "dark"
                                  ? "text-white"
                                  : "text-gray-800"
                              )}
                            >
                              {table}
                            </Label>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* Selected Tables Display */}
                {selectedTables.length > 0 && (
                  <div className="space-y-2">
                    <Label className="modal-label-enhanced text-sm">
                      Selected Tables ({selectedTables.length}):
                    </Label>
                    <div className="flex flex-wrap gap-2">
                      {selectedTables.map((table) => (
                        <Badge
                          key={table}
                          variant="secondary"
                          className="bg-green-600/20 text-green-400 border-green-500"
                        >
                          {table}
                          <button
                            onClick={() => handleRemoveTable(table)}
                            className="ml-2 hover:text-red-400 text-red-400"
                          >
                            ×
                          </button>
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>


              {/* Error Display */}
              {error && (
                <div className="modal-form-group">
                  <div className="flex items-center p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
                    <span className="text-red-400 text-sm">{error}</span>
                  </div>
                </div>
              )}

              {/* Submit Error Display */}
              {submitError && (
                <div className="modal-form-group">
                  <div className="flex items-center p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
                    <span className="text-red-400 text-sm">{submitError}</span>
                  </div>
                </div>
              )}

              {/* Submit Success Display */}
              {submitSuccess && (
                <div className="modal-form-group">
                  <div className="flex items-center p-3 bg-green-900/20 border border-green-500/30 rounded-lg">
                    <span className="text-green-400 text-sm">
                      Vector DB access {editingUser ? "updated" : "created"}{" "}
                      successfully!
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Action Buttons */}
            <div className="flex-shrink-0 px-6 py-6">
              <div className="flex justify-start gap-3">
                <Button
                  variant="outline"
                  onClick={handleClose}
                  disabled={isSubmitting}
                  className="modal-button-secondary"
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleSubmit}
                  disabled={
                    isSubmitting ||
                    !selectedUserId ||
                    !selectedDatabase ||
                    selectedTables.length === 0
                  }
                  className="modal-button-primary"
                >
                  {isSubmitting
                    ? editingUser
                      ? "Updating..."
                      : "Creating..."
                    : editingUser
                    ? "Update Vector Access"
                    : "Create Vector Access"}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
