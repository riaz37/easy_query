"use client";

import { useState } from "react";
import { Database, Loader2, Plus, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { toast } from "sonner";
import { MSSQLConfigData } from "@/types/api";
import { DatabaseConfigStepProps } from "../types";
import { useAuthContext } from "@/components/providers/AuthContextProvider";

export function DatabaseConfigStep({
  selectedDbId,
  setSelectedDbId,
  databases,
  mssqlLoading,
  setConfig,
  setCurrentStep,
  setDatabaseCreationData,
  setCurrentTaskId,
}: DatabaseConfigStepProps) {
  const { user } = useAuthContext();
  const [selectedOption, setSelectedOption] = useState("existing");

  // New database form states
  const [newDbUrl, setNewDbUrl] = useState("");
  const [newDbName, setNewDbName] = useState("");
  const [newDbBusinessRule, setNewDbBusinessRule] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handlePrevious = () => {
    setCurrentStep("company-info");
  };

  const handleNext = () => {
    if (selectedDbId) {
      setCurrentStep("vector-config");
    } else {
      toast.error("Please select or create a database first");
    }
  };

  const handleCreateDatabase = async () => {
    if (!newDbUrl.trim() || !newDbName.trim()) {
      toast.error("Database URL and name are required");
      return;
    }

    if (!user?.user_id) {
      toast.error("User authentication required");
      return;
    }

    try {
      const dbConfig = {
        db_url: newDbUrl.trim(),
        db_name: newDbName.trim(),
        business_rule: newDbBusinessRule.trim() || undefined,
        user_id: user.user_id,
        file: selectedFile,
      };

      setDatabaseCreationData({ dbConfig, selectedFile });

      const taskResponse = await setConfig(dbConfig);

      if (!taskResponse) {
        toast.error("Failed to start database creation - no response received");
        return;
      }

      // Handle different response formats
      let taskId: string | null = null;

      if (typeof taskResponse === "object") {
        if ("task_id" in taskResponse) {
          taskId = taskResponse.task_id;
        } else if (
          "data" in taskResponse &&
          taskResponse.data &&
          "task_id" in taskResponse.data
        ) {
          taskId = taskResponse.data.task_id;
        }
      }

      if (!taskId) {
        console.error("Task response structure:", taskResponse);
        toast.error(
          "Failed to start database creation - invalid task ID in response"
        );
        return;
      }

      setCurrentTaskId(taskId);
      setCurrentStep("database-creation");
      toast.success("Database creation started successfully");
    } catch (error: any) {
      console.error("Database creation error:", error);
      const errorMessage =
        error?.message || error?.toString() || "Failed to create database";
      toast.error(`Database creation failed: ${errorMessage}`);
    }
  };

  const resetNewDbForm = () => {
    setNewDbName("");
    setNewDbUrl("");
    setNewDbBusinessRule("");
    setSelectedFile(null);
  };

  return (
    <div className="space-y-6">
        <div>
          <h3 className="text-lg font-medium text-green-400">
            Database Configuration
          </h3>
          <p className="text-sm text-gray-400 mt-1">
            Choose an existing database or create a new one
          </p>
        </div>

      <div className="space-y-6">
        {/* Radio Button Selection */}
        <div className="space-y-4">
          <Label className="text-sm font-medium text-gray-300">Select Databased</Label>
          <RadioGroup value={selectedOption} onValueChange={setSelectedOption} className="flex gap-6">
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="existing" id="existing" />
              <Label htmlFor="existing" className="text-sm text-gray-300 cursor-pointer">
                Existing
              </Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="new" id="new" />
              <Label htmlFor="new" className="text-sm text-gray-300 cursor-pointer">
                New
              </Label>
            </div>
          </RadioGroup>
      </div>

        {/* Existing Database Content */}
        {selectedOption === "existing" && (
          <div className="space-y-4">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="modal-label-enhanced">Select Database</Label>
            </div>
            {mssqlLoading ? (
              <div className="flex items-center gap-2 p-4 bg-gray-800/50 rounded-lg">
                <Loader2 className="w-4 h-4 animate-spin text-green-400" />
                <span className="text-gray-400">Loading databases...</span>
              </div>
            ) : databases.length > 0 ? (
              <Select
                value={selectedDbId?.toString() || ""}
                onValueChange={(value) => setSelectedDbId(parseInt(value))}
              >
                <SelectTrigger className="modal-select-enhanced w-full">
                  <SelectValue placeholder="Choose a database" />
                </SelectTrigger>
                <SelectContent className="modal-select-content-enhanced">
                  {databases.map((db) => (
                    <SelectItem key={db.db_id} value={db.db_id.toString()} className="dropdown-item">
                      <span>{db.db_name}</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              <div className="text-center py-8 text-gray-400">
                <Database className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>No databases found</p>
                <p className="text-sm">Create one to get started</p>
              </div>
            )}

            {databases.length === 0 && !mssqlLoading && (
              <div className="text-center py-8 text-gray-400">
                <Database className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>No databases found</p>
                <p className="text-sm">Create one to get started</p>
              </div>
            )}
          </div>
          </div>
        )}

        {/* New Database Content */}
        {selectedOption === "new" && (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="newDbName" className="modal-label-enhanced">
                Database Name *
              </Label>
              <Input
                id="newDbName"
                value={newDbName}
                onChange={(e) => setNewDbName(e.target.value)}
                placeholder="MyDatabase"
                className="modal-input-enhanced"
              />
          </div>

          <div className="space-y-2">
            <Label htmlFor="newDbUrl" className="modal-label-enhanced">
              Database URL *
            </Label>
            <Input
              id="newDbUrl"
              value={newDbUrl}
              onChange={(e) => setNewDbUrl(e.target.value)}
              placeholder="mssql+pyodbc://sa:password@server:1433/database..."
              className="modal-input-enhanced"
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label
                htmlFor="newDbBusinessRule"
                className="modal-label-enhanced"
              >
                Business Rules (Optional)
              </Label>
              <Textarea
                id="newDbBusinessRule"
                value={newDbBusinessRule}
                onChange={(e) => setNewDbBusinessRule(e.target.value)}
                placeholder="Enter business rules for this database"
                className="modal-textarea-enhanced min-h-[80px]"
              />
            </div>

            <div className="space-y-2">
              <Label className="modal-label-enhanced">
                Database File (Optional)
              </Label>
              <div className="space-y-2">
                <div className="text-sm text-gray-400">
                  Supported: .bak, .sql, .mdf, .ldf, .trn, .dmp, .dump
                </div>
                <div className="flex items-center gap-2">
                  <Input
                    type="file"
                    accept=".bak,.sql,.mdf,.ldf,.trn,.dmp,.dump"
                    onChange={(e) =>
                      setSelectedFile(e.target.files?.[0] || null)
                    }
                    className="modal-input-enhanced file:bg-[var(--primary-8,rgba(19,245,132,0.08))] file:border-[var(--primary-16,rgba(19,245,132,0.16))] file:text-white file:border file:rounded file:px-3 file:py-1"
                  />
                  {selectedFile && (
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => setSelectedFile(null)}
                      className="modal-button-secondary"
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  )}
                </div>
                {selectedFile && (
                  <div className="text-sm text-green-400">
                    Selected: {selectedFile.name} (
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-3">
            <Button
              type="button"
              onClick={handleCreateDatabase}
              disabled={!newDbUrl.trim() || !newDbName.trim() || !user?.user_id}
              className="modal-button-primary flex-1"
            >
              <Plus className="w-4 h-4 mr-2" />
              Create Database
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={resetNewDbForm}
              className="modal-button-secondary w-full sm:w-auto"
            >
              Reset
            </Button>
          </div>
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="modal-footer-enhanced">
        <Button
          variant="outline"
          onClick={handlePrevious}
          className="modal-button-secondary w-full sm:w-auto"
        >
          Back
        </Button>
        {selectedDbId && (
          <Button
            onClick={handleNext}
            className="modal-button-primary w-full sm:w-auto"
          >
            Next
          </Button>
        )}
      </div>
    </div>
  );
}
