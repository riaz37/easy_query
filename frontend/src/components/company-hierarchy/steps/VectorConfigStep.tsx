"use client";

import { useState } from "react";
import { Brain, Loader2, Plus, ArrowLeft, ArrowRight, X } from "lucide-react";
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
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { toast } from "sonner";
import { DatabaseConfigData } from "@/types/api";
import { DatabaseConfig } from "@/lib/api/services/database-config-service";
import { VectorConfigStepProps } from "../types";
import { useAuthContext } from "@/components/providers/AuthContextProvider";

export function VectorConfigStep({
  selectedUserConfigId,
  setSelectedUserConfigId,
  userConfigs,
  userConfigLoading,
  createDatabaseConfig,
  setCurrentStep,
  refreshUserConfigs,
}: VectorConfigStepProps) {
  const { user } = useAuthContext();
  const [selectedOption, setSelectedOption] = useState("existing");
  const [creatingConfig, setCreatingConfig] = useState(false);

  // New vector database config form states
  const [newConfigHost, setNewConfigHost] = useState("");
  const [newConfigPort, setNewConfigPort] = useState(5432);
  const [newConfigDatabase, setNewConfigDatabase] = useState("");
  const [newConfigUsername, setNewConfigUsername] = useState("");
  const [newConfigPassword, setNewConfigPassword] = useState("");

  const handlePrevious = () => {
    setCurrentStep("database-config");
  };

  const handleNext = () => {
    if (selectedUserConfigId) {
      setCurrentStep("final-creation");
    } else {
      toast.error(
        "Please select or create a vector database configuration first"
      );
    }
  };

  const handleCreateConfig = async () => {
    if (
      !newConfigHost.trim() ||
      !newConfigDatabase.trim() ||
      !newConfigUsername.trim()
    ) {
      toast.error("Host, database, and username are required");
      return;
    }

    if (!user?.user_id) {
      toast.error("User authentication required");
      return;
    }

    setCreatingConfig(true);
    try {
      const configRequest: DatabaseConfig = {
        DB_HOST: newConfigHost.trim(),
        DB_PORT: newConfigPort,
        DB_NAME: newConfigDatabase.trim(),
        DB_USER: newConfigUsername.trim(),
        DB_PASSWORD: newConfigPassword.trim(),
        schema: "public", // Default schema
        user_id: user.user_id, // Add user ID from auth context
      };

      const newConfig = await createDatabaseConfig(configRequest);

      if (newConfig) {
        // Reset form and switch to existing option
        resetForm();
        setSelectedOption("existing");

        // Refresh the configurations list to include the new one
        await refreshUserConfigs();

        // Set the newly created config as selected
        setSelectedUserConfigId(newConfig.db_id);

        toast.success(
          "Vector database configuration created and selected successfully"
        );
      } else {
        toast.error("Failed to create vector database configuration");
      }
    } catch (error) {
      toast.error("Failed to create vector database configuration");
    } finally {
      setCreatingConfig(false);
    }
  };

  const resetForm = () => {
    setNewConfigHost("");
    setNewConfigPort(5432);
    setNewConfigDatabase("");
    setNewConfigUsername("");
    setNewConfigPassword("");
  };

  return (
    <div className="space-y-6">
        <div>
          <h3 className="text-lg font-medium text-green-400">
            Vector Configuration
          </h3>
          <p className="text-sm text-gray-400 mt-1">
            Select existing or create new vector database configuration for AI
            operations
          </p>
        </div>

      <div className="space-y-6">
        {/* Radio Button Selection */}
        <div className="space-y-4">
          <Label className="text-sm font-medium text-gray-300">Select Vector Config</Label>
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
                Create New
              </Label>
            </div>
          </RadioGroup>
      </div>

        {/* Existing Configuration Content */}
        {selectedOption === "existing" && (
          <div className="space-y-4">
          <div className="space-y-3">
            <Label className="modal-label-enhanced">
              Select Vector Database Configuration
            </Label>
            {userConfigLoading ? (
              <div className="flex items-center gap-2 p-4 bg-gray-800/50 rounded-lg">
                <Loader2 className="w-4 h-4 animate-spin text-green-400" />
                <span className="text-gray-400">Loading configurations...</span>
              </div>
            ) : (
              <Select
                value={selectedUserConfigId?.toString() || ""}
                onValueChange={(value) =>
                  setSelectedUserConfigId(parseInt(value))
                }
              >
                <SelectTrigger className="modal-select-enhanced w-full">
                  <SelectValue placeholder="Choose a vector database configuration" />
                </SelectTrigger>
                <SelectContent className="modal-select-content-enhanced">
                  {userConfigs.map((config) => (
                    <SelectItem
                      key={config.db_id}
                      value={config.db_id.toString()}
                      className="dropdown-item"
                    >
                      <span>
                        {config.db_config.DB_NAME || "Unnamed"} -{" "}
                        {config.db_config.DB_HOST}:{config.db_config.DB_PORT}
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}

            {userConfigs.length === 0 && !userConfigLoading && (
              <div className="text-center py-8 text-gray-400">
                <Brain className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>No vector database configurations found</p>
                <p className="text-sm">Create one to get started</p>
              </div>
            )}
          </div>
          </div>
        )}

        {/* New Configuration Content */}
        {selectedOption === "new" && (
          <div className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label htmlFor="newConfigHost" className="modal-label-enhanced">
                Host <span className="text-red-500">*</span>
              </Label>
              <Input
                id="newConfigHost"
                value={newConfigHost}
                onChange={(e) => setNewConfigHost(e.target.value)}
                placeholder="localhost"
                className="modal-input-enhanced"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="newConfigPort" className="modal-label-enhanced">
                Port <span className="text-red-500">*</span>
              </Label>
              <Input
                id="newConfigPort"
                type="number"
                value={newConfigPort}
                onChange={(e) =>
                  setNewConfigPort(parseInt(e.target.value) || 5432)
                }
                placeholder="5432"
                className="modal-input-enhanced"
              />
            </div>

            <div className="space-y-2">
              <Label
                htmlFor="newConfigDatabase"
                className="modal-label-enhanced"
              >
                Database Name <span className="text-red-500">*</span>
              </Label>
              <Input
                id="newConfigDatabase"
                value={newConfigDatabase}
                onChange={(e) => setNewConfigDatabase(e.target.value)}
                placeholder="vectordb"
                className="modal-input-enhanced"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label
                htmlFor="newConfigUsername"
                className="modal-label-enhanced"
              >
                Username *
              </Label>
              <Input
                id="newConfigUsername"
                value={newConfigUsername}
                onChange={(e) => setNewConfigUsername(e.target.value)}
                placeholder="admin"
                className="modal-input-enhanced"
              />
            </div>

            <div className="space-y-2">
              <Label
                htmlFor="newConfigPassword"
                className="modal-label-enhanced"
              >
                Password *
              </Label>
              <Input
                id="newConfigPassword"
                type="password"
                value={newConfigPassword}
                onChange={(e) => setNewConfigPassword(e.target.value)}
                placeholder="Enter password"
                className="modal-input-enhanced"
              />
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-3">
            <Button
              type="button"
              onClick={handleCreateConfig}
              disabled={
                !newConfigHost.trim() ||
                !newConfigDatabase.trim() ||
                !newConfigUsername.trim() ||
                !user?.user_id ||
                creatingConfig
              }
              className="modal-button-primary flex-1"
            >
              {creatingConfig ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Creating Configuration...
                </>
              ) : (
                <>
                  <Plus className="w-4 h-4 mr-2" />
                  Create Vector Database Configuration
                </>
              )}
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={resetForm}
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
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back
        </Button>
        {selectedUserConfigId && (
          <Button
            onClick={handleNext}
            className="modal-button-primary w-full sm:w-auto"
          >
            Next
            <ArrowRight className="w-4 h-4 ml-2" />
          </Button>
        )}
      </div>

    </div>
  );
}
