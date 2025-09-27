"use client";

import React, { useState, useCallback, useEffect } from "react";
import { X, XIcon } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { useDatabaseContext } from "@/components/providers/DatabaseContextProvider";

import { MSSQLConfigData, DatabaseConfigData } from "@/types/api";
import { toast } from "sonner";

// Step components
import { StepIndicator } from "./steps/StepIndicator";
import { CompanyInfoStep } from "./steps/CompanyInfoStep";
import { DatabaseConfigStep } from "./steps/DatabaseConfigStep";
import { DatabaseCreationStep } from "./steps/DatabaseCreationStep";
import { VectorConfigStep } from "./steps/VectorConfigStep";
import { FinalCreationStep } from "./steps/FinalCreationStep";

// Types
import {
  CompanyFormData,
  DatabaseFormData,
  WorkflowStep,
  CompanyCreationModalProps,
} from "./types";
import { ServiceRegistry } from "@/lib/api/services/service-registry";

export function CompanyCreationModal({
  isOpen,
  onClose,
  onSubmit,
  type,
  parentCompanyId,
}: CompanyCreationModalProps) {
  // Remove dependency on context
  // const {
  //   availableDatabases,
  //   isLoading: userConfigLoading,
  // } = useDatabaseContext();

  // Workflow state
  const [currentStep, setCurrentStep] = useState<WorkflowStep>("company-info");
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);

  // Form states
  const [companyName, setCompanyName] = useState("");
  const [description, setDescription] = useState("");
  const [address, setAddress] = useState("");
  const [contactEmail, setContactEmail] = useState("");
  const [selectedDbId, setSelectedDbId] = useState<number | null>(null);
  const [selectedUserConfigId, setSelectedUserConfigId] = useState<
    number | null
  >(null);

  // Data states
  const [databases, setDatabases] = useState<MSSQLConfigData[]>([]);
  const [userConfigs, setUserConfigs] = useState<DatabaseConfigData[]>([]);
  const [creatingCompany, setCreatingCompany] = useState(false);
  const [databaseCreationData, setDatabaseCreationData] = useState<any>(null);

  // Loading states
  const [isLoadingDatabases, setIsLoadingDatabases] = useState(false);
  const [isLoadingUserConfigs, setIsLoadingUserConfigs] = useState(false);
  const [databaseError, setDatabaseError] = useState<string | null>(null);
  const [userConfigError, setUserConfigError] = useState<string | null>(null);

  // Add cache flags to prevent unnecessary reloading
  const [databasesLoaded, setDatabasesLoaded] = useState(false);
  const [userConfigsLoaded, setUserConfigsLoaded] = useState(false);

  const loadInitialData = useCallback(async () => {
    try {
      setIsLoadingDatabases(true);
      setIsLoadingUserConfigs(true);
      setDatabaseError(null);
      setUserConfigError(null);

      // Load both databases and user configs in parallel
      await Promise.all([loadDatabases(), loadUserConfigs()]);
    } catch (error) {
      console.error("Error loading initial data:", error);
      setDatabases([]);
      setUserConfigs([]);
      setDatabaseError("Failed to load databases");
    } finally {
      setIsLoadingDatabases(false);
      setIsLoadingUserConfigs(false);
    }
  }, []);

  const loadDatabases = useCallback(async () => {
    try {
      setIsLoadingDatabases(true);
      setDatabaseError(null);

      // Load databases using API call
      const databasesResponse =
        await ServiceRegistry.database.getAllDatabases();
      if (databasesResponse.success && databasesResponse.data) {
        // Transform DatabaseInfo to MSSQLConfigData format
        const mssqlDatabases: MSSQLConfigData[] = databasesResponse.data.map(
          (db) => ({
            db_id: db.id,
            db_name: db.name,
            db_url: db.url,
            business_rule: db.metadata?.businessRule || "",
            table_info: db.metadata?.tableInfo || {},
            db_schema: db.metadata?.dbSchema || {},
            dbpath: "",
            created_at: db.metadata?.createdAt || new Date().toISOString(),
            updated_at: db.lastUpdated || new Date().toISOString(),
          })
        );
        setDatabases(mssqlDatabases);
        setDatabasesLoaded(true);
      } else {
        setDatabases([]);
        setDatabaseError("Failed to load databases");
        setDatabasesLoaded(false);
      }
    } catch (error) {
      console.error("Error loading databases:", error);
      setDatabases([]);
      setDatabaseError("Failed to load databases");
      setDatabasesLoaded(false);
    } finally {
      setIsLoadingDatabases(false);
    }
  }, []);

  const loadUserConfigs = useCallback(async () => {
    try {
      setIsLoadingUserConfigs(true);
      setUserConfigError(null);

      // Load user configs using API call (vector databases)
      const userConfigsResponse =
        await ServiceRegistry.databaseConfig.getDatabaseConfigs();
      if (userConfigsResponse && userConfigsResponse.configs) {
        setUserConfigs(userConfigsResponse.configs);
        setUserConfigsLoaded(true);
      } else {
        setUserConfigs([]);
        setUserConfigsLoaded(false);
      }
    } catch (error) {
      console.error("Error loading vector database configs:", error);
      setUserConfigs([]);
      setUserConfigError("Failed to load vector database configurations");
      setUserConfigsLoaded(false);
    } finally {
      setIsLoadingUserConfigs(false);
    }
  }, []);

  const resetForm = () => {
    setCompanyName("");
    setDescription("");
    setAddress("");
    setContactEmail("");
    setSelectedDbId(null);
    setSelectedUserConfigId(null);
    setCurrentStep("company-info");
    setCurrentTaskId(null);
    setDatabaseCreationData(null);
  };

  const clearCache = useCallback(() => {
    setDatabasesLoaded(false);
    setUserConfigsLoaded(false);
  }, []);

  // Force refresh - clears cache and reloads data
  const forceRefreshUserConfigs = useCallback(async () => {
    setUserConfigsLoaded(false);
    await loadUserConfigs();
  }, [loadUserConfigs]);

  const forceRefreshDatabases = useCallback(async () => {
    setDatabasesLoaded(false);
    await loadDatabases();
  }, [loadDatabases]);

  const handleTaskComplete = async (success: boolean, result?: any) => {
    console.log("Task completed:", { success, result, databaseCreationData });
    
    if (success) {
      // Try to find and select the newly created database first
      let newDbId = null;

      if (result?.db_id) {
        newDbId = result.db_id;
        console.log("Found database ID from result.db_id:", newDbId);
      } else if (result?.database_id) {
        newDbId = result.database_id;
        console.log("Found database ID from result.database_id:", newDbId);
      }

      // If we found the ID, set it immediately
      if (newDbId) {
        console.log("Setting selected database ID:", newDbId);
        setSelectedDbId(newDbId);
      }

      // Reload databases to get the newly created one
      await loadInitialData();

      // If we didn't find the ID from result, try to find by name after reload
      if (!newDbId && databaseCreationData?.dbConfig?.db_name) {
        // Wait a bit for the databases to be loaded
        await new Promise(resolve => setTimeout(resolve, 500));
        
        const newDb = databases.find(
          (db) => db.db_name === databaseCreationData.dbConfig.db_name
        );
        console.log("Looking for database by name:", databaseCreationData.dbConfig.db_name);
        console.log("Available databases:", databases.map(db => ({ id: db.db_id, name: db.db_name })));
        
        if (newDb) {
          newDbId = newDb.db_id;
          console.log("Found database by name:", newDbId);
          setSelectedDbId(newDbId);
        }
      }

      if (!newDbId) {
        console.log("No database ID found, user will need to select manually");
      }

      // Move to vector config step
      setCurrentStep("vector-config");
      setDatabaseCreationData(null);
    } else {
      setCurrentStep("database-config");
    }

    setCurrentTaskId(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!companyName.trim()) {
      toast.error("Company name is required");
      return;
    }

    if (!selectedDbId) {
      toast.error("Please select or create a database");
      return;
    }

    setCreatingCompany(true);
    try {
      const companyData: CompanyFormData = {
        name: companyName.trim(),
        description: description.trim(),
        address: address.trim(),
        contactEmail: contactEmail.trim(),
        dbId: selectedDbId,
        parentCompanyId,
      };

      await onSubmit(companyData);
      handleClose();
      toast.success(
        `${type === "parent" ? "Parent" : "Sub"} company created successfully`
      );
    } catch (error) {
      toast.error("Failed to create company");
    } finally {
      setCreatingCompany(false);
    }
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  // Load data when modal opens
  useEffect(() => {
    if (isOpen) {
      // Only load data if not already loaded or if we need to refresh
      if (!databasesLoaded || !userConfigsLoaded) {
        loadInitialData();
      }
      resetForm();
    }
  }, [isOpen, databasesLoaded, userConfigsLoaded]);

  const stepProps = {
    currentStep,
    setCurrentStep,
    companyName,
    setCompanyName,
    description,
    setDescription,
    address,
    setAddress,
    contactEmail,
    setContactEmail,
    selectedDbId,
    setSelectedDbId,
    selectedUserConfigId,
    setSelectedUserConfigId,
    databases,
    userConfigs,
    mssqlLoading: isLoadingDatabases,
    userConfigLoading: isLoadingUserConfigs,
    onClose,
    setConfig: async (dbConfig: any) => {
      // Use MSSQLConfigService to create database
      const { MSSQLConfigService } = await import(
        "@/lib/api/services/mssql-config-service"
      );
      return await MSSQLConfigService.setMSSQLConfig(dbConfig);
    },
    loadDatabases: loadDatabases,
    setDatabaseCreationData,
    setCurrentTaskId,
    creatingCompany,
    handleSubmit,
    type,
    refreshUserConfigs: forceRefreshUserConfigs,
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent 
        className="p-0 border-0 bg-transparent" 
        showCloseButton={false}
        style={{
          width: '900px',
          maxWidth: '900px',
          maxHeight: '90vh',
        }}
      >
        <div className="modal-enhanced">
          <div className="modal-content-enhanced overflow-y-auto">
            {/* Header */}
            <DialogHeader className="modal-header-enhanced px-8 pt-6 pb-4">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <DialogTitle className="modal-title-enhanced text-xl">
                    Create {type === "parent" ? "Parent" : "Sub"} Company
                  </DialogTitle>
                  <DialogDescription className="modal-description-enhanced text-sm">
                    Set up your company with database and vector configurations
                  </DialogDescription>
                </div>
                <button
                  onClick={handleClose}
                  className="modal-close-button cursor-pointer"
                >
                  <XIcon className="h-5 w-5" />
                </button>
              </div>
            </DialogHeader>

            {/* Step Indicator */}
            <div className="px-8 pb-2">
              <StepIndicator currentStep={currentStep} />
            </div>

            {/* Content Area */}
            <div className="modal-form-content px-8 pb-6 flex-1 overflow-y-auto max-h-[calc(90vh-200px)]">
              {currentStep === "company-info" && (
                <CompanyInfoStep {...stepProps} />
              )}
              {currentStep === "database-config" && (
                <DatabaseConfigStep {...stepProps} />
              )}
              {currentStep === "database-creation" && (
                <DatabaseCreationStep
                  currentTaskId={currentTaskId}
                  onTaskComplete={handleTaskComplete}
                />
              )}
              {currentStep === "vector-config" && (
                <VectorConfigStep {...stepProps} />
              )}
              {currentStep === "final-creation" && (
                <FinalCreationStep
                  {...stepProps}
                  address={address}
                  contactEmail={contactEmail}
                  selectedUserConfigId={selectedUserConfigId}
                  userConfigs={userConfigs}
                />
              )}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export default CompanyCreationModal;
