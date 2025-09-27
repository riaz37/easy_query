"use client";

import { Button } from "@/components/ui/button";
import { Loader2, ArrowLeft, CheckCircle, Database, Brain } from "lucide-react";
import { MSSQLConfigData, DatabaseConfigData } from "@/types/api";
import { WorkflowStep } from "../types";
import { Label } from "@/components/ui/label";

interface FinalCreationStepProps {
  companyName: string;
  description: string;
  address: string;
  contactEmail: string;
  selectedDbId: number | null;
  selectedUserConfigId: number | null;
  databases: MSSQLConfigData[];
  userConfigs: DatabaseConfigData[];
  creatingCompany: boolean;
  handleSubmit: (e: React.FormEvent) => Promise<void>;
  setCurrentStep: (step: WorkflowStep) => void;
  type: "parent" | "sub";
}

export function FinalCreationStep({
  companyName,
  description,
  address,
  contactEmail,
  selectedDbId,
  selectedUserConfigId,
  databases,
  userConfigs,
  creatingCompany,
  handleSubmit,
  setCurrentStep,
  type,
}: FinalCreationStepProps) {
  const selectedDatabase = databases.find((db) => db.db_id === selectedDbId);
  const selectedUserConfig = userConfigs.find(
    (config) => config.db_id === selectedUserConfigId
  );

  const handlePrevious = () => {
    if (selectedUserConfigId) {
      setCurrentStep("vector-config");
    } else {
      setCurrentStep("database-config");
    }
  };

  return (
    <div className="space-y-6 pb-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Company Information - Left Side */}
        <div className="rounded-lg p-4 space-y-3" style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}>
          <h4 className="text-md font-medium text-green-400">
            Company Information
          </h4>
          <div>
            <Label className="text-xs text-gray-400 uppercase tracking-wide">
              Company Name
            </Label>
            <p className="text-white font-medium">{companyName}</p>
          </div>
          {description && (
            <div>
              <Label className="text-xs text-gray-400 uppercase tracking-wide">
                Description
              </Label>
              <p className="text-gray-300">{description}</p>
            </div>
          )}
          {address && (
            <div>
              <Label className="text-xs text-gray-400 uppercase tracking-wide">
                Address
              </Label>
              <p className="text-gray-300">{address}</p>
            </div>
          )}
          {contactEmail && (
            <div>
              <Label className="text-xs text-gray-400 uppercase tracking-wide">
                Contact Email
              </Label>
              <p className="text-gray-300">{contactEmail}</p>
            </div>
          )}
        </div>

        {/* Right Side - Two Sections Stacked */}
        <div className="space-y-6">
          {/* Database Configuration */}
          <div className="rounded-lg p-4 space-y-3" style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}>
            <h4 className="text-md font-medium text-green-400">
              Database Configuration
            </h4>
            {selectedDatabase ? (
              <div className="space-y-2">
                <p className="text-white font-medium">
                  {selectedDatabase.db_name}
                </p>
              </div>
            ) : (
              <p className="text-sm text-gray-400">No database selected</p>
            )}
          </div>

          {/* Vector Configuration */}
          <div className="rounded-lg p-4 space-y-3" style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}>
            <h4 className="text-md font-medium text-green-400">
              Vector Configuration
            </h4>
            {selectedUserConfig ? (
              <div className="space-y-2">
                <p className="text-white font-medium">
                  {selectedUserConfig.db_config.DB_NAME || "Unnamed"}
                </p>
                <p className="text-sm text-gray-400">
                  Host: {selectedUserConfig.db_config.DB_HOST}:
                  {selectedUserConfig.db_config.DB_PORT}
                </p>
              </div>
            ) : (
              <p className="text-sm text-gray-400">Optional - Not configured</p>
            )}
          </div>
        </div>
      </div>

      {/* Bottom Section */}
      <div className="space-y-6 pt-6 border-t border-gray-700">
        {/* Validation Message */}
        {!selectedDbId && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-red-400 rounded-full"></div>
              <span className="text-sm font-medium text-red-400">
                Configuration Incomplete
              </span>
            </div>
            <p className="text-sm text-red-300 mt-1">
              Please complete the database configuration before creating the
              company.
            </p>
          </div>
        )}

        {/* Action Buttons */}
        <div className="modal-footer-enhanced">
          <Button
            variant="outline"
            onClick={handlePrevious}
            className="modal-button-secondary w-full sm:w-auto"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </Button>

          <Button
            onClick={handleSubmit}
            disabled={!selectedDbId || creatingCompany}
            className="modal-button-primary min-w-[140px] w-full sm:w-auto"
          >
            {creatingCompany ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Creating...
              </>
            ) : (
              <>Create Company</>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}
