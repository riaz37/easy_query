"use client";

import React, { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { FileSpreadsheet, XIcon, ArrowRight, CheckCircle } from "lucide-react";
import { ExcelStepIndicator } from "./ExcelStepIndicator";
import { ExcelStep1UploadFile } from "./ExcelStep1UploadFile";
import { ExcelStep2SelectDestination } from "./ExcelStep2SelectDestination";
import { ExcelStep3Mapping } from "./ExcelStep3Mapping";
import { ExcelStep4Confirm } from "./ExcelStep4Confirm";

interface ExcelImportModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  userId: string;
  availableTables: any[];
  onViewTableData: (tableName: string) => void;
}

export type ExcelImportStep =
  | "upload-file"
  | "select-destination"
  | "mapping"
  | "confirm";

export function ExcelImportModal({
  open,
  onOpenChange,
  userId,
  availableTables,
  onViewTableData,
}: ExcelImportModalProps) {
  const [currentStep, setCurrentStep] =
    useState<ExcelImportStep>("upload-file");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedTable, setSelectedTable] = useState<string>("");
  const [mappingData, setMappingData] = useState<any>(null);

  const handleStepChange = (step: ExcelImportStep) => {
    setCurrentStep(step);
  };

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
  };

  const handleTableSelect = (table: string) => {
    setSelectedTable(table);
  };

  const handleMappingComplete = (data: any) => {
    setMappingData(data);
  };

  const handleReset = () => {
    setCurrentStep("upload-file");
    setSelectedFile(null);
    setSelectedTable("");
    setMappingData(null);
  };

  const handleClose = () => {
    handleReset();
    onOpenChange(false);
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case "upload-file":
        return (
          <ExcelStep1UploadFile
            onFileSelect={handleFileSelect}
            selectedFile={selectedFile}
            onNext={() => setCurrentStep("select-destination")}
          />
        );
      case "select-destination":
        return (
          <ExcelStep2SelectDestination
            userId={userId}
            availableTables={availableTables}
            selectedTable={selectedTable}
            onTableSelect={handleTableSelect}
            onNext={() => setCurrentStep("mapping")}
            onBack={() => setCurrentStep("upload-file")}
          />
        );
      case "mapping":
        return (
          <ExcelStep3Mapping
            selectedFile={selectedFile}
            selectedTable={selectedTable}
            userId={userId}
            onMappingComplete={handleMappingComplete}
            onNext={() => setCurrentStep("confirm")}
            onBack={() => setCurrentStep("select-destination")}
          />
        );
      case "confirm":
        return (
          <ExcelStep4Confirm
            selectedFile={selectedFile}
            selectedTable={selectedTable}
            mappingData={mappingData}
            userId={userId}
            onComplete={() => {
              handleReset();
              onViewTableData(selectedTable);
            }}
            onBack={() => setCurrentStep("mapping")}
          />
        );
      default:
        return null;
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="p-0 border-0 bg-transparent"
        showCloseButton={false}
        style={{
          width: "800px",
          maxWidth: "800px",
          maxHeight: "70vh",
        }}
      >
        <div className="modal-enhanced">
          <div className="modal-content-enhanced flex flex-col overflow-hidden">
            <DialogHeader className="modal-header-enhanced">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <DialogTitle className="modal-title-enhanced flex items-center gap-3">
                    Excel Import
                  </DialogTitle>
                  <p className="modal-description-enhanced">
                    Import data from Excel files to your database tables
                  </p>
                </div>
                <button onClick={handleClose} className="modal-close-button cursor-pointer">
                  <XIcon className="h-5 w-5" />
                </button>
              </div>
            </DialogHeader>

            <div className="modal-form-content flex-1 overflow-y-auto px-6 pb-6 min-h-0">
              {/* Step Indicator */}
              <div className="flex-shrink-0">
                <ExcelStepIndicator
                  currentStep={currentStep}
                  onStepChange={handleStepChange}
                />
              </div>

              {/* Step Content */}
              <div className="mt-8 flex-1 flex flex-col min-h-0 overflow-y-auto">
                {renderStepContent()}
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
