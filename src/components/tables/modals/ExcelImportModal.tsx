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

export type ExcelImportStep = "upload-file" | "select-destination" | "mapping" | "confirm";

export function ExcelImportModal({
  open,
  onOpenChange,
  userId,
  availableTables,
  onViewTableData,
}: ExcelImportModalProps) {
  const [currentStep, setCurrentStep] = useState<ExcelImportStep>("upload-file");
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
      <DialogContent className="max-w-4xl max-h-[90vh] p-0 border-0 bg-transparent" showCloseButton={false}>
        <div className="modal-enhanced">
          <div className="modal-content-enhanced">
            <DialogHeader className="modal-header-enhanced">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <DialogTitle className="modal-title-enhanced flex items-center gap-3">
                    <div className="p-2 bg-green-500/20 rounded-lg">
                      <FileSpreadsheet className="h-6 w-6 text-green-400" />
                    </div>
                    Excel Import
                  </DialogTitle>
                  <p className="modal-description-enhanced">
                    Import data from Excel files to your database tables
                  </p>
                </div>
                <button
                  onClick={handleClose}
                  className="modal-close-button"
                >
                  <XIcon className="h-5 w-5" />
                </button>
              </div>
            </DialogHeader>

            <div className="modal-form-content overflow-y-auto max-h-[calc(90vh-200px)] px-6 pb-6">
              {/* Step Indicator */}
              <ExcelStepIndicator 
                currentStep={currentStep} 
                onStepChange={handleStepChange}
              />

              {/* Step Content */}
              <div className="mt-8">
                {renderStepContent()}
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
