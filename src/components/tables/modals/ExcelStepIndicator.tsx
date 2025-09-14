"use client";

import React from "react";
import { CheckCircle } from "lucide-react";
import { ExcelImportStep } from "./ExcelImportModal";

interface ExcelStepIndicatorProps {
  currentStep: ExcelImportStep;
  onStepChange: (step: ExcelImportStep) => void;
}

const steps = [
  {
    key: "upload-file" as ExcelImportStep,
    number: 1,
    label: "Upload File",
    completed: false,
  },
  {
    key: "select-destination" as ExcelImportStep,
    number: 2,
    label: "Select Destination",
    completed: false,
  },
  {
    key: "mapping" as ExcelImportStep,
    number: 3,
    label: "Mapping",
    completed: false,
  },
  {
    key: "confirm" as ExcelImportStep,
    number: 4,
    label: "Confirm",
    completed: false,
  },
];

export function ExcelStepIndicator({ currentStep, onStepChange }: ExcelStepIndicatorProps) {
  const currentStepIndex = steps.findIndex(step => step.key === currentStep);
  
  return (
    <div className="flex items-center justify-center">
      <div className="flex items-center space-x-4">
        {steps.map((step, index) => {
          const isActive = step.key === currentStep;
          const isCompleted = index < currentStepIndex;
          const isClickable = index <= currentStepIndex;
          
          return (
            <div key={step.key} className="flex items-center">
              <div className="flex items-center">
                {/* Step Circle */}
                <div
                  className={`flex items-center justify-center w-12 h-12 rounded-full border-2 transition-all duration-300 ${
                    isActive
                      ? "border-green-500 bg-green-500/20 text-green-400 shadow-lg shadow-green-500/25"
                      : isCompleted
                      ? "border-green-500 bg-green-500 text-white shadow-lg shadow-green-500/25"
                      : "border-slate-600 bg-slate-700/50 text-slate-500"
                  } ${isClickable ? "cursor-pointer hover:scale-105 hover:shadow-lg" : "cursor-not-allowed"}`}
                  onClick={() => isClickable && onStepChange(step.key)}
                >
                  {isCompleted ? (
                    <CheckCircle className="h-6 w-6" />
                  ) : (
                    <span className="text-lg font-semibold">{step.number}</span>
                  )}
                </div>
                
                {/* Step Label */}
                <span
                  className={`ml-3 text-sm font-medium transition-colors duration-300 ${
                    isActive
                      ? "text-green-400"
                      : isCompleted
                      ? "text-green-400"
                      : "text-slate-500"
                  }`}
                >
                  {step.label}
                </span>
              </div>
              
              {/* Connector Line */}
              {index < steps.length - 1 && (
                <div
                  className={`ml-6 w-8 h-0.5 transition-colors duration-300 ${
                    isCompleted || isActive ? "bg-green-500" : "bg-slate-600"
                  }`}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
