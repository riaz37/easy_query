"use client";

import React from "react";
import { StepIndicator } from "@/components/company-hierarchy/steps/StepIndicator";
import { ExcelImportStep } from "./ExcelImportModal";

interface ExcelStepIndicatorProps {
  currentStep: ExcelImportStep;
  onStepChange: (step: ExcelImportStep) => void;
}

const excelSteps = [
  {
    id: "upload-file",
    title: "Upload File",
    number: 1,
  },
  {
    id: "select-destination",
    title: "Select Destination",
    number: 2,
  },
  {
    id: "mapping",
    title: "Mapping",
    number: 3,
  },
  {
    id: "confirm",
    title: "Confirm",
    number: 4,
  },
];

export function ExcelStepIndicator({ currentStep, onStepChange }: ExcelStepIndicatorProps) {
  return (
    <div className="step-indicator-container">
      <div className="step-indicator-wrapper">
        {excelSteps.map((step, index) => {
          const isCompleted = index < excelSteps.findIndex(s => s.id === currentStep);
          const isCurrent = step.id === currentStep;
          const isUpcoming = index > excelSteps.findIndex(s => s.id === currentStep);

          return (
            <div key={step.id} className="step-item">
              {/* Step Circle */}
              <div className="step-circle-container">
                <div
                  className={`step-circle ${
                    isCompleted
                      ? "completed"
                      : isCurrent
                      ? "current"
                      : "upcoming"
                  }`}
                  onClick={() => onStepChange(step.id as ExcelImportStep)}
                  style={{ cursor: 'pointer' }}
                >
                  <span className="step-number">
                    {isCompleted ? "✓" : step.number}
                  </span>
                </div>
                <div className="step-title-container">
                  <div
                    className={`step-title ${
                      isCompleted
                        ? "completed"
                        : isCurrent
                        ? "current"
                        : "upcoming"
                    }`}
                  >
                    {step.title}
                  </div>
                </div>
              </div>

              {/* Connecting Line */}
              {index < excelSteps.length - 1 && (
                <div
                  className={`step-connector ${
                    isCompleted ? "completed" : "upcoming"
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
