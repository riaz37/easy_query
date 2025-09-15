"use client";

import { WorkflowStep } from "../types";

interface Step {
  id: string;
  title: string;
  number: number;
}

interface StepIndicatorProps {
  currentStep: string;
  steps: Step[];
}

const companySteps = [
  {
    id: "company-info" as WorkflowStep,
    title: "Company Info",
    number: 1,
  },
  {
    id: "database-config" as WorkflowStep,
    title: "Database",
    number: 2,
  },
  {
    id: "vector-config" as WorkflowStep,
    title: "Vector Config",
    number: 3,
  },
  {
    id: "final-creation" as WorkflowStep,
    title: "Create",
    number: 4,
  },
];

export function StepIndicator({ currentStep, steps = companySteps }: StepIndicatorProps) {
  const getCurrentStepIndex = () => {
    return steps.findIndex((step) => step.id === currentStep);
  };

  const currentIndex = getCurrentStepIndex();

  return (
    <div className="step-indicator-container">
      <div className="step-indicator-wrapper">
        {steps.map((step, index) => {
          const isCompleted = index < currentIndex;
          const isCurrent = index === currentIndex;
          const isUpcoming = index > currentIndex;

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
              {index < steps.length - 1 && (
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
