"use client";

import { cn } from "@/lib/utils";
import { PageLoadingProps } from "./types";
import { EasyQueryBrandLoader } from "./EasyQueryBrandLoader";
import { ProgressLoader } from "./ProgressLoader";

export function PageLoader({ 
  size = "lg", 
  variant = "primary", 
  message = "Loading...",
  description,
  showProgress = false,
  progress = 0,
  className 
}: PageLoadingProps) {
  return (
    <div className={cn(
      "w-full flex flex-col items-center justify-center p-4",
      "min-h-[calc(100vh-112px)]", // Account for navbar height (64px + 24px margin + 24px clearance)
      className
    )}>
      <div className="text-center space-y-6 max-w-md">
        {/* Main loader */}
        <div className="flex justify-center">
          <EasyQueryBrandLoader size={size} />
        </div>
        
        {/* Message */}
        <div className="space-y-2">
          <h2 className="text-xl font-semibold text-foreground">
            {message}
          </h2>
          {description && (
            <p className="text-muted-foreground">
              {description}
            </p>
          )}
        </div>
        
        {/* Progress bar */}
        {showProgress && (
          <div className="w-full">
            <ProgressLoader 
              size="md"
              variant={variant}
              progress={progress}
              showPercentage={true}
            />
          </div>
        )}
      </div>
    </div>
  );
}
