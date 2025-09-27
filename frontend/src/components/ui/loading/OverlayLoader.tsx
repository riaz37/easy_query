"use client";

import { cn } from "@/lib/utils";
import { OverlayLoadingProps } from "./types";
import { EasyQueryBrandLoader } from "./EasyQueryBrandLoader";

export function OverlayLoader({ 
  size = "lg", 
  variant = "primary", 
  visible = true,
  backdrop = true,
  message = "Loading...",
  zIndex = 50,
  className 
}: OverlayLoadingProps) {
  if (!visible) return null;

  return (
    <div
      className={cn(
        "fixed inset-0 flex items-center justify-center",
        backdrop && "bg-black/50 backdrop-blur-sm",
        className
      )}
      style={{ zIndex }}
    >
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-8 max-w-sm mx-4">
        <div className="text-center space-y-4">
          {/* Loader */}
          <div className="flex justify-center">
            <EasyQueryBrandLoader size={size} />
          </div>
          
          {/* Message */}
          <p className="text-foreground font-medium">
            {message}
          </p>
        </div>
      </div>
    </div>
  );
}
