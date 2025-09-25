"use client";

import { cn } from "@/lib/utils";
import { ButtonLoadingProps } from "./types";
import { Spinner } from "./Spinner";

const sizeClasses = {
  xs: "px-2 py-1 text-xs",
  sm: "px-3 py-1.5 text-sm",
  md: "px-4 py-2 text-sm",
  lg: "px-6 py-3 text-base",
  xl: "px-8 py-4 text-lg"
};

const spinnerSizes = {
  xs: "xs" as const,
  sm: "sm" as const,
  md: "sm" as const,
  lg: "md" as const,
  xl: "lg" as const
};

export function ButtonLoader({ 
  size = "md", 
  variant = "primary", 
  loading = false,
  disabled = false,
  text = "Loading...",
  className,
  children,
  ...props
}: ButtonLoadingProps) {
  const isDisabled = disabled || loading;
  
  return (
    <button
      {...props}
      className={cn(
        "inline-flex items-center justify-center gap-2 font-medium rounded-md transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none",
        sizeClasses[size],
        variant === "primary" && "bg-[#10b981] text-white hover:bg-[#047857]",
        variant === "primary-dark" && "bg-[#047857] text-white hover:bg-[#065f46]",
        variant === "primary-light" && "bg-[#34d399] text-white hover:bg-[#10b981]",
        variant === "accent-blue" && "bg-[#3b82f6] text-white hover:bg-[#2563eb]",
        variant === "accent-purple" && "bg-[#8b5cf6] text-white hover:bg-[#7c3aed]",
        variant === "accent-orange" && "bg-[#f59e0b] text-white hover:bg-[#d97706]",
        variant === "accent-green" && "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white hover:from-emerald-600 hover:to-emerald-700 shadow-lg shadow-emerald-500/25 border border-emerald-400/30",
        variant === "secondary" && "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        variant === "success" && "bg-green-600 text-white hover:bg-green-700",
        variant === "warning" && "bg-yellow-600 text-white hover:bg-yellow-700",
        variant === "error" && "bg-red-600 text-white hover:bg-red-700",
        variant === "info" && "bg-blue-600 text-white hover:bg-blue-700",
        className
      )}
      disabled={isDisabled}
    >
      {loading && (
        <Spinner 
          size={spinnerSizes[size]}
          variant={variant}
        />
      )}
      {loading ? text : children}
    </button>
  );
}
