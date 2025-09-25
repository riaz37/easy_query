"use client";

import { cn } from "@/lib/utils";
import { LoadingProps } from "./types";
import { Spinner } from "./Spinner";

export function InlineLoader({ 
  size = "sm", 
  variant = "primary", 
  className,
  children
}: LoadingProps) {
  return (
    <div className={cn("inline-flex items-center gap-2", className)}>
      <Spinner size={size} variant={variant} />
      {children && (
        <span className="text-sm text-muted-foreground">
          {children}
        </span>
      )}
    </div>
  );
}
