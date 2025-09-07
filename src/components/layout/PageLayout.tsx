import React from "react";
import { cn } from "@/lib/utils";
import { useTheme } from "@/store/theme-store";
import { EnhancedBackground } from "@/components/ui/enhanced-background";

interface PageLayoutProps {
  children: React.ReactNode;
  className?: string;
  background?: "default" | "gradient" | "enhanced" | "none";
  container?: boolean;
  maxWidth?: "sm" | "md" | "lg" | "xl" | "2xl" | "4xl" | "6xl" | "7xl" | "full";
  backgroundIntensity?: "low" | "medium" | "high";
}

/**
 * Consistent page layout wrapper that provides proper spacing from the navbar
 * across all pages in the application.
 * 
 * Features:
 * - Consistent top padding to account for fixed navbar (112px total: 64px height + 24px margin + 24px clearance)
 * - Multiple background options: default, gradient, enhanced (animated), or none
 * - Enhanced background includes floating particles, animated grids, and glowing effects
 * - Configurable background intensity (low, medium, high)
 * - Responsive container with configurable max-width
 * - Proper bottom padding for content
 */
export function PageLayout({
  children,
  className,
  background = "default",
  container = true,
  maxWidth = "7xl",
  backgroundIntensity = "medium",
}: PageLayoutProps) {
  const theme = useTheme();
  const isDark = theme === 'dark';
  
  const backgroundClasses = {
    default: isDark ? "bg-background" : "bg-gray-50",
    gradient: isDark 
      ? "bg-gradient-to-br from-slate-900 via-slate-800 to-emerald-900" 
      : "bg-gradient-to-br from-gray-50 via-blue-50 to-emerald-50",
    enhanced: "", // Enhanced background handles its own styling
    none: "",
  };

  const maxWidthClasses = {
    sm: "max-w-sm",
    md: "max-w-md", 
    lg: "max-w-lg",
    xl: "max-w-xl",
    "2xl": "max-w-2xl",
    "4xl": "max-w-4xl",
    "6xl": "max-w-6xl",
    "7xl": "max-w-7xl",
    full: "max-w-full",
  };

  const content = (
    <div className="pt-28 pb-8">
      {container ? (
        <div className="container mx-auto px-4">
          <div className={cn("mx-auto", maxWidthClasses[maxWidth])}>
            {children}
          </div>
        </div>
      ) : (
        <div className={cn("mx-auto px-4", maxWidthClasses[maxWidth])}>
          {children}
        </div>
      )}
    </div>
  );

  // If enhanced background is requested, wrap content with EnhancedBackground
  if (background === "enhanced") {
    return (
      <div 
        className={cn("w-full min-h-screen relative", className)}
        style={{
          background:
            isDark
              ? "linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f172a 100%)"
              : "linear-gradient(135deg, #ffffff 0%, #f0f9f5 30%, #e6f7ff 70%, #f0f9f5 100%)",
        }}
      >
        <EnhancedBackground intensity={backgroundIntensity}>
          {content}
        </EnhancedBackground>
      </div>
    );
  }

  // For other background types, use the standard layout
  return (
    <div className={cn("w-full min-h-screen relative", backgroundClasses[background], className)}>
      {content}
    </div>
  );
}

/**
 * Page header component for consistent page titles and descriptions
 */
interface PageHeaderProps {
  title: string;
  description?: string;
  icon?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}

export function PageHeader({
  title,
  description,
  icon,
  actions,
  className,
}: PageHeaderProps) {
  const theme = useTheme();
  const isDark = theme === 'dark';
  
  return (
    <div className={cn("mb-8", className)}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          {icon && (
            <div className={`w-12 h-12 rounded-xl flex items-center justify-center border ${
              isDark 
                ? 'bg-gradient-to-br from-blue-500/30 to-blue-600/20 border-blue-500/40' 
                : 'bg-gradient-to-br from-blue-100 to-blue-200 border-blue-300'
            }`}>
              {icon}
            </div>
          )}
          <div>
            <h1 className={`text-3xl font-bold font-barlow ${isDark ? 'text-white' : 'text-gray-900'}`}>
              {title}
            </h1>
            {description && (
              <p className={`font-public-sans ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                {description}
              </p>
            )}
          </div>
        </div>
        {actions && (
          <div className="flex items-center gap-3">
            {actions}
          </div>
        )}
      </div>
    </div>
  );
}
