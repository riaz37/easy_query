"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { useTheme } from "@/store/theme-store";
import Image from "next/image";

interface PageLayoutProps {
  children: React.ReactNode;
  className?: string;
  background?: "default" | "gradient" | "frame" | "gridframe" | "none" | ("frame" | "gridframe")[];
  container?: boolean;
  maxWidth?: "sm" | "md" | "lg" | "xl" | "2xl" | "4xl" | "6xl" | "7xl" | "full";
}

/**
 * Consistent page layout wrapper that provides proper spacing from the navbar
 * across all pages in the application.
 * 
 * Features:
 * - Consistent top padding to account for fixed navbar (112px total: 64px height + 24px margin + 24px clearance)
 * - Multiple background options: default, gradient, frame (with frame.svg), or none
 * - Frame background includes frame.svg with dark background overlay
 * - Responsive container with configurable max-width
 * - Proper bottom padding for content
 */
export function PageLayout({
  children,
  className,
  background = "default",
  container = true,
  maxWidth = "7xl",
}: PageLayoutProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  
  const backgroundClasses = {
    default: isDark ? "bg-background" : "bg-gray-50",
    gradient: isDark 
      ? "bg-gradient-to-br from-slate-900 via-slate-800 to-emerald-900" 
      : "bg-gradient-to-br from-gray-50 via-blue-50 to-emerald-50",
    frame: "", // Frame background handles its own styling
    gridframe: "", // Gridframe background handles its own styling
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
    <div className="pt-36 pb-8">
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

  // Handle frame and gridframe backgrounds (single or multiple)
  const hasFrame = background === "frame" || (Array.isArray(background) && background.includes("frame"));
  const hasGridframe = background === "gridframe" || (Array.isArray(background) && background.includes("gridframe"));
  
  if (hasFrame || hasGridframe) {
    return (
      <div 
        className={cn("w-full min-h-screen relative", className)}
        style={{
          background: "var(--BG-2-Primary-Color, rgba(0, 0, 0, 0.85))",
        }}
      >
        {/* Frame SVG Background */}
        {hasFrame && (
          <div className="absolute inset-0 pointer-events-none overflow-hidden">
            <Image
              src="/dashboard/frame.svg"
              alt="Background Frame"
              fill
              className="object-cover object-top"
              priority
              style={{
                opacity: 0.6,
                mixBlendMode: 'lighten',
              }}
            />
          </div>
        )}
        
        {/* Gridframe SVG Background */}
        {hasGridframe && (
          <div className="absolute inset-0 pointer-events-none overflow-hidden">
            <Image
              src="/gridframe.svg"
              alt="Background Gridframe"
              fill
              className="object-cover object-center"
              priority
              style={{
                opacity: 0.4,
                mixBlendMode: 'lighten',
              }}
            />
          </div>
        )}
        
        {/* Content */}
        <div className="relative z-10">
          {content}
        </div>
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
  enhancedTitle?: boolean;
}

export function PageHeader({
  title,
  description,
  icon,
  actions,
  className,
  enhancedTitle = false,
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
            <h1 className={cn(
              "text-3xl font-bold font-barlow",
              enhancedTitle 
                ? "modal-title-enhanced" 
                : isDark ? 'text-white' : 'text-gray-900'
            )}>
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
