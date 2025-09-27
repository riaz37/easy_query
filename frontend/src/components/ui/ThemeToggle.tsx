"use client";

import * as React from "react";
import { Moon, Sun } from "lucide-react";
import { useThemeState } from "@/store/theme-store";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface ThemeToggleProps {
  className?: string;
  size?: "sm" | "md" | "lg";
  variant?: "button" | "icon" | "dropdown";
}

export function ThemeToggle({
  className = "",
  size = "md",
  variant = "icon"
}: ThemeToggleProps) {
  const { themeMode, theme, toggleTheme, setThemeMode } = useThemeState();

  const handleToggle = (event: React.MouseEvent) => {
    const button = event.currentTarget as HTMLButtonElement;
    const switchingToDark = theme === 'light';

    // Add ZenUI-style animation classes for smooth transitions
    if (switchingToDark) {
      button.classList.add('switching-to-dark');
      setTimeout(() => button.classList.remove('switching-to-dark'), 1200);
    } else {
      button.classList.add('switching-to-light');
      setTimeout(() => button.classList.remove('switching-to-light'), 1200);
    }

    // Add smooth scaling effect
    button.style.transform = 'scale(0.95)';
    setTimeout(() => {
      button.style.transform = '';
    }, 150);

    toggleTheme(event);
  };

  const handleThemeChange = (newTheme: 'light' | 'dark' | 'system', event?: React.MouseEvent) => {
    if (event) {
      // Use smooth transition for dropdown selections too
      const switchingToDark = (newTheme === 'dark') || (newTheme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);

      if (switchingToDark && theme === 'light') {
        toggleTheme(event);
      } else if (!switchingToDark && theme === 'dark') {
        toggleTheme(event);
      } else {
        setThemeMode(newTheme);
      }
    } else {
      setThemeMode(newTheme);
    }
  };

  const sizeClasses = {
    sm: "w-8 h-8",
    md: "w-12 h-12",
    lg: "w-16 h-16"
  };

  const iconSizes = {
    sm: { width: 16, height: 16 },
    md: { width: 20, height: 20 },
    lg: { width: 24, height: 24 }
  };

  // New dropdown variant using the ModeToggle design
  if (variant === "dropdown") {
    return (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" size="icon" className={`theme-toggle-button cursor-pointer ${className}`}>
            <Sun className="h-[1.2rem] w-[1.2rem] scale-100 rotate-0 transition-all duration-700 ease-[cubic-bezier(0.25,0.46,0.45,0.94)] dark:scale-0 dark:-rotate-90" />
            <Moon className="absolute h-[1.2rem] w-[1.2rem] scale-0 rotate-90 transition-all duration-700 ease-[cubic-bezier(0.25,0.46,0.45,0.94)] dark:scale-100 dark:rotate-0" />
            <span className="sr-only">Toggle theme</span>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onClick={(e) => handleThemeChange("light", e as any)} className="cursor-pointer">
            <Sun className="mr-2 h-4 w-4" />
            Light
          </DropdownMenuItem>
          <DropdownMenuItem onClick={(e) => handleThemeChange("dark", e as any)} className="cursor-pointer">
            <Moon className="mr-2 h-4 w-4" />
            Dark
          </DropdownMenuItem>
          <DropdownMenuItem onClick={(e) => handleThemeChange("system", e as any)} className="cursor-pointer">
            <div className="mr-2 h-4 w-4 flex items-center justify-center">
              <div className="h-2 w-2 rounded-full bg-current" />
            </div>
            System
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    );
  }

  if (variant === "button") {
    return (
      <button
        onClick={handleToggle}
        className={`
          theme-toggle-button
          inline-flex items-center justify-center gap-2 px-4 py-2
          rounded-lg bg-white/50 dark:bg-[#232435]/50
          border border-gray-200 dark:border-white/12
          text-gray-800 dark:text-white
          hover:bg-white/70 dark:hover:bg-[#232435]/70
          transition-all duration-300 ease-in-out
          hover:scale-105 active:scale-95
          backdrop-blur-sm
          ${className}
        `}
        aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
      >
        <div className="relative">
          <Sun className="h-4 w-4 scale-100 rotate-0 transition-all duration-700 ease-[cubic-bezier(0.25,0.46,0.45,0.94)] dark:scale-0 dark:-rotate-90" />
          <Moon className="absolute inset-0 h-4 w-4 scale-0 rotate-90 transition-all duration-700 ease-[cubic-bezier(0.25,0.46,0.45,0.94)] dark:scale-100 dark:rotate-0" />
        </div>
        <span className="text-sm font-medium">
          {theme === 'dark' ? 'Light' : 'Dark'} Mode
        </span>
      </button>
    );
  }

  return (
    <button
      onClick={handleToggle}
      className={`
        theme-toggle-button
        ${sizeClasses[size]}
        rounded-full bg-white/50 dark:bg-[#232435]/50
        border border-gray-200 dark:border-white/12
        flex items-center justify-center cursor-pointer
        hover:bg-white/70 dark:hover:bg-[#232435]/70
        transition-all duration-300 ease-in-out
        hover:scale-110 active:scale-95
        hover:shadow-lg hover:shadow-gray-200/50 dark:hover:shadow-gray-800/50
        backdrop-blur-sm
        ${className}
      `}
      aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
      title={`Current: ${themeMode} mode (${theme})`}
    >
      <div className="relative transition-transform duration-300 ease-in-out hover:rotate-12">
        <Sun
          className={`scale-100 rotate-0 transition-all duration-700 ease-[cubic-bezier(0.25,0.46,0.45,0.94)] dark:scale-0 dark:-rotate-90 text-gray-600 dark:text-gray-300`}
          width={iconSizes[size].width}
          height={iconSizes[size].height}
        />
        <Moon
          className={`absolute inset-0 scale-0 rotate-90 transition-all duration-700 ease-[cubic-bezier(0.25,0.46,0.45,0.94)] dark:scale-100 dark:rotate-0 text-gray-600 dark:text-gray-300`}
          width={iconSizes[size].width}
          height={iconSizes[size].height}
        />
      </div>
    </button>
  );
}

// Modern dropdown-style theme toggle (equivalent to the original ModeToggle)
export function ModeToggle({ className = "" }: { className?: string }) {
  return <ThemeToggle variant="dropdown" className={className} />;
}

// Theme selector dropdown component for more advanced theme switching
export function ThemeSelector({ className = "" }: { className?: string }) {
  const { themeMode, setThemeMode } = useThemeState();

  return (
    <select
      value={themeMode}
      onChange={(e) => setThemeMode(e.target.value as 'light' | 'dark' | 'system')}
      className={`
        px-3 py-2 rounded-lg
        bg-white/50 dark:bg-[#232435]/50
        border border-gray-200 dark:border-white/12
        text-gray-800 dark:text-white
        text-sm font-medium
        cursor-pointer
        transition-all duration-200
        ${className}
      `}
      aria-label="Select theme"
    >
      <option value="light">Light</option>
      <option value="dark">Dark</option>
      <option value="system">System</option>
    </select>
  );
}

// Hook for components that need theme information
export function useThemeInfo() {
  const { themeMode, theme } = useThemeState();
  
  return {
    themeMode,
    theme,
    isDark: theme === 'dark',
    isLight: theme === 'light',
    isSystem: themeMode === 'system'
  };
}