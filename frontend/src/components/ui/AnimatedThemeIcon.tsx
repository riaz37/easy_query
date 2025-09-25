"use client";

import React from 'react';
import { useResolvedTheme } from '@/store/theme-store';

interface AnimatedThemeIconProps {
  width?: number;
  height?: number;
  className?: string;
}

export function AnimatedThemeIcon({ 
  width = 24, 
  height = 24, 
  className = "" 
}: AnimatedThemeIconProps) {
  const resolvedTheme = useResolvedTheme();
  const isDark = resolvedTheme === 'dark';

  return (
    <div className={`relative ${className}`} style={{ width, height }}>
      {/* Sun Icon */}
      <svg
        width={width}
        height={height}
        viewBox="0 0 24 24"
        fill="none"
        className={`absolute inset-0 transition-all duration-700 ease-[cubic-bezier(0.25,0.46,0.45,0.94)] ${
          isDark
            ? 'opacity-0 rotate-90 scale-50'
            : 'opacity-100 rotate-0 scale-100'
        }`}
      >
        <circle
          cx="12"
          cy="12"
          r="4"
          fill="currentColor"
          className="transition-all duration-700 ease-[cubic-bezier(0.25,0.46,0.45,0.94)]"
        />
        <path
          d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 6.34L4.93 4.93M19.07 19.07l-1.41-1.41"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          className="transition-all duration-700 ease-[cubic-bezier(0.25,0.46,0.45,0.94)]"
        />
      </svg>

      {/* Moon Icon */}
      <svg
        width={width}
        height={height}
        viewBox="0 0 24 24"
        fill="none"
        className={`absolute inset-0 transition-all duration-700 ease-[cubic-bezier(0.25,0.46,0.45,0.94)] ${
          isDark
            ? 'opacity-100 rotate-0 scale-100'
            : 'opacity-0 -rotate-90 scale-50'
        }`}
      >
        <path
          d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"
          fill="currentColor"
          className="transition-all duration-700 ease-[cubic-bezier(0.25,0.46,0.45,0.94)]"
        />
        <circle
          cx="17"
          cy="7"
          r="1"
          fill="currentColor"
          opacity="0.6"
          className="transition-all duration-700 ease-[cubic-bezier(0.25,0.46,0.45,0.94)]"
        />
        <circle
          cx="14"
          cy="5"
          r="0.5"
          fill="currentColor"
          opacity="0.4"
          className="transition-all duration-700 ease-[cubic-bezier(0.25,0.46,0.45,0.94)]"
        />
      </svg>
    </div>
  );
}

export default AnimatedThemeIcon;
