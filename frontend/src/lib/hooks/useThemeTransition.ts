"use client";

import { useEffect, useCallback } from "react";

/**
 * Hook to handle smooth theme transitions and prevent flash of unstyled content
 */
export function useThemeTransition() {
  useEffect(() => {
    // Remove no-transition class after initial load to enable smooth transitions
    const timer = setTimeout(() => {
      document.body.classList.remove("no-transition");
    }, 100);

    // Add no-transition class on page load to prevent flash
    document.body.classList.add("no-transition");

    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    // Add support for reduced motion preference
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");

    const handleMotionChange = (e: MediaQueryListEvent) => {
      if (e.matches) {
        document.documentElement.style.setProperty(
          "--transition-duration",
          "0s"
        );
        document.documentElement.classList.add("reduced-motion");
      } else {
        document.documentElement.style.removeProperty("--transition-duration");
        document.documentElement.classList.remove("reduced-motion");
      }
    };

    // Initial check
    handleMotionChange(mediaQuery as any);
    mediaQuery.addEventListener("change", handleMotionChange);

    return () => mediaQuery.removeEventListener("change", handleMotionChange);
  }, []);
}

/**
 * Debounce utility function to prevent rapid theme changes
 */
function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

/**
 * Smart theme transition with expanding circle effect for both directions
 * Light → Dark: New dark theme expands outward
 * Dark → Light: New light theme expands outward
 * Smooth and comfortable transition timing
 */
export function createSemanticThemeTransition(
  event: React.MouseEvent,
  callback: () => void,
  isCurrentlyDark: boolean, // Kept for backward compatibility, not used in expanding-only mode
  duration: number = 1200 // Slower, more comfortable transition
): Promise<void> {
  return new Promise((resolve) => {
    // Check for reduced motion preference
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      callback();
      resolve();
      return;
    }

    // Fallback for browsers that don't support View Transitions
    if (!("startViewTransition" in document)) {
      callback();
      resolve();
      return;
    }

    const rect = event.currentTarget.getBoundingClientRect();
    const x = rect.left + rect.width / 2;
    const y = rect.top + rect.height / 2;

    // Calculate the maximum distance to any corner for full coverage
    const endRadius =
      Math.hypot(
        Math.max(x, window.innerWidth - x),
        Math.max(y, window.innerHeight - y)
      ) * 1.2; // Add 20% buffer for smooth edge coverage

    const transition = (document as any).startViewTransition(callback);

    transition.ready
      .then(() => {
        // Both transitions use expanding circle effect
        // Always animate the new theme expanding from the click point
        const animation = document.documentElement.animate(
          {
            clipPath: [
              `circle(0px at ${x}px ${y}px)`,
              `circle(${endRadius}px at ${x}px ${y}px)`,
            ],
          },
          {
            duration,
            easing: "cubic-bezier(0.25, 0.46, 0.45, 0.94)", // ZenUI-like smooth easing
            pseudoElement: "::view-transition-new(root)", // Always animate the new theme
          }
        );

        animation.addEventListener("finish", () => resolve());
      })
      .catch(() => {
        // Fallback if transition fails
        callback();
        resolve();
      });
  });
}

/**
 * Legacy circular transition for expanding effect
 * @deprecated Use createSemanticThemeTransition instead
 */
export function createCircularTransition(
  event: React.MouseEvent,
  callback: () => void,
  duration: number = 1200
): Promise<void> {
  return createSemanticThemeTransition(event, callback, false, duration);
}

/**
 * Legacy contracting transition
 * @deprecated Use createSemanticThemeTransition instead
 */
export function createContractingTransition(
  event: React.MouseEvent,
  callback: () => void,
  duration: number = 1200
): Promise<void> {
  return createSemanticThemeTransition(event, callback, true, duration);
}

/**
 * Advanced theme transition hook with debouncing and performance optimizations
 */
export function useAdvancedThemeTransition() {
  const debouncedTransition = useCallback(
    debounce(
      (
        event: React.MouseEvent,
        callback: () => void,
        isCurrentlyDark: boolean,
        duration?: number
      ) => {
        createSemanticThemeTransition(
          event,
          callback,
          isCurrentlyDark,
          duration
        );
      },
      150
    ),
    []
  );

  // Initialize theme transition effects
  useThemeTransition();

  return {
    transitionTheme: createSemanticThemeTransition,
    debouncedTransition,
  };
}

/**
 * Utility function to detect if browser supports View Transitions
 */
export function supportsViewTransitions(): boolean {
  return "startViewTransition" in document;
}

/**
 * Utility function to check if user prefers reduced motion
 */
export function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Get optimal transition duration based on user preferences
 */
export function getOptimalTransitionDuration(
  defaultDuration: number = 1200
): number {
  if (prefersReducedMotion()) {
    return 0;
  }
  return defaultDuration;
}

/**
 * Theme transition configuration options
 */
export interface ThemeTransitionConfig {
  duration?: number;
  easing?: string;
  respectReducedMotion?: boolean;
  fallbackTransition?: boolean;
}

/**
 * Configurable theme transition function
 */
export function createConfigurableThemeTransition(
  event: React.MouseEvent,
  callback: () => void,
  isCurrentlyDark: boolean, // Kept for backward compatibility, not used in expanding-only mode
  config: ThemeTransitionConfig = {}
): Promise<void> {
  const {
    duration = 1200, // Updated to match the slower transition
    easing = "cubic-bezier(0.25, 0.46, 0.45, 0.94)",
    respectReducedMotion = true,
    fallbackTransition = true,
  } = config;

  return new Promise((resolve) => {
    // Check for reduced motion preference
    if (respectReducedMotion && prefersReducedMotion()) {
      callback();
      resolve();
      return;
    }

    // Fallback for browsers that don't support View Transitions
    if (!supportsViewTransitions()) {
      if (fallbackTransition) {
        // Simple CSS transition fallback
        document.body.style.transition = `all ${duration}ms ${easing}`;
        callback();
        setTimeout(() => {
          document.body.style.transition = "";
          resolve();
        }, duration);
      } else {
        callback();
        resolve();
      }
      return;
    }

    const rect = event.currentTarget.getBoundingClientRect();
    const x = rect.left + rect.width / 2;
    const y = rect.top + rect.height / 2;

    const endRadius =
      Math.hypot(
        Math.max(x, window.innerWidth - x),
        Math.max(y, window.innerHeight - y)
      ) * 1.2;

    const transition = (document as any).startViewTransition(callback);

    transition.ready
      .then(() => {
        // Both transitions use expanding circle effect
        // Always animate the new theme expanding from the click point
        const animation = document.documentElement.animate(
          {
            clipPath: [
              `circle(0px at ${x}px ${y}px)`,
              `circle(${endRadius}px at ${x}px ${y}px)`,
            ],
          },
          {
            duration,
            easing,
            pseudoElement: "::view-transition-new(root)", // Always animate the new theme
          }
        );

        animation.addEventListener("finish", () => resolve());
      })
      .catch(() => {
        callback();
        resolve();
      });
  });
}
