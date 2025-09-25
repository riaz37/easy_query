import { create } from "zustand";
import { persist } from "zustand/middleware";
import { createSemanticThemeTransition } from "@/lib/hooks/useThemeTransition";

type ThemeMode = 'light' | 'dark' | 'system';

interface ThemeState {
  themeMode: ThemeMode;
  theme: 'light' | 'dark';
  mounted: boolean;
  
  // Actions
  setThemeMode: (themeMode: ThemeMode) => void;
  toggleTheme: (event?: React.MouseEvent) => void;
  setTheme: (theme: 'light' | 'dark') => void;
  setMounted: (mounted: boolean) => void;
  initializeTheme: () => void;
  
  // Internal helper methods
  updateTheme: () => void;
  updateDOM: () => void;
  setupSystemThemeListener: () => (() => void) | undefined;
}

export const useThemeStore = create<ThemeState>()(
  persist(
    (set, get) => ({
      themeMode: 'system',
      theme: 'dark',
      mounted: false,

      setThemeMode: (newThemeMode: ThemeMode) => {
        set({ themeMode: newThemeMode });
        get().updateTheme();
      },

      toggleTheme: (event?: React.MouseEvent) => {
        const { themeMode, theme } = get();
        const newThemeMode = themeMode === 'light' ? 'dark' : themeMode === 'dark' ? 'light' : (theme === 'dark' ? 'light' : 'dark');

        // Use semantic theme transition if event is provided
        if (event && typeof document !== 'undefined') {
          const isCurrentlyDark = theme === 'dark';

          createSemanticThemeTransition(
            event,
            () => get().setThemeMode(newThemeMode),
            isCurrentlyDark,
            1200
          );
        } else {
          get().setThemeMode(newThemeMode);
        }
      },

      setTheme: (theme: 'light' | 'dark') => {
        set({ theme });
        get().updateDOM();
      },

      setMounted: (mounted: boolean) => set({ mounted }),

      initializeTheme: () => {
        if (typeof window === 'undefined') return;
        
        set({ mounted: true });
        get().updateTheme();
        get().setupSystemThemeListener();
      },

      // Internal helper methods (not exposed in interface)
      updateTheme: () => {
        if (typeof window === 'undefined') return;
        
        const { themeMode } = get();
        let resolved: 'light' | 'dark';
        
        if (themeMode === 'system') {
          resolved = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        } else {
          resolved = themeMode;
        }
        
        get().setTheme(resolved);
      },

      updateDOM: () => {
        if (typeof document === 'undefined') return;
        
        const { theme } = get();
        const root = document.documentElement;
        
        // Add no-transition class temporarily to prevent flash
        document.body.classList.add('no-transition');
        
        root.classList.remove('light', 'dark');
        root.classList.add(theme);
        root.setAttribute('data-theme', theme);
        
        // Remove no-transition class after DOM update
        requestAnimationFrame(() => {
          document.body.classList.remove('no-transition');
        });
        
        // Update meta theme-color for mobile browsers
        const metaThemeColor = document.querySelector('meta[name="theme-color"]');
        if (metaThemeColor) {
          metaThemeColor.setAttribute('content', theme === 'dark' ? '#0f172a' : '#ffffff');
        }
      },

      setupSystemThemeListener: () => {
        if (typeof window === 'undefined') return;
        
        const { themeMode } = get();
        
        if (themeMode === 'system') {
          const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
          const handleChange = () => get().updateTheme();
          
          mediaQuery.addEventListener('change', handleChange);
          
          // Store cleanup function (you might want to call this on unmount)
          return () => mediaQuery.removeEventListener('change', handleChange);
        }
      },
    }),
    {
      name: 'easy-query-theme',
      partialize: (state) => ({ themeMode: state.themeMode }),
    }
  )
);

// Selector hooks for better performance
export const useTheme = () => useThemeStore((state) => state.theme);
export const useThemeMode = () => useThemeStore((state) => state.themeMode);
export const useThemeActions = () =>
  useThemeStore((state) => ({
    setThemeMode: state.setThemeMode,
    toggleTheme: state.toggleTheme,
    setTheme: state.setTheme,
    initializeTheme: state.initializeTheme,
  }));

// Combined hook for components that need both theme and actions
export const useThemeState = () =>
  useThemeStore((state) => ({
    themeMode: state.themeMode,
    theme: state.theme,
    mounted: state.mounted,
    setThemeMode: state.setThemeMode,
    toggleTheme: state.toggleTheme,
    setTheme: state.setTheme,
  }));

// Legacy hook for backward compatibility (will be removed)
export const useResolvedTheme = () => useThemeStore((state) => state.theme);