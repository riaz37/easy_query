import { create } from "zustand";

interface UIState {
  showSidebar: boolean;
  setShowSidebar: (show: boolean) => void;
}

export const useUIStore = create<UIState>((set) => ({
  showSidebar: false,
  setShowSidebar: (show) => set({ showSidebar: show }),
}));
