import { create } from "zustand";
import { FileUploadState } from "@/types";
import { FileService } from "../lib/api";

export const useFileUploadStore = create<FileUploadState>((set, get) => ({
  uploadedFiles: [],
  fileMetas: [],
  processing: false,
  status: null,
  bundleId: null,
  bundleStatus: null,
  initialResponse: null,
  error: null,
  polling: false,
  pollingRef: null,
  setFiles: (files, metas) => set({ uploadedFiles: files, fileMetas: metas }),
  setProcessing: (processing) => set({ processing }),
  setStatus: (status) => set({ status }),
  setBundleId: (bundleId) => set({ bundleId }),
  setBundleStatus: (bundleStatus) => set({ bundleStatus }),
  setInitialResponse: (initialResponse) => set({ initialResponse }),
  setError: (error) => set({ error }),
  reset: () => {
    const { stopPolling } = get();
    stopPolling();
    set({
      uploadedFiles: [],
      fileMetas: [],
      processing: false,
      status: null,
      bundleId: null,
      bundleStatus: null,
      initialResponse: null,
      error: null,
      polling: false,
      pollingRef: null,
    });
  },
  startPolling: (bundleId: string) => {
    const { stopPolling, setBundleStatus, setStatus, setProcessing, setError } =
      get();
    stopPolling();
    set({ polling: true });
    const ref = setInterval(async () => {
      try {
        const response = await FileService.getBundleTaskStatus(bundleId);

        console.log("Polling - Full response:", response);
        console.log("Polling - Response type:", typeof response);

        // Check if response exists (with API client interceptor, response is now the data directly)
        if (!response) {
          console.warn("No response from bundle status API", {
            response,
          });
          return;
        }

        const data = response;

        // Add safety checks for data structure
        if (!data || typeof data !== "object") {
          console.warn(
            "Invalid bundle status data:",
            data,
            "Type:",
            typeof data,
          );
          return;
        }

        // Handle error responses from the API (bundle not found, etc.)
        if (
          data.detail &&
          (data.detail.includes("not found") ||
            data.detail.includes("No response from server"))
        ) {
          console.warn("Bundle not ready yet, continuing to poll...");
          return;
        }

        // If this is an error response but not a "not found" error, handle it
        if (data.error && !response.success) {
          console.warn("Bundle status error:", data.detail);
          return;
        }

        setBundleStatus(data);

        // Normalize status with safety checks
        let normalized = "pending";
        const statusValue = data.status || data.state || "pending";

        if (
          statusValue === "PROCESSING" ||
          statusValue === "RUNNING" ||
          statusValue === "processing" ||
          statusValue === "running"
        )
          normalized = "running";
        if (
          statusValue === "COMPLETED" ||
          statusValue === "completed" ||
          statusValue === "FINISHED" ||
          statusValue === "finished"
        )
          normalized = "completed";
        if (
          statusValue === "FAILED" ||
          statusValue === "failed" ||
          statusValue === "ERROR" ||
          statusValue === "error"
        )
          normalized = "completed"; // Treat failed as completed to stop polling

        setStatus(normalized as "pending" | "running" | "completed");

        console.log(
          "Status normalized to:",
          normalized,
          "from original:",
          statusValue,
        );

        if (normalized === "completed") {
          console.log("File processing completed! Stopping polling...");
          setProcessing(false);
          get().stopPolling();
        } else {
          console.log("Still processing, continuing to poll...");
        }
      } catch (e: any) {
        console.error("Polling error:", e);
        // Don't immediately stop on errors, continue polling for a bit
        // Only stop if it's a persistent error
        const currentState = get();
        if (!currentState.error) {
          setError(e.message || "Unknown error");
          setProcessing(false);
          get().stopPolling();
        }
      }
    }, 3000);
    set({ pollingRef: ref });
  },
  stopPolling: () => {
    const { pollingRef } = get();
    console.log("stopPolling called, pollingRef:", pollingRef);
    if (pollingRef) {
      clearInterval(pollingRef);
      console.log("Polling interval cleared");
    }
    set({ polling: false, pollingRef: null });
    console.log("Polling state reset");
  },
}));
