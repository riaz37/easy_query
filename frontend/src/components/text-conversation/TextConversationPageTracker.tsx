"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { useTextConversation } from "@/components/providers/TextConversationContextProvider";

export function TextConversationPageTracker() {
  const pathname = usePathname();
  const { updateCurrentPage, isReady } = useTextConversation();
  const [currentTrackedPage, setCurrentTrackedPage] = useState<string>("");

  useEffect(() => {
    if (!isReady) return;

    // Convert pathname to page name
    let currentPage = "dashboard";

    if (pathname) {
      // Remove leading slash and convert to page name
      const path = pathname.replace(/^\//, "").toLowerCase();

      // Map common paths to page names (matching the service's pageMap)
      const pageMap: Record<string, string> = {
        "": "dashboard",
        dashboard: "dashboard",
        "file-query": "file-query",
        "database-query": "database-query",
        tables: "tables",
        users: "users",
        "ai-reports": "ai-reports",
        "company-structure": "company-structure",
        "voice-control": "voice-control",
        "user-configuration": "user-configuration",
      };

      currentPage = pageMap[path] || path || "dashboard";
    }

    // Only update if the page actually changed
    if (currentPage !== currentTrackedPage) {
      console.log("💬 Page tracker - updating current page:", currentPage);
      setCurrentTrackedPage(currentPage);
      updateCurrentPage(currentPage);
    }
  }, [pathname, isReady, updateCurrentPage, currentTrackedPage]);

  // This component doesn't render anything
  return null;
}
