"use client";
import { Dashboard } from "@/components/dashboard";
import { OpeningAnimation } from "@/components/ui/opening-animation";
import { useEffect, useState } from "react";
import { useDatabaseOperations } from "@/lib/hooks";
import { useAuthContext } from "@/components/providers/AuthContextProvider";

export default function DashboardPage() {
  const [showOpeningAnimation, setShowOpeningAnimation] = useState(false);

  const databaseOps = useDatabaseOperations();
  const { user, isAuthenticated } = useAuthContext();

  // Initialize component
  useEffect(() => {
    const hasSeen =
      typeof window !== "undefined" &&
      localStorage.getItem("welcome-animation-shown");
    if (!hasSeen) {
      setShowOpeningAnimation(true);
    }
  }, [isAuthenticated, databaseOps]);

  const handleOpeningComplete = () => {
    setShowOpeningAnimation(false);
    if (typeof window !== "undefined") {
      localStorage.setItem("welcome-animation-shown", "true");
    }
  };

  return (
    <>
      {showOpeningAnimation ? (
        <OpeningAnimation duration={4000} onComplete={handleOpeningComplete}>
          <div />
        </OpeningAnimation>
      ) : (
        <main className="flex-1 animate-[fadeIn_0.5s_ease-out_forwards]">
          <Dashboard />
        </main>
      )}
    </>
  );
}
