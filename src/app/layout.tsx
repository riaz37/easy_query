"use client";
import { ThemeStoreProvider } from "@/components/ThemeStoreProvider";
import { ThemeTransitionProvider } from "@/components/ThemeTransitionProvider";
import { AuthContextProvider } from "@/components/providers/AuthContextProvider";
import { DatabaseContextProvider } from "@/components/providers/DatabaseContextProvider";
import { BusinessRulesContextProvider } from "@/components/providers/BusinessRulesContextProvider";
import { VoiceAgentProvider } from "@/components/providers/VoiceAgentContextProvider";
import { TextConversationProvider } from "@/components/providers/TextConversationContextProvider";
import { Toaster } from "@/components/ui/sonner";
import { cn } from "@/lib/utils";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";
import Menu from "@/components/Menu";
import {
  FloatingVoiceButton,
  VoiceNavigationHandler,
  CurrentPageIndicator,
} from "@/components/voice-agent";
import {
  FloatingTextButton,
  TextConversationPageTracker,
} from "@/components/text-conversation";
import { useUIStore } from "@/store/uiStore";

const inter = Inter({ subsets: ["latin"] });

function AppContent({ children }: { children: React.ReactNode }) {
  const { showSidebar, setShowSidebar } = useUIStore();

  return (
    <>
      <Navbar />

      {/* Menu Overlay */}
      {showSidebar && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-black/30 backdrop-blur-sm z-40"
            onClick={() => setShowSidebar(false)}
          />

          {/* Menu */}
          <Menu />
        </>
      )}

      <div className="min-h-screen w-full">
        <div className="flex-1 -mt-22 pt-22">{children}</div>
      </div>
    </>
  );
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={cn("min-h-screen bg-background font-sans antialiased")}>
        <ThemeStoreProvider>
          <ThemeTransitionProvider>
            <AuthContextProvider>
              <DatabaseContextProvider>
                <BusinessRulesContextProvider>
                  <VoiceAgentProvider>
                    <TextConversationProvider>
                      <VoiceNavigationHandler>
                        <AppContent>{children}</AppContent>
                      </VoiceNavigationHandler>
                      <FloatingVoiceButton />
                      <FloatingTextButton />
                      <TextConversationPageTracker />
                      <CurrentPageIndicator />
                      <Toaster />
                    </TextConversationProvider>
                  </VoiceAgentProvider>
                </BusinessRulesContextProvider>
              </DatabaseContextProvider>
            </AuthContextProvider>
          </ThemeTransitionProvider>
        </ThemeStoreProvider>
      </body>
    </html>
  );
}
