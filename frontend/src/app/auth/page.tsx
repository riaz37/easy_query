"use client";

import { AuthPage } from '@/components/auth';
import { useResolvedTheme } from '@/store/theme-store';
import Image from "next/image";

export default function AuthPageRoute() {
  const theme = useResolvedTheme();

  return (
    <main className="flex-1 animate-[fadeIn_0.5s_ease-out_forwards]">
      <div
        className="relative w-full h-screen overflow-hidden pt-[124px]"
        style={{
          background:
            theme === "dark"
              ? "linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f172a 100%)"
              : "linear-gradient(135deg, #ffffff 0%, #f0f9f5 30%, #e6f7ff 70%, #f0f9f5 100%)",
        }}
      >
        {/* Frame SVG Background */}
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

        {/* Enhanced Background Effects for Light Mode */}
        {theme === "light" && (
          <>
            {/* Subtle gradient overlays */}
            <div className="absolute inset-0 bg-gradient-to-br from-emerald-50/30 via-transparent to-blue-50/20 pointer-events-none" />
            <div className="absolute inset-0 bg-gradient-to-tl from-green-50/40 via-transparent to-emerald-50/30 pointer-events-none" />

            {/* Floating light particles */}
            <div className="absolute inset-0 pointer-events-none overflow-hidden">
              {[...Array(12)].map((_, i) => (
                <div
                  key={i}
                  className="absolute w-1 h-1 bg-emerald-400/20 rounded-full animate-pulse"
                  style={{
                    left: `${Math.random() * 100}%`,
                    top: `${Math.random() * 100}%`,
                    animation: `float ${
                      4 + Math.random() * 4
                    }s ease-in-out infinite`,
                    animationDelay: `${Math.random() * 2}s`,
                    boxShadow: "0 0 8px rgba(16, 185, 129, 0.3)",
                  }}
                />
              ))}
            </div>
          </>
        )}

        {/* Theme-aware Spotlight Effect */}
        <div className="absolute inset-0 z-[5] pointer-events-none overflow-visible">
          <div
            className="!opacity-100 animate-spotlight absolute -top-20 left-1/2 transform -translate-x-1/2 w-[140%] h-[140%]"
            style={{
              background: `radial-gradient(600px circle at 0px 0px, ${
                theme === "dark" ? "rgba(16, 185, 129, 0.15)" : "rgba(5, 150, 105, 0.1)"
              }, transparent 40%)`,
            }}
          />
        </div>

        {/* Auth Content */}
        <div className="relative z-10 flex items-center justify-center h-full px-4">
          <div className="w-full max-w-md">
            <AuthPage />
          </div>
        </div>
      </div>
    </main>
  );
} 