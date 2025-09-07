"use client";
import React from "react";
import Image from "next/image";
import { Bell, User, LogOut, LogIn, Settings } from "lucide-react";
import { useUIStore } from "@/store/uiStore";
import { ThemeToggle } from "@/components/ui/ThemeToggle";
import { useAuthContext } from "@/components/providers/AuthContextProvider";
import { useTheme } from "@/store/theme-store";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";

export default function Navbar() {
  const { showSidebar, setShowSidebar, showAIAssistant, setShowAIAssistant } = useUIStore();
  const { isAuthenticated, user, logout } = useAuthContext();
  const theme = useTheme();

  const handleMenuClick = () => {
    setShowSidebar(!showSidebar);
  };

  return (
    <nav
      className={cn(
        "fixed top-6 left-4 right-4 z-50 backdrop-blur-xl rounded-full flex items-center justify-between shadow-2xl max-w-7xl mx-auto transition-all duration-300",
        theme === "dark" 
          ? "bg-white/5 border border-white/10 shadow-black/20" 
          : "bg-black/5 border border-black/10 shadow-black/10"
      )}
      style={{
        height: "64px",
        paddingLeft: "20px",
        paddingRight: "20px",
        background: theme === "dark"
          ? "linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05))"
          : "linear-gradient(135deg, rgba(0, 0, 0, 0.05), rgba(0, 0, 0, 0.02))",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        boxShadow: theme === "dark"
          ? "0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1)"
          : "0 8px 32px rgba(0, 0, 0, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.2)",
      }}
    >
      {/* Left side - Logo and Menu */}
      <div className="flex items-center gap-6">
        {/* ESAP Logo - Theme Aware */}
        <div className="flex items-center gap-2 px-4 py-2 rounded-full">
          <Image
            src={theme === "dark" ? "/logo/ESAP_W.png" : "/logo/ESAP_B_PNG.png"}
            alt="ESAP"
            width={120}
            height={40}
            className="h-8 w-auto"
          />
        </div>

        {/* Menu Button */}
        <div
          onClick={handleMenuClick}
          className={cn(
            "flex items-center gap-2 px-4 py-2 rounded-full border transition-all duration-300 cursor-pointer group",
            theme === "dark"
              ? "bg-emerald-500/15 border-emerald-500/25 hover:bg-emerald-500/20"
              : "bg-emerald-500/10 border-emerald-500/20 hover:bg-emerald-500/15"
          )}
        >
          <div className="w-4 h-4 flex flex-col items-center justify-center gap-0.5">
            {/* Three horizontal bars */}
            <div
              className={cn(
                "w-4 h-0.5 rounded-full transition-all duration-300",
                theme === "dark" ? "bg-emerald-400" : "bg-emerald-600",
                showSidebar ? "rotate-45 translate-y-1" : ""
              )}
            ></div>
            <div
              className={cn(
                "w-4 h-0.5 rounded-full transition-all duration-300",
                theme === "dark" ? "bg-emerald-400" : "bg-emerald-600",
                showSidebar ? "opacity-0" : ""
              )}
            ></div>
            <div
              className={cn(
                "w-4 h-0.5 rounded-full transition-all duration-300",
                theme === "dark" ? "bg-emerald-400" : "bg-emerald-600",
                showSidebar ? "-rotate-45 -translate-y-1" : ""
              )}
            ></div>
          </div>
          <span className={cn(
            "text-sm font-medium transition-colors",
            theme === "dark" 
              ? "text-emerald-400 group-hover:text-emerald-300" 
              : "text-emerald-600 group-hover:text-emerald-700"
          )}>
            {showSidebar ? "Close" : "Menu"}
          </span>
        </div>

        {/* Robot Icon */}
        <div
          onClick={() => setShowAIAssistant(!showAIAssistant)}
          className={cn(
            "flex items-center justify-center transition-colors cursor-pointer rounded-full",
            theme === "dark" 
              ? "hover:bg-gray-600/30" 
              : "hover:bg-gray-300/30"
          )}
          style={{
            width: "40px",
            height: "40px",
            opacity: 1,
            padding: "4.85px",
            gap: "3.03px",
          }}
        >
          <Image
            src="/autopilot.svg"
            alt="Robot"
            width={36}
            height={36}
            className="w-full h-auto cursor-pointer"
          />
        </div>
      </div>

      {/* Right side - Notifications, Theme Toggle and User */}
      <div className="flex items-center gap-4">
        {/* Notification Bell */}
        <div className="relative">
          <div className={cn(
            "w-10 h-10 rounded-full flex items-center justify-center border transition-colors cursor-pointer",
            theme === "dark"
              ? "bg-gray-700/50 border-gray-600/30 hover:bg-gray-600/50"
              : "bg-gray-200/50 border-gray-300/30 hover:bg-gray-300/50"
          )}>
            <Bell className={cn(
              "w-5 h-5",
              theme === "dark" ? "text-white/90" : "text-gray-700/90"
            )} />
          </div>
          <div className={cn(
            "absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full flex items-center justify-center border-2",
            theme === "dark" ? "border-black/50" : "border-white/50"
          )}>
            <span className="text-white text-xs font-bold">1</span>
          </div>
        </div>

        {/* Theme Toggle */}
        <ThemeToggle
          size="sm"
          className={cn(
            theme === "dark"
              ? "bg-gray-700/50 border-gray-600/30 hover:bg-gray-600/50"
              : "bg-gray-200/50 border-gray-300/30 hover:bg-gray-300/50"
          )}
        />

        {/* User Avatar */}
        {isAuthenticated && user ? (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <div className={cn(
                "w-10 h-10 rounded-full flex items-center justify-center border transition-colors cursor-pointer",
                theme === "dark"
                  ? "bg-gray-700/50 border-gray-600/30 hover:bg-gray-600/50"
                  : "bg-gray-200/50 border-gray-300/30 hover:bg-gray-300/50"
              )}>
                <User className={cn(
                  "w-5 h-5",
                  theme === "dark" ? "text-white/90" : "text-gray-700/90"
                )} />
              </div>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <div className="px-3 py-2">
                <p className="text-sm font-medium">{user.username}</p>
                <p className="text-xs text-gray-500">{user.email}</p>
              </div>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => window.location.href = '/user-configuration'}>
                <Settings className="w-4 h-4 mr-2" />
                Configuration
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => window.location.href = '/auth'}>
                <User className="w-4 h-4 mr-2" />
                Profile
              </DropdownMenuItem>
              <DropdownMenuItem onClick={logout}>
                <LogOut className="w-4 h-4 mr-2" />
                Sign Out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        ) : (
          <Button
            onClick={() => window.location.href = '/auth'}
            className={cn(
              "transition-colors",
              theme === "dark"
                ? "text-white/90 hover:text-white hover:bg-gray-600/50"
                : "text-gray-700/90 hover:text-gray-900 hover:bg-gray-300/50"
            )}
          >
            <LogIn className="w-4 h-4 mr-2" />
            Sign In
          </Button>
        )}
      </div>
    </nav>
  );
}
