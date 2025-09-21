"use client";
import React from "react";
import Image from "next/image";
import { Bell, User, LogIn, Settings } from "lucide-react";
import { useUIStore } from "@/store/uiStore";
import { useAuthContext } from "@/components/providers/AuthContextProvider";
import { useTheme } from "@/store/theme-store";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { NavbarTaskIndicator } from "@/components/task-manager";

export default function Navbar() {
  const { showSidebar, setShowSidebar } = useUIStore();
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
        background:
          theme === "dark"
            ? "linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05))"
            : "linear-gradient(135deg, rgba(0, 0, 0, 0.05), rgba(0, 0, 0, 0.02))",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        boxShadow:
          theme === "dark"
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
          className="flex items-center gap-2 px-4 py-2 rounded-full transition-all duration-300 cursor-pointer group"
          style={{
            background: "rgba(255, 255, 255, 0.08)",
          }}
        >
          <div className="w-4 h-4 flex flex-col items-center justify-center gap-0.5">
            {/* Three horizontal bars */}
            <div
              className={cn(
                "w-4 h-0.5 rounded-full transition-all duration-300 bg-white",
                showSidebar ? "rotate-45 translate-y-1" : ""
              )}
            ></div>
            <div
              className={cn(
                "w-4 h-0.5 rounded-full transition-all duration-300 bg-white",
                showSidebar ? "opacity-0" : ""
              )}
            ></div>
            <div
              className={cn(
                "w-4 h-0.5 rounded-full transition-all duration-300 bg-white",
                showSidebar ? "-rotate-45 -translate-y-1" : ""
              )}
            ></div>
          </div>
          <span className="text-sm font-medium text-white transition-colors group-hover:text-white/90">
            {showSidebar ? "Close" : "Menu"}
          </span>
        </div>
      </div>

      {/* Right side - Notifications, Task Indicator, Theme Toggle and User */}
      <div className="flex items-center gap-4">
        {/* Notification Bell */}
        <div className="relative">
          <div
            className={cn(
              "w-10 h-10 rounded-full flex items-center justify-center border transition-colors cursor-pointer",
              theme === "dark"
                ? "bg-gray-700/50 border-gray-600/30 hover:bg-gray-600/50"
                : "bg-gray-200/50 border-gray-300/30 hover:bg-gray-300/50"
            )}
          >
            <Bell
              className={cn(
                "w-5 h-5",
                theme === "dark" ? "text-white/90" : "text-gray-700/90"
              )}
            />
          </div>
          <div
            className={cn(
              "absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full flex items-center justify-center border-2",
              theme === "dark" ? "border-black/50" : "border-white/50"
            )}
          >
            <span className="text-white text-xs font-bold">1</span>
          </div>
        </div>

        {/* Task Indicator */}
        <NavbarTaskIndicator />

        {/* User Avatar */}
        {isAuthenticated && user ? (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <div
                className={cn(
                  "w-10 h-10 rounded-full flex items-center justify-center border transition-colors cursor-pointer",
                  theme === "dark"
                    ? "bg-gray-700/50 border-gray-600/30 hover:bg-gray-600/50"
                    : "bg-gray-200/50 border-gray-300/30 hover:bg-gray-300/50"
                )}
              >
                <User
                  className={cn(
                    "w-5 h-5",
                    theme === "dark" ? "text-white/90" : "text-gray-700/90"
                  )}
                />
              </div>
            </DropdownMenuTrigger>
            <DropdownMenuContent
              align="end"
              side="bottom"
              className="w-56 query-content-gradient rounded-[32px] border-0 p-0"
              style={{ marginTop: 0 }}
            >
              <div className="px-2 py-1">
                <DropdownMenuItem
                  onClick={() => (window.location.href = "/user-configuration")}
                  className="text-white active:scale-95 mx-1 transition-all duration-200 flex items-center gap-2 p-2 cursor-pointer"
                  style={{
                    backgroundColor: "transparent",
                    borderRadius: "99px",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor =
                      "var(--item-root-active-bgcolor, rgba(19, 245, 132, 0.08))";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = "transparent";
                  }}
                >
                   <Image
                    src="/dashboard/profile.svg"
                    alt="Log Out"
                    width={16}
                    height={16}
                    className="w-4 h-4"
                  />
                  Profile
                </DropdownMenuItem>

                <DropdownMenuItem
                  onClick={logout}
                  className="text-white active:scale-95 mx-1 transition-all duration-200 flex items-center gap-2 p-2 cursor-pointer"
                  style={{
                    backgroundColor: "transparent",
                    borderRadius: "99px",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor =
                      "var(--item-root-active-bgcolor, rgba(19, 245, 132, 0.08))";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = "transparent";
                  }}
                >
                  <Image
                    src="/dashboard/logout.svg"
                    alt="Log Out"
                    width={16}
                    height={16}
                    className="w-4 h-4"
                  />
                  Log Out
                </DropdownMenuItem>
              </div>
            </DropdownMenuContent>
          </DropdownMenu>
        ) : (
          <Button
            onClick={() => (window.location.href = "/auth")}
            className={cn(
              "transition-colors cursor-pointer",
              theme === "dark"
                ? "text-white/90 hover:text-white hover:bg-gray-600/50"
                : "text-gray-700/90 hover:text-gray-900 hover:bg-gray-300/50"
            )}
            style={{
              background: "rgba(255, 255, 255, 0.08)",
            }}
          >
            <LogIn className="w-4 h-4 mr-2" />
            Sign In
          </Button>
        )}
      </div>
    </nav>
  );
}
