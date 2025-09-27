"use client";
import React, { useState, useEffect } from "react";
import Image from "next/image";
import { Bell, User, LogIn, Settings } from "lucide-react";
import { useUIStore } from "@/store/uiStore";
import { useAuthContext } from "@/components/providers/AuthContextProvider";
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
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY;
      setIsScrolled(scrollTop > 20); // Change to top-0 when scrolled more than 20px
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleMenuClick = () => {
    setShowSidebar(!showSidebar);
  };

  return (
    <nav
      className={cn(
        "fixed left-4 right-4 z-50 backdrop-blur-xl rounded-full flex items-center justify-between shadow-2xl max-w-7xl mx-auto transition-all duration-300",
        isScrolled ? "top-0" : "top-6",
        "bg-white/5 border border-white/10 shadow-black/20"
      )}
      style={{
        height: "64px",
        paddingLeft: "20px",
        paddingRight: "20px",
        background: "linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05))",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1)",
      }}
    >
      {/* Left side - Logo and Menu */}
      <div className="flex items-center gap-6">
        {/* Easy Query Logo */}
        <div className="flex items-center gap-2 px-4 py-2 rounded-full">
          <Image
            src="/logo/logo.svg"
            alt="Easy Query"
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
            className="w-10 h-10 rounded-full flex items-center justify-center transition-colors cursor-pointer"
            style={{
              background: "rgba(255, 255, 255, 0.08)",
            }}
          >
            <Bell className="w-5 h-5 text-white/90" />
          </div>
          <div className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full flex items-center justify-center border-2 border-black/50">
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
                className="w-10 h-10 rounded-full flex items-center justify-center transition-colors cursor-pointer"
                style={{
                  background: "rgba(255, 255, 255, 0.08)",
                }}
              >
                <User className="w-5 h-5 text-white/90" />
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
            className="transition-colors cursor-pointer text-white/90 hover:text-white hover:bg-gray-600/50"
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
