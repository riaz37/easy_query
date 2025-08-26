"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  Upload,
  User,
  Settings,
  History,
  Terminal,
  Square,
  FileText,
  Bot,
  Database,
  Search,
  Bookmark,
  Palette,
  Home,
  Shield,
  Building2,
  Plane,
  Users,
} from "lucide-react";

import { useUIStore } from "@/store/uiStore";
import { cn } from "@/lib/utils";

export default function Menu() {
  const pathname = usePathname();
  const { setShowSidebar } = useUIStore();

  // Menu items with proper icons
  const menuItems = [
    {
      icon: Home,
      name: "Dashboard",
      path: "/",
      isActive: pathname === "/",
    },
    {
      icon: Search,
      name: "Database Query",
      path: "/database-query",
      isActive: pathname === "/database-query",
    },
    {
      icon: BarChart3,
      name: "AI Reports",
      path: "/ai-results",
      isActive: pathname === "/ai-results",
    },
    {
      icon: FileText,
      name: "File Query",
      path: "/file-query",
      isActive: pathname === "/file-query",
    },
    {
      icon: Database,
      name: "Tables",
      path: "/tables",
      isActive: pathname === "/tables",
    },
    {
      icon: Building2,
      name: "Company Struture",
      path: "/company-structure",
      isActive: pathname === "/company-structure",
    },
    {
      icon: Plane,
      name: "User Configuration",
      path: "/user-configuration",
      isActive: pathname === "/user-configuration",
    },
    {
      icon: Users,
      name: "Users",
      path: "/users",
      isActive: pathname === "/users",
    },
  ];

  const handleMenuItemClick = () => {
    setShowSidebar(false);
  };

  const renderMenuItem = (item: any) => {
    const isActive = item.isActive;
    const hasChildren = item.children && item.children.length > 0;
    const IconComponent = item.icon;

    return (
      <div key={item.name}>
        <Link
          href={item.path}
          onClick={handleMenuItemClick}
          className={cn(
            "flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 group relative",
            isActive
              ? "bg-green-500/20 text-green-400 border border-green-500/30 shadow-lg"
              : "text-gray-300 hover:text-white hover:bg-white/5 border border-transparent hover:border-white/10"
          )}
        >
          <IconComponent
            className={cn(
              "h-5 w-5 transition-all duration-200",
              isActive
                ? "text-green-400"
                : "text-gray-400 group-hover:text-white"
            )}
          />
          <span className="flex-1">{item.name}</span>
          {hasChildren && (
            <div
              className={cn(
                "w-2 h-2 rounded-full transition-all duration-200",
                isActive ? "bg-green-400" : "bg-gray-500 group-hover:bg-white"
              )}
            />
          )}
        </Link>

        {/* Render children if they exist and parent is active */}
        {hasChildren && isActive && (
          <div className="ml-6 mt-2 space-y-1">
            {item.children.map((child: any) => {
              const ChildIconComponent = child.icon;
              return (
                <Link
                  key={child.name}
                  href={child.path}
                  onClick={handleMenuItemClick}
                  className={cn(
                    "flex items-center gap-3 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 group relative",
                    pathname === child.path
                      ? "bg-green-500/10 text-green-300 border border-green-500/20"
                      : "text-gray-400 hover:text-gray-200 hover:bg-white/5 border border-transparent hover:border-white/5"
                  )}
                >
                  <ChildIconComponent
                    className={cn(
                      "h-4 w-4 transition-all duration-200",
                      pathname === child.path
                        ? "text-green-300"
                        : "text-gray-500 group-hover:text-gray-300"
                    )}
                  />
                  <span className="flex-1">{child.name}</span>
                </Link>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="fixed top-[80px] left-1/2 -translate-x-[calc(50%+300px)] z-50 animate-in slide-in-from-top-4 duration-300">
      <div
        className="backdrop-blur-2xl border border-green-500/30 rounded-2xl shadow-2xl overflow-hidden w-80 max-h-[80vh] overflow-y-auto"
        style={{
          background:
            "linear-gradient(135deg, rgba(0, 0, 0, 0.7) 0%, rgba(0, 50, 30, 0.8) 50%, rgba(0, 0, 0, 0.9) 100%)",
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          boxShadow:
            "0 25px 50px -12px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(34, 197, 94, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.1)",
        }}
      >
        {/* Menu Items */}
        <div className="p-4">
          <div className="space-y-1">{menuItems.map(renderMenuItem)}</div>
        </div>
      </div>
    </div>
  );
}
