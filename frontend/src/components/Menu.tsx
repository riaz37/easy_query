"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import Image from "next/image";
import { useUIStore } from "@/store/uiStore";
import { cn } from "@/lib/utils";

export default function Menu() {
  const pathname = usePathname();
  const { setShowSidebar } = useUIStore();

  const menuItems = [
    {
      icon: "/dashboard/dashboard.svg",
      name: "Dashboard",
      path: "/",
    },
    {
      icon: "/dashboard/Databased Query.svg",
      name: "Database Query",
      path: "/database-query",
    },
    {
      icon: "/dashboard/Report.svg",
      name: "AI Reports",
      path: "/ai-reports",
    },
    {
      icon: "/dashboard/File Query.svg",
      name: "File Query",
      path: "/file-query",
    },
    {
      icon: "/dashboard/Table.svg",
      name: "Tables",
      path: "/tables",
    },
    {
      icon: "/dashboard/Company.svg",
      name: "Company Structure",
      path: "/company-structure",
    },
    {
      icon: "/dashboard/user.svg",
      name: "Users",
      path: "/users",
    },
  ];

  return (
    <div className="fixed top-[100px] left-1/2 -translate-x-[calc(50%+300px)] z-50 animate-in slide-in-from-top-4 duration-300">
      <div
        className="w-80 max-h-[80vh] overflow-y-auto rounded-[32px] shadow-2xl border"
        style={{
          background: `linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)),
                        linear-gradient(246.02deg, rgba(19, 245, 132, 0) 91.9%, rgba(19, 245, 132, 0.2) 114.38%),
                        linear-gradient(59.16deg, rgba(19, 245, 132, 0) 71.78%, rgba(19, 245, 132, 0.2) 124.92%)`,
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
        }}
      >
        <div className="p-4 space-y-1">
          {menuItems.map((item) => {
            const isActive = pathname === item.path;

            return (
              <Link
                key={item.name}
                href={item.path}
                onClick={() => setShowSidebar(false)}
                className={cn(
                  "flex items-center gap-3 px-4 py-3 rounded-[99px] text-sm font-medium transition-all duration-200",
                  isActive
                    ? "text-green-300"
                    : "text-gray-300 hover:text-green-300"
                )}
                style={{
                  backgroundColor: isActive
                    ? "var(--item-root-active-bgcolor, #13F58414)"
                    : undefined,
                }}
                onMouseEnter={(e) => {
                  if (!isActive) {
                    e.currentTarget.style.backgroundColor =
                      "var(--item-root-active-bgcolor, #13F58414)";
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isActive) {
                    e.currentTarget.style.backgroundColor = "";
                  }
                }}
              >
                <Image
                  src={item.icon}
                  alt={item.name}
                  width={24}
                  height={24}
                  className="h-7 w-7"
                />
                <span className="flex-1">{item.name}</span>
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
}
