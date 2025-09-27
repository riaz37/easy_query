"use client";

import { CompanyCard } from "./ui/CompanyCard";
import { Company, CompanyCardProps, CompanyTreeProps } from "./types";

export function CompanyTree({
  companies,
  onAddSubCompany,
  selectedCompany,
  onSelectCompany,
}: CompanyTreeProps) {
  const renderCompanyNode = (
    company: Company,
    level: number = 0,
    parentX: number = 0,
    parentY: number = 0
  ) => {
    const hasChildren = company.children && company.children.length > 0;
    const cardWidth = 320;
    const cardHeight = 180;
    const verticalSpacing = 200;
    const horizontalSpacing = 400;

    // Calculate position
    const x = level === 0 ? 0 : parentX + horizontalSpacing;
    const y = level === 0 ? 0 : parentY;

    return (
      <div
        key={company.id}
        className="absolute z-30"
        style={{ left: x, top: y }}
      >
        {/* Connection line from parent */}
        {level > 0 && (
          <div className="absolute -left-32 top-1/2 w-32 h-0.5 bg-green-500/40 transform -translate-y-1/2 z-20">
            {/* Connection dots */}
            <div className="absolute left-0 top-1/2 w-2 h-2 bg-green-400 rounded-full transform -translate-y-1/2 shadow-lg shadow-green-400/50" />
            <div className="absolute right-0 top-1/2 w-2 h-2 bg-green-400 rounded-full transform -translate-y-1/2 shadow-lg shadow-green-400/50" />
          </div>
        )}

        {/* Company Card */}
        <div className="relative z-40">
          <CompanyCard
            company={company}
            onAddSubCompany={(parentId: string) => {
              onAddSubCompany("", "", "", parentId);
            }}
            isSelected={selectedCompany === company.id}
            onSelect={() => onSelectCompany(company.id)}
            level={level}
          />
        </div>

        {/* Children */}
        {hasChildren && (
          <div className="relative z-30">
            {/* Vertical line down from card */}
            <div className="absolute left-1/2 top-full w-0.5 h-16 bg-green-500/40 transform -translate-x-1/2 z-20">
              <div className="absolute top-0 left-1/2 w-2 h-2 bg-green-400 rounded-full transform -translate-x-1/2 -translate-y-1/2 shadow-lg shadow-green-400/50" />
              <div className="absolute bottom-0 left-1/2 w-2 h-2 bg-green-400 rounded-full transform -translate-x-1/2 translate-y-1/2 shadow-lg shadow-green-400/50" />
            </div>

            {/* Children container */}
            <div className="relative z-30" style={{ top: cardHeight + 64 }}>
              {company.children!.map((child, index) => {
                const childY = index * verticalSpacing;
                return (
                  <div key={child.id}>
                    {/* Horizontal line to child position */}
                    <div
                      className="absolute w-32 h-0.5 bg-green-500/40 z-20"
                      style={{
                        left: cardWidth / 2,
                        top: childY + cardHeight / 2,
                        transform: "translateY(-50%)",
                      }}
                    >
                      <div className="absolute left-0 top-1/2 w-2 h-2 bg-green-400 rounded-full transform -translate-y-1/2 shadow-lg shadow-green-400/50" />
                      <div className="absolute right-0 top-1/2 w-2 h-2 bg-green-400 rounded-full transform -translate-y-1/2 shadow-lg shadow-green-400/50" />
                    </div>

                    {/* Child node */}
                    {renderCompanyNode(child, level + 1, x, childY)}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="w-full h-full overflow-auto p-8">
      <div
        className="relative min-w-max min-h-max z-30"
        style={{ width: "200vw", height: "200vh" }}
      >
        {companies.map((company, index) => {
          return (
            <div
              key={company.id}
              className="absolute z-40"
              style={{
                left: "50%",
                top: index * 300 + 100,
                transform: "translateX(-50%)",
              }}
            >
              {renderCompanyNode(company, 0, 0, 0)}
            </div>
          );
        })}
      </div>
    </div>
  );
}
