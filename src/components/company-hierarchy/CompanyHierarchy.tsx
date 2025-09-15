"use client";

import { CompanyTreeView } from "./CompanyTreeView";

export function CompanyHierarchy() {
  const handleCompanyCreated = () => {
    // Optional callback when a company is created
  };

  return (
    <CompanyTreeView onCompanyCreated={handleCompanyCreated} />
  );
}
