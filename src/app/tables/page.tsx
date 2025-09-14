"use client";

import React from "react";
import { TablesManager } from "@/components/tables/TablesManager";
import { PageLayout } from "@/components/layout/PageLayout";

export default function TablesPage() {
  return (
    <PageLayout 
      background={["frame", "gridframe"]} 
      container={false} 
      maxWidth="full"
      className="h-screen w-full overflow-hidden"
    >
      <TablesManager />
    </PageLayout>
  );
}
