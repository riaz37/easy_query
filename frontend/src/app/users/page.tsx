"use client";

import React from "react";
import { UsersManager } from "@/components/users/UsersManager";
import { PageLayout } from "@/components/layout/PageLayout";

export default function UsersPage() {
  return (
    <PageLayout background={["frame", "gridframe"]}>
      <UsersManager />
    </PageLayout>
  );
}
