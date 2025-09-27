"use client";

import React from "react";
import { UserConfiguration } from "@/components/user-configuration";
import { PageLayout } from "@/components/layout/PageLayout";

export default function UserConfigurationPage() {
  return (
    <PageLayout background={["frame", "gridframe"]}>
      <UserConfiguration />
    </PageLayout>
  );
}
