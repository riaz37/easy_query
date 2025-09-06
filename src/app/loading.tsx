"use client";
import React from "react";
import { PageLoader } from "@/components/ui/loading";

export default function LoadingPage() {
  return (
    <PageLoader
      size="lg"
      variant="primary"
      message="Loading..."
      description="Please wait while we prepare your content"
      showProgress={false}
    />
  );
}