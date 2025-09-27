"use client";

import React from "react";
import Image from "next/image";

interface FileQueryHeaderProps {
  className?: string;
}

export function FileQueryHeader({ className = "" }: FileQueryHeaderProps) {
  return (
    <div className={`flex items-start ${className}`}>
      <Image
        src="/file-query/filerobot.svg"
        alt="File Robot"
        width={120}
        height={120}
        className="flex-shrink-0 -ml-2"
      />
      <div className="flex flex-col justify-start pt-5 -ml-4 z-10">
        <h3 className="text-white font-semibold text-xl">
          File Query
        </h3>
      </div>
    </div>
  );
}
