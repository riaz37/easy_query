"use client";

import { cn } from "@/lib/utils";
import { LoadingProps } from "./types";
import { Spinner } from "./Spinner";
import { PulseLoader } from "./PulseLoader";
import { DotsLoader } from "./DotsLoader";
import { WaveLoader } from "./WaveLoader";
import { OrbitLoader } from "./OrbitLoader";

export type LoadingStateType = 'spinner' | 'pulse' | 'dots' | 'wave' | 'orbit';

interface LoadingStatesProps extends LoadingProps {
  type?: LoadingStateType;
}

export function LoadingStates({ 
  type = 'spinner',
  size = "md", 
  variant = "primary", 
  className 
}: LoadingStatesProps) {
  const renderLoader = () => {
    switch (type) {
      case 'spinner':
        return <Spinner size={size} variant={variant} className={className} />;
      case 'pulse':
        return <PulseLoader size={size} variant={variant} className={className} />;
      case 'dots':
        return <DotsLoader size={size} variant={variant} className={className} />;
      case 'wave':
        return <WaveLoader size={size} variant={variant} className={className} />;
      case 'orbit':
        return <OrbitLoader size={size} variant={variant} className={className} />;
      default:
        return <Spinner size={size} variant={variant} className={className} />;
    }
  };

  return renderLoader();
}
