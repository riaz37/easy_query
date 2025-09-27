import React from 'react';

interface DashboardIconProps {
  className?: string;
  size?: number;
}

export const DashboardIcon: React.FC<DashboardIconProps> = ({ 
  className = "h-5 w-5", 
  size = 20 
}) => {
  return (
    <svg 
      width={size} 
      height={size} 
      viewBox="0 0 24 24" 
      fill="none" 
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* Dashboard grid icon */}
      <rect x="3" y="3" width="7" height="7" rx="1" fill="currentColor" fillOpacity="0.8"/>
      <rect x="3" y="12" width="7" height="9" rx="1" fill="currentColor" fillOpacity="0.6"/>
      <rect x="12" y="3" width="9" height="7" rx="1" fill="currentColor" fillOpacity="0.4"/>
    </svg>
  );
};
