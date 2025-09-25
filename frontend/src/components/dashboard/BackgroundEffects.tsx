import React from 'react';

export const BackgroundEffects: React.FC = () => {
  return (
    <>
      {/* Enhanced gradient overlays */}
      <div className="absolute inset-0 bg-gradient-to-t from-emerald-950/60 via-transparent to-emerald-950/30 pointer-events-none" />
      <div className="absolute inset-0 bg-gradient-to-r from-emerald-950/20 via-transparent to-teal-950/20 pointer-events-none" />
      
      {/* Animated background particles */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        {[...Array(20)].map((_, i: number) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-emerald-400/30 rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animation: `float ${4 + Math.random() * 4}s ease-in-out infinite`,
              animationDelay: `${Math.random() * 2}s`,
              boxShadow: '0 0 6px currentColor',
            }}
          />
        ))}
      </div>
    </>
  );
};