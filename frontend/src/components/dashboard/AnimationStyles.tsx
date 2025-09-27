import React from 'react';

export const AnimationStyles: React.FC = () => {
  return (
    <style jsx>{`
      @keyframes float {
        0%, 100% { 
          transform: translateY(0px) translateX(0px); 
          opacity: 0.3;
        }
        50% { 
          transform: translateY(-20px) translateX(10px); 
          opacity: 0.8;
        }
      }
      
      @keyframes borderGlow {
        0%, 100% { 
          background-position: 0% 50%; 
        }
        50% { 
          background-position: 100% 50%; 
        }
      }
      
      @keyframes shimmer {
        0% { 
          transform: translateX(-100%); 
        }
        100% { 
          transform: translateX(100%); 
        }
      }
      
      @keyframes borderSweep {
        0%, 100% { 
          background-position: -200% 0; 
        }
        50% { 
          background-position: 200% 0; 
        }
      }
    `}</style>
  );
};