"use client";

import React, { createContext, useContext, ReactNode } from 'react';
import { useAuth } from '@/lib/hooks/use-auth';
import { AuthContextData } from '@/types/auth';

const AuthContext = createContext<AuthContextData | null>(null);

export { AuthContext };

export function useAuthContext() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuthContext must be used within an AuthContextProvider');
  }
  return context;
}

interface AuthContextProviderProps {
  children: ReactNode;
}

export function AuthContextProvider({ children }: AuthContextProviderProps) {
  const authData = useAuth();

  return (
    <AuthContext.Provider value={authData}>
      {children}
    </AuthContext.Provider>
  );
} 