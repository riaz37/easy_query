"use client";

import React, { createContext, useContext, ReactNode, useState, useCallback, useMemo, useEffect } from 'react';
import { useAuthContext } from './AuthContextProvider';
import { STORAGE_KEYS, saveToUserStorage, loadFromUserStorage, clearEasyQueryStorage } from '@/lib/utils/storage';

// Types for database context
export interface DatabaseConfig {
  db_id: number;
  db_name: string;
  db_url: string;
  db_type: string;
  business_rule?: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface MSSQLDatabaseConfig {
  db_id: number;
  db_name: string;
  db_url: string;
  db_type: 'mssql';
  business_rule?: string;
  is_active: boolean;
  created_at?: string;
  updated_at: string;
}

export interface DatabaseContextData {
  // Current database state
  currentDatabase: DatabaseConfig | null;
  currentDatabaseId: number | null;
  currentDatabaseName: string;
  
  // Available databases
  availableDatabases: DatabaseConfig[];
  userDatabases: DatabaseConfig[];
  mssqlDatabases: MSSQLDatabaseConfig[];
  
  // Loading and error states
  isLoading: boolean;
  error: string | null;
  
  // Actions (called by user-configuration page only)
  setCurrentDatabase: (dbId: number, dbName: string) => void;
  loadDatabasesFromConfig: (databases: DatabaseConfig[]) => void;
  clearCurrentDatabase: () => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  
  // Computed values
  hasCurrentDatabase: boolean;
  databaseCount: number;
  activeDatabaseCount: number;
  mssqlDatabaseCount: number;
}

const DatabaseContext = createContext<DatabaseContextData | undefined>(undefined);

export { DatabaseContext };

export function useDatabaseContext() {
  const context = useContext(DatabaseContext);
  if (!context) {
    throw new Error('useDatabaseContext must be used within a DatabaseContextProvider');
  }
  return context;
}

interface DatabaseContextProviderProps {
  children: ReactNode;
}

export function DatabaseContextProvider({ children }: DatabaseContextProviderProps) {
  const { user } = useAuthContext();

  // State
  const [currentDatabase, setCurrentDatabaseState] = useState<DatabaseConfig | null>(null);
  const [currentDatabaseId, setCurrentDatabaseId] = useState<number | null>(null);
  const [currentDatabaseName, setCurrentDatabaseName] = useState<string>('');
  const [availableDatabases, setAvailableDatabases] = useState<DatabaseConfig[]>([]);
  const [userDatabases, setUserDatabases] = useState<DatabaseConfig[]>([]);
  const [mssqlDatabases, setMSSQLDatabases] = useState<MSSQLDatabaseConfig[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Memoized computed values
  const hasCurrentDatabase = useMemo(() => !!currentDatabaseId, [currentDatabaseId]);
  const databaseCount = useMemo(() => availableDatabases.length, [availableDatabases]);
  const activeDatabaseCount = useMemo(() => 
    availableDatabases.filter(db => db.is_active).length, 
    [availableDatabases]
  );
  const mssqlDatabaseCount = useMemo(() => 
    mssqlDatabases.filter(db => db.is_active).length, 
    [mssqlDatabases]
  );

  // Load configuration from localStorage on mount
  useEffect(() => {
    if (user?.user_id) {
      loadConfigurationFromStorage();
    }
  }, [user?.user_id]);

  // Load configuration from localStorage
  const loadConfigurationFromStorage = useCallback(() => {
    try {
      // Load current database
      const storedCurrentDB = loadFromUserStorage(STORAGE_KEYS.CURRENT_DATABASE, user?.user_id || '');
      if (storedCurrentDB) {
        setCurrentDatabaseState(storedCurrentDB);
        setCurrentDatabaseId(storedCurrentDB.db_id);
        setCurrentDatabaseName(storedCurrentDB.db_name);
      }

      // Load available databases
      const storedAvailableDBs = loadFromUserStorage(STORAGE_KEYS.AVAILABLE_DATABASES, user?.user_id || '');
      if (storedAvailableDBs) {
        setAvailableDatabases(storedAvailableDBs);
        setUserDatabases(storedAvailableDBs);
        
        // Filter MSSQL databases
        const mssqlDBs = storedAvailableDBs.filter((db: DatabaseConfig) => db.db_type === 'mssql') as MSSQLDatabaseConfig[];
        setMSSQLDatabases(mssqlDBs);
      }
    } catch (error) {
      // Clear corrupted storage
      clearEasyQueryStorage(user?.user_id);
    }
  }, [user?.user_id]);

  // Load databases from user-configuration page
  const loadDatabasesFromConfig = useCallback((databases: DatabaseConfig[]) => {
    setAvailableDatabases(databases);
    setUserDatabases(databases);
    
    // Filter MSSQL databases
    const mssqlDBs = databases.filter(db => db.db_type === 'mssql') as MSSQLDatabaseConfig[];
    setMSSQLDatabases(mssqlDBs);
    
    // Save to localStorage
    saveToUserStorage(STORAGE_KEYS.AVAILABLE_DATABASES, user?.user_id || '', databases);
    saveToUserStorage(STORAGE_KEYS.USER_DATABASES, user?.user_id || '', databases);
    saveToUserStorage(STORAGE_KEYS.MSSQL_DATABASES, user?.user_id || '', mssqlDBs);
    
    // Update current database if we have a basic config that needs updating
    if (currentDatabaseId && currentDatabase && currentDatabase.db_url === '') {
      const realDbConfig = databases.find(db => db.db_id === currentDatabaseId);
      if (realDbConfig) {
        setCurrentDatabaseState(realDbConfig);
        saveToUserStorage(STORAGE_KEYS.CURRENT_DATABASE, user?.user_id || '', realDbConfig);
      }
    }
  }, [user?.user_id, currentDatabaseId, currentDatabase]);

  // Set current database (called by user-configuration page only)
  const setCurrentDatabase = useCallback((dbId: number, dbName: string) => {
    // Find the database config
    const dbConfig = availableDatabases.find(db => db.db_id === dbId);
    
    if (dbConfig) {
      setCurrentDatabaseState(dbConfig);
      setCurrentDatabaseId(dbId);
      setCurrentDatabaseName(dbName);
      
      // Save to localStorage
      saveToUserStorage(STORAGE_KEYS.CURRENT_DATABASE, user?.user_id || '', dbConfig);
    } else {
      // Database not found in available databases yet, create a basic config
      // This will be updated when availableDatabases is loaded
      const basicConfig: DatabaseConfig = {
        db_id: dbId,
        db_name: dbName,
        db_url: '',
        db_type: 'mssql', // Default type
        business_rule: '',
        is_active: true,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };
      
      setCurrentDatabaseState(basicConfig);
      setCurrentDatabaseId(dbId);
      setCurrentDatabaseName(dbName);
      
      // Save to localStorage
      saveToUserStorage(STORAGE_KEYS.CURRENT_DATABASE, user?.user_id || '', basicConfig);
    }
  }, [availableDatabases, user?.user_id]);

  // Clear current database
  const clearCurrentDatabase = useCallback(() => {
    setCurrentDatabaseState(null);
    setCurrentDatabaseId(null);
    setCurrentDatabaseName('');
    
    // Remove from localStorage
    if (user?.user_id) {
      localStorage.removeItem(`${STORAGE_KEYS.CURRENT_DATABASE}_${user.user_id}`);
    }
  }, [user?.user_id]);

  // Set loading state
  const setLoadingState = useCallback((loading: boolean) => {
    setIsLoading(loading);
  }, []);

  // Set error state
  const setErrorState = useCallback((error: string | null) => {
    setError(error);
  }, []);

  // Memoized context value
  const contextValue = useMemo<DatabaseContextData>(() => ({
    currentDatabase,
    currentDatabaseId,
    currentDatabaseName,
    availableDatabases,
    userDatabases,
    mssqlDatabases,
    isLoading,
    error,
    setCurrentDatabase,
    loadDatabasesFromConfig,
    clearCurrentDatabase,
    setLoading: setLoadingState,
    setError: setErrorState,
    hasCurrentDatabase,
    databaseCount,
    activeDatabaseCount,
    mssqlDatabaseCount,
  }), [
    currentDatabase,
    currentDatabaseId,
    currentDatabaseName,
    availableDatabases,
    userDatabases,
    mssqlDatabases,
    isLoading,
    error,
    setCurrentDatabase,
    loadDatabasesFromConfig,
    clearCurrentDatabase,
    setLoadingState,
    setErrorState,
    hasCurrentDatabase,
    databaseCount,
    activeDatabaseCount,
    mssqlDatabaseCount,
  ]);

  return (
    <DatabaseContext.Provider value={contextValue}>
      {children}
    </DatabaseContext.Provider>
  );
} 