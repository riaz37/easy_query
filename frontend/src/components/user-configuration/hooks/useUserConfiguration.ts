import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { toast } from 'sonner';
import { useAuthContext } from '@/components/providers/AuthContextProvider';
import { useDatabaseContext } from '@/components/providers/DatabaseContextProvider';
import { useBusinessRulesContext } from '@/components/providers/BusinessRulesContextProvider';
import { ServiceRegistry } from '@/lib/api';
import type { DatabaseInfo } from '../types';

export const useUserConfiguration = () => {
  const { user, isAuthenticated } = useAuthContext();
  const {
    currentDatabaseId,
    currentDatabaseName,
    setCurrentDatabase,
    loadDatabasesFromConfig,
    setLoading: setDatabaseLoading,
    setError: setDatabaseError,
    availableDatabases,
  } = useDatabaseContext();
  const {
    businessRules,
    loadBusinessRulesFromConfig,
    updateBusinessRules,
    refreshBusinessRules,
    hasBusinessRules,
    businessRulesCount,
    setLoading: setBusinessRulesLoading,
    setError: setBusinessRulesError,
  } = useBusinessRulesContext();

  // Report structure state
  const [reportStructure, setReportStructure] = useState<string>('');
  const [reportStructureLoading, setReportStructureLoading] = useState(false);
  const [reportStructureError, setReportStructureError] = useState<string | null>(null);

  // State
  const [loading, setLoading] = useState(false);
  const [databases, setDatabases] = useState<DatabaseInfo[]>([]);
  const hasLoadedRef = useRef(false);

  // Load user configuration
  const loadUserConfiguration = useCallback(async () => {
    // Check if we already have configuration loaded from storage
    if (
      currentDatabaseId &&
      availableDatabases &&
      availableDatabases.length > 0
    ) {
      return;
    }

    setLoading(true);
    setDatabaseLoading(true);
    setBusinessRulesLoading(true);
    setReportStructureLoading(true);
    setDatabaseError(null);
    setBusinessRulesError(null);
    setReportStructureError(null);

    try {
      // Load accessible databases
      const databasesResponse =
        await ServiceRegistry.database.getAllDatabases();

      if (databasesResponse.success) {
        const dbList = databasesResponse.data.map((db) => ({
          db_id: db.id,
          db_name: db.name,
          db_url: db.url,
          db_type: db.type,
          is_current: db.id === currentDatabaseId,
          business_rule: db.metadata?.businessRule || '',
        }));

        // Update local state
        setDatabases(dbList);

        // Update database context provider
        const dbConfigs = dbList.map((db) => ({
          db_id: db.db_id,
          db_name: db.db_name,
          db_url: db.db_url,
          db_type: db.db_type,
          business_rule: db.business_rule,
          is_active: true,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        }));
        loadDatabasesFromConfig(dbConfigs);

        // Get current database from backend and load business rules
        const currentDBInfo =
          await ServiceRegistry.userCurrentDB.getUserCurrentDB(user?.user_id);
        if (
          currentDBInfo.success &&
          currentDBInfo.data &&
          currentDBInfo.data.db_id
        ) {
          // Update the current database in context
          setCurrentDatabase(
            currentDBInfo.data.db_id,
            currentDBInfo.data.db_name || 'Unknown',
          );

          // Update local state to mark the current database
          setDatabases((prev) =>
            prev.map((db) => ({
              ...db,
              is_current: db.db_id === currentDBInfo.data.db_id,
            })),
          );

          // Extract business rules and report structure from the response
          const businessRules = currentDBInfo.data.business_rule || '';
          const reportStructureData = currentDBInfo.data.report_structure || '';
          loadBusinessRulesFromConfig(businessRules);
          setReportStructure(reportStructureData);
        } else {
          // No current database set
          loadBusinessRulesFromConfig('');
          setReportStructure('');
        }
      } else {
        throw new Error(databasesResponse.error || 'Failed to load databases');
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Failed to load configuration';
      setDatabaseError(errorMessage);
      setBusinessRulesError(errorMessage);
      setReportStructureError(errorMessage);
      toast.error('Failed to load configuration');
    } finally {
      setLoading(false);
      setDatabaseLoading(false);
      setBusinessRulesLoading(false);
      setReportStructureLoading(false);
    }
  }, [
    user?.user_id,
    setCurrentDatabase,
    loadBusinessRulesFromConfig,
    loadDatabasesFromConfig,
    setDatabaseLoading,
    setBusinessRulesLoading,
    setDatabaseError,
    setBusinessRulesError,
    currentDatabaseId,
    availableDatabases,
  ]);

  // Load user configuration on mount
  useEffect(() => {
    if (isAuthenticated && !loading && !hasLoadedRef.current) {
      hasLoadedRef.current = true;
      loadUserConfiguration();
    }
  }, [isAuthenticated, loadUserConfiguration, loading]);

  // Reset loaded flag when user changes
  useEffect(() => {
    hasLoadedRef.current = false;
  }, [user?.user_id]);

  // Manual refresh function (separate from automatic loading)
  const handleManualRefresh = useCallback(async () => {
    hasLoadedRef.current = false;
    // Force reload by clearing storage check
    setLoading(true);
    setDatabaseLoading(true);
    setBusinessRulesLoading(true);
    await loadUserConfiguration();
  }, [loadUserConfiguration]);

  // Handle database selection
  const handleDatabaseChange = useCallback(
    async (databaseId: number) => {
      try {
        setDatabaseLoading(true);
        setDatabaseError(null);

        const selectedDB = databases.find((db) => db.db_id === databaseId);
        if (selectedDB) {
          // Set the current database in backend
          const response = await ServiceRegistry.userCurrentDB.setUserCurrentDB(
            {
              db_id: databaseId,
            },
            user?.user_id,
          );

          if (response.success) {
            // Update database context provider
            setCurrentDatabase(databaseId, selectedDB.db_name);

            // Update local state
            setDatabases((prev) =>
              prev.map((db) => ({
                ...db,
                is_current: db.db_id === databaseId,
              })),
            );

            // Get the current database info which includes business rules and report structure
            const currentDBInfo =
              await ServiceRegistry.userCurrentDB.getUserCurrentDB(
                user?.user_id,
              );
            if (currentDBInfo.success && currentDBInfo.data) {
              const businessRules = currentDBInfo.data.business_rule || '';
              const reportStructureData = currentDBInfo.data.report_structure || '';
              loadBusinessRulesFromConfig(businessRules);
              setReportStructure(reportStructureData);
            } else {
              loadBusinessRulesFromConfig('');
              setReportStructure('');
            }

            toast.success(`Switched to database: ${selectedDB.db_name}`);
          } else {
            throw new Error(response.error || 'Failed to set current database');
          }
        } else {
          throw new Error('Selected database not found');
        }
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : 'Failed to switch database';
        setDatabaseError(errorMessage);
        toast.error('Failed to switch database');
      } finally {
        setDatabaseLoading(false);
      }
    },
    [
      databases,
      user?.user_id,
      setCurrentDatabase,
      loadBusinessRulesFromConfig,
      setDatabaseLoading,
      setDatabaseError,
    ],
  );

  // Handle business rules refresh
  const handleBusinessRulesRefresh = useCallback(async () => {
    try {
      if (currentDatabaseId) {
        setBusinessRulesLoading(true);
        setBusinessRulesError(null);

        const currentDBInfo =
          await ServiceRegistry.userCurrentDB.getUserCurrentDB(user?.user_id);
        if (currentDBInfo.success && currentDBInfo.data) {
          const businessRules = currentDBInfo.data.business_rule || '';
          loadBusinessRulesFromConfig(businessRules);
          toast.success('Business rules refreshed successfully');
        } else {
          loadBusinessRulesFromConfig('');
          toast.success('Business rules refreshed (no rules found)');
        }
      } else {
        toast.error('No database selected');
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : 'Failed to refresh business rules';
      setBusinessRulesError(errorMessage);
      toast.error('Failed to refresh business rules');
    } finally {
      setBusinessRulesLoading(false);
    }
  }, [
    currentDatabaseId,
    user?.user_id,
    setBusinessRulesLoading,
    setBusinessRulesError,
    loadBusinessRulesFromConfig,
  ]);

  // Handle report structure refresh
  const handleReportStructureRefresh = useCallback(async () => {
    try {
      if (currentDatabaseId) {
        setReportStructureLoading(true);
        setReportStructureError(null);

        const currentDBInfo =
          await ServiceRegistry.userCurrentDB.getUserCurrentDB(user?.user_id);
        if (currentDBInfo.success && currentDBInfo.data) {
          const reportStructureData = currentDBInfo.data.report_structure || '';
          setReportStructure(reportStructureData);
          toast.success('Report structure refreshed successfully');
        } else {
          setReportStructure('');
          toast.success('Report structure refreshed (no structure found)');
        }
      } else {
        toast.error('No database selected');
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : 'Failed to refresh report structure';
      setReportStructureError(errorMessage);
      toast.error('Failed to refresh report structure');
    } finally {
      setReportStructureLoading(false);
    }
  }, [
    currentDatabaseId,
    user?.user_id,
    setReportStructureLoading,
    setReportStructureError,
  ]);

  // Memoize the return object to prevent unnecessary re-renders
  return useMemo(() => ({
    // State
    loading,
    databases,
    user,
    isAuthenticated,
    currentDatabaseId,
    currentDatabaseName,
    businessRules,
    hasBusinessRules,
    businessRulesCount,
    reportStructure,
    reportStructureLoading,
    reportStructureError,
    
    // Actions
    loadUserConfiguration,
    handleManualRefresh,
    handleDatabaseChange,
    handleBusinessRulesRefresh,
    handleReportStructureRefresh,
  }), [
    loading,
    databases,
    user,
    isAuthenticated,
    currentDatabaseId,
    currentDatabaseName,
    businessRules,
    hasBusinessRules,
    businessRulesCount,
    reportStructure,
    reportStructureLoading,
    reportStructureError,
    loadUserConfiguration,
    handleManualRefresh,
    handleDatabaseChange,
    handleBusinessRulesRefresh,
    handleReportStructureRefresh,
  ]);
};
