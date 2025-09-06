import React, { useState, Suspense, lazy } from 'react';
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@/components/ui/tabs';
import { Card, CardContent } from '@/components/ui/card';
import { AlertCircle } from 'lucide-react';
import { Spinner } from '@/components/ui/loading';
import { useUserConfiguration } from '../hooks/useUserConfiguration';
import { useBusinessRulesEditor } from '../hooks/useBusinessRulesEditor';
import { PageLayout } from '@/components/layout/PageLayout';
import type { UserConfigurationProps } from '../types';

// Lazy load tab components for code splitting
const OverviewTab = lazy(() => import('./OverviewTab').then(module => ({ default: module.OverviewTab })));
const DatabaseTab = lazy(() => import('./DatabaseTab').then(module => ({ default: module.DatabaseTab })));
const BusinessRulesTab = lazy(() => import('./BusinessRulesTab').then(module => ({ default: module.BusinessRulesTab })));

// Loading component for Suspense fallback
const TabLoadingFallback = () => (
  <div className="flex items-center justify-center py-8">
    <Spinner size="md" variant="primary" />
    <span className="ml-2 text-gray-400">Loading...</span>
  </div>
);

export const UserConfiguration = React.memo<UserConfigurationProps>(({ className }) => {
  const [activeTab, setActiveTab] = useState('overview');
  
  const {
    loading,
    databases,
    user,
    isAuthenticated,
    currentDatabaseId,
    currentDatabaseName,
    businessRules,
    hasBusinessRules,
    businessRulesCount,
    handleManualRefresh,
    handleDatabaseChange,
    handleBusinessRulesRefresh,
  } = useUserConfiguration();

  const {
    editorState,
    handleRulesEdit,
    handleRulesSave,
    handleRulesCancel,
    handleRulesReset,
    handleRulesContentChange,
  } = useBusinessRulesEditor({
    currentDatabaseId,
    businessRulesContent: businessRules.content,
    onRefresh: handleManualRefresh,
  });

  if (!isAuthenticated) {
    return (
      <PageLayout background="default" maxWidth="4xl">
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <AlertCircle className="w-12 h-12 text-yellow-500 mx-auto mb-4" />
              <h2 className="text-xl font-semibold mb-2">
                Authentication Required
              </h2>
              <p className="text-gray-600">
                Please log in to access your configuration.
              </p>
            </div>
          </CardContent>
        </Card>
      </PageLayout>
    );
  }

  return (
    <PageLayout background="default" className={className}>
      <div className="space-y-6">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2 flex items-center justify-center gap-3">
            <svg className="w-8 h-8 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            User Configuration
          </h1>
          <p className="text-gray-400 text-lg">
            Manage your database settings, business rules, and preferences
          </p>
        </div>

        {/* Main Configuration Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3 bg-slate-800/50">
            <TabsTrigger value="overview" className="flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
              Overview
            </TabsTrigger>
            <TabsTrigger value="database" className="flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
              </svg>
              Database Settings
            </TabsTrigger>
            <TabsTrigger value="business-rules" className="flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
              Business Rules
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview">
            <Suspense fallback={<TabLoadingFallback />}>
              <OverviewTab
                user={user}
                currentDatabaseName={currentDatabaseName}
                businessRules={businessRules}
                businessRulesCount={businessRulesCount}
                hasBusinessRules={hasBusinessRules}
                onNavigateToTab={setActiveTab}
              />
            </Suspense>
          </TabsContent>

          {/* Database Settings Tab */}
          <TabsContent value="database">
            <Suspense fallback={<TabLoadingFallback />}>
              <DatabaseTab
                databases={databases}
                loading={loading}
                onDatabaseChange={handleDatabaseChange}
                onRefresh={handleManualRefresh}
                businessRules={businessRules}
              />
            </Suspense>
          </TabsContent>

          {/* Business Rules Tab */}
          <TabsContent value="business-rules">
            <Suspense fallback={<TabLoadingFallback />}>
              <BusinessRulesTab
                currentDatabaseId={currentDatabaseId}
                currentDatabaseName={currentDatabaseName}
                businessRules={businessRules}
                businessRulesCount={businessRulesCount}
                hasBusinessRules={hasBusinessRules}
                editorState={editorState}
                onRefresh={handleBusinessRulesRefresh}
                onEdit={handleRulesEdit}
                onSave={handleRulesSave}
                onCancel={handleRulesCancel}
                onReset={handleRulesReset}
                onContentChange={handleRulesContentChange}
              />
            </Suspense>
          </TabsContent>
        </Tabs>
      </div>
    </PageLayout>
  );
});

UserConfiguration.displayName = 'UserConfiguration';
