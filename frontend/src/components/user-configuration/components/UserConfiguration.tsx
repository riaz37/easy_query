import React, { useState, Suspense, lazy } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { 
  AlertCircle, 
  User, 
  Phone, 
  MapPin, 
  Mail, 
  Lock, 
  Database, 
  Shield,
  CheckCircle,
  X
} from 'lucide-react';
import { Spinner } from '@/components/ui/loading';
import { useUserConfiguration } from '../hooks/useUserConfiguration';
import { useBusinessRulesEditor } from '../hooks/useBusinessRulesEditor';
import type { UserConfigurationProps } from '../types';

// Lazy load tab components for code splitting
const OverviewTab = lazy(() => import('./OverviewTab').then(module => ({ default: module.OverviewTab })));
const DatabaseTab = lazy(() => import('./DatabaseTab').then(module => ({ default: module.DatabaseTab })));
const BusinessRulesTab = lazy(() => import('./BusinessRulesTab').then(module => ({ default: module.BusinessRulesTab })));
const ReportStructureTab = lazy(() => import('./ReportStructureTab').then(module => ({ default: module.ReportStructureTab })));

// Loading component for Suspense fallback
const TabLoadingFallback = () => (
  <div className="flex items-center justify-center py-8">
    <Spinner size="md" variant="primary" />
    <span className="ml-2 text-gray-400">Loading...</span>
  </div>
);

export const UserConfiguration = React.memo<UserConfigurationProps>(({ className }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [blockProfile, setBlockProfile] = useState(false);
  const [emailNotifications, setEmailNotifications] = useState({
    comments: true,
    answers: true,
    follows: false
  });
  
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
    reportStructure,
    reportStructureLoading,
    reportStructureError,
    handleManualRefresh,
    handleDatabaseChange,
    handleBusinessRulesRefresh,
    handleReportStructureRefresh,
  } = useUserConfiguration();

  const {
    editorState,
    handleRulesEdit,
    handleRulesSave,
    handleRulesCancel,
    handleRulesContentChange,
  } = useBusinessRulesEditor({
    currentDatabaseId,
    businessRulesContent: businessRules.content,
    onRefresh: handleManualRefresh,
  });

  if (!isAuthenticated) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="card-enhanced">
          <div className="card-content-enhanced">
            <div className="text-center">
              <AlertCircle className="w-12 h-12 text-yellow-400 mx-auto mb-4" />
              <h2 className="text-xl font-semibold mb-2 text-white">
                Authentication Required
              </h2>
              <p className="text-gray-300">
                Please log in to access your configuration.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return (
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
        );
      case 'database':
        return (
          <Suspense fallback={<TabLoadingFallback />}>
            <DatabaseTab
              databases={databases}
              loading={loading}
              onDatabaseChange={handleDatabaseChange}
              businessRules={businessRules}
            />
          </Suspense>
        );
      case 'business-rules':
        return (
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
              onContentChange={handleRulesContentChange}
            />
          </Suspense>
        );
      case 'report-structure':
        return (
          <Suspense fallback={<TabLoadingFallback />}>
            <ReportStructureTab
              reportStructure={reportStructure}
              reportStructureLoading={reportStructureLoading}
              reportStructureError={reportStructureError}
              onRefresh={handleReportStructureRefresh}
            />
          </Suspense>
        );
      default:
        return null;
    }
  };

  return (
    <div className={className}>
      <div className="flex gap-6">
        {/* Left Side - Profile Management Section */}
        <div className="w-80 flex-shrink-0">
          <div className="query-content-gradient rounded-[32px] p-6">
            {/* Profile Picture Section */}
            <div className="flex flex-col items-center mb-4">
              <div className="relative w-24 h-24 mb-4">
                <div
                  className="w-full h-full rounded-full flex items-center justify-center"
                  style={{
                    background: "var(--primary-8, rgba(19, 245, 132, 0.08))",
                    color: "var(--primary-main, rgba(19, 245, 132, 1))",
                    border: "1px dashed var(--primary-16, rgba(19, 245, 132, 0.5))",
                    borderDasharray: "8px 4px",
                    borderDashoffset: "0",
                    borderRadius: "99px"
                  }}
                >
                  <User className="w-12 h-12 text-current" />
                </div>
              </div>
              <p className="text-gray-400 text-sm text-center mb-2">Upload photo</p>
              <p className="text-gray-500 text-xs text-center mb-4">
                Allowed *.jpeg, *.jpg, *.png, *.gif<br />
                Max size of 3.1 MB
              </p>
            </div>

            {/* Block Profile Toggle */}
            <div className="mb-8">
              <div className="flex items-center justify-center gap-3">
                <span className="text-white font-medium">Block Profile</span>
                <Switch
                  checked={blockProfile}
                  onCheckedChange={setBlockProfile}
                  className="data-[state=checked]:bg-emerald-500"
                />
              </div>
            </div>

            {/* Delete User Button */}
            <div className="flex justify-center">
              <Button 
                variant="destructive" 
                className="px-6 py-2"
                style={{
                  background: "var(--error-8, rgba(255, 86, 48, 0.08))",
                  color: "var(--error-main, rgba(255, 86, 48, 1))",
                  border: "1px solid var(--error-16, rgba(255, 86, 48, 0.16))",
                  borderRadius: "99px"
                }}
              >
                Delete user
              </Button>
            </div>
          </div>
        </div>

        {/* Right Side - Main Content */}
        <div className="flex-1 space-y-6">
          {/* Tab Navigation Section */}
          <div className="query-content-gradient rounded-[32px] p-6 h-24 flex items-center">
            <div className="flex gap-8">
              {[
                { id: 'overview', label: 'Overview' },
                { id: 'database', label: 'Database Settings' },
                { id: 'business-rules', label: 'Business Rules' },
                { id: 'report-structure', label: 'Report Structure' }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`text-sm font-medium pb-2 border-b-2 transition-colors cursor-pointer ${
                    activeTab === tab.id
                      ? ''
                      : 'text-gray-400 border-transparent hover:text-white'
                  }`}
                  style={activeTab === tab.id ? {
                    color: 'var(--primary-main, rgba(19, 245, 132, 1))',
                    borderBottomColor: 'var(--primary-main, rgba(19, 245, 132, 1))'
                  } : {}}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>

          {/* Tab Content */}
          <div className="space-y-6">
            {activeTab === 'overview' && (
              <>
                {/* User Information Section */}
                <div className="query-content-gradient rounded-[32px] p-6">
                  <div className="space-y-4">
                    <h2 className="text-xl font-semibold text-white">User Information</h2>
                    <p className="text-gray-400 text-sm">Donec mi odio, faucibus at, scelerisque quis</p>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <Label className="text-gray-400">Name</Label>
                        <Input 
                          value={user?.name || 'Jayvion Simon'} 
                          className="modal-input-enhanced"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-gray-400 flex items-center gap-2">
                          <Phone className="w-4 h-4" />
                          Phone number
                        </Label>
                        <Input 
                          value="365-374-4961" 
                          className="modal-input-enhanced"
                        />
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <Label className="text-gray-400 flex items-center gap-2">
                        <MapPin className="w-4 h-4" />
                        Address
                      </Label>
                      <Input 
                        value="19034 Verna Unions Apt. 164 - Honolulu, RI / 87535" 
                        className="modal-input-enhanced"
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <Label className="text-gray-400">About</Label>
                      <Textarea 
                        value="Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Sed aliquam, nisi quis porttitor congue, elit erat euismod orci, ac placerat dolor lectus quis orci." 
                        className="modal-input-enhanced min-h-[100px]"
                      />
                    </div>
                  </div>
                </div>

                {/* Security Section */}
                <div className="query-content-gradient rounded-[32px] p-6">
                  <div className="space-y-4">
                    <h2 className="text-xl font-semibold text-white">Security</h2>
                    <p className="text-gray-400 text-sm">Donec mi odio, faucibus at, scelerisque quis</p>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <Label className="text-gray-400 flex items-center gap-2">
                          <Mail className="w-4 h-4" />
                          Email
                        </Label>
                        <Input 
                          value={user?.email || 'nannie.abernathy70@yahoo.com'} 
                          className="modal-input-enhanced"
                        />
                        <div className="flex items-center gap-1">
                          <a href="#" className="text-gray-400 text-sm hover:underline">Change Email</a>
                          <div className="w-4 h-4 rounded-full bg-gray-400 flex items-center justify-center">
                            <span className="text-xs text-white">i</span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-2">
                        <Label className="text-gray-400 flex items-center gap-2">
                          <Lock className="w-4 h-4" />
                          Password
                        </Label>
                        <Input 
                          type="password"
                          value="••••••••••••" 
                          className="modal-input-enhanced"
                        />
                        <div className="flex items-center gap-1">
                          <a href="#" className="text-gray-400 text-sm hover:underline">Change Password</a>
                          <div className="w-4 h-4 rounded-full bg-gray-400 flex items-center justify-center">
                            <span className="text-xs text-white">i</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Activity Section */}
                <div className="query-content-gradient rounded-[32px] p-6">
                  <div className="space-y-4">
                    <h2 className="text-xl font-semibold text-white">Activity</h2>
                    <p className="text-gray-400 text-sm">Donec mi odio, faucibus at, scelerisque quis</p>
                    
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <span className="text-white">Email me when someone comments on my article</span>
                        <Switch
                          checked={emailNotifications.comments}
                          onCheckedChange={(checked) => setEmailNotifications(prev => ({ ...prev, comments: checked }))}
                          className="data-[state=checked]:bg-emerald-500"
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-white">Email me when someone answers on my form</span>
                        <Switch
                          checked={emailNotifications.answers}
                          onCheckedChange={(checked) => setEmailNotifications(prev => ({ ...prev, answers: checked }))}
                          className="data-[state=checked]:bg-emerald-500"
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-white">Email me when someone follows me</span>
                        <Switch
                          checked={emailNotifications.follows}
                          onCheckedChange={(checked) => setEmailNotifications(prev => ({ ...prev, follows: checked }))}
                          className="data-[state=checked]:bg-emerald-500"
                        />
                      </div>
                    </div>
                  </div>
                </div>

              </>
            )}

            {activeTab !== 'overview' && renderTabContent()}
          </div>
        </div>
      </div>
    </div>
  );
});

UserConfiguration.displayName = 'UserConfiguration';