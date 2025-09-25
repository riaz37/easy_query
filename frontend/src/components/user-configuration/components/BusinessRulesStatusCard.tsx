import React, { useState } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import {
  Shield,
  CheckCircle,
  AlertCircle,
  RefreshCw,
  Edit3,
  Plus,
} from 'lucide-react';
import { BusinessLogicModal } from './BusinessLogicModal';
import type { BusinessRulesStatusCardProps } from '../types';

export const BusinessRulesStatusCard = React.memo<BusinessRulesStatusCardProps>(({
  currentDatabaseName,
  businessRules,
  businessRulesCount,
  hasBusinessRules,
  editorState,
  onRefresh,
  onEdit,
  onSave,
  onCancel,
  onReset,
}) => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleModalSave = (data: any) => {
    // Handle saving the business logic data
    console.log('Saving business logic:', data);
  };

  return (
    <>
      <div className="card-enhanced">
        <div className="card-content-enhanced">
          <div className="card-header-enhanced">
            <div className="card-title-enhanced flex items-center gap-2">
              <Shield className="h-5 w-5 text-emerald-400" />
              Business Rules Status
            </div>
            <p className="card-description-enhanced">
              Current business rules configuration for{' '}
              {currentDatabaseName || 'selected database'}
            </p>
          </div>
          <div className="mt-4 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <Label className="text-gray-400">Status</Label>
                <div className="flex items-center gap-2 mt-1">
                  {businessRules.status === 'loaded' && hasBusinessRules ? (
                    <Badge
                      variant="outline"
                      className="text-green-400 border-green-400"
                    >
                      <CheckCircle className="w-3 h-3 mr-1" />
                      Active
                    </Badge>
                  ) : (
                    <Badge
                      variant="outline"
                      className="text-yellow-400 border-yellow-400"
                    >
                      <AlertCircle className="w-3 h-3 mr-1" />
                      Not Configured
                    </Badge>
                  )}
                </div>
              </div>

              <div>
                <Label className="text-gray-400">Rules Count</Label>
                <div className="text-white font-medium mt-1">
                  {businessRules.status === 'loaded'
                    ? businessRulesCount
                    : '0'}
                </div>
              </div>

              <div>
                <Label className="text-gray-400">Content Length</Label>
                <div className="text-white font-medium mt-1">
                  {businessRules.content
                    ? `${businessRules.content.length} characters`
                    : '0 characters'}
                </div>
              </div>
            </div>

            <Separator className="bg-slate-600" />

            <div className="flex items-center justify-between">
              <div className="text-sm text-gray-400">
                Last updated:{' '}
                {businessRules.lastUpdated
                  ? new Date(businessRules.lastUpdated).toLocaleString()
                  : 'Never'}
              </div>

              <div className="flex gap-2">
                <Button
                  onClick={() => setIsModalOpen(true)}
                  className="card-button-enhanced"
                >
                  <Plus className="w-4 h-4 mr-2" />
                  Business Logic
                </Button>

                <Button
                  variant="outline"
                  onClick={onRefresh}
                  disabled={businessRules.status === 'loading'}
                  className="border-slate-600 text-slate-300 hover:bg-slate-700"
                >
                  <RefreshCw
                    className={`w-4 h-4 mr-2 ${businessRules.status === 'loading' ? 'animate-spin' : ''}`}
                  />
                  Refresh Status
                </Button>

                {!editorState.isEditing ? (
                  <Button
                    onClick={onEdit}
                    className="bg-emerald-600 hover:bg-emerald-700 text-white"
                  >
                    <Edit3 className="w-4 h-4 mr-2" />
                    Edit Rules
                  </Button>
                ) : (
                  <div className="flex gap-2">
                    <Button
                      onClick={onSave}
                      disabled={!editorState.hasUnsavedChanges}
                      className="modal-button-primary"
                    >
                      Save
                    </Button>
                    <Button
                      onClick={onCancel}
                      className="modal-button-secondary"
                    >
                      Cancel
                    </Button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      <BusinessLogicModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSave={handleModalSave}
      />
    </>
  );
});

BusinessRulesStatusCard.displayName = 'BusinessRulesStatusCard';
