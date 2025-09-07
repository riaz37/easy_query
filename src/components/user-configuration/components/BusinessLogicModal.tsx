"use client";

import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { ArrowRight, Edit, X, Plus } from 'lucide-react';

interface BusinessLogicModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave?: (data: BusinessLogicData) => void;
  initialData?: BusinessLogicData;
}

interface BusinessLogicData {
  title: string;
  headline: string;
  body: string;
}

export const BusinessLogicModal = React.memo<BusinessLogicModalProps>(({
  isOpen,
  onClose,
  onSave,
  initialData
}) => {
  const [title, setTitle] = useState(initialData?.title || '');
  const [headline, setHeadline] = useState(initialData?.headline || '');
  const [body, setBody] = useState(initialData?.body || '');
  const [isEditing, setIsEditing] = useState(false);

  const handleSave = () => {
    if (onSave) {
      onSave({ title, headline, body });
    }
    onClose();
  };

  const handleCancel = () => {
    setTitle(initialData?.title || '');
    setHeadline(initialData?.headline || '');
    setBody(initialData?.body || '');
    setIsEditing(false);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="relative w-full max-w-4xl mx-4 bg-gradient-to-br from-slate-800/90 to-slate-900/90 rounded-2xl shadow-2xl shadow-emerald-500/20 backdrop-blur-xl border-l-4 border-emerald-500">
        {/* Left corner borders */}
        <div className="absolute top-0 left-0 w-6 h-6 border-l-4 border-t-4 border-emerald-500 rounded-tl-2xl"></div>
        <div className="absolute bottom-0 left-0 w-6 h-6 border-l-4 border-b-4 border-emerald-500 rounded-bl-2xl"></div>
        
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-2 text-gray-400 hover:text-white transition-colors"
        >
          <X className="h-5 w-5" />
        </button>

        <div className="p-8 space-y-8">
          {/* Top Card - Understanding Business Logic */}
          <div className="relative bg-slate-800/50 rounded-xl border-l-4 border-emerald-500 p-6">
            {/* Left corner borders for top card */}
            <div className="absolute top-0 left-0 w-6 h-6 border-l-4 border-t-4 border-emerald-500 rounded-tl-xl"></div>
            <div className="absolute bottom-0 left-0 w-6 h-6 border-l-4 border-b-4 border-emerald-500 rounded-bl-xl"></div>
            
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-emerald-500/20 rounded-full flex items-center justify-center">
                  <ArrowRight className="h-4 w-4 text-emerald-400" />
                </div>
                <span className="text-emerald-400 font-medium">Logic Name or title</span>
              </div>
              <Button
                onClick={() => setIsEditing(!isEditing)}
                variant="outline"
                size="sm"
                className="border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/10"
              >
                <Edit className="h-4 w-4 mr-2" />
                Edit
              </Button>
            </div>

            <h1 className="text-3xl font-bold text-white mb-6">
              Understand What Business Logic Is
            </h1>

            <div className="space-y-4 text-gray-300">
              <p className="text-lg leading-relaxed">
                Business logic refers to the rules and operations that define how data is created, transformed, and used to meet business objectives.
              </p>

              <div className="space-y-3">
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-emerald-400 rounded-full mt-2 flex-shrink-0" />
                  <p>
                    It's distinct from UI/UX and data storage—it's the "thinking" part of your system.
                  </p>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-emerald-400 rounded-full mt-2 flex-shrink-0" />
                  <p>
                    Examples include pricing rules, user permissions, inventory calculations, or approval workflows.
                  </p>
                </div>
              </div>
            </div>

            {/* Key Principles Section */}
            <div className="mt-8">
              <div className="flex items-center gap-2 mb-4">
                <h2 className="text-xl font-semibold text-white">
                  Key Principles for Designing Business Logic
                </h2>
                <div className="w-2 h-2 bg-red-400 rounded-full" />
              </div>

              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium text-emerald-400 mb-2">
                    1. Keep It Modular:
                  </h3>
                  <div className="space-y-2 text-gray-300">
                    <div className="flex items-start gap-3">
                      <div className="w-2 h-2 bg-emerald-400 rounded-full mt-2 flex-shrink-0" />
                      <p>Break logic into small, reusable components.</p>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-2 h-2 bg-emerald-400 rounded-full mt-2 flex-shrink-0" />
                      <p>Avoid monolithic code—use services, functions, or classes that encapsulate specific rules.</p>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-medium text-emerald-400 mb-2">
                    2. Separate Concerns:
                  </h3>
                  <div className="space-y-2 text-gray-300">
                    <div className="flex items-start gap-3">
                      <div className="w-2 h-2 bg-emerald-400 rounded-full mt-2 flex-shrink-0" />
                      <p>Keep business logic separate from presentation and data layers.</p>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-2 h-2 bg-emerald-400 rounded-full mt-2 flex-shrink-0" />
                      <p>Use design patterns like MVC (Model-View-Controller) or clean architecture to enforce this.</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Card - Input Form */}
          <div className="relative bg-slate-800/50 rounded-xl border-l-4 border-emerald-500 p-6">
            {/* Left corner borders for bottom card */}
            <div className="absolute top-0 left-0 w-6 h-6 border-l-4 border-t-4 border-emerald-500 rounded-tl-xl"></div>
            <div className="absolute bottom-0 left-0 w-6 h-6 border-l-4 border-b-4 border-emerald-500 rounded-bl-xl"></div>
            
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-6 h-6 bg-emerald-500/20 rounded-full flex items-center justify-center">
                  <ArrowRight className="h-3 w-3 text-emerald-400" />
                </div>
                <Label className="text-emerald-400 font-medium">Write a title here</Label>
              </div>
              <div className="flex gap-2">
                <Button
                  onClick={handleCancel}
                  variant="outline"
                  size="sm"
                  className="border-slate-600 text-gray-300 hover:bg-slate-700"
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleSave}
                  size="sm"
                  className="bg-emerald-600 hover:bg-emerald-700 text-white"
                >
                  Save
                </Button>
              </div>
            </div>

            <h3 className="text-xl font-semibold text-white mb-4">
              Logic Headline Write Here
            </h3>
            <Textarea
              value={body}
              onChange={(e) => setBody(e.target.value)}
              placeholder="Start writing body here"
              className="bg-slate-700/50 border-slate-600 text-white placeholder-gray-400 min-h-[120px] resize-none mb-4"
            />
            <div className="flex gap-3">
              <Button
                onClick={handleCancel}
                variant="outline"
                className="border-slate-600 text-gray-300 hover:bg-slate-700"
              >
                Cancel
              </Button>
              <Button
                onClick={handleSave}
                className="bg-emerald-600 hover:bg-emerald-700 text-white"
              >
                <Plus className="h-4 w-4 mr-2" />
                Add New
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

BusinessLogicModal.displayName = 'BusinessLogicModal';
