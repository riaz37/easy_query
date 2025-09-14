"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle 
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from "@/components/ui/select";
import { 
  Database, 
  Brain, 
  Building2, 
  User, 
  Settings, 
  Palette,
  Eye,
  Code,
  Sparkles,
  XIcon
} from "lucide-react";

export default function ModalDemoPage() {
  const [isEnhancedModalOpen, setIsEnhancedModalOpen] = useState(false);
  const [isDatabaseModalOpen, setIsDatabaseModalOpen] = useState(false);
  const [isVectorModalOpen, setIsVectorModalOpen] = useState(false);
  const [isFormModalOpen, setIsFormModalOpen] = useState(false);

  // Form state for demo
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    company: "",
    role: "",
    description: ""
  });

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const resetForm = () => {
    setFormData({
      name: "",
      email: "",
      company: "",
      role: "",
      description: ""
    });
  };

  const handleSubmit = () => {
    console.log("Form submitted:", formData);
    setIsFormModalOpen(false);
    resetForm();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Sparkles className="h-8 w-8 text-green-400" />
            <h1 className="text-4xl font-bold text-white">Enhanced Modal Showcase</h1>
            <Sparkles className="h-8 w-8 text-green-400" />
          </div>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Experience the power of glass morphism and gradient backgrounds in our enhanced modal system. 
            Featuring corner border effects, backdrop blur, and beautiful transparency.
          </p>
        </div>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Palette className="h-5 w-5 text-green-400" />
                Glass Morphism
              </CardTitle>
              <CardDescription className="text-gray-400">
                Beautiful transparent backgrounds with backdrop blur effects
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Badge variant="secondary" className="bg-green-600/20 text-green-400 border-green-500">
                Enhanced
              </Badge>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Code className="h-5 w-5 text-green-400" />
                Corner Borders
              </CardTitle>
              <CardDescription className="text-gray-400">
                Distinctive green gradient corner border effects
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Badge variant="secondary" className="bg-green-600/20 text-green-400 border-green-500">
                Unique
              </Badge>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Eye className="h-5 w-5 text-green-400" />
                Visual Depth
              </CardTitle>
              <CardDescription className="text-gray-400">
                Multi-layered shadows and enhanced visual hierarchy
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Badge variant="secondary" className="bg-green-600/20 text-green-400 border-green-500">
                Premium
              </Badge>
            </CardContent>
          </Card>
        </div>

        {/* Demo Buttons */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Enhanced Form Modal */}
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm hover:bg-slate-800/70 transition-all duration-300">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <User className="h-5 w-5 text-green-400" />
                User Form
              </CardTitle>
              <CardDescription className="text-gray-400">
                Complete user registration form with enhanced styling
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button 
                onClick={() => setIsFormModalOpen(true)}
                className="w-full bg-green-600 hover:bg-green-700 text-white"
              >
                Open Form Modal
              </Button>
            </CardContent>
          </Card>

          {/* Database Access Modal */}
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm hover:bg-slate-800/70 transition-all duration-300">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Database className="h-5 w-5 text-green-400" />
                Database Access
              </CardTitle>
              <CardDescription className="text-gray-400">
                Database access configuration with company selection
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button 
                onClick={() => setIsDatabaseModalOpen(true)}
                className="w-full bg-green-600 hover:bg-green-700 text-white"
              >
                Open Database Modal
              </Button>
            </CardContent>
          </Card>

          {/* Vector DB Modal */}
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm hover:bg-slate-800/70 transition-all duration-300">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Brain className="h-5 w-5 text-green-400" />
                Vector Database
              </CardTitle>
              <CardDescription className="text-gray-400">
                AI vector database access with table selection
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button 
                onClick={() => setIsVectorModalOpen(true)}
                className="w-full bg-green-600 hover:bg-green-700 text-white"
              >
                Open Vector Modal
              </Button>
            </CardContent>
          </Card>

          {/* Settings Modal */}
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm hover:bg-slate-800/70 transition-all duration-300">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Settings className="h-5 w-5 text-green-400" />
                Settings
              </CardTitle>
              <CardDescription className="text-gray-400">
                System configuration with enhanced modal styling
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button 
                onClick={() => setIsEnhancedModalOpen(true)}
                className="w-full bg-green-600 hover:bg-green-700 text-white"
              >
                Open Settings Modal
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Enhanced Form Modal */}
        <Dialog open={isFormModalOpen} onOpenChange={setIsFormModalOpen}>
          <DialogContent className="max-w-2xl max-h-[90vh] p-0 border-0 bg-transparent" showCloseButton={false}>
            <div className="modal-enhanced">
              <div className="modal-content-enhanced max-h-[90vh] overflow-y-auto">
                <DialogHeader className="modal-header-enhanced">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <DialogTitle className="modal-title-enhanced flex items-center gap-2">
                        <User className="h-5 w-5 text-green-400" />
                        Create User Account
                      </DialogTitle>
                      <p className="modal-description-enhanced">
                        Complete user registration with enhanced glass morphism styling
                      </p>
                    </div>
                    <button
                      onClick={() => setIsFormModalOpen(false)}
                      className="modal-close-button"
                    >
                      <XIcon className="h-5 w-5" />
                    </button>
                  </div>
                </DialogHeader>

                <div className="modal-form-content">
                  {/* Name Field */}
                  <div className="modal-form-group">
                    <Label className="modal-form-label">Full Name *</Label>
                    <Input
                      placeholder="Enter your full name"
                      value={formData.name}
                      onChange={(e) => handleInputChange("name", e.target.value)}
                      className="modal-input-enhanced"
                    />
                    <div className="modal-form-description">
                      Enter your complete name as it should appear in the system
                    </div>
                  </div>

                  {/* Email Field */}
                  <div className="modal-form-group">
                    <Label className="modal-form-label">Email Address *</Label>
                    <Input
                      type="email"
                      placeholder="Enter your email address"
                      value={formData.email}
                      onChange={(e) => handleInputChange("email", e.target.value)}
                      className="modal-input-enhanced"
                    />
                    <div className="modal-form-description">
                      This will be used for login and notifications
                    </div>
                  </div>

                  {/* Company Field */}
                  <div className="modal-form-group">
                    <Label className="modal-form-label flex items-center gap-2">
                      <Building2 className="w-4 h-4" />
                      Company
                    </Label>
                    <Input
                      placeholder="Enter company name"
                      value={formData.company}
                      onChange={(e) => handleInputChange("company", e.target.value)}
                      className="modal-input-enhanced"
                    />
                    <div className="modal-form-description">
                      Optional: Your company or organization name
                    </div>
                  </div>

                  {/* Role Selection */}
                  <div className="modal-form-group">
                    <Label className="modal-form-label">Role</Label>
                    <Select value={formData.role} onValueChange={(value) => handleInputChange("role", value)}>
                      <SelectTrigger className="modal-select-enhanced">
                        <SelectValue placeholder="Select your role" />
                      </SelectTrigger>
                      <SelectContent className="modal-select-content-enhanced">
                        <SelectItem value="admin">Administrator</SelectItem>
                        <SelectItem value="manager">Manager</SelectItem>
                        <SelectItem value="user">User</SelectItem>
                        <SelectItem value="guest">Guest</SelectItem>
                      </SelectContent>
                    </Select>
                    <div className="modal-form-description">
                      Choose the role that best describes your position
                    </div>
                  </div>

                  {/* Description Field */}
                  <div className="modal-form-group">
                    <Label className="modal-form-label">Description</Label>
                    <Textarea
                      placeholder="Tell us about yourself..."
                      value={formData.description}
                      onChange={(e) => handleInputChange("description", e.target.value)}
                      className="modal-textarea-enhanced"
                      rows={4}
                    />
                    <div className="modal-form-description">
                      Optional: Brief description about yourself or your role
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="modal-footer-enhanced">
                  <Button
                    variant="outline"
                    onClick={() => setIsFormModalOpen(false)}
                    className="modal-button-secondary"
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={handleSubmit}
                    className="modal-button-primary"
                  >
                    Create Account
                  </Button>
                </div>
              </div>
            </div>
          </DialogContent>
        </Dialog>

        {/* Database Access Modal */}
        <Dialog open={isDatabaseModalOpen} onOpenChange={setIsDatabaseModalOpen}>
          <DialogContent className="max-w-4xl max-h-[90vh] p-0 border-0 bg-transparent" showCloseButton={false}>
            <div className="modal-enhanced">
              <div className="modal-content-enhanced max-h-[90vh] overflow-y-auto">
                <DialogHeader className="modal-header-enhanced">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <DialogTitle className="modal-title-enhanced flex items-center gap-2">
                        <Database className="h-5 w-5 text-green-400" />
                        Database Access Configuration
                      </DialogTitle>
                      <p className="modal-description-enhanced">
                        Configure user access to MSSQL databases with enhanced glass styling
                      </p>
                    </div>
                    <button
                      onClick={() => setIsDatabaseModalOpen(false)}
                      className="modal-close-button"
                    >
                      <XIcon className="h-5 w-5" />
                    </button>
                  </div>
                </DialogHeader>

                <div className="modal-form-content">
                  <div className="modal-form-group">
                    <Label className="modal-form-label">User ID *</Label>
                    <Input
                      placeholder="Enter user ID (email)"
                      className="modal-input-enhanced"
                    />
                    <div className="modal-form-description">
                      Enter the email address of the user you want to grant database access to
                    </div>
                  </div>

                  <div className="modal-form-group">
                    <Label className="modal-form-label flex items-center gap-2">
                      <Building2 className="w-4 h-4" />
                      Parent Company *
                    </Label>
                    <Select>
                      <SelectTrigger className="modal-select-enhanced">
                        <SelectValue placeholder="Select parent company" />
                      </SelectTrigger>
                      <SelectContent className="modal-select-content-enhanced">
                        <SelectItem value="1">Acme Corporation</SelectItem>
                        <SelectItem value="2">Tech Solutions Inc</SelectItem>
                        <SelectItem value="3">Global Systems Ltd</SelectItem>
                      </SelectContent>
                    </Select>
                    <div className="modal-form-description">
                      Select the parent company for this database access
                    </div>
                  </div>

                  <div className="modal-form-group">
                    <Label className="modal-form-label flex items-center gap-2">
                      <Database className="w-4 h-4" />
                      Database
                    </Label>
                    <div className="p-3 modal-input-enhanced rounded-lg">
                      <div className="text-white font-medium">
                        Database 1 - Production
                      </div>
                      <div className="modal-form-description mt-1">
                        Auto-selected based on company choice
                      </div>
                    </div>
                  </div>
                </div>

                <div className="modal-footer-enhanced">
                  <Button
                    variant="outline"
                    onClick={() => setIsDatabaseModalOpen(false)}
                    className="modal-button-secondary"
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={() => setIsDatabaseModalOpen(false)}
                    className="modal-button-primary"
                  >
                    Create Access
                  </Button>
                </div>
              </div>
            </div>
          </DialogContent>
        </Dialog>

        {/* Vector Database Modal */}
        <Dialog open={isVectorModalOpen} onOpenChange={setIsVectorModalOpen}>
          <DialogContent className="max-w-4xl max-h-[90vh] p-0 border-0 bg-transparent" showCloseButton={false}>
            <div className="modal-enhanced">
              <div className="modal-content-enhanced max-h-[90vh] overflow-y-auto">
                <DialogHeader className="modal-header-enhanced">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <DialogTitle className="modal-title-enhanced flex items-center gap-2">
                        <Brain className="h-5 w-5 text-green-400" />
                        Vector Database Access
                      </DialogTitle>
                      <p className="modal-description-enhanced">
                        Configure AI vector database access with table selection
                      </p>
                    </div>
                    <button
                      onClick={() => setIsVectorModalOpen(false)}
                      className="modal-close-button"
                    >
                      <XIcon className="h-5 w-5" />
                    </button>
                  </div>
                </DialogHeader>

                <div className="modal-form-content">
                  <div className="modal-form-group">
                    <Label className="modal-form-label">User ID *</Label>
                    <Input
                      placeholder="Enter user ID (email)"
                      className="modal-input-enhanced"
                    />
                    <div className="modal-form-description">
                      Enter the email address for vector database access
                    </div>
                  </div>

                  <div className="modal-form-group">
                    <Label className="modal-form-label flex items-center gap-2">
                      <Database className="w-4 h-4" />
                      Database
                    </Label>
                    <Select>
                      <SelectTrigger className="modal-select-enhanced">
                        <SelectValue placeholder="Select database" />
                      </SelectTrigger>
                      <SelectContent className="modal-select-content-enhanced">
                        <SelectItem value="1">Vector DB - AI Operations</SelectItem>
                        <SelectItem value="2">Vector DB - ML Models</SelectItem>
                        <SelectItem value="3">Vector DB - Analytics</SelectItem>
                      </SelectContent>
                    </Select>
                    <div className="modal-form-description">
                      Choose the vector database for AI operations
                    </div>
                  </div>

                  <div className="modal-form-group">
                    <Label className="modal-form-label flex items-center gap-2">
                      <Brain className="w-4 h-4" />
                      Tables
                    </Label>
                    <div className="space-y-3">
                      <div className="flex gap-2">
                        <Input
                          placeholder="Enter table name"
                          className="modal-input-enhanced flex-1"
                        />
                        <Button variant="outline" size="sm" className="modal-button-secondary">
                          Add Table
                        </Button>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <Badge variant="secondary" className="bg-green-600/20 text-green-400 border-green-500">
                          embeddings
                        </Badge>
                        <Badge variant="secondary" className="bg-green-600/20 text-green-400 border-green-500">
                          vectors
                        </Badge>
                        <Badge variant="secondary" className="bg-green-600/20 text-green-400 border-green-500">
                          metadata
                        </Badge>
                      </div>
                    </div>
                    <div className="modal-form-description">
                      Add table names for vector database access
                    </div>
                  </div>
                </div>

                <div className="modal-footer-enhanced">
                  <Button
                    variant="outline"
                    onClick={() => setIsVectorModalOpen(false)}
                    className="modal-button-secondary"
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={() => setIsVectorModalOpen(false)}
                    className="modal-button-primary"
                  >
                    Create Vector Access
                  </Button>
                </div>
              </div>
            </div>
          </DialogContent>
        </Dialog>

        {/* Enhanced Settings Modal */}
        <Dialog open={isEnhancedModalOpen} onOpenChange={setIsEnhancedModalOpen}>
          <DialogContent className="max-w-2xl max-h-[90vh] p-0 border-0 bg-transparent" showCloseButton={false}>
            <div className="modal-enhanced">
              <div className="modal-content-enhanced max-h-[90vh] overflow-y-auto">
                <DialogHeader className="modal-header-enhanced">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <DialogTitle className="modal-title-enhanced flex items-center gap-2">
                        <Settings className="h-5 w-5 text-green-400" />
                        System Settings
                      </DialogTitle>
                      <p className="modal-description-enhanced">
                        Configure system preferences with enhanced glass morphism
                      </p>
                    </div>
                    <button
                      onClick={() => setIsEnhancedModalOpen(false)}
                      className="modal-close-button"
                    >
                      <XIcon className="h-5 w-5" />
                    </button>
                  </div>
                </DialogHeader>

                <div className="modal-form-content">
                  <div className="modal-form-group">
                    <Label className="modal-form-label">Theme</Label>
                    <Select>
                      <SelectTrigger className="modal-select-enhanced">
                        <SelectValue placeholder="Select theme" />
                      </SelectTrigger>
                      <SelectContent className="modal-select-content-enhanced">
                        <SelectItem value="dark">Dark Mode</SelectItem>
                        <SelectItem value="light">Light Mode</SelectItem>
                        <SelectItem value="auto">Auto</SelectItem>
                      </SelectContent>
                    </Select>
                    <div className="modal-form-description">
                      Choose your preferred color theme
                    </div>
                  </div>

                  <div className="modal-form-group">
                    <Label className="modal-form-label">Language</Label>
                    <Select>
                      <SelectTrigger className="modal-select-enhanced">
                        <SelectValue placeholder="Select language" />
                      </SelectTrigger>
                      <SelectContent className="modal-select-content-enhanced">
                        <SelectItem value="en">English</SelectItem>
                        <SelectItem value="es">Spanish</SelectItem>
                        <SelectItem value="fr">French</SelectItem>
                        <SelectItem value="de">German</SelectItem>
                      </SelectContent>
                    </Select>
                    <div className="modal-form-description">
                      Select your preferred language
                    </div>
                  </div>

                  <div className="modal-form-group">
                    <Label className="modal-form-label">Notifications</Label>
                    <Textarea
                      placeholder="Configure notification settings..."
                      className="modal-textarea-enhanced"
                      rows={3}
                    />
                    <div className="modal-form-description">
                      Set up your notification preferences
                    </div>
                  </div>
                </div>

                <div className="modal-footer-enhanced">
                  <Button
                    variant="outline"
                    onClick={() => setIsEnhancedModalOpen(false)}
                    className="modal-button-secondary"
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={() => setIsEnhancedModalOpen(false)}
                    className="modal-button-primary"
                  >
                    Save Settings
                  </Button>
                </div>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}
