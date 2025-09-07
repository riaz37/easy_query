"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Spinner, 
  PulseLoader, 
  DotsLoader, 
  WaveLoader, 
  OrbitLoader,
  ProgressLoader,
  SkeletonLoader,
  CardSkeleton,
  TableSkeleton,
  ButtonLoader,
  PageLoader,
  OverlayLoader,
  InlineLoader,
  LoadingStates,
  GradientLoader,
  ESAPLoader,
  ESAPBrandLoader,
  EmeraldLoader
} from '@/components/ui/loading';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  Settings, 
  Eye,
  Code,
  Copy,
  Check
} from 'lucide-react';
import { toast } from 'sonner';

export default function LoadingDemoPage() {
  const [progress, setProgress] = useState(0);
  const [showOverlay, setShowOverlay] = useState(false);
  const [showPageLoader, setShowPageLoader] = useState(false);
  const [selectedVariant, setSelectedVariant] = useState<'primary' | 'primary-dark' | 'primary-light' | 'accent-blue' | 'accent-purple' | 'accent-orange' | 'secondary' | 'success' | 'warning' | 'error' | 'info'>('primary');
  const [selectedSize, setSelectedSize] = useState<'xs' | 'sm' | 'md' | 'lg' | 'xl'>('md');
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  // Simulate progress
  useEffect(() => {
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) return 0;
        return prev + 1;
      });
    }, 100);

    return () => clearInterval(interval);
  }, []);

  const copyToClipboard = (code: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(code);
    toast.success('Code copied to clipboard!');
    setTimeout(() => setCopiedCode(null), 2000);
  };

  const variants = ['primary', 'primary-dark', 'primary-light', 'accent-blue', 'accent-purple', 'accent-orange', 'secondary', 'success', 'warning', 'error', 'info'] as const;
  const sizes = ['xs', 'sm', 'md', 'lg', 'xl'] as const;
  const loaderTypes = ['spinner', 'pulse', 'dots', 'wave', 'orbit'] as const;

  const generateCode = (component: string, props: Record<string, any> = {}) => {
    const propString = Object.entries(props)
      .filter(([_, value]) => value !== undefined && value !== 'md' && value !== 'primary')
      .map(([key, value]) => `${key}="${value}"`)
      .join(' ');
    
    return `<${component}${propString ? ` ${propString}` : ''} />`;
  };

  const LoadingDemoCard = ({ 
    title, 
    description, 
    children, 
    code 
  }: { 
    title: string; 
    description: string; 
    children: React.ReactNode; 
    code: string;
  }) => (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg">{title}</CardTitle>
            <p className="text-sm text-muted-foreground mt-1">{description}</p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => copyToClipboard(code)}
            className="ml-4"
          >
            {copiedCode === code ? (
              <Check className="w-4 h-4" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-center p-8 bg-gray-50 dark:bg-gray-900 rounded-lg">
          {children}
        </div>
        <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-800 rounded text-sm font-mono">
          {code}
        </div>
      </CardContent>
    </Card>
  );

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-[#10b981] via-[#3b82f6] to-[#8b5cf6] bg-clip-text text-transparent">
            ESAP Loading Components
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            A comprehensive collection of beautiful, animated loading components using the ESAP brand color scheme
          </p>
          <div className="flex justify-center items-center gap-4 mt-6">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-[#10b981]"></div>
              <span className="text-sm text-muted-foreground">Primary Emerald</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-[#3b82f6]"></div>
              <span className="text-sm text-muted-foreground">Accent Blue</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-[#8b5cf6]"></div>
              <span className="text-sm text-muted-foreground">Accent Purple</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-[#f59e0b]"></div>
              <span className="text-sm text-muted-foreground">Accent Orange</span>
            </div>
          </div>
        </div>

        {/* Controls */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="w-5 h-5" />
              Demo Controls
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="text-sm font-medium mb-2 block">Variant</label>
                <div className="flex flex-wrap gap-2">
                  {variants.map(variant => (
                    <Button
                      key={variant}
                      variant={selectedVariant === variant ? "default" : "outline"}
                      size="sm"
                      onClick={() => setSelectedVariant(variant)}
                    >
                      {variant}
                    </Button>
                  ))}
                </div>
              </div>
              <div>
                <label className="text-sm font-medium mb-2 block">Size</label>
                <div className="flex flex-wrap gap-2">
                  {sizes.map(size => (
                    <Button
                      key={size}
                      variant={selectedSize === size ? "default" : "outline"}
                      size="sm"
                      onClick={() => setSelectedSize(size)}
                    >
                      {size}
                    </Button>
                  ))}
                </div>
              </div>
              <div>
                <label className="text-sm font-medium mb-2 block">Actions</label>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowOverlay(!showOverlay)}
                  >
                    {showOverlay ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                    Overlay
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowPageLoader(!showPageLoader)}
                  >
                    <Eye className="w-4 h-4" />
                    Page
                  </Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Main Demo Tabs */}
        <Tabs defaultValue="basic" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="basic">Basic Loaders</TabsTrigger>
            <TabsTrigger value="progress">Progress & Skeletons</TabsTrigger>
            <TabsTrigger value="components">Component Loaders</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
          </TabsList>

          {/* Basic Loaders */}
          <TabsContent value="basic" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <LoadingDemoCard
                title="Spinner"
                description="Classic rotating spinner"
                code={generateCode('Spinner', { size: selectedSize, variant: selectedVariant })}
              >
                <Spinner size={selectedSize} variant={selectedVariant} />
              </LoadingDemoCard>

              <LoadingDemoCard
                title="Pulse Loader"
                description="Three pulsing dots"
                code={generateCode('PulseLoader', { size: selectedSize, variant: selectedVariant })}
              >
                <PulseLoader size={selectedSize} variant={selectedVariant} />
              </LoadingDemoCard>

              <LoadingDemoCard
                title="Dots Loader"
                description="Bouncing dots animation"
                code={generateCode('DotsLoader', { size: selectedSize, variant: selectedVariant })}
              >
                <DotsLoader size={selectedSize} variant={selectedVariant} />
              </LoadingDemoCard>

              <LoadingDemoCard
                title="Wave Loader"
                description="Wave-like bar animation"
                code={generateCode('WaveLoader', { size: selectedSize, variant: selectedVariant })}
              >
                <WaveLoader size={selectedSize} variant={selectedVariant} />
              </LoadingDemoCard>

              <LoadingDemoCard
                title="Orbit Loader"
                description="Concentric rotating rings"
                code={generateCode('OrbitLoader', { size: selectedSize, variant: selectedVariant })}
              >
                <OrbitLoader size={selectedSize} variant={selectedVariant} />
              </LoadingDemoCard>

              <LoadingDemoCard
                title="Loading States"
                description="Unified loader component"
                code={generateCode('LoadingStates', { type: 'orbit', size: selectedSize, variant: selectedVariant })}
              >
                <LoadingStates type="orbit" size={selectedSize} variant={selectedVariant} />
              </LoadingDemoCard>

              <LoadingDemoCard
                title="Emerald Loader"
                description="Clean emerald-focused spinner"
                code={generateCode('EmeraldLoader', { size: selectedSize })}
              >
                <EmeraldLoader size={selectedSize} />
              </LoadingDemoCard>

              <LoadingDemoCard
                title="Gradient Loader"
                description="Elegant emerald gradient animation"
                code={generateCode('GradientLoader', { size: selectedSize })}
              >
                <GradientLoader size={selectedSize} />
              </LoadingDemoCard>

              <LoadingDemoCard
                title="ESAP Loader"
                description="Simple logo with emerald rings"
                code={generateCode('ESAPLoader', { size: selectedSize })}
              >
                <ESAPLoader size={selectedSize} />
              </LoadingDemoCard>

              <LoadingDemoCard
                title="ESAP Brand Loader"
                description="Premium branded loader with particles"
                code={generateCode('ESAPBrandLoader', { size: selectedSize })}
              >
                <ESAPBrandLoader size={selectedSize} />
              </LoadingDemoCard>
            </div>
          </TabsContent>

          {/* Progress & Skeletons */}
          <TabsContent value="progress" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <LoadingDemoCard
                title="Progress Loader"
                description="Animated progress bar with percentage"
                code={generateCode('ProgressLoader', { size: selectedSize, variant: selectedVariant, progress: progress, showPercentage: true })}
              >
                <div className="w-full">
                  <ProgressLoader 
                    size={selectedSize} 
                    variant={selectedVariant} 
                    progress={progress}
                    showPercentage={true}
                  />
                </div>
              </LoadingDemoCard>

              <LoadingDemoCard
                title="Skeleton Loader"
                description="Content placeholder animation"
                code={generateCode('SkeletonLoader', { size: selectedSize, lines: 3 })}
              >
                <SkeletonLoader size={selectedSize} lines={3} />
              </LoadingDemoCard>

              <LoadingDemoCard
                title="Card Skeleton"
                description="Complete card placeholder"
                code={generateCode('CardSkeleton', { size: selectedSize })}
              >
                <CardSkeleton size={selectedSize} />
              </LoadingDemoCard>

              <LoadingDemoCard
                title="Table Skeleton"
                description="Table structure placeholder"
                code={generateCode('TableSkeleton', { size: selectedSize, rows: 4, columns: 3 })}
              >
                <TableSkeleton size={selectedSize} rows={4} columns={3} />
              </LoadingDemoCard>
            </div>
          </TabsContent>

          {/* Component Loaders */}
          <TabsContent value="components" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <LoadingDemoCard
                title="Button Loader"
                description="Loading state for buttons"
                code={generateCode('ButtonLoader', { size: selectedSize, variant: selectedVariant, loading: true, text: 'Processing...' })}
              >
                <div className="space-y-4">
                  <ButtonLoader 
                    size={selectedSize} 
                    variant={selectedVariant} 
                    loading={true}
                    text="Processing..."
                  >
                    Submit
                  </ButtonLoader>
                  <ButtonLoader 
                    size={selectedSize} 
                    variant={selectedVariant} 
                    loading={false}
                  >
                    Submit
                  </ButtonLoader>
                </div>
              </LoadingDemoCard>

              <LoadingDemoCard
                title="Inline Loader"
                description="Small inline loading indicator"
                code={generateCode('InlineLoader', { size: selectedSize, variant: selectedVariant })}
              >
                <InlineLoader size={selectedSize} variant={selectedVariant}>
                  Loading data...
                </InlineLoader>
              </LoadingDemoCard>
            </div>
          </TabsContent>

          {/* Advanced */}
          <TabsContent value="advanced" className="space-y-6">
            <div className="grid grid-cols-1 gap-6">
              <LoadingDemoCard
                title="Page Loader"
                description="Full page loading experience"
                code={generateCode('PageLoader', { size: selectedSize, variant: selectedVariant, message: 'Loading your dashboard...', description: 'Please wait while we prepare everything for you' })}
              >
                <div className="relative h-64 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg">
                  {showPageLoader ? (
                    <div className="absolute inset-0 flex items-center justify-center bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm rounded-lg">
                      <div className="text-center space-y-4">
                        <OrbitLoader size="lg" variant={selectedVariant} />
                        <div>
                          <h3 className="font-semibold">Loading your dashboard...</h3>
                          <p className="text-sm text-muted-foreground">Please wait while we prepare everything for you</p>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center h-full text-muted-foreground">
                      Click "Page" button to see page loader
                    </div>
                  )}
                </div>
              </LoadingDemoCard>

              <LoadingDemoCard
                title="Overlay Loader"
                description="Modal-style loading overlay"
                code={generateCode('OverlayLoader', { size: selectedSize, variant: selectedVariant, visible: true, message: 'Processing your request...' })}
              >
                <div className="relative h-64 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg">
                  {showOverlay ? (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm rounded-lg">
                      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6">
                        <div className="text-center space-y-4">
                          <OrbitLoader size="lg" variant={selectedVariant} />
                          <p className="font-medium">Processing your request...</p>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center h-full text-muted-foreground">
                      Click "Overlay" button to see overlay loader
                    </div>
                  )}
                </div>
              </LoadingDemoCard>
            </div>
          </TabsContent>
        </Tabs>

        {/* Usage Examples */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Code className="w-5 h-5" />
              Usage Examples
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h4 className="font-semibold mb-2">Basic Import</h4>
              <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded text-sm font-mono">
                {`import { Spinner, PulseLoader, DotsLoader } from '@/components/ui/loading';`}
              </div>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">In a Component</h4>
              <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded text-sm font-mono">
                {`{isLoading ? (
  <Spinner size="md" variant="primary" />
) : (
  <Button>Submit</Button>
)}`}
              </div>
            </div>

            <div>
              <h4 className="font-semibold mb-2">With Loading Provider</h4>
              <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded text-sm font-mono">
                {`import { LoadingProvider, useLoading } from '@/components/ui/loading';

function App() {
  return (
    <LoadingProvider>
      <YourComponent />
    </LoadingProvider>
  );
}`}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Variant Showcase */}
        <Card>
          <CardHeader>
            <CardTitle>All Variants</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              {variants.map(variant => (
                <div key={variant} className="text-center space-y-2">
                  <Badge variant="outline" className="w-full justify-center">
                    {variant}
                  </Badge>
                  <div className="flex justify-center">
                    <Spinner size="md" variant={variant} />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Size Showcase */}
        <Card>
          <CardHeader>
            <CardTitle>All Sizes</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center gap-8">
              {sizes.map(size => (
                <div key={size} className="text-center space-y-2">
                  <Badge variant="outline">{size}</Badge>
                  <div className="flex justify-center">
                    <Spinner size={size} variant="primary" />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Overlay Loader */}
      <OverlayLoader
        visible={showOverlay}
        size="lg"
        variant={selectedVariant}
        message="Processing your request..."
        backdrop={true}
      />

      {/* Page Loader */}
      {showPageLoader && (
        <div className="fixed inset-0 z-50">
          <PageLoader
            size="lg"
            variant={selectedVariant}
            message="Loading your dashboard..."
            description="Please wait while we prepare everything for you"
            showProgress={true}
            progress={progress}
          />
        </div>
      )}
    </div>
  );
}
