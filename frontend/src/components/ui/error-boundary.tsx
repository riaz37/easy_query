"use client";
import React from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { AlertTriangle, RefreshCw, Home } from "lucide-react";
import Link from "next/link";

interface ErrorBoundaryProps {
  children: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen w-full flex items-center justify-center p-4">
          <Card className="w-full max-w-2xl overflow-hidden relative backdrop-blur-xl bg-white/10 dark:bg-black/10 border-white/20 dark:border-white/10 shadow-2xl">
            {/* Glass effect overlay */}
            <div className="absolute inset-0 bg-gradient-to-br from-white/20 via-white/5 to-transparent dark:from-white/10 dark:via-white/5 dark:to-transparent" />
            {/* Error glow effect */}
            <div className="absolute -inset-1 bg-gradient-to-r from-destructive/20 to-destructive/10 rounded-lg blur-lg opacity-30" />
            
            <CardContent className="relative p-8 text-center">
              <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-gradient-to-br from-destructive/20 to-destructive/10 backdrop-blur-sm flex items-center justify-center border border-white/20 shadow-lg">
                <AlertTriangle className="w-8 h-8 text-destructive" />
              </div>
              
              <h1 className="text-2xl font-bold text-foreground mb-2">
                Something went wrong
              </h1>
              
              <p className="text-muted-foreground mb-6 max-w-md mx-auto">
                We encountered an unexpected error. Don't worry, our team has been notified and we're working on a fix.
              </p>
              
              {process.env.NODE_ENV === 'development' && this.state.error && (
                <div className="mb-6 p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-left">
                  <p className="text-sm font-mono text-destructive">
                    {this.state.error.message}
                  </p>
                </div>
              )}
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button 
                  onClick={() => window.location.reload()}
                  className="gap-2 bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70"
                >
                  <RefreshCw className="w-4 h-4" />
                  Try Again
                </Button>
                
                <Link href="/">
                  <Button 
                    variant="outline"
                    className="gap-2 backdrop-blur-sm bg-white/10 dark:bg-black/10 border-white/20 hover:bg-white/20 dark:hover:bg-black/20"
                  >
                    <Home className="w-4 h-4" />
                    Go Home
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;