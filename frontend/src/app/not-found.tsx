"use client";
import React from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Home, ArrowLeft } from "lucide-react";

const NotFoundPage = () => {
  return (
    <div className="min-h-screen w-full flex items-center justify-center p-4">
      <Card className="w-full max-w-4xl overflow-hidden relative backdrop-blur-xl bg-white/10 dark:bg-black/10 border-white/20 dark:border-white/10 shadow-2xl">
        {/* Glass effect overlay */}
        <div className="absolute inset-0 bg-gradient-to-br from-white/20 via-white/5 to-transparent dark:from-white/10 dark:via-white/5 dark:to-transparent" />
        {/* Glow effect */}
        <div className="absolute -inset-1 bg-gradient-to-r from-primary/20 to-secondary/20 rounded-lg blur-lg opacity-30" />
        
        <CardContent className="relative px-10 w-full lg:flex-row gap-[30px] lg:gap-0 flex-col flex items-center justify-evenly py-20">
          {/* Illustration Section */}
          <div className="w-[80%] lg:w-[40%]">
            <div className="relative">
              {/* Placeholder for illustration - you can replace with your own image */}
              <div className="w-full aspect-square rounded-2xl bg-gradient-to-br from-primary/20 to-secondary/20 backdrop-blur-sm flex items-center justify-center border border-white/20 shadow-lg">
                <div className="text-center">
                  <div className="text-8xl mb-4 opacity-50">🔍</div>
                  <div className="text-6xl font-bold text-primary/60">404</div>
                </div>
              </div>
              {/* Glass shine effect */}
              <div className="absolute inset-0 bg-gradient-to-br from-white/30 via-transparent to-transparent rounded-2xl opacity-50" />
            </div>
          </div>

          {/* Content Section */}
          <div className="w-full lg:w-[40%] text-center lg:text-start">
            <h1 className="text-[2.5rem] sm:text-[4rem] font-[800] bg-gradient-to-r from-primary via-primary to-secondary bg-clip-text text-transparent leading-[80px] mb-4">
              OOPS!
            </h1>
            <h3 className="text-muted-foreground text-[0.9rem] sm:text-[1.2rem] mb-8 max-w-md">
              Looks like this page has wandered off into the digital wilderness. 
              Let's get you back on track!
            </h3>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
              <Link href="/">
                <Button 
                  size="lg"
                  className="gap-2 bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 text-primary-foreground shadow-lg hover:shadow-xl transition-all duration-300"
                >
                  <Home className="w-4 h-4" />
                  Back to Dashboard
                </Button>
              </Link>
              
              <Button 
                variant="outline" 
                size="lg"
                onClick={() => window.history.back()}
                className="gap-2 backdrop-blur-sm bg-white/10 dark:bg-black/10 border-white/20 hover:bg-white/20 dark:hover:bg-black/20"
              >
                <ArrowLeft className="w-4 h-4" />
                Go Back
              </Button>
            </div>

            {/* Additional helpful links */}
            <div className="mt-8 pt-6 border-t border-white/10">
              <p className="text-sm text-muted-foreground mb-4">
                Or try one of these popular sections:
              </p>
              <div className="flex flex-wrap gap-2 justify-center lg:justify-start">
                <Link href="/db-knowledge">
                  <Button variant="ghost" size="sm" className="text-xs hover:bg-white/10 dark:hover:bg-black/10">
                    DB Knowledge
                  </Button>
                </Link>
                <Link href="/file-system">
                  <Button variant="ghost" size="sm" className="text-xs hover:bg-white/10 dark:hover:bg-black/10">
                    File System
                  </Button>
                </Link>
                <Link href="/hr-knowledge">
                  <Button variant="ghost" size="sm" className="text-xs hover:bg-white/10 dark:hover:bg-black/10">
                    HR Knowledge
                  </Button>
                </Link>
                <Link href="/support-team">
                  <Button variant="ghost" size="sm" className="text-xs hover:bg-white/10 dark:hover:bg-black/10">
                    Support Team
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default NotFoundPage;