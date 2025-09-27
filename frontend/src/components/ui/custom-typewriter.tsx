"use client";

import React, { useState, useEffect, useRef } from "react";

interface CustomTypewriterProps {
  texts: string[];
  className?: string;
  typeSpeed?: number;
  deleteSpeed?: number;
  deleteChunkSize?: number;
  pauseTime?: number;
  loop?: boolean;
  startDelay?: number;
}

export function CustomTypewriter({
  texts,
  className = "",
  typeSpeed = 50,
  deleteSpeed = 30,
  deleteChunkSize = 1,
  pauseTime = 2000,
  loop = true,
  startDelay = 1000,
}: CustomTypewriterProps) {
  const [displayText, setDisplayText] = useState("");
  const [currentTextIndex, setCurrentTextIndex] = useState(0);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    if (texts.length === 0) return;

    const typeText = () => {
      const currentText = texts[currentTextIndex];
      
      if (!isDeleting) {
        // Typing phase
        if (displayText.length < currentText.length) {
          setDisplayText(currentText.slice(0, displayText.length + 1));
          timeoutRef.current = setTimeout(typeText, typeSpeed);
        } else {
          // Finished typing, pause before deleting
          setIsPaused(true);
          timeoutRef.current = setTimeout(() => {
            setIsPaused(false);
            setIsDeleting(true);
          }, pauseTime);
        }
      } else {
        // Deleting phase
        if (displayText.length > 0) {
          setDisplayText(displayText.slice(0, -deleteChunkSize));
          timeoutRef.current = setTimeout(typeText, deleteSpeed);
        } else {
          // Finished deleting, move to next text
          setIsDeleting(false);
          setCurrentTextIndex((prevIndex) => {
            const nextIndex = (prevIndex + 1) % texts.length;
            if (!loop && nextIndex === 0) {
              // Stop if not looping and we've completed one cycle
              return prevIndex;
            }
            return nextIndex;
          });
          
          timeoutRef.current = setTimeout(typeText, 500); // Small delay before starting next text
        }
      }
    };

    // Start the typewriter effect
    timeoutRef.current = setTimeout(typeText, startDelay);

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [texts, displayText, isDeleting, isPaused, typeSpeed, deleteSpeed, deleteChunkSize, pauseTime, loop, startDelay]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return (
    <span className={`text-slate-400 ${className}`}>
      {displayText}
      <span className="animate-pulse">|</span>
    </span>
  );
}

// Predefined text sets for different contexts
export const TYPEWRITER_TEXTS = {
  database: [
    "Show me all users from last month",
    "What are the top selling products?",
    "Find customers with orders over $1000",
    "Generate a sales report for Q4",
    "Which products have low inventory?",
    "Show me user engagement metrics",
    "What's the average order value?",
    "Find inactive users from the last 6 months",
    "Show me revenue by region",
    "Which campaigns had the highest conversion?"
  ],
  file: [
    "Ask a question about your uploaded files",
    "What insights can you find in this document?",
    "Summarize the key points from this file",
    "What are the main topics discussed?",
    "Extract important data from this document",
    "Find specific information in the text",
    "What patterns do you see in this data?",
    "Analyze the content and provide insights",
    "What conclusions can be drawn from this?",
    "Help me understand this document better"
  ],
  general: [
    "Ask me anything",
    "How can I help you today?",
    "What would you like to know?",
    "Tell me what you're looking for",
    "I'm here to assist you",
    "What's on your mind?",
    "How can I make your day better?",
    "What information do you need?",
    "I'm ready to help",
    "What can I do for you?"
  ],
  reports: [
    "Generate a comprehensive sales analysis for Q2",
    "Show me the financial report of May",
    "Create a detailed user engagement report",
    "Analyze customer behavior patterns",
    "Generate a marketing performance report",
    "Show me revenue trends over the last 6 months",
    "Create a product performance analysis",
    "Generate a customer satisfaction report",
    "Analyze website traffic and conversion rates",
    "Create a comprehensive business intelligence report"
  ]
} as const;

export type TypewriterTextType = keyof typeof TYPEWRITER_TEXTS;
