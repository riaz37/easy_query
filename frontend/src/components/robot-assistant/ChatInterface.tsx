"use client";

import React, { useRef, useEffect } from "react";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageCircle, X } from "lucide-react";
import { MessageBubble } from "./MessageBubble";

interface Message {
  id: string;
  content: string;
  timestamp: Date | string | number;
  type: "user" | "assistant";
}

interface ChatInterfaceProps {
  isExpanded: boolean;
  isMinimized: boolean;
  isConnected: boolean;
  messages: Message[];
  isTyping: boolean;
  inputMessage: string;
  setInputMessage: (message: string) => void;
  onSendMessage: () => void;
  onClose: () => void;
  onToggleMinimized: () => void;
  formatTime: (date: Date | string | number) => string;
}

export function ChatInterface({
  isExpanded,
  isMinimized,
  isConnected,
  messages,
  isTyping,
  inputMessage,
  setInputMessage,
  onSendMessage,
  onClose,
  onToggleMinimized,
  formatTime,
}: ChatInterfaceProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  // Focus input when expanded
  useEffect(() => {
    if (isExpanded && !isMinimized && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isExpanded, isMinimized]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSendMessage();
    }
  };

  if (!isExpanded) return null;

  return (
    <div
      className={`absolute bottom-20 rounded-2xl shadow-2xl transition-all duration-300 flex flex-col backdrop-blur-xl ${
        isMinimized
          ? "w-80 h-12 right-0"
          : "w-96 sm:w-96 md:w-[28rem] h-[500px] right-0"
      }`}
      style={{
        right: "0",
        maxWidth: "calc(100vw - 2rem)",
        maxHeight: "calc(100vh - 8rem)",
        background:
          "linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)), linear-gradient(246.02deg, rgba(19, 245, 132, 0) 91.9%, rgba(19, 245, 132, 0.2) 114.38%), linear-gradient(59.16deg, rgba(19, 245, 132, 0) 71.78%, rgba(19, 245, 132, 0.2) 124.92%)",
        border: "1.5px solid",
        borderImageSource:
          "linear-gradient(158.39deg, rgba(255, 255, 255, 0.06) 14.19%, rgba(255, 255, 255, 1.5e-05) 50.59%, rgba(255, 255, 255, 1.5e-05) 68.79%, rgba(255, 255, 255, 0.015) 105.18%)",
        borderRadius: "30px",
        backdropFilter: "blur(30px)",
        WebkitBackdropFilter: "blur(30px)",
        boxShadow:
          "0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1)",
      }}
    >
      {/* Chat Container with 30px radius */}
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-2 flex-shrink-0">
          <div className="flex items-center">
            <Image
              src="/autopilot.svg"
              alt="AI Assistant"
              width={48}
              height={48}
              className="flex-shrink-0"
            />
            <div className="flex items-center ml-4">
              <h3 className="text-base font-semibold text-white">
                AI Assistance
              </h3>
            </div>
          </div>
          <Button
            onClick={onClose}
            variant="ghost"
            size="sm"
            className="w-6 h-6 p-0 text-slate-400 hover:text-white hover:bg-slate-700/50"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>

        {!isMinimized && (
          <div className="flex flex-col flex-1 min-h-0 px-4 pb-4 mt-8">
            {/* Connection Loading State */}
            {!isConnected && (
              <div className="p-3 bg-emerald-50 dark:bg-emerald-900/20 rounded-xl mb-4 flex-shrink-0">
                <div className="flex items-center justify-center gap-2 text-emerald-600 dark:text-emerald-400">
                  <div className="w-4 h-4 border-2 border-emerald-600 border-t-transparent rounded-full animate-spin"></div>
                  <span className="text-sm font-medium">
                    Connecting to chat...
                  </span>
                </div>
              </div>
            )}

            {/* Messages Area */}
            <div
              className="flex-1 min-h-0 mb-4 rounded-[30px] overflow-hidden"
              style={{ background: "rgba(255, 255, 255, 0.04)" }}
            >
              <ScrollArea className="h-full">
                <div className="space-y-4 p-4 pt-8">
                  {messages.length === 0 && isConnected && (
                    <div className="text-center text-muted-foreground py-8">
                      <MessageCircle className="w-12 h-12 mx-auto mb-2 opacity-50" />
                      <p>Start a conversation!</p>
                    </div>
                  )}

                  {messages.map((message) => (
                    <MessageBubble
                      key={message.id}
                      message={message}
                      formatTime={formatTime}
                    />
                  ))}

                  {isTyping && (
                    <div className="flex gap-2 mb-4">
                      <div className="flex-shrink-0 -mt-4">
                        <Image
                          src="/autopilot.svg"
                          alt="AI Assistant"
                          width={24}
                          height={24}
                          className="w-6 h-6"
                        />
                      </div>
                      <div className="relative">
                        <div 
                          className="rounded-2xl rounded-tl-sm px-4 py-3 text-sm"
                          style={{
                            background: "linear-gradient(121.65deg, rgba(63, 67, 70, 0.3) 55.58%, rgba(76, 81, 85, 0.3) 97.52%)",
                            border: "1px solid",
                            borderImageSource: "linear-gradient(90deg, rgba(65, 69, 72, 0.5) 12.35%, rgba(109, 115, 120, 0.5) 42.59%, rgba(65, 69, 72, 0.5) 72.84%)",
                            color: "rgba(255, 255, 255, 0.72)"
                          }}
                        >
                          <div className="flex items-center gap-1">
                            <div className="flex gap-1">
                              <div
                                className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce"
                                style={{ animationDelay: "0ms" }}
                              ></div>
                              <div
                                className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce"
                                style={{ animationDelay: "150ms" }}
                              ></div>
                              <div
                                className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce"
                                style={{ animationDelay: "300ms" }}
                              ></div>
                            </div>
                            <span className="text-xs ml-2 text-white/72">
                              typing...
                            </span>
                          </div>
                        </div>
                        
                        {/* Speech bubble tail for typing indicator - pointing upward */}
                        <div
                          className="absolute left-0 -top-3 w-0 h-0 border-r-[12px] border-r-transparent border-b-[12px]"
                          style={{
                            borderBottomColor: "rgba(63, 67, 70, 0.3)"
                          }}
                        />
                      </div>
                    </div>
                  )}

                  <div ref={messagesEndRef} />
                </div>
              </ScrollArea>
            </div>

            {/* Input Area */}
            <div className="flex-shrink-0">
              <div className="relative">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask anything"
                  disabled={!isConnected}
                  className="w-full h-16 px-4 pr-20 text-white placeholder-slate-400 focus:outline-none border-0"
                  style={{
                    background: "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
                    borderRadius: "16px",
                    outline: "none",
                    border: "none",
                  }}
                />
                
                <div className="absolute right-2 top-1/2 transform -translate-y-1/2">
                  <Button
                    onClick={onSendMessage}
                    disabled={!isConnected || !inputMessage.trim()}
                    className="text-xs cursor-pointer"
                    style={{
                      background: "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
                      border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
                      color: "var(--p-main, rgba(19, 245, 132, 1))",
                      borderRadius: "99px",
                      height: "40px",
                      minWidth: "60px",
                    }}
                  >
                    Send
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}