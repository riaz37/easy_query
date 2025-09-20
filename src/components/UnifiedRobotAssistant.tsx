"use client";

import React, { useState, useEffect, useRef } from "react";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { 
  Mic, 
  MicOff, 
  Play, 
  MessageCircle, 
  X, 
  Send, 
  Trash2, 
  Minimize2, 
  Maximize2,
  Navigation, 
  Bot,
} from "lucide-react";
import { useVoiceAgent } from "@/components/providers/VoiceAgentContextProvider";
import { useTextConversation } from "@/components/providers/TextConversationContextProvider";
import { useTheme } from "@/store/theme-store";
import { cn } from "@/lib/utils";

type AssistantMode = "voice" | "text";
type RobotState = "idle" | "selecting" | "chatting" | "talking";

export function UnifiedRobotAssistant() {
  const theme = useTheme();
  const [mode, setMode] = useState<AssistantMode>("voice");
  const [robotState, setRobotState] = useState<RobotState>("idle");
  const [isExpanded, setIsExpanded] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [inputMessage, setInputMessage] = useState("");
  const [unreadCount, setUnreadCount] = useState(0);
  const [voiceNavigationStatus, setVoiceNavigationStatus] = useState("Ready");
  const [lastVoiceCommand, setLastVoiceCommand] = useState<string | null>(null);

  // Voice Agent Context
  const {
    isConnected: isVoiceConnected,
    isInConversation,
    connectionStatus: voiceConnectionStatus,
    messages: voiceMessages,
    isReady: isVoiceReady,
    isLoading: isVoiceLoading,
    connect: connectVoice,
    disconnect: disconnectVoice,
    startConversation,
    stopConversation,
    clearMessages: clearVoiceMessages,
  } = useVoiceAgent();

  // Text Conversation Context
  const {
    isConnected: isTextConnected,
    connectionStatus: textConnectionStatus,
    messages: textMessages,
    isTyping,
    isReady: isTextReady,
    isLoading: isTextLoading,
    connect: connectText,
    disconnect: disconnectText,
    sendMessage,
    clearMessages: clearTextMessages,
  } = useTextConversation();

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [mode === "text" ? textMessages : voiceMessages]);

  // Track unread messages when chat is minimized
  useEffect(() => {
    if (!isExpanded || isMinimized) {
      const currentMessages = mode === "text" ? textMessages : voiceMessages;
      if (currentMessages.length > 0) {
        setUnreadCount((prev) => prev + 1);
      }
    } else {
      setUnreadCount(0);
    }
  }, [
    mode === "text" ? textMessages.length : voiceMessages.length,
    isExpanded,
    isMinimized,
  ]);

  // Focus input when expanded
  useEffect(() => {
    if (isExpanded && !isMinimized && inputRef.current && mode === "text") {
      inputRef.current.focus();
    }
  }, [isExpanded, isMinimized, mode]);

  // Listen for voice navigation events
  useEffect(() => {
    if (isVoiceLoading || !isVoiceReady) {
      return;
    }

    const handleVoiceEvent = (event: CustomEvent) => {
      const { type, element_name, page, user_id } = event.detail;
      setVoiceNavigationStatus(type);
      setLastVoiceCommand(
        `${type}: ${element_name || page || "action executed"}${
          user_id ? ` (User: ${user_id})` : ""
        }`
      );
    };

    window.addEventListener(
      "voice-database-search",
      handleVoiceEvent as EventListener
    );
    window.addEventListener(
      "voice-file-search",
      handleVoiceEvent as EventListener
    );
    window.addEventListener(
      "voice-file-upload",
      handleVoiceEvent as EventListener
    );
    window.addEventListener(
      "voice-generate-report",
      handleVoiceEvent as EventListener
    );
    window.addEventListener(
      "voice-navigate",
      handleVoiceEvent as EventListener
    );

    return () => {
      window.removeEventListener(
        "voice-database-search",
        handleVoiceEvent as EventListener
      );
      window.removeEventListener(
        "voice-file-search",
        handleVoiceEvent as EventListener
      );
      window.removeEventListener(
        "voice-file-upload",
        handleVoiceEvent as EventListener
      );
      window.removeEventListener(
        "voice-generate-report",
        handleVoiceEvent as EventListener
      );
      window.removeEventListener(
        "voice-navigate",
        handleVoiceEvent as EventListener
      );
    };
  }, [isVoiceLoading, isVoiceReady]);

  // Don't render if services are not ready
  if ((isVoiceLoading || !isVoiceReady) && (isTextLoading || !isTextReady)) {
    return null;
  }

  const handleSendMessage = () => {
    if (!inputMessage.trim() || !isTextConnected) return;

    sendMessage(inputMessage.trim());
    setInputMessage("");
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleRobotClick = () => {
    if (robotState === "idle") {
      setRobotState("selecting");
    } else if (robotState === "selecting") {
      setRobotState("idle");
    } else if (robotState === "talking") {
      // Disconnect voice and return to idle
      disconnectVoice();
      setRobotState("idle");
    } else {
      // If in chatting mode, close the interface
      setRobotState("idle");
      setIsExpanded(false);
      setIsMinimized(false);
    }
  };

  const handleModeSelect = (selectedMode: AssistantMode) => {
    setMode(selectedMode);
    if (selectedMode === "voice") {
      setRobotState("talking");
      connectVoice();
    } else {
      setRobotState("chatting");
      setIsExpanded(true);
      connectText();
    }
  };

  const toggleMinimized = () => {
    setIsMinimized(!isMinimized);
    if (isMinimized) {
      setUnreadCount(0);
    }
  };

  const formatTime = (date: Date | string | number) => {
    try {
      const dateObj = date instanceof Date ? date : new Date(date);
      if (isNaN(dateObj.getTime())) {
        return "Invalid time";
      }
      return dateObj.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch (error) {
      console.error("Error formatting time:", error, "Input:", date);
      return "Invalid time";
    }
  };

  const currentMessages = mode === "text" ? textMessages : voiceMessages;
  const isConnected = mode === "text" ? isTextConnected : isVoiceConnected;
  const connectionStatus =
    mode === "text" ? textConnectionStatus : voiceConnectionStatus;

  return (
    <div className="fixed bottom-6 right-6 z-50">
      {/* Main Robot Button */}
      <div className="relative">
        <button
          onClick={handleRobotClick}
          className={cn(
            "transition-all duration-300 hover:scale-110 focus:outline-none active:outline-none",
            "p-2 rounded-lg relative",
            robotState === "talking" && "animate-pulse",
            robotState === "talking" && "cursor-pointer"
          )}
          title={
            robotState === "talking"
              ? "Click to disconnect voice"
              : "Click to interact"
          }
        >
          {/* Pulsing Glow Ring */}
          <div
            className={cn(
              "absolute inset-0 rounded-lg transition-all duration-1000",
              "bg-gradient-to-r from-emerald-400/20 to-emerald-600/20",
              "animate-pulse",
              robotState === "talking" &&
                "from-emerald-400/40 to-emerald-600/40",
              "hover:from-emerald-400/30 hover:to-emerald-600/30"
            )}
            style={{
              animationDuration: "2s",
              animationTimingFunction: "ease-in-out",
            }}
          />
          
          <Image
            src="/autopilot.svg"
            alt="AI Assistant"
            width={48}
            height={48}
            className="w-12 h-12 relative z-10"
          />
        </button>

        {/* Talking indicator dots */}
        {robotState === "talking" && isInConversation && (
          <div className="absolute -top-2 -right-2 flex gap-1">
            <div
              className="w-1 h-1 bg-emerald-400 rounded-full animate-bounce"
              style={{ animationDelay: "0ms" }}
            ></div>
            <div
              className="w-1 h-1 bg-emerald-400 rounded-full animate-bounce"
              style={{ animationDelay: "150ms" }}
            ></div>
            <div
              className="w-1 h-1 bg-emerald-400 rounded-full animate-bounce"
              style={{ animationDelay: "300ms" }}
            ></div>
          </div>
        )}

        {/* Expanding rings for active states */}
        {robotState === "talking" && isInConversation && (
          <div className="absolute inset-0 pointer-events-none">
            <div
              className="absolute inset-0 border-2 border-emerald-400/30 rounded-lg animate-ping"
              style={{ animationDuration: "1.5s" }}
            ></div>
            <div
              className="absolute inset-0 border-2 border-emerald-400/20 rounded-lg animate-ping"
              style={{ animationDuration: "1.5s", animationDelay: "0.5s" }}
            ></div>
          </div>
        )}

        {/* Attention-seeking pulse for idle state */}
        {robotState === "idle" && (
          <div className="absolute inset-0 pointer-events-none">
            <div
              className="absolute inset-0 border-2 border-emerald-400/20 rounded-lg animate-pulse"
              style={{ animationDuration: "3s" }}
            ></div>
          </div>
        )}

        {/* Robot Question Speech Bubble */}
        {robotState === "selecting" && (
          <div className="absolute bottom-6 right-20 flex items-center animate-in slide-in-from-right-2 duration-300">
            {/* Speech bubble */}
            <div
              className={cn(
              "px-4 py-2 rounded-2xl shadow-lg backdrop-blur-xl relative whitespace-nowrap",
              theme === "dark" 
                ? "bg-white/10 border border-emerald-500/20" 
                : "bg-white/95 border border-emerald-500/30 shadow-emerald-500/20"
            )}
            style={{
                background:
                  theme === "dark"
                ? "rgba(255, 255, 255, 0.1)"
                : "linear-gradient(158.39deg, rgba(255, 255, 255, 0.98) 14.19%, rgba(240, 249, 245, 0.95) 50.59%, rgba(255, 255, 255, 0.98) 68.79%, rgba(240, 249, 245, 0.95) 105.18%)",
              backdropFilter: "blur(20px)",
              WebkitBackdropFilter: "blur(20px)",
                boxShadow:
                  theme === "dark"
                ? "0 8px 32px rgba(0, 0, 0, 0.3)"
                    : "0 8px 32px rgba(16, 185, 129, 0.12), 0 1px 0 rgba(255, 255, 255, 0.8) inset",
              }}
            >
              {/* Speech bubble tail pointing to robot */}
              <div
                className={cn(
                "absolute -right-2 top-1/2 transform -translate-y-1/2 w-0 h-0",
                "border-l-[12px] border-l-emerald-500/30 border-t-[8px] border-t-transparent border-b-[8px] border-b-transparent"
                )}
              ></div>
              
              <div className="flex flex-col gap-2">
                <span
                  className={cn(
                  "text-sm font-medium",
                  theme === "dark" ? "text-emerald-400" : "text-emerald-600"
                  )}
                >
                  What would you like to do?
                </span>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleModeSelect("voice")}
                    className={cn(
                      "px-3 py-1 text-xs rounded-md transition-all duration-200",
                      theme === "dark"
                        ? "bg-emerald-500 hover:bg-emerald-600 text-white"
                        : "bg-emerald-500 hover:bg-emerald-600 text-white shadow-sm hover:shadow-md"
                    )}
                  >
                    Talk
                  </button>
                  <button
                    onClick={() => handleModeSelect("text")}
                    className={cn(
                      "px-3 py-1 text-xs rounded-md transition-all duration-200",
                      theme === "dark"
                        ? "bg-emerald-500 hover:bg-emerald-600 text-white"
                        : "bg-emerald-500 hover:bg-emerald-600 text-white shadow-sm hover:shadow-md"
                    )}
                  >
                    Chat
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Voice Mode - Speech Bubble beside existing robot */}
      {robotState === "talking" && (
        <div className="absolute bottom-6 right-20 animate-in slide-in-from-right-2 duration-300">
          {/* Speech bubble */}
          <div
            className={cn(
            "px-4 py-2 rounded-2xl shadow-lg backdrop-blur-xl relative w-80",
            theme === "dark" 
              ? "bg-white/10 border border-emerald-500/20" 
              : "bg-white/95 border border-emerald-500/30 shadow-emerald-500/20"
          )}
          style={{
              background:
                theme === "dark"
              ? "rgba(255, 255, 255, 0.1)"
              : "linear-gradient(158.39deg, rgba(255, 255, 255, 0.98) 14.19%, rgba(240, 249, 245, 0.95) 50.59%, rgba(255, 255, 255, 0.98) 68.79%, rgba(240, 249, 245, 0.95) 105.18%)",
            backdropFilter: "blur(20px)",
            WebkitBackdropFilter: "blur(20px)",
              boxShadow:
                theme === "dark"
              ? "0 8px 32px rgba(0, 0, 0, 0.3)"
                  : "0 8px 32px rgba(16, 185, 129, 0.12), 0 1px 0 rgba(255, 255, 255, 0.8) inset",
            }}
          >
            {/* Speech bubble tail pointing to robot */}
            <div
              className={cn(
              "absolute -right-2 top-1/2 transform -translate-y-1/2 w-0 h-0",
              "border-l-[12px] border-l-emerald-500/30 border-t-[8px] border-t-transparent border-b-[8px] border-b-transparent"
              )}
            ></div>
            
            {voiceMessages.length > 0 ? (
              <div className="flex flex-col gap-2">
                <span
                  className={cn(
                  "text-sm font-medium break-words",
                  theme === "dark" ? "text-emerald-400" : "text-emerald-600"
                  )}
                >
                  {voiceMessages[voiceMessages.length - 1]?.content}
                </span>
                <span
                  className={cn(
                  "text-xs self-end",
                  theme === "dark" ? "text-gray-400" : "text-gray-500"
                  )}
                >
                  {formatTime(
                    voiceMessages[voiceMessages.length - 1]?.timestamp
                  )}
                </span>
              </div>
            ) : (
              <span
                className={cn(
                "text-sm",
                theme === "dark" ? "text-gray-300" : "text-gray-600"
                )}
              >
                {isInConversation ? "Listening..." : "Ready to talk"}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Expanded Panel - Text Chat Mode */}
      {robotState === "chatting" && isExpanded && (
        <div
          className={cn(
          "absolute bottom-20 rounded-2xl shadow-2xl transition-all duration-300 flex flex-col backdrop-blur-xl",
            isMinimized
              ? "w-80 h-12 right-0"
              : "w-96 sm:w-96 md:w-[28rem] h-[500px] right-0"
        )}
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
                onClick={() => setIsExpanded(false)}
                variant="ghost"
                size="sm"
                className="w-6 h-6 p-0 text-slate-400 hover:text-white hover:bg-slate-700/50"
              >
                <X className="w-4 h-4" />
              </Button>
          </div>

          {!isMinimized && (
              <div className="flex flex-col flex-1 min-h-0 px-4 pb-4 mt-4">
              {/* Text Mode Content */}
                {mode === "text" && (
                <div className="flex flex-col flex-1 min-h-0">
                  {/* Connection Loading State */}
                  {!isTextConnected && (
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
                        <div className="space-y-4 p-4">
                      {textMessages.length === 0 && isTextConnected && (
                        <div className="text-center text-muted-foreground py-8">
                          <MessageCircle className="w-12 h-12 mx-auto mb-2 opacity-50" />
                          <p>Start a conversation!</p>
                        </div>
                      )}

                      {textMessages.map((message) => (
                        <div
                          key={message.id}
                          className={cn(
                                "flex gap-2 mb-4",
                                message.type === "user"
                                  ? "justify-end"
                                  : "justify-start"
                              )}
                            >
                              {/* AI Avatar */}
                              {message.type !== "user" && (
                                <div className="flex-shrink-0 -mt-6">
                                  <Image
                                    src="/autopilot.svg"
                                    alt="AI Assistant"
                                    width={30}
                                    height={30}
                                  />
                                </div>
                              )}

                              <div className="flex flex-col max-w-[80%]">
                                <div className="relative">
                          <div
                            className={cn(
                                      "px-4 py-3 text-sm",
                                      message.type === "user"
                                        ? "text-white rounded-2xl rounded-tr-sm"
                                        : "text-white/72 rounded-2xl rounded-tl-sm"
                                    )}
                                    style={message.type === "user" ? {
                                      background: "var(--primary-12, rgba(19, 245, 132, 0.12))",
                                      boxShadow: "0px 1px 1px -0.5px elevation-shadow"
                                    } : {
                                      background: "linear-gradient(121.65deg, rgba(63, 67, 70, 0.3) 55.58%, rgba(76, 81, 85, 0.3) 97.52%)",
                                      border: "1px solid",
                                      borderImageSource: "linear-gradient(90deg, rgba(65, 69, 72, 0.5) 12.35%, rgba(109, 115, 120, 0.5) 42.59%, rgba(65, 69, 72, 0.5) 72.84%)",
                                      color: "rgba(255, 255, 255, 0.72)"
                                    }}
                          >
                            <div className="whitespace-pre-wrap break-words">
                              {message.content}
                                    </div>
                                  </div>
                                  
                                  {/* Speech bubble tail - pointing upward */}
                                  <div
                                    className={cn(
                                      "absolute w-0 h-0",
                                      message.type === "user"
                                        ? "right-0 -top-3 border-l-[12px] border-l-transparent border-b-[12px]"
                                        : "left-0 -top-3 border-r-[12px] border-r-transparent border-b-[12px]"
                                    )}
                                    style={message.type === "user" ? {
                                      borderBottomColor: "var(--primary-12, rgba(19, 245, 132, 0.12))"
                                    } : {
                                      borderBottomColor: "rgba(63, 67, 70, 0.3)"
                                    }}
                                  />
                            </div>
                            <div
                              className={cn(
                                    "flex items-center gap-1 mt-1 text-xs text-slate-400",
                                    message.type === "user"
                                      ? "justify-end"
                                      : "justify-start"
                                  )}
                                >
                                  {message.type === "user" && (
                                    <>
                                      <div className="w-3 h-3 text-slate-400">
                                        ✓
                                      </div>
                                      <span>You</span>
                                    </>
                                  )}
                                  <span>{formatTime(message.timestamp)}</span>
                            </div>
                          </div>

                               {/* User Avatar */}
                               {message.type === "user" && (
                                 <div className="w-8 h-8 rounded-full bg-purple-500 flex items-center justify-center flex-shrink-0 -mt-4">
                                   <div className="w-4 h-4 text-white">👤</div>
                                 </div>
                               )}
                        </div>
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
                        disabled={!isTextConnected}
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
                        onClick={handleSendMessage}
                        disabled={!isTextConnected || !inputMessage.trim()}
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
          )}
          </div>
        </div>
      )}
    </div>
  );
}