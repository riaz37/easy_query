"use client";

import React, { useState, useEffect } from "react";
import { useVoiceAgent } from "@/components/providers/VoiceAgentContextProvider";
import { useTextConversation } from "@/components/providers/TextConversationContextProvider";
import { 
  RobotButton, 
  ModeSelector, 
  VoiceInterface, 
  ChatInterface 
} from "@/components/robot-assistant";

type AssistantMode = "voice" | "text";
type RobotState = "idle" | "selecting" | "chatting" | "talking";

export function UnifiedRobotAssistant() {
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
      <RobotButton
        robotState={robotState}
        isInConversation={isInConversation}
        onClick={handleRobotClick}
      />

      {/* Mode Selector */}
      {robotState === "selecting" && (
        <ModeSelector onModeSelect={handleModeSelect} />
      )}

      {/* Voice Interface */}
      {robotState === "talking" && (
        <VoiceInterface
          messages={voiceMessages}
          isInConversation={isInConversation}
        />
      )}

      {/* Chat Interface */}
      <ChatInterface
        isExpanded={isExpanded}
        isMinimized={isMinimized}
        isConnected={isTextConnected}
        messages={textMessages}
        isTyping={isTyping}
        inputMessage={inputMessage}
        setInputMessage={setInputMessage}
        onSendMessage={handleSendMessage}
        onClose={() => {
          setRobotState("idle");
          setIsExpanded(false);
          setIsMinimized(false);
        }}
        onToggleMinimized={toggleMinimized}
        formatTime={formatTime}
      />
    </div>
  );
}
