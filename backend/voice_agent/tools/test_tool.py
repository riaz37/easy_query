#!/usr/bin/env python3
"""
Mental Health Chatbot Tools for Omani Arabic Voice Assistant
Simple demo functions for intent analysis, safety detection, and session management.
"""

import json
import datetime
from typing import Dict, Any, List, Optional
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIServerMessageFrame
from pipecat.frames.frames import Frame, LLMMessagesAppendFrame
from pipecat.adapters.schemas.function_schema import FunctionSchema
from loguru import logger


class IntentAnalysisTool:
    """Tool for sentiment/emotion detection and therapy context classification."""
    
    def __init__(self, rtvi_processor: RTVIProcessor, task=None):
        self.rtvi = rtvi_processor
        self.task = task
        
    def get_tool_definition(self) -> FunctionSchema:
        """Get the tool definition for intent analysis."""
        return FunctionSchema(
            name="analyze_intent",
            description="Analyze user's emotional state and therapeutic context from their speech",
            properties={
                "text": {
                    "type": "string",
                    "description": "The user's speech text to analyze"
                },
                "language": {
                    "type": "string",
                    "enum": ["omani_arabic", "english", "mixed"],
                    "description": "The language/dialect of the input",
                    "default": "omani_arabic"
                }
            },
            required=["text"]
        )
    
    async def execute(self, text: str, language: str = "omani_arabic") -> str:
        """Analyze user intent and emotional state."""
        try:
            # Simple emotion detection (in real implementation, this would be ML model)
            emotion_keywords = {
                "anxiety": ["قلق", "خوف", "توتر", "stressed", "worried"],
                "sadness": ["حزن", "كآبة", "يأس", "sad", "depressed"],
                "anger": ["غضب", "زعل", "frustrated", "angry"],
                "joy": ["سعادة", "فرح", "happy", "good"],
                "fear": ["خوف", "رعب", "afraid", "scared"]
            }
            
            therapy_contexts = {
                "family": ["أهل", "عائلة", "أم", "أب", "family", "parents"],
                "work": ["شغل", "عمل", "مدير", "work", "job", "boss"],
                "relationships": ["صديق", "حبيب", "زواج", "relationship", "marriage"],
                "health": ["صحة", "مرض", "health", "illness"],
                "spiritual": ["دين", "صلاة", "الله", "religion", "prayer"]
            }
            
            # Detect emotions
            detected_emotions = []
            for emotion, keywords in emotion_keywords.items():
                if any(keyword in text.lower() for keyword in keywords):
                    detected_emotions.append(emotion)
            
            # Detect therapy context
            detected_contexts = []
            for context, keywords in therapy_contexts.items():
                if any(keyword in text.lower() for keyword in keywords):
                    detected_contexts.append(context)
            
            # Simple sentiment scoring
            sentiment_score = self._calculate_sentiment(text)
            
            analysis_result = {
                "emotions": detected_emotions or ["neutral"],
                "therapy_contexts": detected_contexts or ["general"],
                "sentiment_score": sentiment_score,
                "language": language,
                "cultural_markers": self._detect_cultural_markers(text),
                "urgency_level": self._assess_urgency(text, detected_emotions)
            }
            
            logger.info(f"Intent analysis result: {analysis_result}")
            
            return json.dumps(analysis_result)
            
        except Exception as e:
            logger.error(f"Error in intent analysis: {e}")
            return json.dumps({"error": str(e), "emotions": ["neutral"], "therapy_contexts": ["general"]})
    
    def _calculate_sentiment(self, text: str) -> float:
        """Simple sentiment calculation (-1 to 1)."""
        positive_words = ["جيد", "سعيد", "حلو", "good", "happy", "fine", "better"]
        negative_words = ["سيء", "حزين", "صعب", "bad", "sad", "difficult", "worse"]
        
        pos_count = sum(1 for word in positive_words if word in text.lower())
        neg_count = sum(1 for word in negative_words if word in text.lower())
        
        if pos_count + neg_count == 0:
            return 0.0
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    def _detect_cultural_markers(self, text: str) -> List[str]:
        """Detect Omani/Gulf cultural markers."""
        cultural_markers = []
        
        # Religious markers
        if any(word in text for word in ["الله", "إن شاء الله", "الحمد لله", "الله يعطيك العافية"]):
            cultural_markers.append("religious")
        
        # Family hierarchy markers
        if any(word in text for word in ["الوالد", "الوالدة", "الأكبر", "الأصغر"]):
            cultural_markers.append("family_hierarchy")
        
        # Omani dialect markers
        if any(word in text for word in ["زين", "شلون", "وين", "هني"]):
            cultural_markers.append("omani_dialect")
            
        return cultural_markers
    
    def _assess_urgency(self, text: str, emotions: List[str]) -> str:
        """Assess urgency level based on content and emotions."""
        crisis_keywords = ["موت", "انتحار", "أذى", "suicide", "death", "hurt myself"]
        
        if any(keyword in text.lower() for keyword in crisis_keywords):
            return "crisis"
        elif "anxiety" in emotions or "fear" in emotions:
            return "high"
        elif "sadness" in emotions:
            return "medium"
        else:
            return "low"


class SafetyCrisisDetectionTool:
    """Tool for detecting harmful content, suicidal ideation, and crisis situations."""
    
    def __init__(self, rtvi_processor: RTVIProcessor, task=None):
        self.rtvi = rtvi_processor
        self.task = task
        
    def get_tool_definition(self) -> FunctionSchema:
        """Get the tool definition for safety detection."""
        return FunctionSchema(
            name="detect_safety_crisis",
            description="Detect harmful content, suicidal ideation, or crisis situations that need immediate attention",
            properties={
                "text": {
                    "type": "string",
                    "description": "The user's speech text to analyze for safety concerns"
                },
                "context": {
                    "type": "string",
                    "description": "Additional context about the conversation or user state"
                }
            },
            required=["text"]
        )
    
    async def execute(self, text: str, context: str = "") -> str:
        """Detect safety concerns and crisis situations."""
        try:
            # Crisis detection keywords (Arabic and English)
            crisis_indicators = {
                "suicide": ["انتحار", "أقتل نفسي", "أموت", "suicide", "kill myself", "end my life"],
                "self_harm": ["أذى نفسي", "أجرح نفسي", "hurt myself", "cut myself"],
                "violence": ["أقتل", "أضرب", "عنف", "kill", "hurt others", "violence"],
                "substance_abuse": ["مخدرات", "خمر", "إدمان", "drugs", "alcohol", "addiction"],
                "severe_depression": ["لا أستطيع", "انتهيت", "فقدت الأمل", "can't go on", "hopeless"]
            }
            
            # Detect crisis indicators
            detected_risks = []
            risk_level = "low"
            
            for risk_type, keywords in crisis_indicators.items():
                if any(keyword in text.lower() for keyword in keywords):
                    detected_risks.append(risk_type)
                    if risk_type in ["suicide", "self_harm", "violence"]:
                        risk_level = "critical"
                    elif risk_level != "critical":
                        risk_level = "high"
            
            # Additional context analysis
            escalation_needed = risk_level == "critical"
            
            # Generate safety assessment
            safety_result = {
                "risk_level": risk_level,
                "detected_risks": detected_risks,
                "escalation_needed": escalation_needed,
                "recommended_actions": self._get_recommended_actions(risk_level, detected_risks),
                "emergency_contacts": self._get_emergency_contacts() if escalation_needed else [],
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Log critical situations
            if escalation_needed:
                logger.critical(f"CRISIS DETECTED: {safety_result}")
                await self._trigger_escalation(safety_result)
            else:
                logger.info(f"Safety check completed: {safety_result}")
            
            return json.dumps(safety_result)
            
        except Exception as e:
            logger.error(f"Error in safety detection: {e}")
            return json.dumps({"error": str(e), "risk_level": "unknown", "escalation_needed": True})
    
    def _get_recommended_actions(self, risk_level: str, detected_risks: List[str]) -> List[str]:
        """Get recommended actions based on risk assessment."""
        actions = []
        
        if risk_level == "critical":
            actions.extend([
                "immediate_professional_intervention",
                "contact_emergency_services",
                "stay_with_user",
                "document_interaction"
            ])
        elif risk_level == "high":
            actions.extend([
                "professional_referral",
                "follow_up_scheduling",
                "family_contact_if_consented",
                "crisis_resources_sharing"
            ])
        else:
            actions.extend([
                "continue_supportive_conversation",
                "monitor_for_escalation",
                "self_care_recommendations"
            ])
        
        return actions
    
    def _get_emergency_contacts(self) -> List[Dict[str, str]]:
        """Get emergency contact information for Oman."""
        return [
            {
                "service": "Oman Emergency Services",
                "number": "999",
                "description": "General emergency line"
            },
            {
                "service": "Mental Health Crisis Line",
                "number": "24697777",
                "description": "Specialized mental health support"
            },
            {
                "service": "Ministry of Health",
                "number": "24602000",
                "description": "Health services information"
            }
        ]
    
    async def _trigger_escalation(self, safety_result: Dict[str, Any]):
        """Trigger escalation protocols for crisis situations."""
        try:
            # Send alert to monitoring system
            alert_data = {
                "type": "crisis_alert",
                "severity": "critical",
                "details": safety_result,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            logger.critical(f"CRISIS ESCALATION TRIGGERED: {alert_data}")
            
            # In real implementation, this would:
            # 1. Alert human supervisors
            # 2. Contact emergency services if needed
            # 3. Log to secure crisis database
            # 4. Trigger immediate intervention protocols
            
        except Exception as e:
            logger.error(f"Error triggering escalation: {e}")


class SessionManagementTool:
    """Tool for session logging, consent tracking, and data security."""
    
    def __init__(self, rtvi_processor: RTVIProcessor, task=None):
        self.rtvi = rtvi_processor
        self.task = task
        self.session_data = {}
        
    def get_tool_definition(self) -> FunctionSchema:
        """Get the tool definition for session management."""
        return FunctionSchema(
            name="manage_session",
            description="Manage therapy session logging, consent tracking, and data security",
            properties={
                "action": {
                    "type": "string",
                    "enum": ["start_session", "log_interaction", "update_consent", "end_session", "emergency_contact"],
                    "description": "The session management action to perform"
                },
                "data": {
                    "type": "object",
                    "description": "Data relevant to the action being performed"
                },
                "consent_type": {
                    "type": "string",
                    "enum": ["recording", "data_storage", "emergency_contact", "professional_referral"],
                    "description": "Type of consent being managed"
                }
            },
            required=["action"]
        )
    
    async def execute(self, action: str, data: Dict[str, Any] = None, consent_type: str = None) -> str:
        """Execute session management actions."""
        try:
            if action == "start_session":
                return await self._start_session(data or {})
            elif action == "log_interaction":
                return await self._log_interaction(data or {})
            elif action == "update_consent":
                return await self._update_consent(consent_type, data or {})
            elif action == "end_session":
                return await self._end_session(data or {})
            elif action == "emergency_contact":
                return await self._emergency_contact(data or {})
            else:
                return json.dumps({"error": "Invalid action", "valid_actions": ["start_session", "log_interaction", "update_consent", "end_session", "emergency_contact"]})
                
        except Exception as e:
            logger.error(f"Error in session management: {e}")
            return json.dumps({"error": str(e), "action": action})
    
    async def _start_session(self, data: Dict[str, Any]) -> str:
        """Start a new therapy session."""
        session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.session_data[session_id] = {
            "session_id": session_id,
            "start_time": datetime.datetime.now().isoformat(),
            "user_id": data.get("user_id", "anonymous"),
            "language": data.get("language", "omani_arabic"),
            "consent_status": {
                "recording": False,
                "data_storage": False,
                "emergency_contact": False,
                "professional_referral": False
            },
            "interactions": [],
            "safety_flags": [],
            "cultural_context": data.get("cultural_context", {})
        }
        
        logger.info(f"Started session: {session_id}")
        
        return json.dumps({
            "session_id": session_id,
            "status": "started",
            "message": "Session started successfully",
            "consent_required": ["recording", "data_storage"]
        })
    
    async def _log_interaction(self, data: Dict[str, Any]) -> str:
        """Log a therapy interaction."""
        session_id = data.get("session_id")
        if not session_id or session_id not in self.session_data:
            return json.dumps({"error": "Invalid or missing session_id"})
        
        interaction = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user_input": data.get("user_input", ""),
            "bot_response": data.get("bot_response", ""),
            "emotion_detected": data.get("emotion_detected", []),
            "therapy_context": data.get("therapy_context", []),
            "safety_level": data.get("safety_level", "low")
        }
        
        self.session_data[session_id]["interactions"].append(interaction)
        
        logger.info(f"Logged interaction for session {session_id}")
        
        return json.dumps({
            "session_id": session_id,
            "interaction_logged": True,
            "total_interactions": len(self.session_data[session_id]["interactions"])
        })
    
    async def _update_consent(self, consent_type: str, data: Dict[str, Any]) -> str:
        """Update user consent status."""
        session_id = data.get("session_id")
        if not session_id or session_id not in self.session_data:
            return json.dumps({"error": "Invalid or missing session_id"})
        
        consent_given = data.get("consent_given", False)
        
        if consent_type in self.session_data[session_id]["consent_status"]:
            self.session_data[session_id]["consent_status"][consent_type] = consent_given
            
            logger.info(f"Updated consent for {consent_type}: {consent_given}")
            
            return json.dumps({
                "session_id": session_id,
                "consent_type": consent_type,
                "consent_given": consent_given,
                "all_consents": self.session_data[session_id]["consent_status"]
            })
        else:
            return json.dumps({"error": f"Invalid consent type: {consent_type}"})
    
    async def _end_session(self, data: Dict[str, Any]) -> str:
        """End a therapy session."""
        session_id = data.get("session_id")
        if not session_id or session_id not in self.session_data:
            return json.dumps({"error": "Invalid or missing session_id"})
        
        session = self.session_data[session_id]
        session["end_time"] = datetime.datetime.now().isoformat()
        session["duration_minutes"] = self._calculate_duration(session["start_time"], session["end_time"])
        
        # Generate session summary
        summary = {
            "session_id": session_id,
            "duration_minutes": session["duration_minutes"],
            "total_interactions": len(session["interactions"]),
            "primary_emotions": self._get_primary_emotions(session["interactions"]),
            "therapy_contexts": self._get_therapy_contexts(session["interactions"]),
            "safety_incidents": len(session["safety_flags"]),
            "follow_up_needed": self._assess_follow_up_needed(session)
        }
        
        logger.info(f"Ended session {session_id}: {summary}")
        
        return json.dumps({
            "session_ended": True,
            "summary": summary,
            "data_retention": "7_days" if session["consent_status"]["data_storage"] else "immediate_deletion"
        })
    
    async def _emergency_contact(self, data: Dict[str, Any]) -> str:
        """Handle emergency contact procedures."""
        session_id = data.get("session_id")
        emergency_type = data.get("emergency_type", "crisis")
        
        contact_info = {
            "emergency_services": "999",
            "mental_health_crisis": "24697777",
            "family_contact": data.get("family_contact", ""),
            "professional_referral": data.get("professional_referral", "")
        }
        
        logger.critical(f"Emergency contact triggered for session {session_id}: {emergency_type}")
        
        return json.dumps({
            "emergency_activated": True,
            "emergency_type": emergency_type,
            "contact_info": contact_info,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def _calculate_duration(self, start_time: str, end_time: str) -> float:
        """Calculate session duration in minutes."""
        try:
            start = datetime.datetime.fromisoformat(start_time)
            end = datetime.datetime.fromisoformat(end_time)
            return (end - start).total_seconds() / 60
        except:
            return 0.0
    
    def _get_primary_emotions(self, interactions: List[Dict]) -> List[str]:
        """Get primary emotions from session interactions."""
        emotion_counts = {}
        for interaction in interactions:
            for emotion in interaction.get("emotion_detected", []):
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return sorted(emotion_counts.keys(), key=lambda x: emotion_counts[x], reverse=True)[:3]
    
    def _get_therapy_contexts(self, interactions: List[Dict]) -> List[str]:
        """Get therapy contexts from session interactions."""
        context_counts = {}
        for interaction in interactions:
            for context in interaction.get("therapy_context", []):
                context_counts[context] = context_counts.get(context, 0) + 1
        
        return sorted(context_counts.keys(), key=lambda x: context_counts[x], reverse=True)[:3]
    
    def _assess_follow_up_needed(self, session: Dict[str, Any]) -> bool:
        """Assess if follow-up is needed based on session data."""
        return (
            len(session["safety_flags"]) > 0 or
            session["duration_minutes"] > 60 or
            any(interaction.get("safety_level") in ["high", "critical"] for interaction in session["interactions"])
        )


# Factory functions to create the tools
def create_intent_analysis_tool(rtvi_processor: RTVIProcessor, task=None) -> IntentAnalysisTool:
    """Create and return an IntentAnalysisTool instance."""
    return IntentAnalysisTool(rtvi_processor, task)

def create_safety_crisis_detection_tool(rtvi_processor: RTVIProcessor, task=None) -> SafetyCrisisDetectionTool:
    """Create and return a SafetyCrisisDetectionTool instance."""
    return SafetyCrisisDetectionTool(rtvi_processor, task)

def create_session_management_tool(rtvi_processor: RTVIProcessor, task=None) -> SessionManagementTool:
    """Create and return a SessionManagementTool instance."""
    return SessionManagementTool(rtvi_processor, task)


# Example usage and testing
if __name__ == "__main__":
    # Demo function to show tool capabilities
    async def demo_tools():
        """Demonstrate the mental health tools functionality."""
        print("🧠 Mental Health Chatbot Tools Demo")
        print("=" * 50)
        
        # Mock RTVI processor for demo
        class MockRTVIProcessor:
            async def push_frame(self, frame):
                pass
        
        rtvi = MockRTVIProcessor()
        
        # Create tools
        intent_tool = create_intent_analysis_tool(rtvi)
        safety_tool = create_safety_crisis_detection_tool(rtvi)
        session_tool = create_session_management_tool(rtvi)
        
        print("\n1. Intent Analysis Tool Demo:")
        result = await intent_tool.execute("أنا حزين جداً وأشعر بالقلق من العمل")
        print(f"Result: {result}")
        
        print("\n2. Safety Crisis Detection Tool Demo:")
        result = await safety_tool.execute("أريد أن أؤذي نفسي")
        print(f"Result: {result}")
        
        print("\n3. Session Management Tool Demo:")
        # Start session
        result = await session_tool.execute("start_session", {"user_id": "user123", "language": "omani_arabic"})
        session_data = json.loads(result)
        session_id = session_data["session_id"]
        print(f"Start session result: {result}")
        
        # Log interaction
        result = await session_tool.execute("log_interaction", {
            "session_id": session_id,
            "user_input": "أنا أشعر بالقلق",
            "bot_response": "أفهم شعورك، هل يمكنك أن تخبرني أكثر؟",
            "emotion_detected": ["anxiety"],
            "therapy_context": ["general"]
        })
        print(f"Log interaction result: {result}")
        
        # End session
        result = await session_tool.execute("end_session", {"session_id": session_id})
        print(f"End session result: {result}")
        
        print("\n✅ Demo completed successfully!")
    
    # Run demo if this script is executed directly
    import asyncio
    asyncio.run(demo_tools())