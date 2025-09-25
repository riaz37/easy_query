import os
import json
import time
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import traceback
from pathlib import Path
import difflib
from enum import Enum

# Load environment variables
load_dotenv(override=True)

class ActionType(Enum):
    EXACT_MATCH = "exact_match"
    PARTIAL_MATCH_UPDATE = "partial_match_update"
    CREATE_NEW = "create_new"

@dataclass
class IntentMatch:
    """Represents a match between a table and an intent"""
    intent_name: str
    confidence_score: float
    action_type: ActionType
    suggested_description: Optional[str] = None
    reasoning: Optional[str] = None

@dataclass
class Intent:
    """Represents an intent category"""
    name: str
    description: str
    tables: List[str]
    keywords: List[str]
    created_date: str
    last_modified: str
    table_count: int = 0
    
    def __post_init__(self):
        self.table_count = len(self.tables)

@dataclass
class SubIntent:
    """Represents a sub-intent category"""
    name: str
    description: str
    parent_intent: str
    tables: List[str]
    keywords: List[str]
    created_date: str
    last_modified: str
    table_count: int = 0
    
    def __post_init__(self):
        self.table_count = len(self.tables)

class IntentClassificationSystem:
    """Self-learning intent classification system for database tables"""
    
    def __init__(self, 
                 input_file: str = r"C:\Users\Nilab\Desktop\Esap_test\KnowladgeBase\all_tables_with_descriptions_v1.json",
                 intents_file: str = "intent_classification.json",
                 sub_intents_file: str = "sub_intent_classification.json",
                 confidence_threshold: float = 0.7,
                 rate_limit_delay: int = 3):
        
        self.input_file = Path(input_file)
        self.intents_file = Path(intents_file)
        self.sub_intents_file = Path(sub_intents_file)
        self.confidence_threshold = confidence_threshold
        self.rate_limit_delay = rate_limit_delay
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Load existing classifications
        self.intents: Dict[str, Intent] = self._load_intents()
        self.sub_intents: Dict[str, SubIntent] = self._load_sub_intents()
        
        # Statistics tracking
        self.stats = {
            "tables_processed": 0,
            "intents_created": 0,
            "intents_updated": 0,
            "sub_intents_created": 0,
            "sub_intents_updated": 0,
            "exact_matches": 0,
            "partial_matches": 0
        }
    
    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """Initialize the Gemini LLM"""
        try:
            gemini_apikey = os.getenv("google_gemini_key")
            gemini_model_name = os.getenv("google_gemini_name", "gemini-pro")
            
            if not gemini_apikey:
                raise ValueError("Google Gemini API key not found in environment variables")
                
            print(f"🔧 Initializing LLM: {gemini_model_name}")
            llm = ChatGoogleGenerativeAI(
                model=gemini_model_name,
                temperature=0.1,  # Low temperature for consistent classification
                google_api_key=gemini_apikey
            )
            
            # Test the connection
            test_response = llm.invoke("Test connection")
            print("✅ LLM initialized successfully")
            return llm
            
        except Exception as e:
            print(f"❌ Failed to initialize LLM: {e}")
            sys.exit(1)
    
    def _load_intents(self) -> Dict[str, Intent]:
        """Load existing intents from file"""
        if not self.intents_file.exists():
            print("📝 No existing intents file found, starting fresh")
            return {}
        
        try:
            with open(self.intents_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                intents = {}
                for intent_data in data.get('intents', []):
                    intent = Intent(**intent_data)
                    intents[intent.name.lower()] = intent
                print(f"📚 Loaded {len(intents)} existing intents")
                return intents
        except Exception as e:
            print(f"⚠️ Error loading intents: {e}")
            return {}
    
    def _load_sub_intents(self) -> Dict[str, SubIntent]:
        """Load existing sub-intents from file"""
        if not self.sub_intents_file.exists():
            print("📝 No existing sub-intents file found, starting fresh")
            return {}
        
        try:
            with open(self.sub_intents_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                sub_intents = {}
                for sub_intent_data in data.get('sub_intents', []):
                    sub_intent = SubIntent(**sub_intent_data)
                    sub_intents[sub_intent.name.lower()] = sub_intent
                print(f"📚 Loaded {len(sub_intents)} existing sub-intents")
                return sub_intents
        except Exception as e:
            print(f"⚠️ Error loading sub-intents: {e}")
            return {}
    
    def _save_intents(self):
        """Save intents to file"""
        try:
            intent_list = [asdict(intent) for intent in self.intents.values()]
            data = {
                "metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "total_intents": len(intent_list),
                    "total_tables_classified": sum(intent.table_count for intent in self.intents.values())
                },
                "intents": intent_list
            }
            
            with open(self.intents_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved {len(intent_list)} intents to {self.intents_file}")
        except Exception as e:
            print(f"❌ Error saving intents: {e}")
    
    def _save_sub_intents(self):
        """Save sub-intents to file"""
        try:
            sub_intent_list = [asdict(sub_intent) for sub_intent in self.sub_intents.values()]
            data = {
                "metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "total_sub_intents": len(sub_intent_list),
                    "total_tables_classified": sum(sub_intent.table_count for sub_intent in self.sub_intents.values())
                },
                "sub_intents": sub_intent_list
            }
            
            with open(self.sub_intents_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved {len(sub_intent_list)} sub-intents to {self.sub_intents_file}")
        except Exception as e:
            print(f"❌ Error saving sub-intents: {e}")
    
    def _extract_table_context(self, table_data: Dict[str, Any]) -> str:
        """Extract meaningful context from table data for classification"""
        context_parts = []
        
        # Table name and description
        table_name = table_data.get('table_name', '')
        description = table_data.get('description', '')
        
        context_parts.append(f"Table: {table_name}")
        context_parts.append(f"Description: {description}")
        
        # Column information (focus on key columns)
        columns = table_data.get('columns', [])
        key_columns = []
        
        for col in columns[:15]:  # Limit to prevent token overflow
            col_name = col.get('name', '')
            col_desc = col.get('description', '')
            col_type = col.get('type', '')
            
            # Prioritize key columns
            if any(keyword in col_name.lower() for keyword in ['id', 'name', 'type', 'status', 'date', 'amount']):
                key_columns.append(f"{col_name} ({col_type}): {col_desc}")
            elif col.get('is_primary') or col.get('is_foreign'):
                key_columns.append(f"{col_name} ({col_type}): {col_desc}")
        
        if key_columns:
            context_parts.append("Key Columns:")
            context_parts.extend(key_columns[:10])  # Limit key columns
        
        # Relationships
        relationships = table_data.get('relationships', [])
        if relationships:
            rel_info = []
            for rel in relationships[:5]:  # Limit relationships
                rel_table = rel.get('related_table', '')
                rel_type = rel.get('type', '')
                rel_info.append(f"{rel_type} with {rel_table}")
            context_parts.append("Relationships: " + ", ".join(rel_info))
        
        return "\n".join(context_parts)
    
    def _classify_intent(self, table_context: str, table_name: str) -> IntentMatch:
        """Classify a table into an intent using LLM"""
        try:
            # Prepare existing intents for context
            existing_intents_info = ""
            if self.intents:
                intent_summaries = []
                for intent_name, intent in self.intents.items():
                    intent_summaries.append(f"- {intent.name}: {intent.description} (Keywords: {', '.join(intent.keywords[:5])})")
                existing_intents_info = "EXISTING INTENTS:\n" + "\n".join(intent_summaries)
            
            prompt = f"""
You are an expert database analyst tasked with classifying database tables into business domain intents.
Your goal is to determine the best intent category for the given table.

{existing_intents_info}

TABLE TO CLASSIFY:
{table_context}

CLASSIFICATION RULES:
1. EXACT_MATCH: Table clearly fits an existing intent (confidence ≥ 0.8)
2. PARTIAL_MATCH: Table somewhat fits but intent description needs updating (confidence 0.6-0.79)
3. CREATE_NEW: Table doesn't fit any existing intent well (confidence < 0.6)

IMPORTANT GUIDELINES:
- Consider table name, description, columns, and relationships
- Focus on business domain (HR, Finance, Sales, Inventory, etc.)
- Be specific about why a table fits or doesn't fit
- For partial matches, suggest how to improve the intent description
- For new intents, suggest a comprehensive description and keywords

Respond in JSON format:
{{
    "intent_name": "suggested or existing intent name",
    "confidence_score": 0.85,
    "action_type": "exact_match|partial_match_update|create_new",
    "suggested_description": "detailed intent description",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "reasoning": "detailed explanation of classification decision"
}}
"""
            
            print(f"🧠 Classifying table: {table_name}")
            response = self.llm.invoke(prompt)
            
            # Parse response
            response_text = response.content.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            result = json.loads(response_text.strip())
            
            # Create IntentMatch object
            action_type = ActionType(result['action_type'])
            intent_match = IntentMatch(
                intent_name=result['intent_name'],
                confidence_score=result['confidence_score'],
                action_type=action_type,
                suggested_description=result.get('suggested_description'),
                reasoning=result.get('reasoning')
            )
            
            # Store keywords for later use
            intent_match.keywords = result.get('keywords', [])
            
            return intent_match
            
        except Exception as e:
            print(f"❌ Error classifying intent for {table_name}: {e}")
            # Fallback to create new intent
            return IntentMatch(
                intent_name=f"Unknown_{table_name.split('_')[0]}",
                confidence_score=0.3,
                action_type=ActionType.CREATE_NEW,
                reasoning=f"Error in classification: {str(e)}"
            )
    
    def _classify_sub_intent(self, table_context: str, table_name: str, parent_intent: str) -> IntentMatch:
        """Classify a table into a sub-intent within the parent intent"""
        try:
            # Get existing sub-intents for this parent intent
            existing_sub_intents = {k: v for k, v in self.sub_intents.items() 
                                  if v.parent_intent.lower() == parent_intent.lower()}
            
            existing_sub_intents_info = ""
            if existing_sub_intents:
                sub_intent_summaries = []
                for sub_intent_name, sub_intent in existing_sub_intents.items():
                    sub_intent_summaries.append(f"- {sub_intent.name}: {sub_intent.description}")
                existing_sub_intents_info = f"EXISTING SUB-INTENTS FOR {parent_intent.upper()}:\n" + "\n".join(sub_intent_summaries)
            
            prompt = f"""
You are classifying a database table into sub-intents within the "{parent_intent}" domain.
Sub-intents are more specific categories within the main business domain.

{existing_sub_intents_info}

TABLE TO CLASSIFY:
{table_context}

PARENT INTENT: {parent_intent}

CLASSIFICATION RULES:
1. EXACT_MATCH: Table clearly fits an existing sub-intent
2. PARTIAL_MATCH: Table somewhat fits but sub-intent needs updating  
3. CREATE_NEW: Table needs a new sub-intent category

GUIDELINES:
- Sub-intents should be specific functional areas within {parent_intent}
- Examples for HR: Employee Management, Payroll, Recruitment, Performance
- Examples for Finance: Accounting, Budgeting, Invoicing, Expenses
- Examples for Sales: Customer Management, Orders, Products, Campaigns
- Consider table's specific business function

Respond in JSON format:
{{
    "intent_name": "sub-intent name",
    "confidence_score": 0.85,
    "action_type": "exact_match|partial_match_update|create_new",
    "suggested_description": "specific sub-intent description",
    "keywords": ["keyword1", "keyword2"],
    "reasoning": "classification reasoning"
}}
"""
            
            print(f"🎯 Classifying sub-intent for: {table_name} under {parent_intent}")
            response = self.llm.invoke(prompt)
            
            # Parse response
            response_text = response.content.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            result = json.loads(response_text.strip())
            
            action_type = ActionType(result['action_type'])
            sub_intent_match = IntentMatch(
                intent_name=result['intent_name'],
                confidence_score=result['confidence_score'],
                action_type=action_type,
                suggested_description=result.get('suggested_description'),
                reasoning=result.get('reasoning')
            )
            
            sub_intent_match.keywords = result.get('keywords', [])
            return sub_intent_match
            
        except Exception as e:
            print(f"❌ Error classifying sub-intent for {table_name}: {e}")
            return IntentMatch(
                intent_name=f"{parent_intent}_Specific",
                confidence_score=0.3,
                action_type=ActionType.CREATE_NEW,
                reasoning=f"Error in sub-intent classification: {str(e)}"
            )
    
    def _process_intent_match(self, intent_match: IntentMatch, table_name: str) -> str:
        """Process the intent match and update intents accordingly"""
        intent_key = intent_match.intent_name.lower()
        current_time = datetime.now().isoformat()
        
        if intent_match.action_type == ActionType.EXACT_MATCH:
            # Add table to existing intent
            if intent_key in self.intents:
                if table_name not in self.intents[intent_key].tables:
                    self.intents[intent_key].tables.append(table_name)
                    self.intents[intent_key].last_modified = current_time
                    self.intents[intent_key].table_count = len(self.intents[intent_key].tables)
                    self.stats["exact_matches"] += 1
                    print(f"  ✅ Added to existing intent: {intent_match.intent_name}")
                return intent_match.intent_name
            else:
                # Intent doesn't exist, create it
                intent_match.action_type = ActionType.CREATE_NEW
        
        elif intent_match.action_type == ActionType.PARTIAL_MATCH_UPDATE:
            # Update existing intent description and add table
            if intent_key in self.intents:
                # Update description if suggested
                if intent_match.suggested_description:
                    self.intents[intent_key].description = intent_match.suggested_description
                
                # Add new keywords
                if hasattr(intent_match, 'keywords'):
                    for keyword in intent_match.keywords:
                        if keyword.lower() not in [k.lower() for k in self.intents[intent_key].keywords]:
                            self.intents[intent_key].keywords.append(keyword)
                
                # Add table
                if table_name not in self.intents[intent_key].tables:
                    self.intents[intent_key].tables.append(table_name)
                    self.intents[intent_key].last_modified = current_time
                    self.intents[intent_key].table_count = len(self.intents[intent_key].tables)
                    self.stats["partial_matches"] += 1
                    self.stats["intents_updated"] += 1
                    print(f"  🔄 Updated and added to intent: {intent_match.intent_name}")
                return intent_match.intent_name
            else:
                # Intent doesn't exist, create it
                intent_match.action_type = ActionType.CREATE_NEW
        
        if intent_match.action_type == ActionType.CREATE_NEW:
            # Create new intent
            new_intent = Intent(
                name=intent_match.intent_name,
                description=intent_match.suggested_description or f"Tables related to {intent_match.intent_name}",
                tables=[table_name],
                keywords=getattr(intent_match, 'keywords', []),
                created_date=current_time,
                last_modified=current_time
            )
            
            self.intents[intent_key] = new_intent
            self.stats["intents_created"] += 1
            print(f"  🆕 Created new intent: {intent_match.intent_name}")
            return intent_match.intent_name
        
        return intent_match.intent_name
    
    def _process_sub_intent_match(self, sub_intent_match: IntentMatch, table_name: str, parent_intent: str) -> str:
        """Process the sub-intent match and update sub-intents accordingly"""
        sub_intent_key = sub_intent_match.intent_name.lower()
        current_time = datetime.now().isoformat()
        
        if sub_intent_match.action_type == ActionType.EXACT_MATCH:
            # Add table to existing sub-intent
            if sub_intent_key in self.sub_intents:
                if table_name not in self.sub_intents[sub_intent_key].tables:
                    self.sub_intents[sub_intent_key].tables.append(table_name)
                    self.sub_intents[sub_intent_key].last_modified = current_time
                    self.sub_intents[sub_intent_key].table_count = len(self.sub_intents[sub_intent_key].tables)
                    print(f"    ✅ Added to existing sub-intent: {sub_intent_match.intent_name}")
                return sub_intent_match.intent_name
            else:
                sub_intent_match.action_type = ActionType.CREATE_NEW
        
        elif sub_intent_match.action_type == ActionType.PARTIAL_MATCH_UPDATE:
            # Update existing sub-intent
            if sub_intent_key in self.sub_intents:
                if sub_intent_match.suggested_description:
                    self.sub_intents[sub_intent_key].description = sub_intent_match.suggested_description
                
                if hasattr(sub_intent_match, 'keywords'):
                    for keyword in sub_intent_match.keywords:
                        if keyword.lower() not in [k.lower() for k in self.sub_intents[sub_intent_key].keywords]:
                            self.sub_intents[sub_intent_key].keywords.append(keyword)
                
                if table_name not in self.sub_intents[sub_intent_key].tables:
                    self.sub_intents[sub_intent_key].tables.append(table_name)
                    self.sub_intents[sub_intent_key].last_modified = current_time
                    self.sub_intents[sub_intent_key].table_count = len(self.sub_intents[sub_intent_key].tables)
                    self.stats["sub_intents_updated"] += 1
                    print(f"    🔄 Updated sub-intent: {sub_intent_match.intent_name}")
                return sub_intent_match.intent_name
            else:
                sub_intent_match.action_type = ActionType.CREATE_NEW
        
        if sub_intent_match.action_type == ActionType.CREATE_NEW:
            # Create new sub-intent
            new_sub_intent = SubIntent(
                name=sub_intent_match.intent_name,
                description=sub_intent_match.suggested_description or f"Specific {parent_intent} functionality",
                parent_intent=parent_intent,
                tables=[table_name],
                keywords=getattr(sub_intent_match, 'keywords', []),
                created_date=current_time,
                last_modified=current_time
            )
            
            self.sub_intents[sub_intent_key] = new_sub_intent
            self.stats["sub_intents_created"] += 1
            print(f"    🆕 Created new sub-intent: {sub_intent_match.intent_name}")
            return sub_intent_match.intent_name
        
        return sub_intent_match.intent_name
    
    def process_all_tables(self):
        """Process all tables and classify them into intents and sub-intents"""
        if not self.input_file.exists():
            print(f"❌ Input file '{self.input_file}' not found!")
            return
        
        print("🚀 Starting Self-Learning Intent Classification")
        print("=" * 60)
        
        # Load table data
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tables = data.get('tables', [])
        total_tables = len(tables)
        
        print(f"📊 Found {total_tables} tables to classify")
        print(f"📚 Starting with {len(self.intents)} existing intents")
        print(f"🎯 Starting with {len(self.sub_intents)} existing sub-intents")
        
        # Process each table
        for i, table_data in enumerate(tables):
            table_name = table_data.get('table_name', f'table_{i}')
            
            try:
                # Print progress
                percent = ((i + 1) / total_tables) * 100
                print(f"\n[{i+1}/{total_tables}] ({percent:.1f}%) Processing: {table_name}")
                
                # Extract table context
                table_context = self._extract_table_context(table_data)
                
                # Classify into intent
                intent_match = self._classify_intent(table_context, table_name)
                print(f"  🎯 Intent: {intent_match.intent_name} (confidence: {intent_match.confidence_score:.2f})")
                
                # Process intent match
                assigned_intent = self._process_intent_match(intent_match, table_name)
                
                # Classify into sub-intent
                sub_intent_match = self._classify_sub_intent(table_context, table_name, assigned_intent)
                print(f"    🎯 Sub-intent: {sub_intent_match.intent_name} (confidence: {sub_intent_match.confidence_score:.2f})")
                
                # Process sub-intent match
                self._process_sub_intent_match(sub_intent_match, table_name, assigned_intent)
                
                self.stats["tables_processed"] += 1
                
                # Rate limiting
                if i < total_tables - 1:
                    time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                print(f"❌ Error processing {table_name}: {e}")
                traceback.print_exc()
                continue
        
        # Save results
        print("\n" + "=" * 60)
        print("💾 Saving classification results...")
        
        self._save_intents()
        self._save_sub_intents()
        
        # Print statistics
        self._print_statistics()
        
        print("✨ Intent classification completed successfully!")
    
    def _print_statistics(self):
        """Print processing statistics"""
        print("\n📊 CLASSIFICATION STATISTICS")
        print("=" * 40)
        print(f"Tables Processed: {self.stats['tables_processed']}")
        print(f"Total Intents: {len(self.intents)} ({self.stats['intents_created']} new, {self.stats['intents_updated']} updated)")
        print(f"Total Sub-intents: {len(self.sub_intents)} ({self.stats['sub_intents_created']} new, {self.stats['sub_intents_updated']} updated)")
        print(f"Exact Matches: {self.stats['exact_matches']}")
        print(f"Partial Matches: {self.stats['partial_matches']}")
        
        # Intent distribution
        print(f"\n📈 INTENT DISTRIBUTION")
        print("-" * 30)
        intent_stats = [(intent.name, intent.table_count) for intent in self.intents.values()]
        intent_stats.sort(key=lambda x: x[1], reverse=True)
        
        for intent_name, table_count in intent_stats[:10]:  # Top 10
            print(f"{intent_name}: {table_count} tables")
    
    def generate_summary_report(self, output_file: str = "intent_classification_report.json"):
        """Generate a comprehensive summary report"""
        try:
            report = {
                "metadata": {
                    "generated_date": datetime.now().isoformat(),
                    "total_intents": len(self.intents),
                    "total_sub_intents": len(self.sub_intents),
                    "total_tables_classified": self.stats["tables_processed"]
                },
                "statistics": self.stats,
                "intent_summary": [],
                "sub_intent_summary": [],
                "recommendations": []
            }
            
            # Intent summary
            for intent in sorted(self.intents.values(), key=lambda x: x.table_count, reverse=True):
                intent_summary = {
                    "name": intent.name,
                    "description": intent.description,
                    "table_count": intent.table_count,
                    "tables": intent.tables,
                    "keywords": intent.keywords,
                    "sub_intents": [si.name for si in self.sub_intents.values() if si.parent_intent.lower() == intent.name.lower()]
                }
                report["intent_summary"].append(intent_summary)
            
            # Sub-intent summary  
            for sub_intent in sorted(self.sub_intents.values(), key=lambda x: x.table_count, reverse=True):
                sub_intent_summary = {
                    "name": sub_intent.name,
                    "parent_intent": sub_intent.parent_intent,
                    "description": sub_intent.description,
                    "table_count": sub_intent.table_count,
                    "tables": sub_intent.tables,
                    "keywords": sub_intent.keywords
                }
                report["sub_intent_summary"].append(sub_intent_summary)
            
            # Add recommendations
            report["recommendations"] = self._generate_recommendations()
            
            # Save report
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"📋 Summary report saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error generating summary report: {e}")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on classification results"""
        recommendations = []
        
        # Check for intents with too few tables
        small_intents = [intent for intent in self.intents.values() if intent.table_count <= 2]
        if small_intents:
            recommendations.append(f"Consider reviewing {len(small_intents)} intents with ≤2 tables for potential consolidation")
        
        # Check for intents with too many tables
        large_intents = [intent for intent in self.intents.values() if intent.table_count >= 20]
        if large_intents:
            recommendations.append(f"Consider breaking down {len(large_intents)} intents with ≥20 tables into more specific sub-categories")
        
        # Check for sub-intents without parent relationship
        orphaned_sub_intents = []
        for sub_intent in self.sub_intents.values():
            parent_exists = any(intent.name.lower() == sub_intent.parent_intent.lower() for intent in self.intents.values())
            if not parent_exists:
                orphaned_sub_intents.append(sub_intent.name)
        
        if orphaned_sub_intents:
            recommendations.append(f"Found {len(orphaned_sub_intents)} sub-intents with missing parent intents - review relationships")
        
        # Check intent keyword overlap
        keyword_overlap = self._check_keyword_overlap()
        if keyword_overlap:
            recommendations.append(f"Found {len(keyword_overlap)} intent pairs with significant keyword overlap - consider consolidation")
        
        return recommendations
    
    def _check_keyword_overlap(self) -> List[Tuple[str, str]]:
        """Check for intents with overlapping keywords"""
        overlapping_pairs = []
        intent_list = list(self.intents.values())
        
        for i, intent1 in enumerate(intent_list):
            for intent2 in intent_list[i+1:]:
                # Calculate keyword overlap
                keywords1 = set(kw.lower() for kw in intent1.keywords)
                keywords2 = set(kw.lower() for kw in intent2.keywords)
                
                if keywords1 and keywords2:
                    overlap = len(keywords1.intersection(keywords2))
                    total_unique = len(keywords1.union(keywords2))
                    overlap_ratio = overlap / total_unique if total_unique > 0 else 0
                    
                    if overlap_ratio > 0.5:  # 50% overlap threshold
                        overlapping_pairs.append((intent1.name, intent2.name))
        
        return overlapping_pairs

class IntentValidationSystem:
    """System to validate and optimize intent classifications"""
    
    def __init__(self, classification_system: IntentClassificationSystem):
        self.system = classification_system
        self.llm = classification_system.llm
    
    def validate_classifications(self) -> Dict[str, Any]:
        """Validate current intent classifications and suggest improvements"""
        print("\n🔍 Starting Classification Validation...")
        
        validation_results = {
            "intent_quality_scores": {},
            "sub_intent_quality_scores": {},
            "consolidation_suggestions": [],
            "split_suggestions": [],
            "reclassification_suggestions": []
        }
        
        # Validate each intent
        for intent_name, intent in self.system.intents.items():
            quality_score = self._evaluate_intent_quality(intent)
            validation_results["intent_quality_scores"][intent_name] = quality_score
            
            # Suggest improvements based on quality score
            if quality_score < 0.6:
                suggestions = self._suggest_intent_improvements(intent)
                validation_results["reclassification_suggestions"].extend(suggestions)
        
        # Check for consolidation opportunities
        consolidation_suggestions = self._suggest_consolidations()
        validation_results["consolidation_suggestions"] = consolidation_suggestions
        
        # Check for split opportunities
        split_suggestions = self._suggest_splits()
        validation_results["split_suggestions"] = split_suggestions
        
        return validation_results
    
    def _evaluate_intent_quality(self, intent: Intent) -> float:
        """Evaluate the quality of an intent classification"""
        try:
            # Create context about the intent and its tables
            tables_info = []
            for table_name in intent.tables[:5]:  # Sample first 5 tables
                tables_info.append(f"- {table_name}")
            
            tables_context = "\n".join(tables_info)
            
            prompt = f"""
Evaluate the quality and coherence of this intent classification on a scale of 0.0 to 1.0.

INTENT: {intent.name}
DESCRIPTION: {intent.description}
KEYWORDS: {', '.join(intent.keywords)}
TABLE COUNT: {intent.table_count}

SAMPLE TABLES IN THIS INTENT:
{tables_context}

EVALUATION CRITERIA:
1. Coherence: Do all tables logically belong to this business domain?
2. Specificity: Is the intent description specific enough to be useful?
3. Completeness: Does the description accurately cover all table types?
4. Keywords: Are keywords relevant and comprehensive?
5. Size: Is the intent appropriately sized (not too broad/narrow)?

Rate from 0.0 (poor) to 1.0 (excellent) and provide reasoning.

Respond in JSON:
{{
    "quality_score": 0.85,
    "reasoning": "detailed explanation of the score",
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"]
}}
"""
            
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            result = json.loads(response_text.strip())
            return result.get("quality_score", 0.5)
            
        except Exception as e:
            print(f"⚠️ Error evaluating intent {intent.name}: {e}")
            return 0.5
    
    def _suggest_intent_improvements(self, intent: Intent) -> List[Dict[str, Any]]:
        """Suggest improvements for a low-quality intent"""
        try:
            prompt = f"""
Suggest specific improvements for this intent classification:

INTENT: {intent.name}
DESCRIPTION: {intent.description}
KEYWORDS: {', '.join(intent.keywords)}
TABLES: {', '.join(intent.tables[:10])}

Suggest improvements in these areas:
1. Better intent name
2. Improved description
3. Additional/better keywords
4. Tables that might not belong
5. Potential sub-intent categories

Respond in JSON:
{{
    "suggested_name": "improved intent name",
    "suggested_description": "improved description",
    "suggested_keywords": ["keyword1", "keyword2"],
    "misplaced_tables": ["table1", "table2"],
    "sub_intent_opportunities": ["sub1", "sub2"]
}}
"""
            
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            result = json.loads(response_text.strip())
            
            return [{
                "intent_name": intent.name,
                "improvement_type": "quality_enhancement",
                "suggestions": result
            }]
            
        except Exception as e:
            print(f"⚠️ Error generating improvements for {intent.name}: {e}")
            return []
    
    def _suggest_consolidations(self) -> List[Dict[str, Any]]:
        """Suggest intent consolidation opportunities"""
        consolidation_suggestions = []
        
        # Find intents with similar keywords or small table counts
        intent_pairs = []
        intent_list = list(self.system.intents.values())
        
        for i, intent1 in enumerate(intent_list):
            for intent2 in intent_list[i+1:]:
                # Check similarity
                similarity_score = self._calculate_intent_similarity(intent1, intent2)
                if similarity_score > 0.7:
                    intent_pairs.append((intent1, intent2, similarity_score))
        
        for intent1, intent2, score in intent_pairs:
            consolidation_suggestions.append({
                "intent1": intent1.name,
                "intent2": intent2.name,
                "similarity_score": score,
                "combined_table_count": intent1.table_count + intent2.table_count,
                "suggestion": f"Consider consolidating '{intent1.name}' and '{intent2.name}' (similarity: {score:.2f})"
            })
        
        return consolidation_suggestions
    
    def _suggest_splits(self) -> List[Dict[str, Any]]:
        """Suggest intent split opportunities"""
        split_suggestions = []
        
        # Find intents with many tables that might need splitting
        for intent in self.system.intents.values():
            if intent.table_count >= 15:  # Threshold for large intents
                split_suggestion = self._analyze_split_opportunity(intent)
                if split_suggestion:
                    split_suggestions.append(split_suggestion)
        
        return split_suggestions
    
    def _calculate_intent_similarity(self, intent1: Intent, intent2: Intent) -> float:
        """Calculate similarity between two intents"""
        # Keyword similarity
        keywords1 = set(kw.lower() for kw in intent1.keywords)
        keywords2 = set(kw.lower() for kw in intent2.keywords)
        
        if keywords1 and keywords2:
            keyword_similarity = len(keywords1.intersection(keywords2)) / len(keywords1.union(keywords2))
        else:
            keyword_similarity = 0
        
        # Description similarity (simple word overlap)
        desc1_words = set(intent1.description.lower().split())
        desc2_words = set(intent2.description.lower().split())
        
        if desc1_words and desc2_words:
            desc_similarity = len(desc1_words.intersection(desc2_words)) / len(desc1_words.union(desc2_words))
        else:
            desc_similarity = 0
        
        # Weighted average
        return (keyword_similarity * 0.6 + desc_similarity * 0.4)
    
    def _analyze_split_opportunity(self, intent: Intent) -> Optional[Dict[str, Any]]:
        """Analyze if an intent should be split into smaller intents"""
        try:
            tables_sample = intent.tables[:10]  # Sample for analysis
            
            prompt = f"""
Analyze if this intent should be split into smaller, more specific intents:

INTENT: {intent.name}
DESCRIPTION: {intent.description}
TABLE COUNT: {intent.table_count}
SAMPLE TABLES: {', '.join(tables_sample)}

Based on the table names and current description, suggest if this intent should be split.
If yes, suggest 2-4 more specific sub-categories.

Respond in JSON:
{{
    "should_split": true/false,
    "reasoning": "explanation",
    "suggested_splits": [
        {{
            "name": "sub-intent name",
            "description": "sub-intent description",
            "estimated_tables": ["table1", "table2"]
        }}
    ]
}}
"""
            
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            result = json.loads(response_text.strip())
            
            if result.get("should_split", False):
                return {
                    "intent_name": intent.name,
                    "table_count": intent.table_count,
                    "split_recommendation": result
                }
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error analyzing split for {intent.name}: {e}")
            return None

def main():
    """Main execution function"""
    print("🚀 Self-Learning Intent Classification System")
    print("🧠 Powered by Advanced LLM Analysis")
    print("=" * 60)
    
    # Initialize the classification system
    classifier = IntentClassificationSystem(
        input_file= r"C:\Users\Nilab\Desktop\Esap_test\KnowladgeBase\all_tables_with_descriptions_v1.json",
        intents_file="intent_classification.json",
        sub_intents_file="sub_intent_classification.json",
        confidence_threshold=0.7,
        rate_limit_delay=1
    )
    
    try:
        # Process all tables
        classifier.process_all_tables()
        
        # Generate summary report
        classifier.generate_summary_report("intent_classification_report.json")
        
        # Run validation system
        print("\n🔍 Running Classification Validation...")
        validator = IntentValidationSystem(classifier)
        validation_results = validator.validate_classifications()
        
        # Save validation results
        with open("intent_validation_report.json", 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False)
        
        print("📋 Validation report saved to: intent_validation_report.json")
        
        # Print final summary
        print("\n" + "=" * 60)
        print("✨ Intent Classification System Completed Successfully!")
        print("\n📁 Generated Files:")
        print("  - intent_classification.json (Main intents)")
        print("  - sub_intent_classification.json (Sub-intents)")
        print("  - intent_classification_report.json (Summary report)")
        print("  - intent_validation_report.json (Validation & optimization)")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Process interrupted by user")
        print("💾 Saving current progress...")
        classifier._save_intents()
        classifier._save_sub_intents()
        print("✅ Progress saved successfully")
        
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        traceback.print_exc()
        # Still try to save progress
        try:
            classifier._save_intents()
            classifier._save_sub_intents()
            print("💾 Emergency save completed")
        except:
            print("❌ Could not save progress")

if __name__ == "__main__":
    main()