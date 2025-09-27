# Create a new file memory_manager.py in your experiment_1 folder
import json
import threading
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

class ConversationMemory:
    def __init__(self, max_messages: int = 5):
        self.memory_file = Path(__file__).parent / "conversation_memory.json"
        self.max_messages = max_messages
        self.lock = threading.Lock()

    def load_memory(self) -> Dict:
        with self.lock:
            try:
                if self.memory_file.exists():
                    with open(self.memory_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                return {}
            except Exception as e:
                print(f"Error loading memory: {str(e)}")
                return {}

    def save_memory(self, memory: Dict):
        with self.lock:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory, f, indent=2)

    def add_conversation(self, user_id: str, question: str, query: str, results: Optional[List] = None):
        memory = self.load_memory()
        
        # Initialize user memory if not exists
        if user_id not in memory:
            memory[user_id] = []

        # Create new conversation entry
        conversation = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "query": query,
            "results_summary": f"Found {len(results) if results else 0} records"
        }

        # Add new conversation and keep only last max_messages
        memory[user_id].append(conversation)
        memory[user_id] = memory[user_id][-self.max_messages:]
        
        self.save_memory(memory)

    def get_conversation_history(self, user_id: str) -> List:
        memory = self.load_memory()
        return memory.get(user_id, [])

    def clear_conversation_history(self, user_id: str):
        memory = self.load_memory()
        if user_id in memory:
            memory[user_id] = []
            self.save_memory(memory)