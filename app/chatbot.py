 # Complete Conversational Chatbot System
# Install required packages first:
# pip install transformers torch sqlite3 sentence-transformers accelerate

import os
import json
import sqlite3
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings
warnings.filterwarnings("ignore")

class DatabaseManager:
    """Handles all database operations for conversation persistence"""
    
    def __init__(self, db_path: str = "chatbot_conversations.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
                system_prompt TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def create_session(self, system_prompt: str = None) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO sessions (session_id, system_prompt)
            VALUES (?, ?)
        ''', (session_id, system_prompt))
        conn.commit()
        conn.close()
        return session_id
    
    def save_message(self, session_id: str, role: str, content: str):
        """Save a message to the database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO conversations (session_id, role, content)
            VALUES (?, ?, ?)
        ''', (session_id, role, content))
        
        # Update session last_active
        conn.execute('''
            UPDATE sessions SET last_active = CURRENT_TIMESTAMP
            WHERE session_id = ?
        ''', (session_id,))
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Tuple[str, str, str]]:
        """Get conversation history for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT role, content, timestamp FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
        ''', (session_id, limit))
        history = cursor.fetchall()
        conn.close()
        return history
    
    def get_system_prompt(self, session_id: str) -> Optional[str]:
        """Get system prompt for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT system_prompt FROM sessions
            WHERE session_id = ?
        ''', (session_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    
    def list_sessions(self) -> List[Dict]:
        """List all conversation sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT s.session_id, s.created_at, s.last_active,
                   COUNT(c.id) as message_count
            FROM sessions s
            LEFT JOIN conversations c ON s.session_id = c.session_id
            GROUP BY s.session_id, s.created_at, s.last_active
            ORDER BY s.last_active DESC
        ''')
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'session_id': row[0],
                'created_at': row[1],
                'last_active': row[2],
                'message_count': row[3]
            })
        conn.close()
        return sessions
    
    def delete_session(self, session_id: str):
        """Delete a conversation session"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('DELETE FROM conversations WHERE session_id = ?', (session_id,))
        conn.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
        conn.commit()
        conn.close()
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
import torch
import uuid
from datetime import datetime

class ConversationalChatbot:
    """Main chatbot class with conversation memory"""

    def __init__(self,
                 model_name: str = "google/gemma-3-1b-it",
                 max_length: int = 512,
                 temperature: float = 0.7,
                 max_history_tokens: int = 1000,
                 use_database: bool = True):

        print(f"Initializing chatbot with model: {model_name}")

        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.max_history_tokens = max_history_tokens

        self._load_model()

        self.use_database = use_database
        if use_database:
            self.db = DatabaseManager()
        else:
            self.memory_conversations = {}

        print("Chatbot initialized successfully!")

    def _load_model(self):
        """Load the Gemma3 model and tokenizer following Google's official setup"""
        try:
            print("Loading Gemma tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            print("Loading Gemma model...")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = Gemma3ForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config
            ).eval()

        except Exception as e:
            print(f"Error loading Gemma model: {e}")
            raise

    def create_session(self, system_prompt: str = None) -> str:
        if self.use_database:
            return self.db.create_session(system_prompt)
        else:
            session_id = str(uuid.uuid4())
            self.memory_conversations[session_id] = {
                'history': [],
                'system_prompt': system_prompt,
                'created_at': datetime.now()
            }
            return session_id

    def _get_conversation_history(self, session_id: str):
        if self.use_database:
            raw_history = self.db.get_conversation_history(session_id)
            return [{'role': role, 'content': content, 'timestamp': timestamp}
                    for role, content, timestamp in raw_history]
        else:
            return self.memory_conversations.get(session_id, {}).get('history', [])

    def _save_message(self, session_id: str, role: str, content: str):
        if self.use_database:
            self.db.save_message(session_id, role, content)
        else:
            if session_id in self.memory_conversations:
                self.memory_conversations[session_id]['history'].append({
                    'role': role,
                    'content': content,
                    'timestamp': datetime.now()
                })

    def chat(self, session_id: str, user_message: str) -> str:
        try:
            self._save_message(session_id, 'user', user_message)

            system_prompt = self.db.get_system_prompt(session_id) if self.use_database else \
                self.memory_conversations.get(session_id, {}).get('system_prompt')

            history = self._get_conversation_history(session_id)
            message_list = []

            if system_prompt:
                message_list.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
            for msg in history[-10:]:
                message_list.append({
                    "role": msg['role'],
                    "content": [{"type": "text", "text": msg['content']}]
                })
            message_list.append({"role": "user", "content": [{"type": "text", "text": user_message}]})

            inputs = self.tokenizer.apply_chat_template(
                [message_list],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            if torch.cuda.is_available():
                inputs = inputs.to(torch.bfloat16)

            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=150)

            response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            response = response[len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()

            if len(response) < 3:
                response = "I understand. Could you tell me more about that?"

            self._save_message(session_id, 'assistant', response)
            return response

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error. Could you please try again."

    def export_conversation(self, session_id: str):
        return self._get_conversation_history(session_id)

    def delete_session(self, session_id: str):
        if self.use_database:
            self.db.delete_session(session_id)
        else:
            if session_id in self.memory_conversations:
                del self.memory_conversations[session_id]

    def list_sessions(self):
        if self.use_database:
            return self.db.list_sessions()
        else:
            sessions = []
            for session_id, conv in self.memory_conversations.items():
                sessions.append({
                    'session_id': session_id,
                    'created_at': conv['created_at'],
                    'message_count': len(conv['history'])
                })
            return sessions

    def get_session_info(self, session_id: str):
        if self.use_database:
            sessions = self.db.list_sessions()
            for session in sessions:
                if session['session_id'] == session_id:
                    return session
            return {}
        else:
            if session_id in self.memory_conversations:
                conv = self.memory_conversations[session_id]
                return {
                    'session_id': session_id,
                    'created_at': conv['created_at'],
                    'message_count': len(conv['history'])
                }
            return {}


# class ConversationalChatbot:
#     """Main chatbot class with conversation memory"""
    
#     def __init__(self, 
#                  model_name: str = "microsoft/DialoGPT-small",
#                  max_length: int = 512,
#                  temperature: float = 0.7,
#                  max_history_tokens: int = 1000,
#                  use_database: bool = True):
        
#         print(f"Initializing chatbot with model: {model_name}")
        
#         # Initialize model and tokenizer
#         self.model_name = model_name
#         self.max_length = max_length
#         self.temperature = temperature
#         self.max_history_tokens = max_history_tokens
        
#         # Load model
#         self._load_model()
        
#         # Initialize database or in-memory storage
#         self.use_database = use_database
#         if use_database:
#             self.db = DatabaseManager()
#         else:
#             self.memory_conversations = {}
        
#         print("Chatbot initialized successfully!")
    
#     def _load_model(self):
#         """Load the language model and tokenizer"""
#         try:
#             print("Loading tokenizer...")
#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
#             # Add padding token if it doesn't exist
#             if self.tokenizer.pad_token is None:
#                 self.tokenizer.pad_token = self.tokenizer.eos_token
            
#             print("Loading model...")
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 self.model_name,
#                 torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#                 device_map="auto" if torch.cuda.is_available() else None
#             )
            
#             # Alternative: Use pipeline for easier usage
#             self.generator = pipeline(
#                 "text-generation",
#                 model=self.model,
#                 tokenizer=self.tokenizer,
#                 torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#                 device_map="auto" if torch.cuda.is_available() else None
#             )
            
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             print("Falling back to a lighter model...")
#             # Fallback to a smaller model
#             self.model_name = "gpt2"
#             self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
#             self.tokenizer.pad_token = self.tokenizer.eos_token
            
#             self.generator = pipeline(
#                 "text-generation",
#                 model="gpt2",
#                 tokenizer=self.tokenizer
#             )
    
#     def create_session(self, system_prompt: str = None) -> str:
#         """Create a new conversation session"""
#         if self.use_database:
#             return self.db.create_session(system_prompt)
#         else:
#             session_id = str(uuid.uuid4())
#             self.memory_conversations[session_id] = {
#                 'history': [],
#                 'system_prompt': system_prompt,
#                 'created_at': datetime.now()
#             }
#             return session_id
    
#     def _get_conversation_history(self, session_id: str) -> List[Dict]:
#         """Get formatted conversation history"""
#         if self.use_database:
#             raw_history = self.db.get_conversation_history(session_id)
#             return [{'role': role, 'content': content, 'timestamp': timestamp} 
#                    for role, content, timestamp in raw_history]
#         else:
#             return self.memory_conversations.get(session_id, {}).get('history', [])
    
#     def _save_message(self, session_id: str, role: str, content: str):
#         """Save a message to storage"""
#         if self.use_database:
#             self.db.save_message(session_id, role, content)
#         else:
#             if session_id in self.memory_conversations:
#                 self.memory_conversations[session_id]['history'].append({
#                     'role': role,
#                     'content': content,
#                     'timestamp': datetime.now()
#                 })
    
#     def _format_conversation_for_model(self, session_id: str, new_message: str) -> str:
#         """Format conversation history for model input"""
#         history = self._get_conversation_history(session_id)
        
#         # Get system prompt
#         system_prompt = None
#         if self.use_database:
#             system_prompt = self.db.get_system_prompt(session_id)
#         else:
#             system_prompt = self.memory_conversations.get(session_id, {}).get('system_prompt')
        
#         # Build conversation text
#         conversation_parts = []
        
#         if system_prompt:
#             conversation_parts.append(f"System: {system_prompt}")
        
#         # Add conversation history
#         for msg in history[-10:]:  # Keep last 10 exchanges
#             role = "Human" if msg['role'] == 'user' else "Assistant"
#             conversation_parts.append(f"{role}: {msg['content']}")
        
#         # Add new user message
#         conversation_parts.append(f"Human: {new_message}")
#         conversation_parts.append("Assistant:")
        
#         return "\n".join(conversation_parts)
    
#     def _trim_response(self, response: str) -> str:
#         """Clean and trim the model response"""
#         # Remove the input prompt from response if present
#         if "Human:" in response:
#             response = response.split("Human:")[0]
        
#         if "Assistant:" in response:
#             response = response.replace("Assistant:", "").strip()
        
#         # Remove incomplete sentences at the end
#         sentences = response.split('. ')
#         if len(sentences) > 1 and not sentences[-1].endswith('.'):
#             response = '. '.join(sentences[:-1]) + '.'
        
#         return response.strip()
    
#     def chat(self, session_id: str, user_message: str) -> str:
#         """Generate a response to user input"""
#         try:
#             # Save user message
#             self._save_message(session_id, 'user', user_message)
            
#             # Format conversation for model
#             prompt = self._format_conversation_for_model(session_id, user_message)
            
#             # Generate response
#             outputs = self.generator(
#                 prompt,
#                 max_length=len(self.tokenizer.encode(prompt)) + 150,
#                 num_return_sequences=1,
#                 temperature=self.temperature,
#                 do_sample=True,
#                 pad_token_id=self.tokenizer.eos_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id,
#                 no_repeat_ngram_size=2
#             )
            
#             # Extract response
#             generated_text = outputs[0]['generated_text']
#             response = generated_text[len(prompt):].strip()
#             response = self._trim_response(response)
            
#             # If response is empty or too short, provide a default
#             if len(response) < 3:
#                 response = "I understand. Could you tell me more about that?"
            
#             # Save assistant response
#             self._save_message(session_id, 'assistant', response)
            
#             return response
            
#         except Exception as e:
#             print(f"Error generating response: {e}")
#             return "I'm sorry, I encountered an error. Could you please try again?"
    
#     def get_session_info(self, session_id: str) -> Dict:
#         """Get information about a conversation session"""
#         if self.use_database:
#             sessions = self.db.list_sessions()
#             for session in sessions:
#                 if session['session_id'] == session_id:
#                     return session
#             return {}
#         else:
#             if session_id in self.memory_conversations:
#                 conv = self.memory_conversations[session_id]
#                 return {
#                     'session_id': session_id,
#                     'created_at': conv['created_at'],
#                     'message_count': len(conv['history'])
#                 }
#             return {}
    
#     def list_sessions(self) -> List[Dict]:
#         """List all conversation sessions"""
#         if self.use_database:
#             return self.db.list_sessions()
#         else:
#             sessions = []
#             for session_id, conv in self.memory_conversations.items():
#                 sessions.append({
#                     'session_id': session_id,
#                     'created_at': conv['created_at'],
#                     'message_count': len(conv['history'])
#                 })
#             return sessions
    
#     def export_conversation(self, session_id: str) -> List[Dict]:
#         """Export conversation history"""
#         return self._get_conversation_history(session_id)
    
#     def delete_session(self, session_id: str):
#         """Delete a conversation session"""
#         if self.use_database:
#             self.db.delete_session(session_id)
#         else:
#             if session_id in self.memory_conversations:
#                 del self.memory_conversations[session_id]

class ChatbotInterface:
    """Simple command-line interface for the chatbot"""
    
    def __init__(self, chatbot: ConversationalChatbot):
        self.chatbot = chatbot
        self.current_session = None
    
    def print_menu(self):
        """Print the main menu"""
        print("\n" + "="*50)
        print("🤖 CONVERSATIONAL CHATBOT")
        print("="*50)
        print("1. Start new conversation")
        print("2. Continue existing conversation")
        print("3. List all conversations")
        print("4. Export conversation")
        print("5. Delete conversation")
        print("6. Quit")
        print("="*50)
    
    def start_new_conversation(self):
        """Start a new conversation"""
        print("\n🆕 Starting new conversation...")
        
        # Ask for system prompt
        system_prompt = input("Enter system prompt (optional, press Enter to skip): ").strip()
        if not system_prompt:
            system_prompt = "You are a helpful AI assistant. Be friendly and informative."
        
        self.current_session = self.chatbot.create_session(system_prompt)
        print(f"✅ New conversation started! Session ID: {self.current_session[:8]}...")
        print(f"💡 System prompt: {system_prompt}")
        
        self.chat_loop()
    
    def continue_conversation(self):
        """Continue an existing conversation"""
        sessions = self.chatbot.list_sessions()
        
        if not sessions:
            print("\n❌ No existing conversations found.")
            return
        
        print("\n📋 Existing conversations:")
        for i, session in enumerate(sessions):
            created = session['created_at'][:19] if len(session['created_at']) > 19 else session['created_at']
            print(f"{i+1}. {session['session_id'][:8]}... ({session['message_count']} messages, {created})")
        
        try:
            choice = int(input(f"\nSelect conversation (1-{len(sessions)}): ")) - 1
            if 0 <= choice < len(sessions):
                self.current_session = sessions[choice]['session_id']
                print(f"✅ Continuing conversation {self.current_session[:8]}...")
                
                # Show last few messages
                history = self.chatbot.export_conversation(self.current_session)
                if history:
                    print("\n📜 Recent conversation:")
                    for msg in history[-4:]:  # Show last 4 messages
                        role_emoji = "👤" if msg['role'] == 'user' else "🤖"
                        print(f"{role_emoji} {msg['role'].title()}: {msg['content']}")
                
                self.chat_loop()
            else:
                print("❌ Invalid selection.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    def list_conversations(self):
        """List all conversations"""
        sessions = self.chatbot.list_sessions()
        
        if not sessions:
            print("\n❌ No conversations found.")
            return
        
        print(f"\n📋 Found {len(sessions)} conversation(s):")
        print("-" * 80)
        for session in sessions:
            created = session['created_at'][:19] if len(session['created_at']) > 19 else session['created_at']
            last_active = session.get('last_active', 'N/A')[:19] if session.get('last_active') else 'N/A'
            print(f"ID: {session['session_id'][:8]}...")
            print(f"Messages: {session['message_count']}")
            print(f"Created: {created}")
            print(f"Last active: {last_active}")
            print("-" * 80)
    
    def export_conversation(self):
        """Export a conversation"""
        sessions = self.chatbot.list_sessions()
        
        if not sessions:
            print("\n❌ No conversations found.")
            return
        
        print("\n📋 Select conversation to export:")
        for i, session in enumerate(sessions):
            print(f"{i+1}. {session['session_id'][:8]}... ({session['message_count']} messages)")
        
        try:
            choice = int(input(f"\nSelect conversation (1-{len(sessions)}): ")) - 1
            if 0 <= choice < len(sessions):
                session_id = sessions[choice]['session_id']
                history = self.chatbot.export_conversation(session_id)
                
                filename = f"conversation_{session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(history, f, indent=2, default=str, ensure_ascii=False)
                
                print(f"✅ Conversation exported to {filename}")
            else:
                print("❌ Invalid selection.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    def delete_conversation(self):
        """Delete a conversation"""
        sessions = self.chatbot.list_sessions()
        
        if not sessions:
            print("\n❌ No conversations found.")
            return
        
        print("\n📋 Select conversation to delete:")
        for i, session in enumerate(sessions):
            print(f"{i+1}. {session['session_id'][:8]}... ({session['message_count']} messages)")
        
        try:
            choice = int(input(f"\nSelect conversation (1-{len(sessions)}): ")) - 1
            if 0 <= choice < len(sessions):
                session_id = sessions[choice]['session_id']
                confirm = input(f"⚠️  Delete conversation {session_id[:8]}...? (yes/no): ")
                if confirm.lower() in ['yes', 'y']:
                    self.chatbot.delete_session(session_id)
                    print("✅ Conversation deleted.")
                else:
                    print("❌ Deletion cancelled.")
            else:
                print("❌ Invalid selection.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    def chat_loop(self):
        """Main chat loop"""
        print("\n💬 Chat started! Type 'quit' to return to menu, 'help' for commands.")
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Returning to main menu...")
                    break
                elif user_input.lower() == 'help':
                    print("\n📖 Commands:")
                    print("- quit/exit/q: Return to main menu")
                    print("- help: Show this help")
                    print("- clear: Clear screen")
                    continue
                elif user_input.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                elif not user_input:
                    continue
                
                # Generate response
                print("🤖 Assistant: ", end="", flush=True)
                response = self.chatbot.chat(self.current_session, user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 Returning to main menu...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def run(self):
        """Run the chatbot interface"""
        print("🚀 Welcome to the Conversational Chatbot!")
        print("This chatbot remembers your conversation history.")
        
        while True:
            try:
                self.print_menu()
                choice = input("\nSelect option (1-6): ").strip()
                
                if choice == '1':
                    self.start_new_conversation()
                elif choice == '2':
                    self.continue_conversation()
                elif choice == '3':
                    self.list_conversations()
                elif choice == '4':
                    self.export_conversation()
                elif choice == '5':
                    self.delete_conversation()
                elif choice == '6':
                    print("\n👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid option. Please select 1-6.")
                
                if choice != '6':
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

# Test Functions
def test_basic_functionality():
    """Test basic chatbot functionality"""
    print("🧪 Testing basic functionality...")
    
    # Create chatbot instance
    chatbot = ConversationalChatbot(use_database=False)
    
    # Create session
    session_id = chatbot.create_session("You are a helpful assistant.")
    
    # Test conversation
    test_messages = [
        "Hello, what's your name?",
        "Can you remember what I just asked?",
        "Tell me a joke"
    ]
    
    for msg in test_messages:
        print(f"\n👤 User: {msg}")
        response = chatbot.chat(session_id, msg)
        print(f"🤖 Bot: {response}")
    
    # Check conversation history
    history = chatbot.export_conversation(session_id)
    print(f"\n📊 Conversation has {len(history)} messages")
    
    print("✅ Basic functionality test completed!")

if __name__ == "__main__":
    # You can run different modes:
    
    # 1. Interactive mode (default)
    try:
        # Initialize chatbot (this might take a moment to download the model)
        chatbot = ConversationalChatbot(
            # model_name="microsoft/DialoGPT-small",  # Small model for testing
            use_database=True,  # Set to False for memory-only mode
            temperature=0.8
        )
        
        # Start interface
        interface = ChatbotInterface(chatbot)
        interface.run()
        
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        print("\n🔧 Running basic test instead...")
        test_basic_functionality()
    
    # 2. Test mode (uncomment to run)
    # test_basic_functionality()
    
    # 3. Simple chat mode (uncomment to run)
    # chatbot = ConversationalChatbot(use_database=False)
    # session_id = chatbot.create_session()
    # while True:
    #     user_input = input("You: ")
    #     if user_input.lower() == 'quit':
    #         break
    #     response = chatbot.chat(session_id, user_input)
    #     print(f"Bot: {response}")