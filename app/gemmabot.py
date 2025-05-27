# Modified Conversational Chatbot for Gemma Models
# Install required packages:
# pip install transformers torch accelerate bitsandbytes

import os
import json
import sqlite3
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
warnings.filterwarnings("ignore")

class DatabaseManager:
    """Handles all database operations for conversation persistence"""
    
    def __init__(self, db_path: str = "gemma_chatbot_conversations.db"):
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

class GemmaConversationalChatbot:
    """Chatbot specifically configured for Gemma models"""
    
    def __init__(self, 
                 model_name: str = "google/gemma-3-4b-it",  # Gemma 3 4B Instruction-tuned
                 max_new_tokens: int = 256,
                 temperature: float = 0.7,
                 use_4bit: bool = True,  # Enable 4-bit quantization for memory efficiency
                 use_database: bool = True,
                 force_cpu: bool = False):  # Force CPU mode for compatibility
        
        print(f"🔥 Initializing Gemma 3 chatbot with model: {model_name}")
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_4bit = use_4bit
        self.force_cpu = force_cpu
        
        # Load model with optimizations
        self._load_gemma_model()
        
        # Initialize storage
        self.use_database = use_database
        if use_database:
            self.db = DatabaseManager()
        else:
            self.memory_conversations = {}
        
        print("✅ Gemma 3 chatbot initialized successfully!")
    
    def _load_gemma_model(self):
        """Load Gemma model with proper configuration"""
        try:
            print("📥 Loading Gemma 3 tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Gemma 3 tokenizer configuration
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization for memory efficiency (skip if force_cpu)
            quantization_config = None
            if self.use_4bit and torch.cuda.is_available() and not self.force_cpu:
                print("⚡ Enabling 4-bit quantization for better memory usage...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            
            print("📥 Loading Gemma 3 model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                # Gemma 3 specific configurations - disable flash attention to avoid installation issues
                attn_implementation="eager",
            )
            
            print(f"🎯 Model loaded on: {next(self.model.parameters()).device}")
            
        except Exception as e:
            print(f"❌ Error loading Gemma 3 model: {e}")
            print("🔄 Falling back to smaller Gemma 3 model...")
            # Fallback to Gemma 3 1B
            self.model_name = "google/gemma-3-1b-it"
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load fallback model if main model fails"""
        try:
            print(f"Loading fallback model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Use simpler configuration for fallback
            quantization_config = None
            if self.use_4bit and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                attn_implementation="eager",  # Use eager attention for compatibility
            )
        except Exception as e:
            print(f"❌ Fallback also failed: {e}")
            print("🔄 Trying basic CPU configuration...")
            # Final fallback - basic CPU configuration
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    "google/gemma-3-1b-it",
                    torch_dtype=torch.float32,
                    device_map=None,
                    trust_remote_code=True,
                    attn_implementation="eager"
                )
                self.model_name = "google/gemma-3-1b-it"
                print("✅ Successfully loaded Gemma 3 1B on CPU")
            except Exception as final_error:
                print(f"❌ All fallbacks failed: {final_error}")
                raise RuntimeError("Could not load any Gemma model")
    
    def create_session(self, system_prompt: str = None) -> str:
        """Create a new conversation session"""
        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant. Be friendly, informative, and concise."
        
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
    
    def _get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get formatted conversation history"""
        if self.use_database:
            raw_history = self.db.get_conversation_history(session_id)
            return [{'role': role, 'content': content, 'timestamp': timestamp} 
                   for role, content, timestamp in raw_history]
        else:
            return self.memory_conversations.get(session_id, {}).get('history', [])
    
    def _save_message(self, session_id: str, role: str, content: str):
        """Save a message to storage"""
        if self.use_database:
            self.db.save_message(session_id, role, content)
        else:
            if session_id in self.memory_conversations:
                self.memory_conversations[session_id]['history'].append({
                    'role': role,
                    'content': content,
                    'timestamp': datetime.now()
                })
    
    def _format_gemma_prompt(self, session_id: str, new_message: str) -> str:
        """Format conversation for Gemma 3 model using proper chat template"""
        history = self._get_conversation_history(session_id)
        
        # Get system prompt
        system_prompt = None
        if self.use_database:
            system_prompt = self.db.get_system_prompt(session_id)
        else:
            system_prompt = self.memory_conversations.get(session_id, {}).get('system_prompt')
        
        # Build conversation in Gemma 3 chat format
        messages = []
        
        # Add system message if exists
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history (keep last 10 exchanges)
        for msg in history[-10:]:
            messages.append({"role": msg['role'], "content": msg['content']})
        
        # Add new user message
        messages.append({"role": "user", "content": new_message})
        
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback to manual formatting
            prompt_parts = []
            for msg in messages:
                if msg['role'] == 'system':
                    prompt_parts.append(f"<start_of_turn>system\n{msg['content']}<end_of_turn>")
                elif msg['role'] == 'user':
                    prompt_parts.append(f"<start_of_turn>user\n{msg['content']}<end_of_turn>")
                elif msg['role'] == 'assistant':
                    prompt_parts.append(f"<start_of_turn>model\n{msg['content']}<end_of_turn>")
            
            prompt_parts.append("<start_of_turn>model\n")
            prompt = "\n".join(prompt_parts)
        
        return prompt
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using Gemma 3 model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated part
            response = full_response[len(prompt):].strip()
            
            # Clean up response
            response = self._clean_gemma_response(response)
            
            return response
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return "I apologize, but I encountered an error generating a response. Could you please try again?"
    
    def _clean_gemma_response(self, response: str) -> str:
        """Clean and format Gemma 3 model response"""
        # Remove any remaining special tokens
        response = response.replace("<start_of_turn>", "").replace("<end_of_turn>", "")
        response = response.replace("model\n", "").strip()
        
        # Remove incomplete sentences at the end
        if response and not response.endswith(('.', '!', '?', ':', ';')):
            sentences = response.split('. ')
            if len(sentences) > 1:
                response = '. '.join(sentences[:-1]) + '.'
        
        # Ensure minimum response length
        if len(response.strip()) < 3:
            response = "I understand. Could you please tell me more about that?"
        
        return response.strip()
    
    def chat(self, session_id: str, user_message: str) -> str:
        """Generate a response to user input"""
        try:
            # Save user message
            self._save_message(session_id, 'user', user_message)
            
            # Format prompt for Gemma 3
            prompt = self._format_gemma_prompt(session_id, user_message)
            
            # Generate response
            response = self._generate_response(prompt)
            
            # Save assistant response
            self._save_message(session_id, 'assistant', response)
            
            return response
            
        except Exception as e:
            print(f"❌ Chat error: {e}")
            return "I'm sorry, I encountered an error. Could you please try again?"
    
    def get_session_info(self, session_id: str) -> Dict:
        """Get information about a conversation session"""
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
    
    def list_sessions(self) -> List[Dict]:
        """List all conversation sessions"""
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
    
    def export_conversation(self, session_id: str) -> List[Dict]:
        """Export conversation history"""
        return self._get_conversation_history(session_id)
    
    def delete_session(self, session_id: str):
        """Delete a conversation session"""
        if self.use_database:
            self.db.delete_session(session_id)
        else:
            if session_id in self.memory_conversations:
                del self.memory_conversations[session_id]

class GemmaChatInterface:
    """Command-line interface optimized for Gemma 3 chatbot"""
    
    def __init__(self, chatbot: GemmaConversationalChatbot):
        self.chatbot = chatbot
        self.current_session = None
    
    def print_menu(self):
        """Print the main menu"""
        print("\n" + "="*50)
        print("🔥 GEMMA 3 CONVERSATIONAL CHATBOT")
        print("="*50)
        print("1. Start new conversation")
        print("2. Continue existing conversation")
        print("3. List all conversations")
        print("4. Export conversation")
        print("5. Delete conversation")
        print("6. Model info")
        print("7. Quit")
        print("="*50)
    
    def show_model_info(self):
        """Show model information"""
        print(f"\n🔥 Model Information:")
        print(f"Model: {self.chatbot.model_name}")
        print(f"Device: {next(self.chatbot.model.parameters()).device}")
        print(f"Memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB" if torch.cuda.is_available() else "CPU mode")
        print(f"4-bit quantization: {'Enabled' if self.chatbot.use_4bit else 'Disabled'}")
    
    def start_new_conversation(self):
        """Start a new conversation"""
        print("\n🆕 Starting new Gemma 3 conversation...")
        
        system_prompt = input("Enter system prompt (press Enter for default): ").strip()
        if not system_prompt:
            system_prompt = "You are Gemma 3, a helpful AI assistant created by Google. Be friendly, informative, and concise in your responses."
        
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
                
                # Show recent messages
                history = self.chatbot.export_conversation(self.current_session)
                if history:
                    print("\n📜 Recent conversation:")
                    for msg in history[-4:]:
                        role_emoji = "👤" if msg['role'] == 'user' else "🔥"
                        print(f"{role_emoji} {msg['role'].title()}: {msg['content']}")
                
                self.chat_loop()
            else:
                print("❌ Invalid selection.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    def chat_loop(self):
        """Main chat loop optimized for Gemma"""
        print("\n💬 Gemma 3 chat started! Type 'quit' to return to menu.")
        print("🔥 Gemma 3 is ready to assist you!")
        
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
                print("🔥 Gemma 3: ", end="", flush=True)
                response = self.chatbot.chat(self.current_session, user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 Returning to main menu...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
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
                
                filename = f"gemma3_conversation_{session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    
    def run(self):
        """Run the Gemma chatbot interface"""
        print("🔥 Welcome to the Gemma 3 Conversational Chatbot!")
        print("This chatbot uses Google's Gemma 3 model with full conversation memory.")
        
        while True:
            try:
                self.print_menu()
                choice = input("\nSelect option (1-7): ").strip()
                
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
                    self.show_model_info()
                elif choice == '7':
                    print("\n👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid option. Please select 1-7.")
                
                if choice != '7':
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

# Test function for Gemma 3
def test_gemma_chatbot():
    """Test Gemma 3 chatbot functionality"""
    print("🧪 Testing Gemma 3 chatbot...")
    
    try:
        # Create Gemma 3 chatbot
        chatbot = GemmaConversationalChatbot(
            model_name="google/gemma-3-4b-it",  # Using Gemma 3 4B
            use_database=False,
            use_4bit=True
        )
        
        # Create test session
        session_id = chatbot.create_session("You are a helpful assistant.")
        
        # Test conversation
        test_messages = [
            "Hello! What's your name?",
            "Can you remember what I just asked you?",
            "Tell me something interesting about AI."
        ]
        
        for msg in test_messages:
            print(f"\n👤 User: {msg}")
            response = chatbot.chat(session_id, msg)
            print(f"🔥 Gemma 3: {response}")
        
        print("\n✅ Gemma 3 test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

# Simple initialization function with error handling
def create_gemma_chatbot():
    """Create Gemma chatbot with automatic fallback options"""
    
    # Try different configurations in order of preference
    configs = [
        # 1. Try Gemma 3 4B with GPU and quantization
        {
            "model_name": "google/gemma-3-4b-it",
            "use_4bit": True,
            "force_cpu": False,
            "description": "Gemma 3 4B with GPU and 4-bit quantization"
        },
        # 2. Try Gemma 3 4B with CPU
        {
            "model_name": "google/gemma-3-4b-it", 
            "use_4bit": False,
            "force_cpu": True,
            "description": "Gemma 3 4B on CPU"
        },
        # 3. Try Gemma 3 1B with GPU
        {
            "model_name": "google/gemma-3-1b-it",
            "use_4bit": True, 
            "force_cpu": False,
            "description": "Gemma 3 1B with GPU and 4-bit quantization"
        },
        # 4. Try Gemma 3 1B with CPU
        {
            "model_name": "google/gemma-3-1b-it",
            "use_4bit": False,
            "force_cpu": True,
            "description": "Gemma 3 1B on CPU"
        }
    ]
    
    for config in configs:
        try:
            print(f"🔄 Trying: {config['description']}")
            chatbot = GemmaConversationalChatbot(
                model_name=config["model_name"],
                use_4bit=config["use_4bit"],
                force_cpu=config["force_cpu"],
                use_database=True
            )
            print(f"✅ Successfully initialized: {config['description']}")
            return chatbot
        except Exception as e:
            print(f"❌ Failed: {e}")
            continue
    
    # If all else fails, show helpful error message
    print("\n❌ Could not initialize any Gemma model.")
    print("\n🔧 Troubleshooting suggestions:")
    print("1. Make sure you're logged into Hugging Face: huggingface-cli login")
    print("2. Accept the Gemma license at: https://huggingface.co/google/gemma-3-4b-it")
    print("3. Install missing dependencies: pip install transformers torch accelerate bitsandbytes")
    print("4. Try with force_cpu=True if you have GPU issues")
    
    raise RuntimeError("Could not initialize any Gemma configuration")

if __name__ == "__main__":
    try:
        print("🔥 Initializing Gemma 3 Conversational Chatbot...")
        
        # Available Gemma 3 models (choose one):
        # "google/gemma-3-1b-it" - 1B Instruction-tuned (lightest)
        # "google/gemma-3-4b-it" - 4B Instruction-tuned (recommended)
        # "google/gemma-3-12b-it" - 12B Instruction-tuned (more powerful)
        # "google/gemma-3-27b-it" - 27B Instruction-tuned (largest, requires more memory)
        
        chatbot = GemmaConversationalChatbot(
            model_name="google/gemma-3-4b-it",  # Using the exact model you requested!
            use_database=True,
            use_4bit=True,  # Enable for memory efficiency
            temperature=0.7,
            max_new_tokens=256
        )
        
        interface = GemmaChatInterface(chatbot)
        interface.run()
        
    except Exception as e:
        print(f"❌ Error initializing Gemma 3 chatbot: {e}")
        print("\n🔧 Running basic test instead...")
        test_gemma_chatbot()