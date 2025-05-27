# COMPLETE SETUP INSTRUCTIONS

## 1. REQUIREMENTS.TXT
Create a file named `requirements.txt` with this content:

transformers>=4.30.0
torch>=2.0.0
accelerate>=0.20.0
sentence-transformers>=2.2.0
sqlite3
numpy
huggingface-hub

## 2. INSTALLATION STEPS

### Step 1: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv chatbot_env

# Activate it
# On Windows:
chatbot_env\Scripts\activate
# On macOS/Linux:
source chatbot_env/bin/activate
```

### Step 2: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Alternative: Install manually
pip install transformers torch accelerate sentence-transformers huggingface-hub
```

### Step 3: Download and Run
```bash
# Save the main code as 'chatbot.py'
# Then run:
python chatbot.py
```

## 3. ALTERNATIVE LIGHTWEIGHT VERSION

If you encounter issues with the main version, use this simplified version:

```python
import json
import uuid
from datetime import datetime
from typing import List, Dict
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

class SimpleChatbot:
    def __init__(self):
        print("Loading chatbot model...")
        # Use a simple, reliable model
        self.generator = pipeline("text-generation", model="gpt2")
        self.conversations = {}
        print("Chatbot ready!")
    
    def create_session(self):
        session_id = str(uuid.uuid4())
        self.conversations[session_id] = []
        return session_id
    
    def chat(self, session_id, message):
        if session_id not in self.conversations:
            return "Session not found!"
        
        # Add user message to history
        self.conversations[session_id].append(f"Human: {message}")
        
        # Create prompt with history
        history = "\n".join(self.conversations[session_id][-5:])  # Last 5 exchanges
        prompt = f"{history}\nAI:"
        
        # Generate response
        try:
            result = self.generator(prompt, max_length=len(prompt.split()) + 50, 
                                  num_return_sequences=1, temperature=0.7)
            response = result[0]['generated_text'][len(prompt):].strip()
            
            # Clean response
            if '\n' in response:
                response = response.split('\n')[0]
            
            # Add AI response to history
            self.conversations[session_id].append(f"AI: {response}")
            
            return response
        except Exception as e:
            return f"Error: {str(e)}"

# Simple test
if __name__ == "__main__":
    bot = SimpleChatbot()
    session = bot.create_session()
    
    print("Simple chatbot ready! Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        response = bot.chat(session, user_input)
        print(f"Bot: {response}")
```

## 4. TROUBLESHOOTING

### Common Issues:

**Issue: "No module named 'transformers'"**
Solution: `pip install transformers torch`

**Issue: "CUDA out of memory"**
Solution: The code automatically handles this and falls back to CPU

**Issue: "Model download is too slow"**
Solution: The code uses small models by default (DialoGPT-small, GPT-2)

**Issue: "Database locked"**
Solution: Set `use_database=False` in the chatbot initialization

### Alternative Models You Can Try:
- `"gpt2"` - Fastest, basic conversations
- `"microsoft/DialoGPT-small"` - Better conversations, small size
- `"microsoft/DialoGPT-medium"` - Good balance
- `"facebook/blenderbot-400M-distill"` - Conversation-optimized

### Running Options:

**Option 1: Full Interactive Mode**
```python
chatbot = ConversationalChatbot(use_database=True)
interface = ChatbotInterface(chatbot)
interface.run()
```

**Option 2: Simple Chat**
```python
chatbot = ConversationalChatbot(use_database=False)
session_id = chatbot.create_session()
response = chatbot.chat(session_id, "Hello!")
print(response)
```

**Option 3: Test Mode**
```python
test_basic_functionality()
```

## 5. FEATURES INCLUDED

✅ Conversation memory (remembers chat history)
✅ Multiple conversation sessions
✅ Database persistence (SQLite)
✅ Memory-only mode (no database)
✅ Export conversations to JSON
✅ System prompts
✅ Interactive CLI interface
✅ Session management
✅ Automatic model fallback
✅ Error handling
✅ Cross-platform compatibility

## 6. QUICK START

1. Save the main code as `chatbot.py`
2. Install: `pip install transformers torch`
3. Run: `python chatbot.py`
4. Choose option 1 to start chatting!

The chatbot will remember everything you say within each session and can maintain multiple separate conversations.