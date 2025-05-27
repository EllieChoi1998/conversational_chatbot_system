# Working Gemma 3 Chatbot - Based on your working pipeline code
# Uses the same approach as your successful example

import torch
from transformers import pipeline
import uuid
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

class WorkingGemmaChat:
    def __init__(self, model_name="google/gemma-3-4b-it"):
        print(f"🔥 Loading {model_name} with your working pipeline method...")
        
        self.model_name = model_name
        self.conversations = {}
        
        # Check GPU
        if not torch.cuda.is_available():
            raise RuntimeError("❌ No CUDA GPU available!")
        
        print(f"🎯 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Use the EXACT same pipeline configuration that works for you
        print("📥 Loading pipeline (using your working method)...")
        
        try:
            self.pipe = pipeline(
                "image-text-to-text",  # Same as your working code
                model=model_name,
                device="cuda",  # Same as your working code
                torch_dtype=torch.bfloat16  # Same as your working code
            )
            print("✅ Pipeline loaded successfully!")
            print("📏 Will use longer generation settings")
            
        except Exception as e:
            print(f"❌ Failed to load pipeline: {e}")
            raise
        
        print("🎉 Model ready!")
    
    def create_session(self, system_prompt=None):
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        self.conversations[session_id] = {
            'messages': [],
            'system_prompt': system_prompt or "You are Gemma, a helpful AI assistant. Respond naturally and helpfully."
        }
        return session_id
    
    def chat(self, session_id, user_message):
        """Generate response using the working pipeline method"""
        if session_id not in self.conversations:
            return "Session not found"
        
        conv = self.conversations[session_id]
        
        try:
            # Build conversation context
            context_parts = []
            
            # Add system prompt
            if conv['system_prompt']:
                context_parts.append(f"System: {conv['system_prompt']}")
            
            # Add recent conversation history
            recent_messages = conv['messages'][-8:]  # Last 4 exchanges
            for msg in recent_messages:
                context_parts.append(f"{msg['role']}: {msg['content']}")
            
            # Create the text prompt (similar to your working example)
            conversation_context = "\n".join(context_parts) if context_parts else ""
            
            if conversation_context:
                full_prompt = f"{conversation_context}\nHuman: {user_message}\nAssistant:"
            else:
                full_prompt = f"Human: {user_message}\nAssistant:"
            
            print(f"🔍 Generating response for: '{user_message[:50]}...'")
            
            # Use the same pipeline call that works for you - with longer output
            try:
                # First try with extended parameters
                output = self.pipe(
                    text=full_prompt,
                    max_new_tokens=500,  # Generate longer responses
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            except Exception as param_error:
                print(f"⚠️  Extended parameters failed: {param_error}")
                print("🔄 Falling back to simple call...")
                # Fallback to your original working method
                output = self.pipe(text=full_prompt)
            
            # Extract the response from the output
            if isinstance(output, list) and len(output) > 0:
                generated_text = output[0].get('generated_text', '')
            elif isinstance(output, dict):
                generated_text = output.get('generated_text', '')
            else:
                generated_text = str(output)
            
            print(f"🔍 Raw output: {generated_text[:100]}...")
            
            # Clean the response
            response = self._clean_response(generated_text, full_prompt)
            
            # Validate response
            if not response or len(response.strip()) < 3:
                print("⚠️  Response too short, using fallback")
                response = self._get_fallback_response(user_message)
            else:
                print(f"✅ Using generated response (length: {len(response)})")
            
            # Save to conversation history
            conv['messages'].append({"role": "Human", "content": user_message})
            conv['messages'].append({"role": "Assistant", "content": response})
            
            return response
            
        except Exception as e:
            print(f"❌ Chat error: {e}")
            
            # Save failed attempt to history with fallback
            fallback = self._get_fallback_response(user_message)
            conv['messages'].append({"role": "Human", "content": user_message})
            conv['messages'].append({"role": "Assistant", "content": fallback})
            
            return fallback
    
    def _clean_response(self, generated_text, original_prompt):
        """Clean the generated response - MINIMAL cleaning to preserve content"""
        try:
            print(f"🔧 CLEANING PROCESS:")
            print(f"Original prompt length: {len(original_prompt)}")
            print(f"Generated text length: {len(generated_text)}")
            
            # Try to find where the actual response starts
            response = generated_text
            
            # Only remove the exact original prompt if it appears at the start
            if response.startswith(original_prompt):
                response = response[len(original_prompt):].strip()
                print(f"✂️  Removed original prompt, new length: {len(response)}")
            
            # Very minimal cleaning - only remove obvious prefixes if they're at the very start
            simple_prefixes = ["Assistant:", "AI:", "Gemma:"]
            for prefix in simple_prefixes:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
                    print(f"✂️  Removed prefix '{prefix}', new length: {len(response)}")
                    break  # Only remove one prefix
            
            # Remove image-related artifacts only if they're at the start
            image_prefixes = ["<start_of_image>", "in this image, there is"]
            for prefix in image_prefixes:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
                    print(f"✂️  Removed image prefix '{prefix}', new length: {len(response)}")
                    break
            
            print(f"🔧 FINAL CLEANED RESPONSE:")
            print(f"Length: {len(response)}")
            print(f"Content: {repr(response)}")
            print("-" * 30)
            
            return response
            
        except Exception as e:
            print(f"⚠️  Error cleaning response: {e}")
            return generated_text
    
    def _get_fallback_response(self, user_message):
        """Get contextual fallback response"""
        # Detect language and provide appropriate response
        if any(korean_char in user_message for korean_char in "가나다라마바사아자차카타파하한국"):
            if "이름" in user_message or "뭐야" in user_message:
                return "안녕하세요! 저는 Gemma입니다. 구글에서 만든 AI 어시스턴트예요."
            else:
                return "네, 무엇을 도와드릴까요?"
        else:
            if "name" in user_message.lower():
                return "Hello! I'm Gemma, an AI assistant created by Google."
            elif "who" in user_message.lower():
                return "I'm Gemma, a helpful AI assistant. How can I assist you today?"
            elif "sql" in user_message.lower():
                return "I can help with SQL! For example: SELECT * FROM users; gets all attributes from the users table."
            else:
                return "I'm here to help! Could you please tell me what you'd like to know?"
    
    def get_conversation_history(self, session_id):
        """Get conversation history for a session"""
        if session_id in self.conversations:
            return self.conversations[session_id]['messages']
        return []
    
    def clear_session(self, session_id):
        """Clear a conversation session"""
        if session_id in self.conversations:
            self.conversations[session_id]['messages'] = []

def main():
    """Main chat interface using your working pipeline method"""
    print("🔥 WORKING GEMMA 3 CHATBOT (PIPELINE METHOD)")
    print("=" * 50)
    print("📝 Based on your successful pipeline code")
    
    # GPU check
    if not torch.cuda.is_available():
        print("❌ No CUDA GPU detected!")
        return
    
    try:
        # Initialize chatbot with your working method
        chatbot = WorkingGemmaChat("google/gemma-3-4b-it")
        
        # Create session
        session = chatbot.create_session()
        
        print(f"\n✅ Ready! Using {chatbot.model_name}")
        print("💬 Korean/English supported. Type 'quit' to exit.")
        print("🔧 Using the same pipeline method that works in your environment")
        print("-" * 50)
        
        # Chat loop
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye! / 안녕히 가세요!")
                    break
                
                if not user_input:
                    continue
                
                print("🤖 Gemma: ", end="", flush=True)
                response = chatbot.chat(session, user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye! / 안녕히 가세요!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("\n🔧 Make sure you can run your original pipeline code first!")

# Test function using your exact working code
def test_your_working_code():
    """Test using your exact working pipeline code"""
    print("🧪 Testing your exact working pipeline code...")
    
    try:
        pipe = pipeline(
            "image-text-to-text",
            model="google/gemma-3-4b-it",
            device="cuda",
            torch_dtype=torch.bfloat16
        )
        
        # Test with longer output
        output = pipe(
            text="Hello, what is your name?",
            max_new_tokens=300  # Try longer output
        )
        
        print(f"✅ Your pipeline works! Output: {output}")
        return True
        
    except Exception as e:
        print(f"❌ Your pipeline test failed: {e}")
        # Try basic version
        try:
            print("🔄 Trying basic version...")
            pipe = pipeline(
                "image-text-to-text",
                model="google/gemma-3-4b-it",
                device="cuda",
                torch_dtype=torch.bfloat16
            )
            
            output = pipe(text="Hello, what is your name?")
            print(f"✅ Basic pipeline works! Output: {output}")
            return True
            
        except Exception as basic_error:
            print(f"❌ Basic pipeline also failed: {basic_error}")
            return False

if __name__ == "__main__":
    # First test your working code
    if test_your_working_code():
        print("\n✅ Your pipeline works, starting chatbot...")
        main()
    else:
        print("\n❌ Need to fix the basic pipeline first!")