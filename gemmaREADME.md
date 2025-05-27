# GEMMA 3 MODEL SETUP INSTRUCTIONS

## 1. UPDATED REQUIREMENTS FOR GEMMA 3

Create `requirements.txt`:
```
transformers>=4.50.0
torch>=2.1.0
accelerate>=0.25.0
bitsandbytes>=0.41.0
flash-attn>=2.5.0  # Optional, for faster attention
huggingface-hub>=0.20.0
sqlite3
numpy
```

## 2. INSTALLATION STEPS

### Step 1: Install Dependencies
```bash
# Basic installation
pip install transformers torch accelerate bitsandbytes huggingface-hub

# For GPU optimization (optional)
pip install flash-attn --no-build-isolation

# Or install all at once
pip install -r requirements.txt
```

### Step 2: Hugging Face Authentication (Required for Gemma 3)
```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub

# Login to access Gemma 3 models
huggingface-cli login
```

**Important**: You need to:
1. Create a Hugging Face account at https://huggingface.co
2. Accept the Gemma 3 license at https://huggingface.co/google/gemma-3-4b-it
3. Use your Hugging Face token when prompted

### Step 3: Run the Gemma 3 Chatbot
```bash
python gemma3_chatbot.py
```

## 3. AVAILABLE GEMMA 3 MODELS

You can modify the `model_name` parameter to use different Gemma 3 variants:

```python
# In the code, change this line:
chatbot = GemmaConversationalChatbot(
    model_name="google/gemma-3-4b-it",  # Change this
    use_4bit=True,
    use_database=True
)
```

**Available Gemma 3 Models:**
- `"google/gemma-3-1b-it"` - **Lightest** - 1B instruction-tuned (1GB VRAM)
- `"google/gemma-3-4b-it"` - **Recommended** - 4B instruction-tuned (2.6GB VRAM with 4-bit)
- `"google/gemma-3-12b-it"` - **More Powerful** - 12B instruction-tuned (6GB VRAM with 4-bit)
- `"google/gemma-3-27b-it"` - **Largest** - 27B instruction-tuned (14GB VRAM with 4-bit)

## 4. MEMORY REQUIREMENTS

With 4-bit quantization enabled:
- **Gemma 3 1B**: ~1GB VRAM
- **Gemma 3 4B**: ~2.6GB VRAM  
- **Gemma 3 12B**: ~6GB VRAM
- **Gemma 3 27B**: ~14GB VRAM

Without quantization (BF16):
- **Gemma 3 1B**: ~2GB VRAM
- **Gemma 3 4B**: ~8GB VRAM
- **Gemma 3 12B**: ~24GB VRAM
- **Gemma 3 27B**: ~54GB VRAM

## 5. KEY GEMMA 3 FEATURES

✅ **Multimodal**: Handles text and images (4B, 12B, 27B models)
✅ **128K Context**: Massive context window
✅ **140+ Languages**: Multilingual support
✅ **Function Calling**: Structured output support
✅ **State-of-the-art**: Best performance in size class

## 6. QUICK START

1. Save the code as `gemma3_chatbot.py`
2. Install: `pip install transformers torch accelerate bitsandbytes`
3. Login: `huggingface-cli login`
4. Accept license at: https://huggingface.co/google/gemma-3-4b-it
5. Run: `python gemma3_chatbot.py`

## 7. TROUBLESHOOTING

**Issue: "Repository not found"**
Solution: Make sure you've accepted the license and are logged in

**Issue: "Out of memory"**
Solution: Enable 4-bit quantization (already enabled by default)

**Issue: "Access denied"**
Solution: Accept the Gemma license on HuggingFace

**Issue: "Model loading too slow"**
Solution: The first time takes longer as it downloads the model

## 8. SIMPLE TEST

If you want to test quickly without the full interface:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Gemma 3 4B
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Simple chat
messages = [{"role": "user", "content": "Hello! What's your name?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
    
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

The Gemma 3 4B model you requested offers excellent performance with multimodal capabilities and a large context window!