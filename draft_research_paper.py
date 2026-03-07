from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig

import torch

# 1. Configuration
MODEL_DIR = "./qwen-xray-researcher"

# The section of the research paper you want the AI to write
RESEARCH_PROMPT_TOPIC = "Write a 400-word Methodology section detailing how we balanced the pneumonia classifications using Data Augmentation"

print(f"Loading Fine-Tuned NLP Model from: {MODEL_DIR}")

# 2. Load Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Optimize Latency: Use 4-bit quantization to double inference speed and halve memory
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load the fine-tuned Qwen PEFT model
model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="sdpa"
)
model.eval()

# 3. Format Request using Llama 3's Official Chat Template
messages = [
    {"role": "system", "content": "You are a distinguished AI medical researcher writing a peer-reviewed academic paper."},
    {"role": "user", "content": RESEARCH_PROMPT_TOPIC}
]

# The tokenizer converts the roles into the exact special tokens Llama expects
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# 4. Generate the Research Text
print(f"\n[AI] Drafting Academic Text for: '{RESEARCH_PROMPT_TOPIC}'...\n")
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=800, # Allow a large response for a full paper section
        temperature=0.7, # Lower temperature makes the text more academic and factual
        do_sample=True,
        repetition_penalty=1.1 # Prevents the model from repeating the same phrases
    )

# 5. Decode and Print the Finished Document
# Slice off the prompt so it only prints the new text
input_len = inputs["input_ids"].shape[1]
generated_text = tokenizer.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]

print("="*60)
print("   GENERATED RESEARCH PAPER SECTION:")
print("="*60)
print(generated_text.strip())
print("="*60)
