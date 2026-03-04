import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# 1. Configuration
# Change MODEL_DIR to wherever you downloaded the fine-tuned folder from Google Colab
MODEL_DIR = "./qwen-xray-researcher"

# The section of the research paper you want the AI to write
RESEARCH_PROMPT_TOPIC = "Write a 400-word Methodology section detailing how we balanced the pneumonia classifications using Data Augmentation"

print(f"Loading Fine-Tuned NLP Model from: {MODEL_DIR}")

# 2. Load Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# We load the PEFT adapter and let it automatically fetch the base Qwen model
model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
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
