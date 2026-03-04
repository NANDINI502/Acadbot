import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image

# 1. Paths (Update this to wherever you download your fine-tuned model folder from Colab)
MODEL_DIR = "./qwen-xray-pneumonia"
TEST_IMAGE_PATH = "./sample_images/sample_0_class_0.jpeg"
PROMPT = "Does this chest X-ray show signs of pneumonia? Answer only 'normal' or 'pneumonia'."

print(f"Loading Fine-Tuned Model from: {MODEL_DIR}")

# 2. Load the processor and the fine-tuned model
processor = AutoProcessor.from_pretrained(MODEL_DIR)

# Load the base model with the LoRA adapters merged in
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.float16 # Standard float16 for inference speed
)
model.eval()

# 3. Load the Image
print(f"Loading Test Image: {TEST_IMAGE_PATH}")
image = Image.open(TEST_IMAGE_PATH)
if image.mode != "RGB":
    image = image.convert("RGB")

# 4. Process Inputs via Chat Template
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT}
        ]
    }
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

# 5. Generate Inference
print("Analyzing X-Ray...")
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=20, # The answer will just be "pneumonia" or "normal"
        do_sample=False # We want deterministic, greedy decoding for classification
    )

# 6. Decode the Output
# The output includes the prompt, so we slice it off to just get the answer
input_len = inputs["input_ids"].shape[1]
generated_text = processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]

print("\n" + "="*40)
print("🩺 CLINICAL AI RESULT:")
print(f"Image: {TEST_IMAGE_PATH}")
print(f"Question: {PROMPT}")
print(f"AI Classification: {generated_text.strip().upper()}")
print("="*40 + "\n")
