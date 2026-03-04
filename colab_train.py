# Install necessary libraries if running in Colab
# !pip install -q -U accelerate bitsandbytes datasets peft transformers trl pillow wandb qwen-vl-utils torchvision

import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from PIL import Image

# 1. Configuration & Setup
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"  # Fully open-weights, no HF authentication required!
DATASET_ID = "hf-vision/chest-xray-pneumonia"
OUTPUT_DIR = "./qwen-xray-pneumonia"
MAX_LENGTH = 128
PROMPT = "Does this chest X-ray show signs of pneumonia? Answer only 'normal' or 'pneumonia'."

print("✅ Using Fully Open Model (No token required!)")
print(f"Loading Base Model: {MODEL_ID}")

# 2. Load 4-Bit Quantized Base Model to save Colab VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. Setup LoRA (Low-Rank Adaptation)
model = prepare_model_for_kbit_training(model)

# For Qwen2, we target linear layers in the attention blocks
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Load & Prepare Dataset
print(f"Loading Dataset: {DATASET_ID}")
dataset = load_dataset(DATASET_ID)

# Map numeric labels to readable text
def format_data(example):
    label_num = example.get('label', 0)
    label_text = "pneumonia" if label_num == 1 else "normal"
    
    # Qwen2-VL uses a specific chat template format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example['image'].convert('RGB')},
                {"type": "text", "text": PROMPT}
            ]
        }
    ]
    # The expected output the model must learn
    output_text = label_text
    
    return {
        "messages": messages,
        "output_text": output_text
    }

print("Formatting Dataset...")
# Just taking a subset for faster demonstration in Colab. 
# Remove the .select(range(500)) for full training.
train_dataset = dataset["train"].select(range(500)).map(format_data, remove_columns=dataset["train"].column_names)
eval_dataset = dataset["validation"].map(format_data, remove_columns=dataset["validation"].column_names)

# 5. Collation Function
def collate_fn(examples):
    texts = [processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True) + example["output_text"] for example in examples]
    
    # We need to extract the raw PIL images from the structured dictionary
    image_inputs = []
    for example in examples:
        for msg in example["messages"]:
            for content in msg["content"]:
                if content["type"] == "image":
                    image_inputs.append(content["image"])
    
    # The processor handles converting text and images into numerical token tensors
    inputs = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
    
    # For causal LM, labels are typically shifted versions of inputs; High-level trainers handle this if labels = input_ids
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2, # Keep batch size small for 16GB VRAM Colab GPUs
    gradient_accumulation_steps=8, # Accumulate gradients to simulate larger batch size
    learning_rate=2e-4,
    weight_decay=0.01,
    num_train_epochs=3, # 3 epochs is standard for fine-tuning
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=False,
    bf16=True, # bfloat16 is better for A100/L4 GPUs, fallback to fp16 if T4
    optim="paged_adamw_8bit",
    report_to="none" # Set to "wandb" to log charts if you use Weights and Biases
)

# 7. Start Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn
)

print("\nStarting Training!!!")
trainer.train()

# 8. Save the Fine-Tuned Adapters
print(f"Saving Fine-Tuned Model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("\n✅ Training Complete!")
