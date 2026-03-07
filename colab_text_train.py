import os
os.system("pip install -q -U accelerate bitsandbytes datasets peft 'transformers>=4.45.0' trl wandb")
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer

# 1. Configuration & Setup
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" # Highly capable, fully open-weights NLP model (NO TOKEN NEEDED!)

import glob
# In Kaggle, datasets you upload show up somewhere inside '/kaggle/input/'
# Since names can vary, we will auto-detect the exact path for you:
possible_files = glob.glob("/kaggle/input/**/*.jsonl", recursive=True)
if not possible_files:
    raise FileNotFoundError("Could not find any .jsonl file in /kaggle/input/. Make sure you uploaded the dataset to Kaggle!")
DATASET_PATH = possible_files[0]
print(f"Auto-Detected Dataset at: {DATASET_PATH}")

# In Kaggle, we save the model to the '/kaggle/working/' directory
OUTPUT_DIR = "/kaggle/working/qwen-xray-researcher"

print(f"Loading Fully Open Base NLP Model: {MODEL_ID}")

# 2. Load Tokenizer & 4-Bit Quantized Base Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix for fp16 mixed precision training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32 # Use float32 to prevent CUBLAS errors on older T4 GPUs
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. Setup LoRA (Low-Rank Adaptation) for NLP
model = prepare_model_for_kbit_training(model)

# For Llama 3 NLP fine-tuning, we target all linear layers to achieve the best linguistic nuance capture
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Load & Prepare Dataset
print(f"Loading Dataset from local file: {DATASET_PATH}")
# load_dataset can load local JSONL files easily
dataset = load_dataset('json', data_files=DATASET_PATH)

# Format the dataset using Qwen's specific chat template and tokenize it
def format_chat_template(example):
    # Apply the Chat Template
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    # Tokenize the resulting string into input_ids and attention_mask tensors
    tokenized = tokenizer(text, truncation=True, max_length=1024, padding="max_length")
    return tokenized

print("Applying Chat Template logic to dataset and tokenizing...")
# Map the formatting and drop the original text columns so the Trainer only sees tensors
full_dataset = dataset["train"].map(format_chat_template, remove_columns=["messages", "text"] if "text" in dataset["train"].column_names else ["messages"])

# Split 10% for evaluation to enable early stopping
split = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    num_train_epochs=3, # 3 epochs allows it to properly memorize the academic terminology
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    lr_scheduler_type="cosine",
    warmup_ratio=0.06,
    fp16=True, # Standard for free T4 GPUs
    optim="paged_adamw_8bit",
    report_to="none" 
)

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer

# 6. Start Supervised Fine-Tuning (SFT)
# We use standard Trainer with a causal LM collator since we applied chat formatting manually above
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    args=training_args
)

print("\nStarting NLP Fine-Tuning Training!!! This will take some time.")
trainer.train()

# 7. Save the Fine-Tuned Brain
print(f"Saving Fine-Tuned NLP Model to {OUTPUT_DIR}")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("\n✅ Training Complete! Your model is now a domain expert in Chest X-Ray AI Literature.")
