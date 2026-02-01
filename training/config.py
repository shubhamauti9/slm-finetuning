"""
Training Configuration for Unsloth Fine-tuning
LLaMA 7B for Broking Domain
"""

# Model Configuration
MODEL_CONFIG = {
    "base_model": "unsloth/llama-2-7b-bnb-4bit",  # 4-bit quantized LLaMA 2 7B
    "max_seq_length": 2048,
    "load_in_4bit": True,
    "dtype": None,  # Auto-detect
}

# LoRA Configuration
LORA_CONFIG = {
    "r": 16,  # LoRA rank
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "bias": "none",
    "use_gradient_checkpointing": "unsloth",  # Efficient memory usage
    "random_state": 42,
}

# Training Arguments
TRAINING_CONFIG = {
    "output_dir": "./models/broking-llama-7b",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 10,
    "logging_steps": 10,
    "save_steps": 100,
    "save_total_limit": 2,
    "fp16": True,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 42,
}

# Data Configuration
DATA_CONFIG = {
    "train_data_path": "./data/broking_training_data.json",
    "max_seq_length": 2048,
    "packing": False,  # Set to True for better efficiency with short examples
}

# Prompt Template (Alpaca format)
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

# System prompt for inference
SYSTEM_PROMPT = """You are an expert AI assistant specialized in Indian stock market and broking services. You provide accurate, helpful information about:
- Stock analysis and trade recommendations
- Market trends and sector analysis
- Trading procedures and account management
- Investment strategies and portfolio building
- Regulatory information and compliance

Always include appropriate disclaimers for investment advice. Be professional, accurate, and helpful."""
