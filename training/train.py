"""
Unsloth Fine-tuning Script for LLaMA 7B
Broking Domain - Trade Recommendations & Conversational Assistant
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """Main training function."""
    
    print("=" * 60)
    print("Unsloth Fine-tuning: LLaMA 7B for Broking Domain")
    print("=" * 60)
    
    # Import after path setup
    try:
        from unsloth import FastLanguageModel
        from unsloth import is_bfloat16_supported
    except ImportError:
        print("\n[ERROR] Unsloth not installed. Please install with:")
        print("pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git")
        return
    
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    from training.config import (
        MODEL_CONFIG,
        LORA_CONFIG,
        TRAINING_CONFIG,
        DATA_CONFIG,
        ALPACA_PROMPT,
    )
    
    # =========================================================================
    # Step 1: Load Base Model
    # =========================================================================
    print("\n[1/5] Loading base model...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_CONFIG["base_model"],
        max_seq_length=MODEL_CONFIG["max_seq_length"],
        load_in_4bit=MODEL_CONFIG["load_in_4bit"],
        dtype=MODEL_CONFIG["dtype"],
    )
    
    print(f"   ✓ Loaded: {MODEL_CONFIG['base_model']}")
    
    # =========================================================================
    # Step 2: Apply LoRA Adapters
    # =========================================================================
    print("\n[2/5] Applying LoRA adapters...")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        bias=LORA_CONFIG["bias"],
        use_gradient_checkpointing=LORA_CONFIG["use_gradient_checkpointing"],
        random_state=LORA_CONFIG["random_state"],
    )
    
    print(f"   ✓ LoRA rank: {LORA_CONFIG['r']}")
    print(f"   ✓ Target modules: {len(LORA_CONFIG['target_modules'])} layers")
    
    # =========================================================================
    # Step 3: Load and Prepare Training Data
    # =========================================================================
    print("\n[3/5] Loading training data...")
    
    data_path = Path(DATA_CONFIG["train_data_path"])
    if not data_path.exists():
        # Try relative to script location
        data_path = Path(__file__).parent.parent / "data" / "broking_training_data.json"
    
    if not data_path.exists():
        print(f"   [ERROR] Training data not found at: {data_path}")
        return
    
    with open(data_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)
    
    print(f"   ✓ Loaded {len(training_data)} training examples")
    
    # Format data with Alpaca prompt template
    def format_example(example):
        """Format a single example with the Alpaca prompt template."""
        text = ALPACA_PROMPT.format(
            instruction=example.get("instruction", ""),
            input=example.get("input", ""),
            output=example.get("output", ""),
        )
        return {"text": text}
    
    formatted_data = [format_example(ex) for ex in training_data]
    dataset = Dataset.from_list(formatted_data)
    
    print(f"   ✓ Formatted dataset with Alpaca template")
    
    # =========================================================================
    # Step 4: Configure and Run Training
    # =========================================================================
    print("\n[4/5] Starting training...")
    print(f"   • Epochs: {TRAINING_CONFIG['num_train_epochs']}")
    print(f"   • Batch size: {TRAINING_CONFIG['per_device_train_batch_size']}")
    print(f"   • Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"   • Output directory: {TRAINING_CONFIG['output_dir']}")
    
    # Create output directory
    os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=DATA_CONFIG["max_seq_length"],
        dataset_num_proc=2,
        packing=DATA_CONFIG["packing"],
        args=TrainingArguments(
            output_dir=TRAINING_CONFIG["output_dir"],
            num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
            per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
            learning_rate=TRAINING_CONFIG["learning_rate"],
            warmup_steps=TRAINING_CONFIG["warmup_steps"],
            logging_steps=TRAINING_CONFIG["logging_steps"],
            save_steps=TRAINING_CONFIG["save_steps"],
            save_total_limit=TRAINING_CONFIG["save_total_limit"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim=TRAINING_CONFIG["optim"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
            lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
            seed=TRAINING_CONFIG["seed"],
        ),
    )
    
    # Train!
    print("\n   Training started...")
    trainer_stats = trainer.train()
    
    print("\n   ✓ Training completed!")
    print(f"   • Training time: {trainer_stats.metrics['train_runtime']:.2f}s")
    print(f"   • Final loss: {trainer_stats.metrics['train_loss']:.4f}")
    
    # =========================================================================
    # Step 5: Save the Model
    # =========================================================================
    print("\n[5/5] Saving model...")
    
    # Save LoRA adapters
    lora_path = os.path.join(TRAINING_CONFIG["output_dir"], "lora_adapters")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"   ✓ LoRA adapters saved to: {lora_path}")
    
    # Save merged model (optional but recommended for inference)
    print("\n   Merging LoRA weights with base model...")
    merged_path = os.path.join(TRAINING_CONFIG["output_dir"], "merged_model")
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    print(f"   ✓ Merged model saved to: {merged_path}")
    
    # Optional: Save to GGUF format for llama.cpp deployment
    # Uncomment if you want GGUF format
    # print("\n   Converting to GGUF format...")
    # gguf_path = os.path.join(TRAINING_CONFIG["output_dir"], "model.gguf")
    # model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method="q4_k_m")
    # print(f"   ✓ GGUF model saved to: {gguf_path}")
    
    print("\n" + "=" * 60)
    print("Fine-tuning completed successfully!")
    print("=" * 60)
    print(f"\nModel outputs:")
    print(f"  • LoRA adapters: {lora_path}")
    print(f"  • Merged model: {merged_path}")
    print(f"\nTo start the API server:")
    print(f"  cd {Path(TRAINING_CONFIG['output_dir']).parent.parent}")
    print(f"  uvicorn api.main:app --host 0.0.0.0 --port 8000")


def test_model():
    """Quick test of the trained model."""
    from unsloth import FastLanguageModel
    from training.config import TRAINING_CONFIG, ALPACA_PROMPT
    
    model_path = os.path.join(TRAINING_CONFIG["output_dir"], "merged_model")
    
    if not os.path.exists(model_path):
        print("Model not found. Please run training first.")
        return
    
    print("Loading trained model for testing...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    
    FastLanguageModel.for_inference(model)
    
    # Test prompt
    test_prompt = ALPACA_PROMPT.format(
        instruction="Recommend a good banking stock for investment.",
        input="I'm looking for a stable dividend-paying bank stock.",
        output="",
    )
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n--- Test Response ---")
    print(response.split("### Response:")[-1].strip())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unsloth Fine-tuning for Broking Domain")
    parser.add_argument("--test", action="store_true", help="Test the trained model")
    args = parser.parse_args()
    
    if args.test:
        test_model()
    else:
        main()
