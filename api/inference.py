"""
Model Inference Handler
Loads fine-tuned model and handles generation
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Generator
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ModelManager:
    """Manages model loading and inference."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one model instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.model_path = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = True
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the fine-tuned model.
        
        Args:
            model_path: Path to the model. If None, uses default path.
            
        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self.model_loaded and self.model is not None:
            print("Model already loaded.")
            return True
        
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            print("[ERROR] Unsloth not installed.")
            return False
        
        # Determine model path
        if model_path is None:
            # Try merged model first, then LoRA adapters
            base_path = Path(__file__).parent.parent / "models" / "broking-llama-7b"
            merged_path = base_path / "merged_model"
            lora_path = base_path / "lora_adapters"
            
            if merged_path.exists():
                model_path = str(merged_path)
            elif lora_path.exists():
                model_path = str(lora_path)
            else:
                # Fall back to base model for testing
                model_path = "unsloth/llama-2-7b-bnb-4bit"
                print(f"[WARNING] No fine-tuned model found. Using base model: {model_path}")
        
        try:
            print(f"Loading model from: {model_path}")
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=2048,
                load_in_4bit=True,
                dtype=None,
            )
            
            # Set to inference mode
            FastLanguageModel.for_inference(self.model)
            
            self.model_path = model_path
            self.model_loaded = True
            print(f"✓ Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.model_loaded = False
            return False
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text from the model.
        
        Args:
            prompt: The input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to use sampling
            
        Returns:
            Generated text
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - max_new_tokens,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        
        # Decode and remove input prompt from output
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        if prompt in full_response:
            response = full_response[len(prompt):].strip()
        else:
            response = full_response.strip()
        
        return response
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """
        Stream text generation token by token.
        
        Args:
            prompt: The input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature  
            top_p: Top-p sampling
            
        Yields:
            Generated text chunks
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - max_new_tokens,
        ).to(self.device)
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        
        # Run generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield tokens as they're generated
        for text in streamer:
            yield text
        
        thread.join()
    
    def format_chat_prompt(self, messages: List[dict]) -> str:
        """
        Format chat messages into a prompt.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Formatted prompt string
        """
        from training.config import ALPACA_PROMPT, SYSTEM_PROMPT
        
        # Extract system message if present
        system_msg = SYSTEM_PROMPT
        user_input = ""
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            elif msg["role"] == "user":
                user_input = msg["content"]
        
        # Format as Alpaca prompt
        prompt = ALPACA_PROMPT.format(
            instruction=user_input,
            input=f"System: {system_msg}" if system_msg != SYSTEM_PROMPT else "",
            output="",
        )
        
        return prompt
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer is None:
            # Rough estimate if tokenizer not loaded
            return len(text.split()) * 1.3
        return len(self.tokenizer.encode(text))
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
            "device": self.device,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }


# Global model manager instance
model_manager = ModelManager()
