"""
Modal-based inference for batch LLM generation.
Uses left-padding for safe batched generation.
"""

import os
from dataclasses import dataclass
from typing import Any, List

import modal
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ====== Model bundle & inference ====== #

@dataclass
class ModelBundle:
    tokenizer: Any
    model: Any
    device: torch.device
    model_name: str
    torch_dtype: torch.dtype


def apply_chat_template_safe(tokenizer: AutoTokenizer, prompt: str) -> str:
    """Apply chat template if available, otherwise return prompt as-is."""
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    return prompt


def model_generate_batch(
    bundle: ModelBundle,
    prompts: List[str],
    max_new_tokens: int = 256,
    min_tokens: int = 100,
) -> List[str]:
    """
    Run batched generation with left-padding.
    Requires tokenizer.padding_side = "left" (set in load_model).
    """
    tokenizer = bundle.tokenizer
    model = bundle.model

    # Ensure pad_token is set
    if getattr(model.config, "pad_token_id", None) is None:
        eos_id = getattr(model.config, "eos_token_id", None)
        if eos_id is None and hasattr(tokenizer, "eos_token_id"):
            eos_id = tokenizer.eos_token_id
        model.config.pad_token_id = eos_id

    chats = [apply_chat_template_safe(tokenizer, p) for p in prompts]

    # Left-padding: tokenizer.padding_side must be "left" (set in load_model)
    inputs = tokenizer(
        chats,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(bundle.device)

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_tokens,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
        )

    new_tokens = out[:, input_length:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [d.strip() for d in decoded]


# ====== Modal setup ====== #

app = modal.App("batch-llm")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "accelerate")
)

model_volume = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
MODEL_DIR = "/models"


@app.cls(
    gpu="A10G",
    image=image,
    volumes={MODEL_DIR: model_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=60 * 30,
)
class ModelRunner:
    model_name: str = modal.parameter()

    @modal.enter()
    def load_model(self):
        print(f"Initializing model: {self.model_name}", flush=True)
        os.makedirs(MODEL_DIR, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=MODEL_DIR
        )
        tokenizer.padding_side = "left"  # Required for batched generation
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            cache_dir=MODEL_DIR
        )
        model.eval()
        model.to("cuda")

        model_volume.commit()

        self.bundle = ModelBundle(
            tokenizer=tokenizer,
            model=model,
            device=torch.device("cuda"),
            model_name=self.model_name,
            torch_dtype=torch.float16,
        )

    @modal.method()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        min_tokens: int = 100,
    ) -> List[str]:
        """Generate for a single batch. Use for client-side progress tracking."""
        return model_generate_batch(
            self.bundle,
            prompts,
            max_new_tokens=max_new_tokens,
            min_tokens=min_tokens,
        )
