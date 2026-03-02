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


def _get_assistant_prefix_for_strip(tokenizer: AutoTokenizer) -> str:
    """
    Get how the assistant header appears in decoded output (with skip_special_tokens=True).
    The chat template adds e.g. <|start_header_id|>assistant<|end_header_id|>\\n\\n; when we
    decode generated tokens, special tokens are skipped, leaving e.g. 'assistant\\n\\n'.
    This derives that string from the tokenizer so we can strip it without hardcoding.
    """
    with_add = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=False,
        add_generation_prompt=True,
    )
    without_add = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=False,
        add_generation_prompt=False,
    )
    raw_prefix = with_add[len(without_add) :]
    # Decode as it would appear in our output (special tokens skipped)
    ids = tokenizer.encode(raw_prefix, add_special_tokens=False)
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _strip_chat_template_prefix(text: str, prefix: str) -> str:
    """
    Strip the assistant prefix from decoded output. Token boundaries can concatenate
    the last prompt token with the prefix (e.g. 'device' + 'assistant' -> 'deviceassistant').
    """
    text = text.strip()
    if not prefix or not text:
        return text
    if text.startswith(prefix):
        return text[len(prefix) :].strip()
    # Prefix may appear after a concatenated word (no space between tokens)
    idx = text.find(prefix)
    if idx >= 0 and idx < 30:
        return text[idx + len(prefix) :].strip()
    return text


def _ensure_pad_token(model, tokenizer):
    """Set pad_token_id on model config if missing."""
    if getattr(model.config, "pad_token_id", None) is None:
        eos_id = getattr(model.config, "eos_token_id", None)
        if eos_id is None and hasattr(tokenizer, "eos_token_id"):
            eos_id = tokenizer.eos_token_id
        model.config.pad_token_id = eos_id


def model_generate_batch_gemma(
    bundle: ModelBundle,
    prompts: List[str],
    max_new_tokens: int = 256,
    min_tokens: int = 100,
) -> List[str]:
    """
    Batched generation for Gemma. Uses left-padding.
    Requires tokenizer.padding_side = "left" (set in load_model).
    """
    tokenizer = bundle.tokenizer
    model = bundle.model
    _ensure_pad_token(model, tokenizer)

    chats = [apply_chat_template_safe(tokenizer, p) for p in prompts]

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


def model_generate_batch_llama(
    bundle: ModelBundle,
    prompts: List[str],
    max_new_tokens: int = 256,
    min_tokens: int = 100,
) -> List[str]:
    """
    Batched generation for Llama. Avoids padding issues by:
    - Skipping padding when batch_size=1
    - Using per-sequence prompt length (attention_mask) for correct slicing
    """
    tokenizer = bundle.tokenizer
    model = bundle.model
    _ensure_pad_token(model, tokenizer)

    chats = [apply_chat_template_safe(tokenizer, p) for p in prompts]

    # Skip padding for single prompt to avoid Llama tokenizer quirks
    use_padding = len(chats) > 1
    inputs = tokenizer(
        chats,
        return_tensors="pt",
        padding=True if use_padding else False,
        truncation=False,
    ).to(bundle.device)

    if "attention_mask" not in inputs:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
        inputs["attention_mask"] = (inputs["input_ids"] != pad_id).long()

    prompt_lengths = inputs["attention_mask"].sum(dim=1)
    assistant_prefix = _get_assistant_prefix_for_strip(tokenizer)

    # Llama 3.1 has eos_token_id as a list [128001, 128009]; passing it explicitly
    # can cause TypeError in transformers. Use generation_config instead.
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_tokens,
        )

    decoded = []
    for i in range(out.shape[0]):
        plen = prompt_lengths[i].item()
        new_tokens = out[i, plen:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = _strip_chat_template_prefix(text, assistant_prefix)
        decoded.append(text)
    return decoded


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
        """Generate for a single batch. Dispatches to model-specific function."""
        if "llama" in self.model_name.lower():
            return model_generate_batch_llama(
                self.bundle,
                prompts,
                max_new_tokens=max_new_tokens,
                min_tokens=min_tokens,
            )
        else:
            return model_generate_batch_gemma(
                self.bundle,
                prompts,
                max_new_tokens=max_new_tokens,
                min_tokens=min_tokens,
            )
