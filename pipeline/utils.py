import torch
from typing import Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.1-8B"
QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 300

@dataclass
class ModelBundle:
    tokenizer: Any
    model: Any
    device: torch.device
    model_name: str
    torch_dtype: torch.dtype


def load_model_and_tokenizer(
    model_name: str = MODEL_ID,
) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch_dtype = torch.bfloat16 if "qwen" in model_name.lower() else torch.float16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        torch_dtype = torch.float32
    else:
        device = torch.device("cpu")
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )
    model.to(device)

    # pad_token_id fallback
    if getattr(model.config, "pad_token_id", None) is None:
        eos_id = getattr(model.config, "eos_token_id", None)
        if eos_id is None and hasattr(tokenizer, "eos_token_id"):
            eos_id = tokenizer.eos_token_id
        model.config.pad_token_id = eos_id

    return ModelBundle(
        tokenizer=tokenizer,
        model=model,
        device=device,
        model_name=model_name,
        torch_dtype=torch_dtype,
    )


def apply_chat_template(tokenizer: AutoTokenizer, prompt: str) -> str:
    """
    Apply the chat template to a prompt string.

    Inputs:
        tokenizer [AutoTokenizer]: HuggingFace tokenizer with chat template support
        prompt [str]: The raw prompt string to wrap

    Returns:
        The formatted prompt string with chat template applied
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    chat_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False, # might want to tokenize the raw prompt?
        add_generation_prompt=True
    )
    return chat_str

def model_generate(
    bundle: ModelBundle,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    chat = apply_chat_template(bundle.tokenizer, prompt)
    inputs = bundle.tokenizer(chat, return_tensors="pt").to(bundle.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = bundle.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            # Let the model config handle eos_token_id/pad_token_id unless you have a specific override
        )
    new_tokens = out[0][input_len:] # may have problem when batching/padding is involved
    text = bundle.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()