import typing
from utils import *
import torch

LLAMA_ID = "meta-llama/Llama-3.1-8B" # unavailable for now
GEMMA_ID = "google/gemma-3-1b-it"

MAX_TOKENS = 300


bundle = load_model_and_tokenizer(GEMMA_ID)
prompt = "Explain the doppler effect in simple terms."
result = model_generate(bundle, prompt)

print(result)


# def prompt_model(model_id: str, prompt: str) -> str:
#     '''
#     Given the MODEL_ID and a prompt, prompt the model and return 
#     the result as a string

#     Args:
#         model_id: st
#     '''
#     bundle = load_model_and_tokenizer(model_id)

#     result = model_generate(bundle, prompt)