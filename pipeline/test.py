"""
Test script: run inference on 10 Advench samples via Modal.
Prints prompts and model outputs in a readable format.
"""

import os
import random
import modal
from modal_inference import app, ModelRunner


NUM_SAMPLES = 5
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "Data", "nlp-queries-dataset.xlsx")
BATCH_SIZE = 4  # process one at a time to avoid padding issues
MAX_TOKENS = 200
MIN_TOKENS = 100

# Use one model for a quick test
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME = "Qwen"


def main():
    import pandas as pd

    advench = pd.read_excel(DATA_PATH, sheet_name=0)
    all_prompts = advench["goal"].astype(str).tolist()
    prompts = random.sample(all_prompts, min(NUM_SAMPLES, len(all_prompts)))

    print(f"Running {len(prompts)} prompts through {MODEL_NAME} (Modal)...\n")

    modal.enable_output()
    with app.run():
        runner = ModelRunner(model_name=MODEL_ID)

        all_outputs = []
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i : i + BATCH_SIZE]
            outputs = runner.generate_batch.remote(
                batch,
                max_new_tokens=MAX_TOKENS,
                min_tokens=MIN_TOKENS,
            )
            all_outputs.extend(outputs)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for i, (prompt, output) in enumerate(zip(prompts, all_outputs), 1):
        print(f"\n--- Sample {i} ---\n")
        print(f"PROMPT:\n{prompt}\n")
        print(f"OUTPUT:\n{output}\n")
        print("-" * 80)


if __name__ == "__main__":
    main()
