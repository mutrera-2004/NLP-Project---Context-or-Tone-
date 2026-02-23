"""
Run the inference pipeline: load data, run models via Modal, save results.
Adjust hyperparameters below before running.
"""

from pathlib import Path

import modal
import pandas as pd
from tqdm import tqdm

from modal_inference import app, ModelRunner


# =============================================================================
# HYPERPARAMETERS (edit these before running)
# =============================================================================
MODEL_CONFIGS = [
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama"),
    ("google/gemma-7b-it", "Gemma"),
]
BATCH_SIZE = 8 # process one at a time to avoid padding issues
MAX_TOKENS = 200
MIN_TOKENS = 100

# Paths
DATA_PATH = "Data/nlp-queries-dataset.xlsx"
RESULTS_DIR = "results"
RESULTS_PATH = "results/results.xlsx"

# Dataset config: (sheet_index, prompt_column) per sheet
# Sheets: Advench, Context Shift, Polite Shift, Both Shift
DATASET_CONFIG = [
    ("goal", "Advench"),
    ("context_goal", "Context Shift"),
    ("polite_goal", "Polite Shift"),
    ("context_polite_goal", "Both Shift"),
]


def load_data(data_path: str) -> tuple[list[pd.DataFrame], list[str], list[str]]:
    """Load Excel sheets and return (dataframes, prompt columns, sheet names)."""
    xl = pd.ExcelFile(data_path)
    sheets = xl.sheet_names
    dfs = [pd.read_excel(data_path, sheet_name=s) for s in sheets]
    prompt_cols = [c for c, _ in DATASET_CONFIG]
    names = [n for _, n in DATASET_CONFIG]
    return dfs, prompt_cols, names


def run_inference():
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    dfs, prompt_cols, names = load_data(DATA_PATH)
    all_prompts = [df[col].astype(str).tolist() for df, col in zip(dfs, prompt_cols)]

    modal.enable_output()
    with app.run():
        for model_id, col_name in MODEL_CONFIGS:
            print(f"\n{'='*60}")
            print(f"Model: {col_name} ({model_id})")
            print("="*60)
            runner = ModelRunner(model_name=model_id)

            for i, (df, prompts, name) in enumerate(zip(dfs, all_prompts, names)):
                n_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE
                outputs = []

                with tqdm(total=len(prompts), desc=f"{name} ({col_name})", unit="prompt") as pbar:
                    for j in range(0, len(prompts), BATCH_SIZE):
                        batch = prompts[j : j + BATCH_SIZE]
                        batch_out = runner.generate_batch.remote(
                            batch,
                            max_new_tokens=MAX_TOKENS,
                            min_tokens=MIN_TOKENS,
                        )
                        outputs.extend(batch_out)
                        pbar.update(len(batch))

                dfs[i][col_name] = outputs
                print(f"  {name}: {len(outputs)} responses")

    with pd.ExcelWriter(RESULTS_PATH, engine="openpyxl") as writer:
        for df, name in zip(dfs, names):
            df.to_excel(writer, sheet_name=name, index=False)

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_inference()
