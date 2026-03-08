"""
Translation pipeline: translate English prompts into Spanish, Chinese, and Japanese
using Helsinki-NLP MarianMT models via Modal.

Usage: modal run pipeline/translate_pipeline.py
"""

import os
import io

import modal
import pandas as pd
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================
TRANSLATION_MODELS = [
    ("Helsinki-NLP/opus-mt-en-es", "es"),
    ("Helsinki-NLP/opus-mt-en-zh", "zh"),
    ("Helsinki-NLP/opus-mt-en-jap", "ja"),
]
BATCH_SIZE = 32

PROMPT_COL_BY_SHEET = {
    "case-a": "goal",
    "case-b": "context_shifted_target",
    "case-c": "politeness_shifted_target",
    "case-d": "both_shifted_target",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "Data", "nlp-queries-dataset-v3.xlsx")

# =============================================================================
# Modal setup
# =============================================================================
app = modal.App("translation-pipeline")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "transformers",
    "torch",
    "sentencepiece",
    "pandas",
    "openpyxl",
)

model_volume = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
MODEL_DIR = "/models"


@app.cls(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=60 * 30,
)
class TranslationRunner:
    model_name: str = modal.parameter()

    @modal.enter()
    def load_model(self):
        from transformers import MarianMTModel, MarianTokenizer

        print(f"Loading translation model: {self.model_name}", flush=True)
        self.tokenizer = MarianTokenizer.from_pretrained(
            self.model_name, cache_dir=MODEL_DIR
        )
        self.model = MarianMTModel.from_pretrained(
            self.model_name, cache_dir=MODEL_DIR
        )
        self.model.eval()
        model_volume.commit()

    @modal.method()
    def translate_batch(self, texts: list[str]) -> list[str]:
        import torch

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            translated = self.model.generate(**inputs)
        return self.tokenizer.batch_decode(translated, skip_special_tokens=True)


# =============================================================================
# Entrypoint
# =============================================================================
@app.local_entrypoint()
def main():
    xl = pd.ExcelFile(DATA_PATH)
    sheet_names = xl.sheet_names
    dfs = {name: pd.read_excel(DATA_PATH, sheet_name=name) for name in sheet_names}

    modal.enable_output()

    for model_id, lang_suffix in TRANSLATION_MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_id} (-> {lang_suffix})")
        print("=" * 60)
        runner = TranslationRunner(model_name=model_id)

        for sheet_name, df in dfs.items():
            prompt_col = PROMPT_COL_BY_SHEET.get(sheet_name)
            if prompt_col is None or prompt_col not in df.columns:
                print(f"  Skipping sheet '{sheet_name}': no matching prompt column")
                continue

            prompts = df[prompt_col].astype(str).tolist()
            out_col = f"{prompt_col}_{lang_suffix}"
            translations = []

            desc = f"{sheet_name} ({lang_suffix})"
            with tqdm(total=len(prompts), desc=desc, unit="prompt") as pbar:
                for j in range(0, len(prompts), BATCH_SIZE):
                    batch = prompts[j : j + BATCH_SIZE]
                    batch_out = runner.translate_batch.remote(batch)
                    translations.extend(batch_out)
                    pbar.update(len(batch))

            df[out_col] = translations
            print(f"  {sheet_name}: {len(translations)} translations -> '{out_col}'")

    with pd.ExcelWriter(DATA_PATH, engine="openpyxl") as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nResults saved to {DATA_PATH}")
