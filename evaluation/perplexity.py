"""
Evaluate target-token loss and perplexity for an ES→EN translation model.

The script compares:
1. Llama 3.1 8B Instruct + QLoRA adapter
2. Base Llama 3.1 8B Instruct

The evaluation uses the same held-out ES→EN dataset for both models.

The fine-tuned model is loaded using the same 4-bit NF4 configuration
used in the original evaluation notebook. The base model is evaluated
in float16, matching the original experiment.

The evaluation metric is target-only causal language-model loss:
source/prompt tokens are masked with -100 and do not contribute to loss.

The documented preliminary experiment evaluated 360 held-out examples.
"""

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Replace these paths for your own environment.
ADAPTER_DIR = Path("/path/to/runs/tmx-es-en-lr1em04-bs2-r4-a16-e1/adapter")
EVAL_DATASET_DIR = Path("/path/to/tmx_eval")

OUTPUT_DIR = Path("results")

MAX_LENGTH = 512

# The original evaluation notebook ultimately used float16 for the
# 4-bit fine-tuned model evaluation.
COMPUTE_DTYPE = torch.float16


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ES→EN target-token loss and perplexity."
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help=(
            "Evaluate only the first N examples. "
            "Useful for a quick test before running the full evaluation."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

def create_bnb_config():
    """Return the 4-bit configuration used for the fine-tuned model."""

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_target_loss(model, tokenizer, dataset, max_length=512):
    """
    Calculate mean target-token loss and perplexity.

    The source/prompt portion is masked with -100 so that only target
    translation tokens contribute to the causal LM loss.
    """

    model.eval()

    total_loss = 0.0
    total_target_tokens = 0

    for example in tqdm(dataset, desc="Evaluating"):
        source = example["source"]
        target = example["target"]

        # Tokenize source/prompt.
        source_tokens = tokenizer(
            source,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )

        # Tokenize target without adding another BOS/EOS sequence.
        target_tokens = tokenizer(
            target,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )

        source_ids = source_tokens["input_ids"]
        target_ids = target_tokens["input_ids"]

        # Concatenate source and target.
        input_ids = torch.cat([source_ids, target_ids], dim=1)

        # Ignore source tokens when calculating loss.
        source_labels = torch.full_like(source_ids, -100)
        labels = torch.cat([source_labels, target_ids], dim=1)

        # Keep the same maximum sequence length used by the original
        # evaluation. This can truncate the target if the combined
        # source + target exceeds MAX_LENGTH.
        input_ids = input_ids[:, :max_length]
        labels = labels[:, :max_length]

        input_ids = input_ids.to(model.device)
        labels = labels.to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                labels=labels,
            )

        valid_target_tokens = (labels != -100).sum().item()

        if valid_target_tokens == 0:
            continue

        # Hugging Face returns mean loss over the valid prediction tokens.
        total_loss += outputs.loss.item() * valid_target_tokens
        total_target_tokens += valid_target_tokens

    if total_target_tokens == 0:
        raise ValueError("No target tokens were available for evaluation.")

    mean_loss = total_loss / total_target_tokens
    perplexity = math.exp(mean_loss)

    return mean_loss, perplexity


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_evaluation_dataset(max_examples=None):
    """Load the held-out evaluation dataset."""

    dataset = load_from_disk(str(EVAL_DATASET_DIR))

    if max_examples is not None:
        max_examples = min(max_examples, len(dataset))
        dataset = dataset.select(range(max_examples))

    print(f"Evaluation examples: {len(dataset)}")

    return dataset


# ---------------------------------------------------------------------------
# Fine-tuned model
# ---------------------------------------------------------------------------

def evaluate_fine_tuned_model(dataset, tokenizer):
    """Evaluate the base model with the saved QLoRA adapter."""

    print("\nLoading base model in 4-bit mode...")

    bnb_config = create_bnb_config()

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=COMPUTE_DTYPE,
    )

    print("Loading QLoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_DIR),
    )

    model.eval()

    print("\nEvaluating fine-tuned model...")

    mean_loss, perplexity = evaluate_target_loss(
        model,
        tokenizer,
        dataset,
        max_length=MAX_LENGTH,
    )

    print(f"Fine-tuned mean target loss: {mean_loss:.4f}")
    print(f"Fine-tuned target perplexity: {perplexity:.4f}")

    result = {
        "task": "ES>EN",
        "evaluation_examples": len(dataset),
        "mean_target_loss": mean_loss,
        "target_perplexity": perplexity,
        "model": "Llama-3.1-8B-Instruct + QLoRA adapter",
    }

    return result, model


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

def evaluate_base_model(dataset, tokenizer):
    """Evaluate the base Llama 3.1 8B Instruct model."""

    print("\nLoading base model in float16 mode...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model.eval()

    print("\nEvaluating base model...")

    mean_loss, perplexity = evaluate_target_loss(
        model,
        tokenizer,
        dataset,
        max_length=MAX_LENGTH,
    )

    print(f"Base mean target loss: {mean_loss:.4f}")
    print(f"Base target perplexity: {perplexity:.4f}")

    result = {
        "mean_target_loss": mean_loss,
        "target_perplexity": perplexity,
    }

    return result, model


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def save_json(data, filename):
    """Save a result dictionary as JSON."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / filename

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved: {output_path}")


def create_comparison(fine_tuned_result, base_result, evaluation_examples):
    """Create a comparison of base and fine-tuned results."""

    base_loss = base_result["mean_target_loss"]
    fine_tuned_loss = fine_tuned_result["mean_target_loss"]

    base_ppl = base_result["target_perplexity"]
    fine_tuned_ppl = fine_tuned_result["target_perplexity"]

    loss_reduction = (
        (base_loss - fine_tuned_loss) / base_loss
    ) * 100

    perplexity_reduction = (
        (base_ppl - fine_tuned_ppl) / base_ppl
    ) * 100

    return {
        "task": "ES>EN",
        "evaluation_examples": evaluation_examples,
        "base_model": base_result,
        "fine_tuned_model": {
            "mean_target_loss": fine_tuned_loss,
            "target_perplexity": fine_tuned_ppl,
        },
        "loss_reduction_percent": loss_reduction,
        "perplexity_reduction_percent": perplexity_reduction,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 70)
    print("ES→EN TARGET-ONLY PERPLEXITY EVALUATION")
    print("=" * 70)

    print(f"Model: {MODEL_ID}")
    print(f"Evaluation dataset: {EVAL_DATASET_DIR}")
    print(f"Adapter: {ADAPTER_DIR}")

    # Load tokenizer once and reuse it for both evaluations.
    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load evaluation dataset.
    dataset = load_evaluation_dataset(
        max_examples=args.max_examples
    )

    # -----------------------------------------------------------------------
    # Fine-tuned model
    # -----------------------------------------------------------------------

    fine_tuned_result, fine_tuned_model = evaluate_fine_tuned_model(
        dataset,
        tokenizer,
    )

    save_json(
        fine_tuned_result,
        "tmx_es_en_perplexity.json",
    )

    # Free GPU memory before loading the base model.
    del fine_tuned_model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Base model
    # -----------------------------------------------------------------------

    base_result, base_model = evaluate_base_model(
        dataset,
        tokenizer,
    )

    # -----------------------------------------------------------------------
    # Comparison
    # -----------------------------------------------------------------------

    comparison = create_comparison(
        fine_tuned_result,
        base_result,
        len(dataset),
    )

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(
        f"Base loss:        "
        f"{comparison['base_model']['mean_target_loss']:.4f}"
    )

    print(
        f"Fine-tuned loss:  "
        f"{comparison['fine_tuned_model']['mean_target_loss']:.4f}"
    )

    print(
        f"Loss reduction:   "
        f"{comparison['loss_reduction_percent']:.2f}%"
    )

    print()

    print(
        f"Base perplexity:       "
        f"{comparison['base_model']['target_perplexity']:.4f}"
    )

    print(
        f"Fine-tuned perplexity: "
        f"{comparison['fine_tuned_model']['target_perplexity']:.4f}"
    )

    print(
        f"Perplexity reduction:  "
        f"{comparison['perplexity_reduction_percent']:.2f}%"
    )

    save_json(
        comparison,
        "tmx_es_en_perplexity_comparison.json",
    )

    del base_model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
