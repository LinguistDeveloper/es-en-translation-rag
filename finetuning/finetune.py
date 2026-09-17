import os
import json
import random
from pathlib import Path

import numpy as np
import torch


# ============================================================
# Experiment configuration
# ============================================================

SEED = 42

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Input data
TRAINING_FILE = Path("/path/to/training_samples.jsonl")

# Optional limit for small-scale experiments.
# Set to None to use the complete input file.
# Small-scale experiment used for the preliminary evaluation:
# 3,600 examples -> 3,240 training + 360 validation.
#
# Set to None for the full Phase 1 training pool.
MAX_TRAINING_EXAMPLES = 3_600

# Validation split
VAL_FRACTION = 0.10

# QLoRA
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

LORA_RANK = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.10

NUM_EPOCHS = 1
WEIGHT_DECAY = 0.01

MAX_LENGTH = 512

# 4-bit quantization
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"

# Output
RUN_ROOT = Path("/path/to/runs")
TAG = "es-en-translation"

# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Load and split dataset
# ============================================================

from datasets import load_dataset


dataset = load_dataset(
    "json",
    data_files=str(TRAINING_FILE),
)["train"]

# Remove records with missing source or target text.
dataset = dataset.filter(
    lambda x: bool(x["source"]) and bool(x["target"])
)

# Use a reproducible subset for small-scale experiments.
if MAX_TRAINING_EXAMPLES is not None:
    if MAX_TRAINING_EXAMPLES > len(dataset):
        raise ValueError(
            f"Requested {MAX_TRAINING_EXAMPLES:,} examples, "
            f"but only {len(dataset):,} are available."
        )

    dataset = dataset.shuffle(seed=SEED).select(
        range(MAX_TRAINING_EXAMPLES)
    )

# Split into training and held-out validation data.
split = dataset.train_test_split(
    test_size=VAL_FRACTION,
    seed=SEED,
)

train_ds = split["train"]
eval_ds = split["test"]

print(f"Total examples:      {len(dataset):,}")
print(f"Training examples:   {len(train_ds):,}")
print(f"Validation examples: {len(eval_ds):,}")

# ============================================================
# Tokenizer and prompt formatting
# ============================================================

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


def make_messages(example):
    return [
        {
            "role": "system",
            "content": (
                "You are a professional translation engine. "
                "Translate from Spanish to English. "
                "Preserve the meaning, terminology, formatting, "
                "and named entities of the source. "
                "Output only the translation."
            ),
        },
        {
            "role": "user",
            "content": example["source"],
        },
        {
            "role": "assistant",
            "content": example["target"],
        },
    ]


def tokenize_example(example):
    messages = make_messages(example)

    # Complete conversation, including the reference translation.
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Prompt only, ending immediately before the assistant response.
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )

    full = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False,
    )

    prompt = tokenizer(
        prompt_text,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False,
    )

    labels = full["input_ids"].copy()

    # Do not calculate loss on the system/user prompt.
    prompt_len = min(
        len(prompt["input_ids"]),
        len(labels),
    )

    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids": full["input_ids"],
        "attention_mask": full["attention_mask"],
        "labels": labels,
    }


tokenized_train = train_ds.map(
    tokenize_example,
    remove_columns=train_ds.column_names,
    desc="Tokenizing training data",
)

tokenized_eval = eval_ds.map(
    tokenize_example,
    remove_columns=eval_ds.column_names,
    desc="Tokenizing validation data",
)

print("Tokenization complete.")
print("Training examples:", len(tokenized_train))
print("Validation examples:", len(tokenized_eval))

# ============================================================
# Load 4-bit base model
# ============================================================

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. QLoRA training requires a GPU."
    )

print("GPU:", torch.cuda.get_device_name(0))
print(
    f"GPU memory: "
    f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
)


compute_dtype = (
    torch.bfloat16
    if BNB_4BIT_COMPUTE_DTYPE == "bfloat16"
    else torch.float16
)


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
    bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    bnb_4bit_compute_dtype=compute_dtype,
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=compute_dtype,
)

model.config.use_cache = False

print("4-bit model loaded.")


# ============================================================
# Configure QLoRA
# ============================================================

model = prepare_model_for_kbit_training(model)


lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)


model = get_peft_model(
    model,
    lora_config,
)


model.print_trainable_parameters()

trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Trainable parameters: {trainable:,}")
print(f"Total parameters:     {total:,}")
print(
    f"Trainable percentage:  "
    f"{100 * trainable / total:.4f}%"
)

# ============================================================
# Data collator
# ============================================================

class CausalLMCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(
            len(x["input_ids"])
            for x in features
        )

        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_masks = []
        labels = []

        for x in features:
            padding = max_len - len(x["input_ids"])

            input_ids.append(
                x["input_ids"] + [pad_id] * padding
            )

            attention_masks.append(
                x["attention_mask"] + [0] * padding
            )

            labels.append(
                x["labels"] + [-100] * padding
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }


collator = CausalLMCollator(tokenizer)


# ============================================================
# Training configuration
# ============================================================

from transformers import TrainingArguments, Trainer


training_args = TrainingArguments(
    output_dir=str(RUN_ROOT),

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    num_train_epochs=NUM_EPOCHS,

    weight_decay=WEIGHT_DECAY,

    logging_steps=10,

    eval_strategy="steps",

    save_strategy="steps",
    save_steps=20,
    save_total_limit=2,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    fp16=(
        BNB_4BIT_COMPUTE_DTYPE == "float16"
    ),

    bf16=(
        BNB_4BIT_COMPUTE_DTYPE == "bfloat16"
    ),

    optim="paged_adamw_8bit",

    report_to="none",

    remove_unused_columns=False,

    seed=SEED,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=collator,
)

# ============================================================
# Train
# ============================================================

print("Starting training...")

train_result = trainer.train()

print("Training complete.")

print(
    f"Training loss: "
    f"{train_result.training_loss:.4f}"
)


# ============================================================
# Save adapter and tokenizer
# ============================================================

RUN_NAME = (
    f"{TAG}"
    f"-lr{LEARNING_RATE:.0e}"
    f"-bs{BATCH_SIZE}"
    f"-r{LORA_RANK}"
    f"-a{LORA_ALPHA}"
    f"-e{NUM_EPOCHS}"
)

RUN_DIR = RUN_ROOT / RUN_NAME
ADAPTER_DIR = RUN_DIR / "adapter"

ADAPTER_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

model.save_pretrained(
    ADAPTER_DIR,
)

tokenizer.save_pretrained(
    ADAPTER_DIR,
)

print(
    f"Adapter saved to: {ADAPTER_DIR}"
)

# ============================================================
# Save experiment metadata
# ============================================================

experiment_config = {
    "model_id": MODEL_ID,

    "training_examples": len(train_ds),
    "validation_examples": len(eval_ds),

    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": (
        GRADIENT_ACCUMULATION_STEPS
    ),
    "effective_batch_size": (
        BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    ),

    "lora_rank": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,

    "epochs": NUM_EPOCHS,
    "weight_decay": WEIGHT_DECAY,
    "max_length": MAX_LENGTH,

    "seed": SEED,

    "quantization": {
        "type": BNB_4BIT_QUANT_TYPE,
        "double_quant": BNB_4BIT_USE_DOUBLE_QUANT,
        "compute_dtype": BNB_4BIT_COMPUTE_DTYPE,
    },
}

with open(
    RUN_DIR / "config.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        experiment_config,
        f,
        indent=2,
    )

print(
    f"Configuration saved to: "
    f"{RUN_DIR / 'config.json'}"
)
