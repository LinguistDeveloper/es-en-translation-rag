import json
import random
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

SEED = 42

CORPORA = {
    "dgt": {
        "path": Path("/path/to/dgt-es-en.jsonl"),
        "sample_size": 100_000,
    },
    "jrc_acquis": {
        "path": Path("/path/to/jrc-acquis-es-en.jsonl"),
        "sample_size": 50_000,
    },
    "gnome": {
        "path": Path("/path/to/gnome-es-en.jsonl"),
        "sample_size": 50_000,
    },
}

OUTPUT_DIR = Path("/content/drive/MyDrive/Colab Notebooks/translation_rag_data")

TRAINING_OUTPUT = OUTPUT_DIR / "training_samples.jsonl"
VECTOR_DB_OUTPUT = OUTPUT_DIR / "vector_db_samples.jsonl"


# ============================================================
# Reservoir sampling
# ============================================================

def reservoir_sample(jsonl_file, n, seed=42):
    """
    Select n records from a JSONL file using reservoir sampling.

    The sampling process is deterministic when the same seed,
    input file and sample size are used.
    """

    rng = random.Random(seed)
    sample = []
    valid_records = 0

    with open(jsonl_file, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            record = json.loads(line)

            if len(sample) < n:
                sample.append(record)

            else:
                j = rng.randint(0, valid_records)

                if j < n:
                    sample[j] = record

            valid_records += 1

    if valid_records < n:
        raise ValueError(
            f"{jsonl_file} contains only {valid_records:,} valid records, "
            f"but {n:,} were requested."
        )

    return sample


# ============================================================
# Main data preparation pipeline
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # --------------------------------------------------------
    # Sample each corpus
    # --------------------------------------------------------

    for name, config in CORPORA.items():

        filepath = config["path"]
        sample_size = config["sample_size"]

        print(f"Sampling {sample_size:,} from {name}...")

        samples = reservoir_sample(
            filepath,
            sample_size,
            seed=SEED
        )

        # Add corpus provenance
        for record in samples:
            record["corpus"] = name

        all_samples.extend(samples)

        print(f"  Added {len(samples):,}")

    # --------------------------------------------------------
    # Shuffle combined dataset
    # --------------------------------------------------------

    random.Random(SEED).shuffle(all_samples)

    # --------------------------------------------------------
    # Split into training/evaluation pool and vector DB
    # --------------------------------------------------------

    split = int(0.90 * len(all_samples))

    training_samples = all_samples[:split]
    vector_db_samples = all_samples[split:]

    print()
    print("Dataset split")
    print("-------------")
    print(f"Total:     {len(all_samples):,}")
    print(f"Training:  {len(training_samples):,}")
    print(f"Vector DB: {len(vector_db_samples):,}")

    # --------------------------------------------------------
    # Write output datasets
    # --------------------------------------------------------

    with open(TRAINING_OUTPUT, "w", encoding="utf-8") as f:

        for record in training_samples:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    with open(VECTOR_DB_OUTPUT, "w", encoding="utf-8") as f:

        for record in vector_db_samples:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    print()
    print("Output")
    print("------")
    print(f"Training dataset:  {TRAINING_OUTPUT}")
    print(f"Vector DB dataset: {VECTOR_DB_OUTPUT}")


if __name__ == "__main__":
    main()
