# ES→EN Translation: Fine-Tuning, Retrieval and Evaluation

A reproducible experiment investigating whether **domain-specific fine-tuning and translation-memory retrieval** can improve Spanish→English translation of formal technical, institutional and enterprise IT content.

The project compares a base LLM and a fine-tuned model, and subsequently investigates whether retrieval of relevant translation-memory segments provides additional improvements.

## Experimental Pipeline

The experiment is divided into four phases:

| Phase                                  | Description                                                                            | Status      |
| -------------------------------------- | -------------------------------------------------------------------------------------- | ----------- |
| **1. Data Preparation**                | Select, sample, combine and split bilingual corpora                                    | In progress |
| **2. Model Fine-Tuning**               | Fine-tune an LLM for ES→EN translation using QLoRA                                     | Placeholder |
| **3. Model Evaluation**                | Evaluate the base and fine-tuned models on held-out data                               | Placeholder |
| **4. Retrieval-Augmented Translation** | Add translation-memory retrieval and compare base/FT models with and without retrieval | Placeholder |

---

# Phase 1 — Data Preparation

The first phase creates a reproducible dataset from three bilingual corpora representing complementary aspects of the intended translation domain.

### Source corpora

| Corpus         |      Sample | Role                                            |
| -------------- | ----------: | ----------------------------------------------- |
| **DGT**        |     100,000 | Main institutional/technical backbone           |
| **JRC-Acquis** |      50,000 | Legal, regulatory and administrative language   |
| **GNOME**      |      50,000 | Software localization and technical terminology |
| **Total**      | **200,000** |                                                 |

The corpora were selected for **domain complementarity and data quality**, rather than simply taking the largest available datasets.

## Sampling

Because the source corpora are substantially larger than the dataset required for the experiment, reservoir sampling is used to select a reproducible subset without loading the complete files into memory.

The sampler uses a fixed random seed:

```python
seed = 42
```

This makes the sampling process reproducible: given the same input file, sample size and seed, the same records will be selected.

Each corpus is sampled independently according to its target size:

```text
DGT       → 100,000
JRC-Acquis → 50,000
GNOME     → 50,000
-------------------
Total     → 200,000
```

Each sampled record is also given a `corpus` field identifying its source corpus.

## Combining and shuffling

The three samples are combined into a single 200,000-segment dataset.

The resulting dataset is then shuffled using the same fixed seed:

```python
random.Random(42).shuffle(all_samples)
```

This is important because the corpora are initially processed sequentially. Without shuffling, a simple final split could disproportionately contain segments from whichever corpus was processed last.

## Dataset split

The combined 200,000 sampled segments are divided into two datasets:

```text
180,000 → training/evaluation pool
 20,000 → translation-memory vector database
```

The **180,000-segment training/evaluation pool** is subsequently divided into training and held-out evaluation data as part of the fine-tuning workflow:

```text
162,000 → fine-tuning
 18,000 → held-out evaluation
```

The resulting experimental datasets are therefore:

| Dataset             |    Segments | Purpose                                      |
| ------------------- | ----------: | -------------------------------------------- |
| Training            |     162,000 | QLoRA fine-tuning                            |
| Held-out evaluation |      18,000 | Evaluation of the base and fine-tuned models |
| Vector database     |      20,000 | Translation-memory retrieval                 |
| **Total**           | **200,000** |                                              |

The first split is performed by `prepare_data.py`. The subsequent 162,000/18,000 split is performed during the fine-tuning workflow.
                                             |

### Random seeds

Three stages of the process use deterministic randomisation:

1. **Reservoir sampling** — determines which records are selected from each source corpus.
2. **Combined-data shuffle** — determines which 200,000 records enter the training/evaluation pool versus the vector database.
3. **Train/evaluation split** — determines which records from the 180,000-record pool become training versus held-out evaluation data.

All currently use seed `42`.

### Data leakage considerations

Random splitting does not by itself guarantee that there are no duplicate or near-duplicate translation units across the training, evaluation and retrieval datasets.

For a rigorous comparison, the project therefore treats **duplicate and near-duplicate detection** as an important data-quality consideration. Exact or highly similar source segments appearing in multiple experimental partitions could otherwise make retrieval or evaluation results artificially favourable.

The data-preparation pipeline will document any deduplication or leakage checks applied to the final datasets.

## Phase 1 Output

The data-preparation phase produces the datasets required by the subsequent stages:

```text
Raw bilingual corpora
        │
        ▼
Reservoir sampling
        │
        ▼
200,000 combined segments
        │
        ▼
Shuffle
        │
        ├──────────────► 20,000 Vector DB
        │
        ▼
180,000 training/evaluation pool
        │
        ├──────────────► 18,000 Held-out evaluation
        │
        ▼
162,000 Fine-tuning
```

**Implementation:** [`data/prepare_data.py`](data/prepare_data.py)

---

### Phase 2 — Model Fine-Tuning

**Status: Documented; clean script not yet independently tested**

The fine-tuning workflow uses QLoRA to adapt Llama 3.1 8B Instruct for Spanish→English translation.

The documented pipeline includes:

* 4-bit NF4 quantization with double quantization
* LoRA adaptation of the attention and MLP projection layers
* Spanish→English instruction formatting
* masked causal-language-model loss, calculated on the target translation
* configurable dataset size for small-scale experiments or the full fine-tuning pool
* reproducible train/validation splitting using seed 42
* adapter and experiment-configuration saving

The repository pipeline is designed to use the 180,000-example fine-tuning pool produced during Phase 1, which is subsequently divided into 162,000 training examples and 18,000 held-out evaluation examples.

### Preliminary fine-tuning experiment

Before creating the cleaned repository script, the QLoRA workflow was experimentally run in Google Colab using a much smaller subset:

| Dataset             | Examples |
| ------------------- | -------: |
| Training            |    3,240 |
| Held-out validation |      360 |
| Total               |    3,600 |

Configuration included:

* **Base model:** Llama 3.1 8B Instruct
* **Learning rate:** 1e-4
* **Effective batch size:** 8
* **LoRA rank:** 4
* **LoRA alpha:** 16
* **LoRA dropout:** 0.10
* **Epochs:** 1
* **Maximum sequence length:** 512
* **Seed:** 42

On the 360-example held-out evaluation set:

| Metric     | Base model | Fine-tuned model |
| ---------- | ---------: | ---------------: |
| Loss       |     1.4506 |           1.2324 |
| Perplexity |     4.2657 |           3.4294 |

This corresponds to a **15.04% reduction in loss** and a **19.61% reduction in perplexity** for this preliminary experiment.

These results should be interpreted as preliminary because they were obtained from the 3,600-example subset rather than the full 180,000-example fine-tuning pool. The cleaned `finetune.py` script documents the experimental workflow but has not yet been independently rerun from the repository.

### Phase 3 — Model Evaluation

**Status: Placeholder**

The evaluation phase will use the held-out evaluation data to compare the base and fine-tuned models.

Initial evaluation will use perplexity, followed by translation-quality metrics such as COMET, BERTScore and BLEU where appropriate.

### Phase 4 — Retrieval-Augmented Translation

**Status: Placeholder**

The final experiment will investigate whether translation-memory retrieval improves Spanish→English translation quality, comparing:

1. Base model
2. Base model + retrieval
3. Fine-tuned model
4. Fine-tuned model + retrieval

The retrieval dataset consists of the 20,000 examples reserved during Phase 1.
