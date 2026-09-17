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

# Phase 2 — Model Fine-Tuning

> **Status: Placeholder — implementation to be added**

This phase fine-tunes the base LLM for **Spanish→English translation** using the 162,000-segment training dataset produced in Phase 1.

The experiment uses **QLoRA** so that the model can be fine-tuned within the available GPU resources while keeping the number of trainable parameters small.

### Planned workflow

```text
Prepared training data
        │
        ▼
Tokenization / prompt formatting
        │
        ▼
QLoRA fine-tuning
        │
        ▼
Fine-tuned adapter
```

Further details will be added once the final fine-tuning script and configuration have been committed.

**Implementation:** [`finetuning/finetune.py`](finetuning/finetune.py)

---

# Phase 3 — Model Evaluation

> **Status: Placeholder — implementation to be added**

This phase evaluates the base and fine-tuned models using the **18,000-segment held-out evaluation dataset**.

The evaluation is designed to measure whether fine-tuning produces an improvement on data that was not used during training.

### Initial metric

The first evaluation metric is **perplexity**, derived from the model's loss on the held-out target text.

The project will subsequently incorporate translation-quality metrics such as:

* COMET
* BERTScore
* BLEU

where appropriate.

### Planned comparison

```text
                 ┌──► Base model
Held-out data ───┤
                 └──► Fine-tuned model
```

The results will be reported comparatively rather than relying on a single metric.

**Implementation:** [`evaluation/`](evaluation/)

---

# Phase 4 — Retrieval-Augmented Translation

> **Status: Placeholder — implementation to be added**

The final phase investigates whether retrieving relevant translation-memory segments can further improve translation quality.

The 20,000-segment retrieval dataset produced during Phase 1 will be used as the translation-memory/vector database.

### Planned experiment

The retrieval experiment will compare four configurations:

```text
                    No Retrieval       With Retrieval
                    ────────────       ──────────────

Base model              │                    │
                        ▼                    ▼
                     Base                 Base + RAG


Fine-tuned model        │                    │
                        ▼                    ▼
                    Fine-tuned          Fine-tuned + RAG
```

The purpose is to distinguish between:

1. improvements attributable to fine-tuning;
2. improvements attributable to retrieval;
3. any additional benefit from combining fine-tuning and retrieval.

The retrieval system will use semantic similarity between the incoming Spanish source segment and the translation-memory corpus, with the retrieved bilingual context supplied to the LLM during translation.

Further details of the embedding model, vector index, retrieval strategy and evaluation methodology will be added once the implementation is finalized.

---

# Overall Experimental Design

The complete experiment can therefore be represented as:

```text
                    ┌─────────────────────┐
                    │  DGT / JRC / GNOME  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Data Preparation   │
                    │     200k segments   │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        162k Training + 18k Eval       20k Vector DB
                 │                           │
                 ▼                           │
          ┌──────────────┐                   │
          │  Fine-tuning │                   │
          └──────┬───────┘                   │
                 │                           │
                 ▼                           │
          Fine-tuned model                   │
                 │                           │
                 └─────────────┬─────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Evaluation     │
                    │ Perplexity / COMET  │
                    │ BERTScore / BLEU    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Retrieval experiment│
                    │ Base + RAG           │
                    │ FT + RAG             │
                    └─────────────────────┘
```

# Reproducibility

The experiment uses fixed random seeds and separates data preparation, fine-tuning and evaluation into distinct stages.

Large datasets, model weights and generated artifacts are not stored directly in the repository. The repository contains the code, configuration and documentation required to understand and reproduce the experiment.

---

## Repository Structure

```text
es-en-translation-rag/
│
├── README.md
│
├── data/
│   └── prepare_data.py
│
├── finetuning/
│   └── finetune.py
│
├── evaluation/
│   └── perplexity.py
│
├── rag/
│   └── [to be added]
│
├── configs/
│   └── [to be added]
│
└── requirements.txt
```

# Project Status

* [x] Select source corpora
* [x] Implement reproducible reservoir sampling
* [x] Combine and shuffle sampled data
* [x] Create training/evaluation/retrieval partitions
* [ ] Add final data-leakage/deduplication checks
* [ ] Add QLoRA fine-tuning script
* [ ] Add held-out evaluation
* [ ] Add translation-quality metrics
* [ ] Add vector database / retrieval pipeline
* [ ] Compare base vs fine-tuned models with and without retrieval
* [ ] Analyse and document final results
