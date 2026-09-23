# ES→EN Translation: Fine-Tuning, Retrieval and Evaluation

A reproducible experiment investigating whether **domain-specific fine-tuning and translation-memory retrieval** can improve Spanish→English translation of formal technical, institutional and enterprise IT content.

The project compares a base LLM and a fine-tuned model, and subsequently investigates whether retrieval of relevant translation-memory segments provides additional improvements.

## Experimental Pipeline

The experiment is divided into four phases:

| Phase                                  | Description                                                                            | Status      |
| -------------------------------------- | -------------------------------------------------------------------------------------- | ----------- |
| **1. Data Preparation**                | Select, sample, combine and split bilingual corpora                                    | In progress |
| **2. Model Fine-Tuning**               | Fine-tune an LLM for ES→EN translation using QLoRA                                     | In progress |
| **3. Model Evaluation**                | Evaluate the base and fine-tuned models on held-out data                               | In progress |
| **4. Retrieval-Augmented Translation** | Add translation-memory retrieval and compare base/FT models with and without retrieval | In progress |

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
| Loss       |     1.4506 |           1.2295 |
| Perplexity |     4.2657 |           3.4194 |

This corresponds to a **15.04% reduction in loss** and a **19.61% reduction in perplexity** for this preliminary experiment.

These results should be interpreted as preliminary because they were obtained from the 3,600-example subset rather than the full 180,000-example fine-tuning pool. The cleaned `finetune.py` script documents the experimental workflow but has not yet been independently rerun from the repository.

---

## Phase 3 — Evaluation

**Status: Preliminary evaluation complete; extended COMET and token-level analysis ongoing**

The project evaluates Spanish→English translation using held-out data that was not used for fine-tuning.

### Perplexity evaluation

`evaluation/perplexity.py` calculates **target-only causal language-model loss and perplexity**, masking the source/prompt tokens so that only the English translation contributes to the evaluation loss.

The evaluation compares:

* Base `Llama-3.1-8B-Instruct`
* `Llama-3.1-8B-Instruct` + QLoRA adapter

The preliminary experiment used **3,600 examples**, split into:

* 3,240 fine-tuning examples
* 360 held-out evaluation examples

Configuration:

* Learning rate: `1e-4`
* LoRA rank: `4`
* LoRA alpha: `16`
* LoRA dropout: `0.10`
* Epochs: `1`
* Effective batch size: `8`
* Maximum sequence length: `512`
* Random seed: `42`

Preliminary held-out results:

| **Model**        | **Mean target loss** | **Target perplexity** |
| ---------------- | -------------------: | --------------------: |
| Base model       |               1.4506 |                4.2657 |
| Fine-tuned model |               1.2324 |                3.4294 |

This corresponds to a **15.04% reduction in mean target loss** and a **19.61% reduction in perplexity** on the 360-example held-out set.

### Translation and COMET evaluation

The project also evaluates generated Spanish→English translations against the English reference translations using **COMET**.

Because COMET scoring is substantially more computationally expensive than the perplexity evaluation, an initial **50-example subset** of the held-out evaluation data is being used as a preliminary test. This provides an efficient way to validate the translation-generation and COMET scoring pipeline and obtain an initial comparison before running the full 360-example evaluation.

Preliminary results on the 50-example sample:

| **Model**        | **Mean COMET** |
| ---------------- | -------------: |
| Base model       |         0.8490 |
| Fine-tuned model |     **0.8692** |
| Mean difference  |    **+0.0202** |

The fine-tuned model achieved a **+0.0202 absolute increase in mean COMET** on this preliminary sample. Individual examples showed both improvements and regressions, so the result is treated as an initial signal rather than a definitive measurement of overall translation-quality improvement.

The full **360-example held-out set** will be evaluated once the COMET pipeline has been fully validated.

### Logit and token-level analysis

A direct comparison of the base and fine-tuned model logits confirms that the QLoRA adapter is active and materially changes the model's output probability distribution.

Initial token-level analysis also indicates that the fine-tuned model can assign substantially higher probability to reference translation tokens.

A full token-level analysis across the 360-example evaluation set is ongoing. This will examine whether the reduction in perplexity corresponds to a systematic increase in the probability assigned to the reference translations, and whether those changes are concentrated in particular types of tokens or translation segments.

### Next evaluation stages

Further evaluation will investigate translation quality using additional metrics, including:

* COMET
* BERTScore
* BLEU
* Reference-token probability / negative log-likelihood

The immediate next step is to extend the preliminary COMET evaluation from **50 examples to the full 360-example held-out set**, using a consistent generation and scoring configuration.

The next major experiment will then test whether translation-memory retrieval provides an additional benefit beyond fine-tuning.

The RAG experiments will compare four configurations:

1. Base model
2. Base model + translation-memory RAG
3. Fine-tuned model
4. Fine-tuned model + translation-memory RAG

The RAG experiments will use the separate **20,000-segment translation-memory retrieval set**.

This will allow the project to investigate not only whether fine-tuning and retrieval improve performance, but also whether their effects are **complementary**.

---

### Phase 4 — Retrieval-Augmented Translation

**Status: Vector database implemented; base and adapter hybrid predictions generated; initial COMET evaluation completed**

The retrieval experiment investigates whether translation-memory retrieval improves Spanish→English translation quality and whether retrieval has complementary effects when combined with domain-specific fine-tuning.

The current implementation uses a translation-memory vector database containing the segments reserved during Phase 1. The workflow supports generating and comparing predictions for four configurations:

1. Base model
2. Base model + retrieval
3. Fine-tuned adapter model
4. Fine-tuned adapter model + retrieval

The retrieval experiment currently focuses on semantic vector retrieval, with fuzzy matching and alternative retrieval-weight configurations planned for subsequent investigation.

### Retrieval and hybrid prediction workflow

[svg](https://github.com/LinguistDeveloper/es-en-translation-rag#retrieval-and-hybrid-prediction-workflow)

The workflow separates model generation from retrieval-based post-processing. This preserves the original model predictions and makes it possible to assess the contribution of translation-memory retrieval independently of the underlying model.

The implemented workflow includes:

1. Generate translations with the **base Llama 3.1 8B Instruct model**.
2. Generate translations with the **fine-tuned model using the QLoRA adapter**.
3. Query the translation-memory vector database for relevant Spanish source segments.
4. Produce hybrid predictions by combining model-generated translations with retrieved translation-memory evidence.
5. Save base, adapter and hybrid outputs in separate files for subsequent comparison and evaluation.
6. Evaluate the resulting predictions using COMET and qualitative inspection.

The hybrid approach is being investigated as a practical translation-memory augmentation strategy. The model generates a translation, while retrieval supplies potentially relevant bilingual evidence that can be used to support or revise the output.

The current implementation uses a weighted retrieval configuration:

* **Fuzzy retrieval weight:** 0.4
* **Semantic retrieval weight:** 0.6

These weights are experimental and are not yet established as optimal. Further testing will investigate whether increasing the fuzzy retrieval contribution produces better translation quality, particularly for repeated or closely related source segments.

### Initial COMET evaluation

The first comparison evaluates the base model against the fine-tuned adapter model, followed by a comparison between the adapter-only and adapter hybrid configurations.

| Configuration                         | COMET score |
| ------------------------------------- | ----------: |
| Base model                            |    0.849007 |
| Fine-tuned adapter                    |    0.869157 |
| Fine-tuned adapter + hybrid retrieval |    0.867365 |

The adapter model achieved a COMET score of **0.869157**, compared with **0.849007** for the base model, representing an absolute improvement of **+0.020151**.

With the current hybrid retrieval configuration, the adapter hybrid model achieved a COMET score of **0.867365**, which is **0.001793 lower** than the adapter-only score.

These initial results indicate that fine-tuning improved the measured translation quality on the evaluated dataset, whereas the current hybrid retrieval configuration did not provide an additional COMET improvement over the fine-tuned model alone. The small difference between the adapter and adapter hybrid scores requires further investigation rather than being interpreted as evidence that retrieval cannot improve the model.

The results are exploratory and should be interpreted in the context of the evaluation dataset, retrieval configuration, prompt design and hybrid post-processing strategy.

### Retrieval weighting and dynamic translation memory

A further research direction is to investigate whether the relative contribution of fuzzy and semantic retrieval affects hybrid translation quality.

The current configuration assigns a weight of 0.4 to fuzzy retrieval and 0.6 to semantic retrieval. Subsequent experiments will test alternative weight combinations to determine whether stronger fuzzy matching can improve the selection of translation-memory evidence.

One hypothesis is that **dynamic translation-memory retrieval may be particularly beneficial when translating single documents or groups of related documents**. Repeated terminology, recurring segments and document-specific translation patterns could potentially make closely matching translation-memory entries more valuable than semantically similar but less contextually specific results.

This hypothesis remains to be tested. The investigation will consider whether dynamically updated translation memories, document-level context and retrieval weighting can improve consistency and translation quality over time.

Potential future experiments include:

* Testing higher fuzzy retrieval weights.
* Comparing semantic-only, fuzzy-only and hybrid retrieval configurations.
* Investigating document-level or related-document translation memories.
* Assessing whether dynamic TMX updates improve terminology consistency and translation quality.
* Examining retrieval relevance alongside final translation quality.
* Comparing performance across repeated, similar and previously unseen source segments.

### Evaluation roadmap

The next evaluation stage will compare the following configurations where the required predictions are available:

* Base model predictions.
* Base model + semantic retrieval hybrid predictions.
* Fine-tuned adapter predictions.
* Fine-tuned adapter + semantic retrieval hybrid predictions.
* Alternative fuzzy and weighted-retrieval configurations.

The comparison will consider translation quality, retrieval relevance, consistency with reference translations and possible regressions introduced by retrieval.

Relevant retrieval metrics may include recall@k, precision@k and MRR, while translation-quality evaluation may include COMET, BERTScore, BLEU and targeted qualitative error analysis.

The project will retain separate prediction and evaluation files so that model generation, retrieval behaviour and hybrid post-processing can be investigated independently rather than treated as a single opaque pipeline.

---

## Current Implementation Snapshot

[svg](https://github.com/LinguistDeveloper/es-en-translation-rag#current-implementation-snapshot)

The latest development work has moved the project beyond a retrieval placeholder into an initial vector-database and hybrid-prediction workflow, supported by comparative COMET evaluation.

Current capabilities include:

* A translation-memory vector database built from the reserved retrieval set.
* Semantic retrieval for Spanish source segments.
* Separate prediction generation for the base Llama 3.1 8B Instruct model and the QLoRA-adapted model.
* Hybrid prediction generation combining model outputs with translation-memory retrieval.
* Separate prediction files for base, adapter and hybrid configurations.
* Comparative COMET evaluation of the base and adapter models.
* Comparative COMET evaluation of the adapter-only and adapter hybrid configurations.
* Experimental fuzzy and semantic retrieval weighting.
* A workflow that preserves intermediate predictions so that retrieval behaviour, model generation and post-processing can be evaluated independently.

### Current findings

The initial COMET results show a measurable improvement from fine-tuning over the base model on the evaluated dataset. However, the first adapter hybrid configuration, using fuzzy and semantic retrieval weights of 0.4 and 0.6 respectively, produced a slightly lower COMET score than the adapter-only configuration.

This provides a basis for further investigation into retrieval selection, weighting, prompt design and document-level translation-memory strategies.

The current findings are preliminary. Additional experiments are needed to establish whether retrieval can provide consistent improvements, under which conditions it is most beneficial, and whether dynamic translation-memory updates can improve performance for single or related documents over time.
