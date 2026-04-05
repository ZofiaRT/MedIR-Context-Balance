## Dense retrieval: `dpr_dense.py` and `dpr_finetune.py`

Baseline **DPR (MiniLM)** is created with **`make_dpr_pipeline`** in `dpr_dense.py` (see the setup cells earlier in this notebook): it builds or loads a **pyterrier-dr FlexIndex** over **title + abstract**, same text as BM25. Fine-tuning is **not** run inside the notebook here; use the **script** instead (below).

### `dpr_dense.py` (indexing + retrieval)

| Piece | Role |
|-------|------|
| **`iter_flex_docs(dataset)`** | Streams `{"docno", "text"}` for the indexer (pyterrier-dr expects **`docno`**). |
| **`make_dpr_pipeline(...)`** | If `pt_meta.json` is missing, runs `doc_encoder >> FlexIndex.indexer(overwrite)` over the corpus, then returns **`query_encoder >> retriever`** plus a small transform so rankings expose **`docno`** for `pt.Experiment`. |
| **`retriever_for_loaded_encoder(enc, index_dir)`** | After you swap in another `SBertBiEncoder` checkpoint, load the matching on-disk index and get the same retrieval transformer. |
| **`make_biodpr_pipeline`** | Same pattern for a **BioMed** encoder when a pre-built FlexIndex exists. |

### `dpr_finetune.py` (supervision from qrels)

| Piece | Role |
|-------|------|
| **`build_training_pairs(dataset, df_all_topics, qrels, verbosity=..., min_label=1)`** | For each positive qrel, pairs **topic text** (title / description / narrative, or **all** non-empty fields) with the **positive document** text. Used for contrastive training. |
| **`fit_sentence_transformer(pairs, base_model, output_dir, ...)`** | **MultipleNegativesRankingLoss** fine-tuning via sentence-transformers. |
| **CLI** | From `src/`: `python dpr_finetune.py --pairs-smoke` (tiny sanity check), `python dpr_finetune.py --train --verbosity all` (full run; use **GPU** if possible). After training, build a **new** FlexIndex with that checkpoint (same `iter_flex_docs` as BM25) and plug it in with `retriever_for_loaded_encoder`. |

### Fine-tuning protocol (verbosity)

- **Unified:** `verbosity="all"` → more (query, doc+) pairs per judgment; one model; evaluate on Title / Description / Narrative queries as usual.  
- **Specialist:** `verbosity="title"` (or `description` / `narrative`) → separate checkpoints; each needs its **own** saved model directory **and** dense index because query and document towers share weights.