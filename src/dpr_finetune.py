
from __future__ import annotations

from typing import Literal, Sequence

import pandas as pd

Verbosity = Literal["title", "description", "narrative", "all"]


def _qrels_positive(qrels: pd.DataFrame, min_label: int) -> pd.DataFrame:
    q = qrels.copy()
    if "label" not in q.columns:
        raise ValueError("Expected qrels with a 'label' column")
    return q[q["label"] >= min_label][["qid", "docno", "label"]]


def _load_doc_text(dataset, docnos: set[str]) -> dict[str, str]:
    """Map docno -> title + abstract for documents we need for training."""
    need = set(docnos)
    out: dict[str, str] = {}
    for doc in dataset.get_corpus_iter():
        dno = doc.get("docno")
        if dno not in need:
            continue
        title = str(doc.get("title", "") or "")
        abstract = str(doc.get("abstract", "") or "")
        out[dno] = (title + " " + abstract).strip()
        if len(out) == len(need):
            break
    missing = need - set(out.keys())
    if missing:
        ex = next(iter(missing))
        raise RuntimeError(f"Missing {len(missing)} corpus docs (e.g. {ex})")
    return out


def build_training_pairs(
    dataset,
    df_all_topics: pd.DataFrame,
    qrels: pd.DataFrame,
    *,
    verbosity: Verbosity = "all",
    min_label: int = 1,
) -> list[tuple[str, str]]:
    """
    Build (query_text, doc_text) pairs for contrastive fine-tuning.

    """
    pos = _qrels_positive(qrels, min_label)
    if pos.empty:
        raise ValueError("No positive qrels for the given min_label")

    docnos = set(pos["docno"].astype(str))
    doc_text = _load_doc_text(dataset, docnos)

    qid_to_fields: dict[str, dict[str, str]] = {}
    for row in df_all_topics.itertuples(index=False):
        qid = str(row.qid)
        qid_to_fields[qid] = {
            "title": str(row.title or "").strip(),
            "description": str(row.description or "").strip(),
            "narrative": str(row.narrative or "").strip(),
        }

    pairs: list[tuple[str, str]] = []
    for row in pos.itertuples(index=False):
        qid = str(row.qid)
        dno = str(row.docno)
        fields = qid_to_fields.get(qid)
        if not fields:
            continue
        dtxt = doc_text.get(dno, "")
        if not dtxt:
            continue

        if verbosity == "all":
            for key in ("title", "description", "narrative"):
                qtxt = fields[key]
                if qtxt:
                    pairs.append((qtxt, dtxt))
        else:
            qtxt = fields.get(verbosity, "")
            if qtxt:
                pairs.append((qtxt, dtxt))

    if not pairs:
        raise ValueError(
            "No training pairs produced"
        )
    return pairs


def fit_sentence_transformer(
    pairs: Sequence[tuple[str, str]],
    base_model: str,
    output_path: str,
    *,
    epochs: int = 2,
    batch_size: int = 16,
    warmup_ratio: float = 0.1,
) -> str:
    """
    Fine-tune using sentence-transformers (MultipleNegativesRankingLoss).

    """
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError as e:
        raise ImportError(
            "Install sentence-transformers and torch to run fine-tuning."
        ) from e

    model = SentenceTransformer(base_model)
    examples = [InputExample(texts=[q, d]) for q, d in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)
    steps_per_epoch = max(1, len(loader))
    warmup_steps = int(steps_per_epoch * epochs * warmup_ratio)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
    )
    return output_path


def _smoke_test_training() -> None:
    """1–2 step fine-tune on toy pairs; writes a temp model dir."""
    import tempfile

    tiny_pairs = [
        ("covid-19 treatment trials", "randomized trial of remdesivir abstract"),
        ("coronavirus vaccine efficacy", "mrna vaccine phase 3 results abstract"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out = fit_sentence_transformer(
            tiny_pairs,
            "sentence-transformers/all-MiniLM-L6-v2",
            tmp,
            epochs=1,
            batch_size=2,
            warmup_ratio=0.0,
        )
        assert out == tmp
    print("smoke_test_training: OK (model.fit finished, temp dir was written)")


def cord19_corpus_adapter(irds):
    """Wrap ir_datasets cord19 so ``build_training_pairs`` can call ``get_corpus_iter``."""

    class _Adapter:
        def __init__(self, data):
            self._data = data

        def get_corpus_iter(self):
            for doc in self._data.docs_iter():
                yield {
                    "docno": doc.doc_id,
                    "title": doc.title or "",
                    "abstract": doc.abstract or "",
                }

    return _Adapter(irds)


def topics_df_from_irds(irds) -> pd.DataFrame:
    rows = []
    for q in irds.queries_iter():
        rows.append(
            {
                "qid": str(q.query_id).strip(),
                "title": q.title or "",
                "description": q.description or "",
                "narrative": q.narrative or "",
            }
        )
    return pd.DataFrame(rows)


def qrels_df_from_irds(irds) -> pd.DataFrame:
    rows = []
    for r in irds.qrels_iter():
        rows.append(
            {
                "qid": str(r.query_id).strip(),
                "docno": str(r.doc_id).strip(),
                "label": int(r.relevance),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description="DPR fine-tune helpers for TREC-COVID")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny training step (needs sentence-transformers + torch)",
    )
    p.add_argument(
        "--pairs-smoke",
        action="store_true",
        help=(
            "Load cord19/trec-covid via ir_datasets, build a few training pairs "
            "(scans corpus for qrels docnos)"
        ),
    )
    p.add_argument(
        "--train",
        action="store_true",
        help="Full run: build pairs from qrels + corpus, then fine-tune MiniLM",
    )
    p.add_argument(
        "--output",
        type=str,
        default="../models/dpr_covid_minilm_all",
        help="Directory to save the fine-tuned model (default: ../models/...)",
    )
    p.add_argument(
        "--verbosity",
        type=str,
        choices=["title", "description", "narrative", "all"],
        default="all",
        help="Which topic field(s) to use when building (query, doc) pairs",
    )
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--base-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    p.add_argument("--min-label", type=int, default=1)
    p.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="After building pairs, shuffle and keep at most this many (for faster runs)",
    )
    args = p.parse_args()
    if args.smoke:
        _smoke_test_training()
    elif args.pairs_smoke:
        import ir_datasets

        irds = ir_datasets.load("cord19/trec-covid")
        qrels = qrels_df_from_irds(irds)
        qrels_pos = qrels[qrels["label"] >= 1].head(8)
        df_all = topics_df_from_irds(irds)
        adapter = cord19_corpus_adapter(irds)
        pairs = build_training_pairs(
            adapter, df_all, qrels_pos, verbosity="title", min_label=1
        )
        print(f"pairs-smoke: OK, built {len(pairs)} training pairs (subset)")
        if pairs:
            print("  example:", pairs[0][0][:60], "...", pairs[0][1][:60], "...")
    elif args.train:
        import random

        import ir_datasets

        irds = ir_datasets.load("cord19/trec-covid")
        df_all = topics_df_from_irds(irds)
        qrels = qrels_df_from_irds(irds)
        adapter = cord19_corpus_adapter(irds)
        print("Building training pairs (corpus scan + qrels)...", flush=True)
        pairs = build_training_pairs(
            adapter,
            df_all,
            qrels,
            verbosity=args.verbosity,  # type: ignore[arg-type]
            min_label=args.min_label,
        )
        if args.max_pairs is not None and len(pairs) > args.max_pairs:
            rng = random.Random(0)
            order = list(range(len(pairs)))
            rng.shuffle(order)
            pairs = [pairs[i] for i in order[: args.max_pairs]]
        print(f"Built {len(pairs)} pairs. Starting fine-tuning...", flush=True)
        out = Path(args.output)
        if not out.is_absolute():
            out = (Path(__file__).resolve().parent / out).resolve()
        out.mkdir(parents=True, exist_ok=True)
        fit_sentence_transformer(
            pairs,
            args.base_model,
            str(out),
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        print("Fine-tune finished:", out, flush=True)
    else:
        p.print_help()
