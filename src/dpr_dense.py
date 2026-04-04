from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyterrier as pt


def iter_flex_docs(dataset: Any) -> Iterator[dict[str, str]]:
    """
    Yield docno/text records for FlexIndex (title + abstract; same as BM25).

    FlexIndexer expects the key ``docno`` (not ``docid``) after encoding.
    """
    for doc in dataset.get_corpus_iter():
        doc_key = doc.get("docno") or doc.get("docid")
        if not doc_key:
            continue
        title = str(doc.get("title", "") or "")
        abstract = str(doc.get("abstract", "") or "")
        body = doc.get("text")
        if body is not None and str(body).strip():
            text = str(body).strip()
        else:
            text = (title + " " + abstract).strip()
        if not text:
            continue
        yield {"docno": str(doc_key), "text": text}


def _ensure_docno(res):
    if res is None or len(res) == 0:
        return res
    if "docno" not in res.columns and "docid" in res.columns:
        res = res.copy()
        res["docno"] = res["docid"].astype(str)
    return res


def retriever_for_loaded_encoder(
    encoder: Any,
    index_dir: str | Path,
):
    """
    Retrieval pipeline from an ``SBertBiEncoder`` instance and an on-disk FlexIndex
    (e.g. after fine-tuning and ``iter_flex_docs`` indexing).
    """
    import pyterrier_dr as ptdr

    index_path = Path(index_dir).resolve()
    flex = ptdr.FlexIndex(str(index_path))
    retr = encoder.query_encoder() >> flex.retriever()
    return retr >> pt.apply.generic(_ensure_docno)


def make_dpr_pipeline(
    dataset: Any,
    index_dir: str | Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    *,
    rebuild: bool = False,
):
    """Return a pt.Experiment retriever; build FlexIndex if ``pt_meta.json`` is absent."""
    import pyterrier_dr as ptdr

    index_path = Path(index_dir).resolve()
    meta = index_path / "pt_meta.json"

    if rebuild and index_path.exists():
        shutil.rmtree(index_path)

    if not meta.exists():
        if index_path.exists():
            shutil.rmtree(index_path)
        index_path.mkdir(parents=True, exist_ok=True)
        encoder = ptdr.SBertBiEncoder(model_name)
        flex = ptdr.FlexIndex(str(index_path))
        indexer = encoder.doc_encoder() >> flex.indexer(mode="overwrite")
        msg = (
            "Building dense FlexIndex (first run is slow on CPU; use GPU if available)."
        )
        print(msg, flush=True)
        indexer.index(iter_flex_docs(dataset))
        print("Dense index build finished.", flush=True)
        del encoder, flex, indexer

    encoder = ptdr.SBertBiEncoder(model_name)
    return retriever_for_loaded_encoder(encoder, index_path)


def make_biodpr_pipeline(
    index_dir: str | Path,
    model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
):
    """Return BioDPR retriever when the FlexIndex exists, else None."""
    import pyterrier_dr as ptdr

    index_path = Path(index_dir).resolve()
    if not (index_path / "pt_meta.json").exists():
        return None
    encoder = ptdr.SBertBiEncoder(model_name)
    return retriever_for_loaded_encoder(encoder, index_path)
