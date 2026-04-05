# How Much Context is Enough? Investigating the Balance Between Lexical and Semantic Medical Retrieval

**Course:** DSAIT4050-Q3-26: Information Retrieval, TU Delft  
**Team (Group 7):** Nehir Altınkaya, Zofia Rogacka-Trojak, Varuni Sood, and Natalie Mladenova

## Project Overview
In the medical domain, information retrieval operates under extremely high stakes where failures can directly impact 
clinical decisions. Real-world medical queries vary wildly from short exact keywords to lengthy, nuanced symptom descriptions. 

This project is an empirical comparative evaluation of Information Retrieval (IR) methods designed to find the optimal
balance between exact lexical matching and semantic intent. We study the **TREC-COVID** dataset over the CORD-19
collection, evaluating query performance across three distinct levels of verbosity: **Title (Short)**, **Description
(Medium)**, and **Narrative (Long)**.

## Models Evaluated
To understand the impact of medical jargon and underlying user intents, we systematically compare:
* **BM25:** A lexical baseline testing exact keyword matching.
* **BM25 + RM3:** Testing if query expansion (pseudo-relevance feedback) can bridge the vocabulary gap.
* **Standard Dense Retrieval (DPR):** Using a general semantic bi-encoder (MiniLM).
* **BioDPR:** A specialized, domain-adapted model (PubMedBERT) to see how medical jargon influences retrieval.
* **Hybrid (BM25 + BioDPR):** Combining lexical and semantic approaches using Reciprocal Rank Fusion (RRF).

Performance is evaluated using **nDCG@10** (to ensure the most highly relevant documents appear at the top) and **MAP**
(to assess overall ranking quality).

## Repository Structure
```text
ir_project/
├── .gitignore
├── requirements.txt            # Python dependencies for the project
├── src/
│   ├── data_exploration.ipynb  # Data analysis, query verbosity prep, and PyTerrier indexing 
│   ├── experiment.ipynb        # Main evaluation notebook comparing all IR models
│   ├── dpr_dense.py            # Dense retrieval pipeline setup (PyTerrier-DR FlexIndex)
│   └── dpr_finetune.py         # Script for fine-tuning dense models on TREC-COVID
```

## Setup and Installation
1. Clone the repository.
2. Install the required dependencies from the `requirements.txt` file. *(Note: This project relies heavily on `python-terrier`, `pyterrier-dr`, `sentence-transformers`, and Java for PyTerrier's backend)*.
```bash
pip install -r requirements.txt
```

## How to Run
TODO

## Key Findings
* **Verbosity Matters:** Providing more query context does not always improve performance. Description queries (medium length) consistently outperformed both Title (short) and Narrative (long) queries across models.
* **Lexical Baselines are Strong:** BM25 remains a powerful and interpretable baseline, excelling on short, exact-match queries.
* **The Hybrid Advantage:** The Hybrid model (BM25 + BioDPR) achieved the highest overall performance on Description queries (nDCG@10: 0.7791), successfully balancing exact constraints with semantic understanding.
* **Domain Adaptation is Necessary:** General dense models (DPR) underperformed significantly in this specialized domain, whereas the medical-specific BioDPR model improved semantic retrieval.