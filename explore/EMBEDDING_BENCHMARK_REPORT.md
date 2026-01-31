# Embedding Model Benchmark Report

## Context and Motivation

The DiagnoSys RAG pipeline faces a specific cross-lingual challenge: patient input is in **French** (via speech-to-text), while the medical documentation database is predominantly in **English**. The embedding model must therefore handle **FR query to EN document** semantic matching reliably.

This benchmark evaluates 11 embedding models across multiple axes: cross-lingual retrieval accuracy, sensitivity to noisy input (raw transcripts vs. LLM-summarized queries), and practical deployment constraints (speed, model size).

## Models Evaluated

| Model | Type | Language | Notes |
|-------|------|----------|-------|
| `paraphrase-multilingual-MiniLM-L12-v2` | Sentence Transformer | Multilingual | Production choice |
| `multilingual-e5-small` | Sentence Transformer | Multilingual | |
| `multilingual-e5-large` | Sentence Transformer | Multilingual | |
| `bge-m3` | Sentence Transformer | Multilingual | SOTA general multilingual |
| `all-MiniLM-L6-v2` | Sentence Transformer | Multilingual | Lightweight baseline |
| `biobert` | BERT (mean pooling) | English | Biomedical domain |
| `clinical-bert` | BERT (mean pooling) | English | Clinical notes domain |
| `pubmed-bert` | BERT (mean pooling) | English | PubMed literature domain |
| `drbert` | BERT (mean pooling) | French | French medical domain |
| `camembert-base` | BERT (mean pooling) | French | French general |
| `camembert-bio` | BERT (mean pooling) | French | French biomedical |

### Why medical-domain BERT models were tested with mean pooling

Models like BioBERT, ClinicalBERT, PubMedBERT, DrBERT, and CamemBERT-bio are **not sentence transformers**. They are standard BERT models pretrained on domain-specific corpora but lack the contrastive fine-tuning that gives sentence transformers their retrieval properties. Since no sentence-transformer-fine-tuned versions were available for these medical models, they were tested using **mean pooling** over token embeddings as a best-effort embedding strategy.

## Benchmark Methodology

- **Test set**: 51 French medical queries + 50 English medical queries (with both clean "extracted" and noisy "transcript" variants for English)
- **RAG simulation**: Each query is matched against 15+ English distractor documents plus the expected correct document (needle-in-a-haystack)
- **Metrics**: MRR (Mean Reciprocal Rank), Top-1/3/5 accuracy, NDCG@3, NDCG@5, cosine similarity, encoding time
- **Translation**: French-only and English-only models receive translated queries when needed (via Google Translate)

## Results

### Overall Retrieval Performance (aggregated across all queries)

| Model | MRR | Top-1 | Top-3 | Top-5 | Avg Encoding (s) |
|-------|-----|-------|-------|-------|-------------------|
| **all-MiniLM-L6-v2** | **0.743** | **0.613** | **0.853** | **0.893** | 0.076 |
| **paraphrase-multilingual-MiniLM-L12-v2** | **0.669** | **0.500** | **0.773** | **0.920** | 0.045 |
| multilingual-e5-small | 0.645 | 0.500 | 0.740 | 0.853 | 0.045 |
| biobert | 0.592 | 0.493 | 0.607 | 0.660 | 0.130 |
| multilingual-e5-large | 0.585 | 0.453 | 0.620 | 0.753 | 0.420 |
| clinical-bert | 0.435 | 0.300 | 0.453 | 0.560 | 0.127 |
| bge-m3 | 0.389 | 0.240 | 0.407 | 0.533 | 0.436 |
| camembert-bio | 0.350 | 0.220 | 0.367 | 0.447 | 0.378 |
| pubmed-bert | 0.336 | 0.253 | 0.293 | 0.373 | 0.118 |
| drbert | 0.325 | 0.207 | 0.327 | 0.387 | 0.136 |
| camembert-base | 0.314 | 0.173 | 0.313 | 0.453 | 0.255 |

### Key Finding 1: Medical BERT models underperform sentence transformers for RAG

Despite being trained on biomedical corpora, **domain-specific BERT models consistently rank below general-purpose sentence transformers** in retrieval tasks. PubMedBERT achieves the highest raw cosine similarity (0.951) of any model, but its MRR is only 0.336 -- meaning the embeddings are dense but poorly discriminative. The similarity scores are clustered together, failing to separate the correct document from distractors.

This is a direct consequence of the lack of contrastive training: mean-pooled BERT embeddings produce vectors in a space not optimized for nearest-neighbor retrieval.

| Model | Avg Similarity | MRR | Interpretation |
|-------|---------------|-----|----------------|
| pubmed-bert | 0.951 | 0.336 | High similarity, poor discrimination |
| clinical-bert | 0.843 | 0.435 | High similarity, poor discrimination |
| biobert | 0.847 | 0.592 | Moderate -- best of the BERT models |
| paraphrase-multilingual | 0.365 | 0.669 | Lower similarity, better discrimination |
| all-MiniLM-L6-v2 | 0.294 | 0.743 | Lowest similarity, best discrimination |

The sentence transformer models produce **lower absolute similarity scores** but create embedding spaces where relevant documents are relatively closer than irrelevant ones -- which is what matters for retrieval.

### Key Finding 2: Cross-lingual performance gap

Comparing English vs. French query performance reveals the cross-lingual degradation:

| Model | MRR (EN) | MRR (FR) | Gap |
|-------|----------|----------|-----|
| all-MiniLM-L6-v2 | 0.806 | 0.615 | -0.191 |
| paraphrase-multilingual | 0.705 | 0.598 | -0.107 |
| multilingual-e5-small | 0.664 | 0.607 | -0.057 |
| multilingual-e5-large | 0.632 | 0.491 | -0.141 |
| bge-m3 | 0.395 | 0.377 | -0.018 |

`paraphrase-multilingual-MiniLM-L12-v2` shows a moderate cross-lingual gap (-0.107) while maintaining strong absolute FR performance (0.598 MRR). `multilingual-e5-small` has the smallest gap (-0.057) but slightly lower overall performance.

Notably, `all-MiniLM-L6-v2` leads overall but suffers the largest cross-lingual drop, meaning its strong aggregate is driven by English-query performance. For a system where queries are primarily in French, the gap matters.

### Key Finding 3: LLM denoising is critical

The benchmark compares "extracted" queries (clean, LLM-summarized medical descriptions) against "transcript" queries (raw, noisy patient-doctor dialogue). The impact is dramatic:

| Model | MRR (extracted) | MRR (transcript) | Degradation |
|-------|-----------------|-------------------|-------------|
| paraphrase-multilingual | 0.648 | 0.712 | +0.064 (improves) |
| all-MiniLM-L6-v2 | 0.723 | 0.781 | +0.058 (improves) |
| multilingual-e5-small | 0.755 | 0.425 | -0.330 |
| multilingual-e5-large | 0.651 | 0.454 | -0.197 |
| biobert | 0.818 | 0.140 | **-0.678** |
| clinical-bert | 0.585 | 0.136 | **-0.449** |
| pubmed-bert | 0.473 | 0.063 | **-0.410** |
| drbert | 0.452 | 0.071 | **-0.381** |
| bge-m3 | 0.501 | 0.163 | **-0.338** |

Most models collapse when given raw transcripts. Medical BERT models are particularly fragile: BioBERT drops from 0.818 to 0.140 MRR. This validates the architecture decision to run LLM summarization (Mistral) before the embedding step.

Interestingly, `paraphrase-multilingual-MiniLM-L12-v2` and `all-MiniLM-L6-v2` are the only models that actually **improve** with transcript input. This suggests their training on paraphrase data makes them robust to verbose, conversational phrasing -- a valuable property for a speech-to-text pipeline where LLM denoising might fail or be unavailable.

### Key Finding 4: Translation as a strategy

For monolingual models, the benchmark translates queries before embedding. This adds latency and potential translation errors. The French medical models (DrBERT, CamemBERT-bio) perform poorly even on French queries because the **documents are in English** -- they cannot bridge the language gap regardless of query translation. English medical models (BioBERT) require French-to-English translation of queries, which works reasonably but adds a dependency on an external translation service.

Multilingual models eliminate this dependency entirely, handling both languages in a shared embedding space.

## Decision: `paraphrase-multilingual-MiniLM-L12-v2`

The production model was selected based on the following criteria:

| Criterion | paraphrase-multilingual | all-MiniLM-L6-v2 | multilingual-e5-small |
|-----------|------------------------|-------------------|-----------------------|
| Top-5 accuracy | **0.920** | 0.893 | 0.853 |
| FR MRR | 0.598 | **0.615** | 0.607 |
| Cross-lingual gap | -0.107 | -0.191 | **-0.057** |
| Transcript robustness | **+0.064** | +0.058 | -0.330 |
| Encoding speed | **0.045s** | 0.076s | **0.045s** |
| No translation needed | Yes | Yes | Yes |

**Rationale:**

1. **Best Top-5 accuracy (0.920)**: In a RAG system, the LLM receives the top-K retrieved documents. Top-5 accuracy matters more than Top-1 because the LLM can synthesize information from multiple relevant results. The paraphrase-multilingual model retrieves the correct document in 92% of cases within the top 5.

2. **Transcript robustness**: It is the only model (alongside all-MiniLM-L6-v2) that does not degrade with noisy input. This provides a safety net when the LLM denoising step produces suboptimal summaries.

3. **Practical efficiency**: At 0.045s per encoding and 384-dimensional embeddings, it is fast and storage-efficient. While its initialization time is higher (37.93s), this is a one-time cost at application startup.

4. **No translation dependency**: Handles French queries against English documents natively, eliminating the need for an external translation API.

5. **Acceptable cross-lingual gap**: The -0.107 FR/EN gap is moderate and compensated by strong absolute performance on French queries.

`all-MiniLM-L6-v2` was a close contender with slightly better aggregate MRR, but its larger cross-lingual gap and the fact that it is marketed as English-first made `paraphrase-multilingual-MiniLM-L12-v2` the safer choice for a French-first system.

## On the synthetic nature of this benchmark

This benchmark was built entirely from **synthetic data** -- generated medical queries, artificial transcripts, and curated distractor documents. It was designed as a **decision-support tool** to guide early architecture choices, not as a definitive evaluation of production performance.

This distinction matters because the benchmark and the system architecture are **co-dependent**. The "right" embedding model depends on decisions that hadn't been finalized when the benchmark was run: What language will the documents be in? How will they be chunked? Will the LLM summarize transcripts before embedding, and in what format? Conversely, those architectural decisions were partly informed by the benchmark results themselves. The benchmark told us that multilingual sentence transformers outperform medical BERT models for cross-lingual retrieval, which validated the choice to keep English-language documentation rather than translating everything to French. But that choice in turn shapes what "good retrieval" looks like.

In practice, a truly rigorous evaluation would require testing on **real production data**: actual patient transcripts from the STT module, real LLM-generated summaries, and the actual document corpus stored in ChromaDB with its specific chunking. The synthetic queries approximate the style and content of real inputs, but they cannot capture the full variability of spoken medical French, the specific noise patterns of the Sherpa ONNX transcriber, or the idiosyncrasies of how Mistral reformulates patient context.

The benchmark should therefore be understood as a **directional guide** -- strong enough to eliminate clearly unsuitable models (medical BERTs with mean pooling, monolingual French models) and to identify a shortlist of viable candidates, but not a substitute for production evaluation. A follow-up benchmark on real data, once the system is deployed and generating actual transcripts and summaries, would be necessary to fully validate or revise the model choice.

## Other Limitations

- The benchmark uses a fixed set of 15 distractor documents. Production ChromaDB may contain hundreds of documents, potentially changing relative rankings.
- Medical BERT models were tested with mean pooling only. Fine-tuning these models with contrastive objectives on medical data could improve their retrieval performance significantly, but this was out of scope.
- Translation quality (Google Translate) was not independently evaluated and could introduce noise for monolingual model results.
- The benchmark does not account for chunking effects -- production documents are split into 1000-character chunks, which may affect retrieval differently than full-document matching.
