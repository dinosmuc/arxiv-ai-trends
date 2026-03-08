# Implementation Plan — ArXiv AI Trends Project

This document is a step-by-step guide for completing the project. Each phase builds on the previous one. Do them in order. Each step includes exactly what to do, what files to touch, and what the expected outcome is.

**Current project state:**
- 5 notebooks: `01_data_collection`, `02_exploratory_analysis`, `03_preprocessing`, `04_dimensionality_reduction`, `05_clustering_and_results`
- `src/` has `collect.py` and `label.py` (working), plus `preprocess.py`, `features.py`, `visualize.py` (all empty)
- 181,294 papers, 3 embeddings (TF-IDF, MiniLM, KaLM), UMAP 20d for clustering, 2d/3d for viz
- KaLM + HDBSCAN selected as primary (41 clusters, 37% noise), KaLM + K-Means as secondary (24 clusters)
- No external validation, no noise analysis, naive growth calculation, empty README, unpinned requirements

**Target final notebook structure:**
```
notebooks/
  01_data_collection.ipynb
  02_exploratory_analysis.ipynb
  03_preprocessing.ipynb
  04_dimensionality_reduction.ipynb
  05_clustering.ipynb
  06_validation.ipynb
  07_trends_and_insights.ipynb
  archive/
    05_clustering_and_results.ipynb   (original monolith, kept for reference)
```

---

## How to Handle Existing Notebooks

Before starting any phase, read this section to understand the overall file management strategy.

### Notebooks 01-04: Modify in place, rerun

These are the data pipeline. They don't need splitting, just modifications:

- **01_data_collection.ipynb** — No changes needed. Don't touch it.
- **02_exploratory_analysis.ipynb** — No code changes. Only markdown rewrite in the final phase. Rerun at the end to regenerate plots with clean outputs.
- **03_preprocessing.ipynb** — Replace inline code with `src.preprocess` imports, add `embed_text` column and `is_survey` flag. Rerun to regenerate `arxiv_clean.csv` and re-embed with title+abstract.
- **04_dimensionality_reduction.ipynb** — Replace inline code with `src.features.reduce_umap` imports, add 30d/40d + n_neighbors sweep. Rerun.

### Notebook 05: Archive the old one, create new ones from it

This is the only notebook that needs splitting. It's a monolith doing 5 jobs (clustering, evaluation, labeling, trends, recommendations). Surgically removing sections is messier than starting clean.

1. **Create** `notebooks/archive/` directory
2. **Move** `05_clustering_and_results.ipynb` into `notebooks/archive/` (keep it for reference — you'll copy cells from it)
3. **Create fresh** `05_clustering.ipynb` — copy over the clustering + evaluation cells from the old 05, then add BERTopic, topic coherence, and the dendrogram. Use `src/` imports from the start.
4. **Create fresh** `06_validation.ipynb` — entirely new content (noise analysis, ARI/NMI, cross-embedding agreement). Use `src/` imports from the start.
5. **Create fresh** `07_trends_and_insights.ipynb` — copy the trend/recommendation cells from the old 05, then rework them with regression, sensitivity analysis, seasonality normalization. Use `src/` imports from the start.

### Why src/ modules are built FIRST (Phase 0)

The function signatures are already known — `filter_papers()`, `clean_abstracts()`, `reduce_umap()`, `get_cluster_top_terms()`, `compute_coherence()`, etc. They won't change. By building `src/` modules first, every notebook uses shared imports from day one. This eliminates the risk of refactoring inline code later and silently introducing differences (a changed default parameter, a different sort order, a missing filter).

If you build notebooks with inline code first and refactor later, you'd have to do an expensive full rerun (~5+ hours) just to verify the refactored imports produce identical output. Building `src/` first avoids that entirely.

### Execution order summary

```
1. Fill src/ modules (preprocess, features, visualize)    (~30 min, no rerun)
2. Pin requirements, Makefile, README skeleton             (~15 min, no rerun)
3. Modify 03 with src/ imports + new features → rerun     (~2.5 hours)
4. Modify 04 with src/ imports + new features → rerun     (~2-3 hours)
   ⚠ CHECKPOINT: Quick-test HDBSCAN on new KaLM UMAP.
     Compare silhouette to old 0.5073. If worse, adjust before continuing.
5. Archive old 05, create new 05 with src/ imports → run  (~1 hour)
6. Create new 06 with src/ imports → run                  (~10 minutes)
7. Create new 07 with src/ imports → run                  (~5 minutes)
8. Rewrite markdown in ALL notebooks (01-07)
9. Final rerun of ALL notebooks                           (ensures outputs match markdown)
```

The checkpoint after step 4 is important — before spending hours on steps 5-7, verify that the new embeddings + UMAP settings actually improved results. If silhouette dropped, investigate before building everything on top of bad foundations.

The final rerun in step 9 ensures all cell outputs in the notebooks match the rewritten markdown narrative. You don't want stale outputs from an earlier run sitting next to new commentary.

---

## Phase 0 — Build src/ Modules + Reproducibility

**Why this is Phase 0:** Building the shared modules first means every notebook uses `src/` imports from day one. No "refactor later and hope it matches" risk. The function signatures are known — only the notebooks that call them will evolve.

### Step 0.1 — Fill in src/preprocess.py

**File:** `src/preprocess.py`

```python
"""Preprocessing pipeline for ArXiv papers."""

import logging

import pandas as pd
import spacy

logger = logging.getLogger(__name__)

TARGET_CATEGORIES = {"cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.RO", "cs.NE", "cs.MA", "cs.IR"}
MIN_ABSTRACT_WORDS = 30
SURVEY_PATTERN = (
    r"\b(survey|review|overview|tutorial|systematic review"
    r"|literature review|comprehensive review|state of the art|sota review)\b"
)


def filter_papers(df, categories=TARGET_CATEGORIES, min_words=MIN_ABSTRACT_WORDS):
    """Filter to target categories and drop short abstracts."""
    df = df[df["primary_category"].isin(categories)].copy()
    df = df[df["abstract"].str.split().str.len() >= min_words].copy()
    df = df.reset_index(drop=True)
    logger.info(f"Filtered to {len(df):,} papers")
    return df


def clean_abstracts(df):
    """Add abstract_clean (whitespace normalized) and embed_text (title+abstract) columns."""
    df["abstract_clean"] = df["abstract"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["embed_text"] = df["title"] + ". " + df["abstract_clean"]
    return df


def flag_surveys(df):
    """Flag survey/review papers."""
    df["is_survey"] = df["title"].str.contains(SURVEY_PATTERN, case=False, na=False)
    logger.info(f"Flagged {df['is_survey'].sum():,} survey papers ({df['is_survey'].mean():.1%})")
    return df


def lemmatize_abstracts(df, batch_size=1000):
    """Add abstract_lemma (lowercased, no stopwords, lemmatized) column."""
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    df["abstract_lemma"] = [
        " ".join(t.lemma_ for t in doc if not t.is_stop and t.is_alpha and len(t) > 2)
        for doc in nlp.pipe(df["abstract_clean"], batch_size=batch_size)
    ]
    df["abstract_lemma"] = df["abstract_lemma"].str.lower()
    return df


def preprocess_pipeline(raw_path, output_path):
    """Full preprocessing pipeline: load, filter, clean, flag, lemmatize, save."""
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df):,} papers from {raw_path}")
    df = filter_papers(df)
    df = clean_abstracts(df)
    df = flag_surveys(df)
    df = lemmatize_abstracts(df)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df):,} papers to {output_path}")
    return df
```

### Step 0.2 — Fill in src/features.py

**File:** `src/features.py`

```python
"""Feature extraction: TF-IDF, embeddings, dimensionality reduction, topic coherence."""

import logging

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def compute_tfidf(texts, max_features=50000, min_df=5, max_df=0.95):
    """Compute TF-IDF matrix from texts."""
    tfidf = TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=max_df)
    matrix = tfidf.fit_transform(texts)
    logger.info(f"TF-IDF shape: {matrix.shape}")
    return matrix, tfidf


def reduce_svd(sparse_matrix, n_components=500, random_state=42):
    """Reduce sparse matrix with TruncatedSVD."""
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    reduced = svd.fit_transform(sparse_matrix)
    logger.info(
        f"SVD: {sparse_matrix.shape[1]} -> {n_components}, "
        f"variance explained: {svd.explained_variance_ratio_.sum():.1%}"
    )
    return reduced, svd


def compute_embeddings(texts, model_name, device="cuda", batch_size=256):
    """Compute sentence embeddings using a SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    logger.info(f"{model_name}: {embeddings.shape}")
    return embeddings


def reduce_umap(data, n_components=20, n_neighbors=15, min_dist=0.0, random_state=42):
    """Reduce with UMAP."""
    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    reduced = reducer.fit_transform(data)
    logger.info(f"UMAP: {data.shape[1]}d -> {n_components}d")
    return reduced


def get_cluster_top_terms(df, labels, text_col="abstract", n_terms=10):
    """Extract top TF-IDF terms per cluster."""
    cluster_docs = {}
    for label in sorted(set(labels)):
        if label == -1:
            continue
        mask = labels == label
        cluster_docs[label] = " ".join(df.loc[mask, text_col].tolist())

    tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(cluster_docs.values())
    terms = tfidf.get_feature_names_out()

    cluster_terms = {}
    for i, label in enumerate(cluster_docs.keys()):
        scores = tfidf_matrix[i].toarray().flatten()
        top_idx = scores.argsort()[-n_terms:][::-1]
        cluster_terms[label] = [(terms[j], scores[j]) for j in top_idx]

    return cluster_terms


def compute_coherence(df, labels, text_col="abstract", n_terms=10):
    """Compute c_v topic coherence for a clustering."""
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel

    texts = [doc.split() for doc in df[text_col]]
    dictionary = Dictionary(texts)

    topics = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        mask = labels == label
        cluster_text = " ".join(df.loc[mask, text_col].tolist())
        tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
        vec = tfidf.fit_transform([cluster_text])
        terms = tfidf.get_feature_names_out()
        scores = vec.toarray().flatten()
        top_idx = scores.argsort()[-n_terms:][::-1]
        topics.append([terms[j] for j in top_idx])

    cm = CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
    )
    return cm.get_coherence()
```

### Step 0.3 — Fill in src/visualize.py

**File:** `src/visualize.py`

```python
"""Reusable visualization functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_cluster_scatter_2d(data_2d, labels, title="", ax=None, cmap="tab20"):
    """Scatter plot of 2D UMAP with cluster colors."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    colors = labels.copy().astype(float)
    if -1 in set(labels):
        colors[labels == -1] = -1
    ax.scatter(data_2d[:, 0], data_2d[:, 1], c=colors, cmap=cmap, s=0.5, alpha=0.3)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_growth_bars(growth_series, title="Research Area Growth Rates"):
    """Horizontal bar chart of cluster growth rates, green=positive, red=negative."""
    fig, ax = plt.subplots(figsize=(12, max(6, len(growth_series) * 0.35)))
    colors = ["#2ecc71" if g > 0 else "#e74c3c" for g in growth_series.values]
    growth_series.sort_values().plot(kind="barh", ax=ax, color=colors)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("Growth %")
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def plot_monthly_trends(monthly_data, columns, ncols=4, title=""):
    """Grid of monthly bar charts for selected clusters."""
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for ax, name in zip(axes, columns):
        data = monthly_data[name]
        data.plot(kind="bar", ax=ax, color="steelblue", width=0.8)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_visible(i % 4 == 0)
    for ax in axes[len(columns):]:
        ax.set_visible(False)
    if title:
        plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_cluster_category_heatmap(df, cluster_col, category_col="primary_category"):
    """Heatmap showing cluster composition by ArXiv category."""
    ct = pd.crosstab(df[cluster_col], df[category_col], normalize="index")
    fig, ax = plt.subplots(figsize=(12, max(8, len(ct) * 0.4)))
    sns.heatmap(ct, cmap="Blues", ax=ax, annot=True, fmt=".2f")
    ax.set_title("Cluster Composition by ArXiv Category")
    plt.tight_layout()
    return fig, ax


def plot_comparison_scatter(comparison_df, x_col, y_col, size_col=None,
                            color_col=None, label_col=None, title=""):
    """Generic scatter/bubble chart for comparing clusters."""
    import plotly.express as px

    fig = px.scatter(
        comparison_df,
        x=x_col,
        y=y_col,
        size=size_col,
        color=color_col,
        color_continuous_scale="RdYlGn",
        hover_name=label_col,
        text=label_col,
        size_max=60,
    )
    fig.update_traces(textposition="top center", textfont_size=8)
    fig.update_layout(title=title, width=1000, height=700)
    return fig
```

### Step 0.4 — Pin requirements.txt

**File:** `requirements.txt`

Replace the current bare package names with pinned versions. Run `pip freeze > requirements_full.txt` in your conda/venv, then manually curate to include only packages you actually use:

```
# Pin exact versions for reproducibility
# Python 3.11 tested
# GPU required for sentence-transformers (CUDA)

anthropic==0.XX.X
arxiv==2.X.X
bertopic==0.16.X
gensim==4.X.X
hdbscan==0.8.X
langdetect==1.0.X
matplotlib==3.X.X
matplotlib-venn==0.11.X
numpy==1.XX.X
pandas==2.X.X
plotly==5.X.X
pydantic==2.X.X
scikit-learn==1.X.X
scipy==1.X.X
seaborn==0.1X.X
sentence-transformers==3.X.X
spacy==3.X.X
umap-learn==0.5.X
```

**Note:** Remove `sqarify` — it's unused. Add `bertopic` and `gensim` (new dependencies).

### Step 0.5 — Add Makefile

**File:** `Makefile` (NEW)

```makefile
.PHONY: setup collect preprocess reduce cluster validate trends all clean

setup:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

collect:
	python -c "from src.collect import collect_papers; df = collect_papers(); df.to_csv('data/raw/arxiv_papers.csv', index=False)"

preprocess:
	jupyter nbconvert --execute notebooks/03_preprocessing.ipynb --to notebook --inplace

reduce:
	jupyter nbconvert --execute notebooks/04_dimensionality_reduction.ipynb --to notebook --inplace

cluster:
	jupyter nbconvert --execute notebooks/05_clustering.ipynb --to notebook --inplace

validate:
	jupyter nbconvert --execute notebooks/06_validation.ipynb --to notebook --inplace

trends:
	jupyter nbconvert --execute notebooks/07_trends_and_insights.ipynb --to notebook --inplace

all: collect preprocess reduce cluster validate trends

clean:
	rm -rf data/processed/*.npy data/processed/*.npz data/processed/*.json
	@echo "Cleaned processed data. Raw data preserved."
```

### Step 0.6 — Write README skeleton

**File:** `README.md`

Write a skeleton README now with the project structure, reproducibility instructions, and method overview. Leave the "Key Findings" section as a placeholder — you'll fill it in after the analysis is complete in the final phase.

```markdown
# ArXiv AI Research Trends (2024-2026)

Unsupervised clustering analysis of 226k+ AI research papers from ArXiv
to discover and track emerging research trends.

## What This Project Does

[TODO: fill after analysis is complete]

## Key Findings

[TODO: fill after analysis is complete]

## Method

- 3 embedding types (TF-IDF, MiniLM, KaLM) x 4 clustering algorithms
  (K-Means, GMM, HDBSCAN, BERTopic) = 12 combinations
- UMAP dimensionality reduction with hyperparameter tuning
- Evaluated with silhouette, Davies-Bouldin, Calinski-Harabasz, and topic coherence (c_v)
- Validated with ARI/NMI against ArXiv categories and cross-embedding agreement
- Growth trends stress-tested with linear regression, sensitivity analysis, and seasonality normalization

## Project Structure

notebooks/
  01_data_collection.ipynb          — ArXiv API collection (226k papers)
  02_exploratory_analysis.ipynb     — EDA, category/term trends, deep dives
  03_preprocessing.ipynb            — Text cleaning, survey flagging, embeddings
  04_dimensionality_reduction.ipynb — PCA/SVD, UMAP (dimension + hyperparameter tuning)
  05_clustering.ipynb               — K-Means, GMM, HDBSCAN, BERTopic comparison
  06_validation.ipynb               — Noise analysis, ARI/NMI, cross-embedding agreement
  07_trends_and_insights.ipynb      — Growth analysis, recommendations

src/
  collect.py       — ArXiv API data collection with checkpointing
  preprocess.py    — Text cleaning, filtering, lemmatization
  features.py      — TF-IDF, embeddings, UMAP, topic coherence
  visualize.py     — Reusable plotting functions
  label.py         — Cluster labeling via Claude Opus API

data/              — gitignored, regenerated by pipeline
report/            — final report / exported figures

## Reproducibility

1. Clone this repo
2. Create a Python 3.11 environment (conda or venv)
3. `make setup`
4. Set `ANTHROPIC_API_KEY` env var (needed for cluster labeling only)
5. `make all` (or run notebooks 01-07 in order)

Note: Full pipeline takes ~4-5 hours:
- Data collection: ~1h (internet required)
- KaLM embeddings: ~2h (CUDA GPU required)
- UMAP reduction: ~1h
- Clustering + labeling: ~30min
```

**Expected outcome:** `src/` has five working modules (collect, preprocess, features, visualize, label). Requirements are pinned. Makefile and README exist. All infrastructure is in place before any notebook changes.

---

## Phase 1 — Preprocessing Improvement (Title + Abstract Embeddings)

**Why:** Titles are often more discriminative than abstracts alone — a title like "DPO-Augmented RLHF for Code Generation" carries dense topic signal that might be diluted in a 200-word abstract. This is a small change that could meaningfully improve every downstream result.

### Step 1.1 — Update notebook 03 to use src/ imports + add new features

**File:** `notebooks/03_preprocessing.ipynb`

**What to do:**

Replace the inline preprocessing code with `src.preprocess` imports. The notebook becomes thin orchestration:

```python
import sys
from pathlib import Path

project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.preprocess import filter_papers, clean_abstracts, flag_surveys, lemmatize_abstracts
from src.features import compute_tfidf, compute_embeddings

# Load raw data
df = pd.read_csv(project_root / "data" / "raw" / "arxiv_papers.csv")
print(f"Loaded: {len(df):,} papers")

# Preprocessing pipeline
df = filter_papers(df)
df = clean_abstracts(df)      # creates abstract_clean AND embed_text
df = flag_surveys(df)          # creates is_survey
df = lemmatize_abstracts(df)   # creates abstract_lemma

# Save cleaned data
processed_dir = project_root / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(processed_dir / "arxiv_clean.csv", index=False)
```

For embeddings, use `src.features.compute_embeddings` but note: MiniLM and KaLM now use `embed_text` (title+abstract) while TF-IDF still uses `abstract_lemma`:

```python
# TF-IDF (uses lemmatized abstract only — titles add noise to bag-of-words)
tfidf_matrix, tfidf_model = compute_tfidf(df["abstract_lemma"])

# MiniLM (uses title + abstract for richer signal)
minilm_embeddings = compute_embeddings(
    df["embed_text"].tolist(),
    model_name="all-MiniLM-L6-v2",
    batch_size=256,
)

# KaLM (uses title + abstract for richer signal)
kalm_embeddings = compute_embeddings(
    df["embed_text"].tolist(),
    model_name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
    batch_size=64,
)
```

Save all outputs as before (`tfidf_vectors.npz`, `minilm_embeddings.npy`, `kalm_embeddings.npy`).

**Note:** KaLM embedding takes ~2 hours on GPU. MiniLM takes ~5 minutes. Plan accordingly.

**Expected outcome:** `arxiv_clean.csv` now has `embed_text` and `is_survey` columns. Embeddings are re-generated with richer input.

---

## Phase 2 — UMAP Hyperparameter & Dimension Experiment

**Why:** UMAP settings are the most upstream change after embeddings. If different parameters give better clusters, it changes ALL downstream analysis.

### Step 2.1 — Update notebook 04 to use src/ imports + add experiments

**File:** `notebooks/04_dimensionality_reduction.ipynb`

**What to do:**

Replace inline UMAP code with `src.features.reduce_umap` and `src.features.reduce_svd` imports. Then add the new experiments.

**Existing 20d runs** (now using `reduce_umap`):

```python
from src.features import reduce_svd, reduce_umap

# TF-IDF: SVD 500d first, then UMAP
tfidf_svd, svd_model = reduce_svd(tfidf_matrix, n_components=500)

# UMAP 20d for clustering
tfidf_umap20 = reduce_umap(tfidf_svd, n_components=20)
minilm_umap20 = reduce_umap(minilm_emb, n_components=20)
kalm_umap20 = reduce_umap(kalm_emb, n_components=20)
```

**New: 30d and 40d UMAP:**

```python
# UMAP 30d
tfidf_umap30 = reduce_umap(tfidf_svd, n_components=30)
minilm_umap30 = reduce_umap(minilm_emb, n_components=30)
kalm_umap30 = reduce_umap(kalm_emb, n_components=30)

# UMAP 40d
tfidf_umap40 = reduce_umap(tfidf_svd, n_components=40)
minilm_umap40 = reduce_umap(minilm_emb, n_components=40)
kalm_umap40 = reduce_umap(kalm_emb, n_components=40)
```

Save all to `data/processed/` (`tfidf_umap30.npy`, `kalm_umap30.npy`, etc.).

Keep the 2d and 3d UMAP outputs for visualization — regenerate them since embeddings changed in Phase 1.

### Step 2.2 — UMAP n_neighbors sweep

**File:** `notebooks/04_dimensionality_reduction.ipynb`

**What to do:**

You currently fixed `n_neighbors=15` everywhere. With 181k papers, this is quite local. Higher values capture more global structure.

Run a sweep on KaLM embeddings only (to save time):

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

results = []
for n_neighbors in [15, 30, 50]:
    for min_dist in [0.0, 0.05]:
        reduced = reduce_umap(
            kalm_emb, n_components=20,
            n_neighbors=n_neighbors, min_dist=min_dist,
        )
        km = KMeans(n_clusters=24, random_state=42, n_init=10)
        labels = km.fit_predict(reduced)
        sil = silhouette_score(reduced, labels, sample_size=10000)
        results.append({
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "silhouette": sil,
        })
        print(f"n_neighbors={n_neighbors}, min_dist={min_dist}: sil={sil:.4f}")
```

Pick the best combo. If `n_neighbors=30` or `50` wins, regenerate ALL UMAP outputs with the winning parameters.

**Note:** Each UMAP run on 181k x 896d takes ~15-30 minutes. Full sweep is ~2-3 hours. If time is tight, just test `n_neighbors` in [15, 30] with `min_dist=0.0` (2 runs).

### Step 2.3 — Compare clustering quality across dimensions

**File:** `notebooks/04_dimensionality_reduction.ipynb` (add at the end)

**What to do:**

Run HDBSCAN (`min_cluster_size=1000`, `min_samples=10`) on KaLM embeddings at all three dimensionalities (20d, 30d, 40d), using the best `n_neighbors`/`min_dist` from Step 2.2. For each, compute:
- Number of clusters
- Noise percentage
- Silhouette score (on non-noise points, `sample_size=10000`)
- Davies-Bouldin index (on non-noise points)

Also run K-Means (k=24) on each dimensionality and compare. This gives you a 2x3 grid.

```
Dims  Algorithm  Clusters  Noise%  Silhouette  Davies-Bouldin
20    HDBSCAN    41        36.9%   0.5073      0.7358
30    HDBSCAN    ??        ??      ??          ??
40    HDBSCAN    ??        ??      ??          ??
20    K-Means    24        0%      0.3936      0.9448
30    K-Means    24        0%      ??          ??
40    K-Means    24        0%      ??          ??
```

**Decision:** Pick the dimensionality that gives the best silhouette/DB tradeoff. Add a markdown cell documenting your decision and reasoning.

**Expected outcome:** You know the optimal UMAP dimensionality AND n_neighbors/min_dist, with data to justify both.

---

## Phase 3 — Final Clustering (Full 12-Combination Comparison)

**Why 12 instead of 9:** Adding BERTopic as a fourth algorithm. BERTopic is the state-of-the-art for neural topic modeling — it wraps HDBSCAN + c-TF-IDF and provides automatic topic labels, hierarchy visualization, and topic evolution tracking for free.

### Step 3.1 — Archive old notebook 05 and create new one

**What to do:**

1. Create `notebooks/archive/` directory
2. Move `05_clustering_and_results.ipynb` into `notebooks/archive/`
3. Create fresh `05_clustering.ipynb`

Copy the K-Means, GMM, and HDBSCAN cells from the old notebook into the new one. Replace inline `get_cluster_top_terms` with `from src.features import get_cluster_top_terms, compute_coherence`. Replace inline scatter plots with `from src.visualize import plot_cluster_scatter_2d`.

### Step 3.2 — Add BERTopic

**File:** `notebooks/05_clustering.ipynb`

**What to do:**

After the HDBSCAN section, add BERTopic:

```python
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

for name in ["TF-IDF", "MiniLM", "KaLM"]:
    if name == "TF-IDF":
        emb = tfidf_svd
    elif name == "MiniLM":
        emb = minilm_emb
    else:
        emb = kalm_emb

    umap_model = UMAP(
        n_components=WINNING_DIM,      # from Phase 2
        n_neighbors=WINNING_NEIGHBORS, # from Phase 2
        min_dist=0.0,
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=1000,
        min_samples=10,
        metric="euclidean",
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(
        df["abstract_clean"].tolist(),
        embeddings=emb,
    )

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    noise_pct = (np.array(topics) == -1).mean() * 100
    print(f"BERTopic + {name}: {n_topics} topics, {noise_pct:.1f}% noise")
    print(topic_model.get_topic_info().head(15))
```

**Key BERTopic features to show:**
- `topic_model.get_topic_info()` — automatic topic labels with representative words
- `topic_model.visualize_topics()` — interactive topic similarity map
- `topic_model.visualize_hierarchy()` — hierarchical topic structure (dendrogram)
- `topic_model.topics_over_time(df["abstract_clean"].tolist(), df["published"].tolist())` — topic evolution chart

### Step 3.3 — Full comparison table (12 combinations)

**File:** `notebooks/05_clustering.ipynb`

**What to do:**

Update the evaluation comparison to include BERTopic. 3 embeddings x 4 algorithms = 12 combinations. Compute silhouette, Davies-Bouldin, Calinski-Harabasz, AND **topic coherence (c_v)** for all 12:

```python
from src.features import compute_coherence

coherence = compute_coherence(df, labels, text_col="abstract")
```

The final table:

```
Embedding  Algorithm  k   Silhouette  Davies-Bouldin  Calinski-Harabasz  Coherence  Noise%
TF-IDF     K-Means    25  0.3299      0.9536          33534              ?.????     0.0
TF-IDF     GMM        22  0.3059      1.0927          29914              ?.????     0.0
TF-IDF     HDBSCAN    39  0.5139      0.6605          33993              ?.????     45.1
TF-IDF     BERTopic   ??  ?.????      ?.????          ?.????             ?.????     ?.?
MiniLM     K-Means    29  ...
...
KaLM       BERTopic   ??  ...
```

### Step 3.4 — Select best combination and re-label

**File:** `notebooks/05_clustering.ipynb`, `src/label.py`

Based on the 12-combination table, select the best primary and secondary clustering. If BERTopic wins, compare its auto-labels vs Claude Opus labels as a sanity check. If HDBSCAN wins, re-run Claude Opus labeling (clusters likely changed since embeddings changed).

### Step 3.5 — Hierarchical topic structure

**File:** `notebooks/05_clustering.ipynb`

Build a dendrogram showing how topics relate:

```python
from scipy.cluster.hierarchy import linkage, dendrogram

centroids = []
cluster_names = []
for label in sorted(set(final_labels)):
    if label == -1:
        continue
    mask = final_labels == label
    centroids.append(kalm_umap_final[mask].mean(axis=0))
    cluster_names.append(label_info[label]["label"])

centroids = np.array(centroids)
Z = linkage(centroids, method="ward")

fig, ax = plt.subplots(figsize=(16, 10))
dendrogram(Z, labels=cluster_names, leaf_rotation=90, leaf_font_size=8, ax=ax)
ax.set_title("Hierarchical Structure of AI Research Topics")
ax.set_ylabel("Ward Distance")
plt.tight_layout()
plt.show()
```

If BERTopic was used, you get this for free with `topic_model.visualize_hierarchy()`.

**Expected outcome:** Complete, well-justified clustering with 4 algorithms compared, topic coherence measured, and hierarchical topic structure visualized.

---

## Phase 4 — Noise Investigation & External Validation

**Why:** 37% of papers are labeled as noise. This phase turns the assertion "it's a feature" into a demonstrated finding, and validates clusters against external labels.

### Step 4.1 — Create notebook 06_validation.ipynb

**File:** `notebooks/06_validation.ipynb` (NEW)

Create a new notebook. Load the cleaned dataset, all cluster labels (HDBSCAN, K-Means, GMM, BERTopic for all embeddings), and label JSONs. Use `src/` imports throughout.

### Step 4.2 — Noise composition analysis

**File:** `notebooks/06_validation.ipynb`

Take all noise papers (`hdbscan_label == -1`). Analyze:

1. **Category distribution of noise vs non-noise:**
   ```python
   noise_cats = df[df["cluster"] == -1]["primary_category"].value_counts(normalize=True)
   clean_cats = df[df["cluster"] != -1]["primary_category"].value_counts(normalize=True)
   ```
   Plot side by side. Is cs.AI (broad, interdisciplinary) over-represented in noise?

2. **Noise rate over time:**
   ```python
   monthly_noise_rate = df.groupby("month").apply(lambda x: (x["cluster"] == -1).mean())
   ```
   Is noise growing (new topics emerging) or stable?

3. **Top TF-IDF terms in noise papers** — are they generic or do they hint at hidden topics?

4. **Key term noise rates:**
   ```python
   key_terms = {
       "LLM": r"\bLLM[s]?\b",
       "diffusion": r"\bdiffusion\b",
       "transformer": r"\btransformer[s]?\b",
       "agent": r"\bagent[s]?\b",
       "multimodal": r"\bmultimodal\b",
       "RAG": r"\bRAG\b",
       "fine-tun": r"\bfine.tun",
       "reasoning": r"\breasoning\b",
       "reinforcement learning": r"\breinforcement learning\b",
       "robot": r"\brobot[s]?\b",
   }
   for term, pattern in key_terms.items():
       hits = df[df["abstract"].str.contains(pattern, case=False, na=False)]
       noise_rate = (hits["cluster"] == -1).mean()
       print(f"{term}: {noise_rate:.1%} noise rate ({len(hits):,} papers)")
   ```

5. **Survey papers in noise:**
   ```python
   survey_noise = df[df["is_survey"]]["cluster"].eq(-1).mean()
   non_survey_noise = df[~df["is_survey"]]["cluster"].eq(-1).mean()
   ```
   If surveys have much higher noise rate, that explains part of the 37%.

### Step 4.3 — Sub-clustering the noise

**File:** `notebooks/06_validation.ipynb`

```python
noise_mask = hdbscan_labels == -1
noise_embeddings = kalm_umap_final[noise_mask]

for ms in [200, 300, 500]:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=ms, min_samples=10)
    sub_labels = clusterer.fit_predict(noise_embeddings)
    n_sub = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
    still_noise = (sub_labels == -1).mean() * 100
    print(f"min_size={ms}: {n_sub} sub-clusters, {still_noise:.1f}% still noise")
```

If sub-clusters emerge, extract top terms with `get_cluster_top_terms` from `src/features`. Write conclusion: "X% of noise papers form coherent sub-topics below my size threshold. The remaining Y% are genuinely interdisciplinary."

### Step 4.4 — ARI/NMI vs ArXiv categories

**File:** `notebooks/06_validation.ipynb`

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# For each algorithm (HDBSCAN non-noise, K-Means all, GMM all, BERTopic non-noise):
mask = hdbscan_labels != -1
ari = adjusted_rand_score(df.loc[mask, "primary_category"], hdbscan_labels[mask])
nmi = normalized_mutual_info_score(df.loc[mask, "primary_category"], hdbscan_labels[mask])
```

Print comparison table for all 4 algorithms. Create confusion-style heatmap with `plot_cluster_category_heatmap` from `src/visualize`.

**Interpretation:** ARI ~0.05-0.15 expected (your clusters are finer-grained than 8 ArXiv categories). NMI ~0.2-0.4 suggests meaningful alignment.

### Step 4.5 — Cross-embedding agreement

**File:** `notebooks/06_validation.ipynb`

Compute ARI/NMI between clusterings from different embeddings:

```python
# HDBSCAN: only papers non-noise in BOTH
both_non_noise = (hdbscan_kalm != -1) & (hdbscan_minilm != -1)
ari = adjusted_rand_score(hdbscan_kalm[both_non_noise], hdbscan_minilm[both_non_noise])
```

Present as matrices (HDBSCAN ARI, HDBSCAN NMI, K-Means ARI, K-Means NMI).

**Interpretation:** High ARI (>0.3) between KaLM and MiniLM = cluster structure is real. Low ARI with TF-IDF = expected (lexical vs semantic).

**Expected outcome:** Evidence-based statements about noise rate and cluster validity.

---

## Phase 5 — Stress-Test Business Recommendations

### Step 5.1 — Create notebook 07_trends_and_insights.ipynb

**File:** `notebooks/07_trends_and_insights.ipynb` (NEW)

Copy trend analysis and recommendation cells from the archived `05_clustering_and_results.ipynb`. Use `src/` imports for plotting (`plot_growth_bars`, `plot_monthly_trends`, `plot_comparison_scatter`).

### Step 5.2 — Replace naive growth with regression

**File:** `notebooks/07_trends_and_insights.ipynb`

```python
from scipy import stats

growth_stats = []
months = sorted(monthly_clusters.index)
x = np.arange(len(months))

for cluster_name in monthly_clusters.columns:
    y = monthly_clusters[cluster_name].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    mean_val = y.mean()
    monthly_growth_pct = (slope / mean_val) * 100 if mean_val > 0 else 0

    growth_stats.append({
        "Cluster": cluster_name,
        "Slope (papers/month)": round(slope, 2),
        "Monthly Growth %": round(monthly_growth_pct, 2),
        "R-squared": round(r_value**2, 4),
        "p-value": round(p_value, 6),
        "Significant (p<0.05)": p_value < 0.05,
    })

growth_df = pd.DataFrame(growth_stats).sort_values("Slope (papers/month)", ascending=False)
```

Keep the old first-half vs last-half calculation too, for comparison.

### Step 5.3 — Sensitivity analysis

**File:** `notebooks/07_trends_and_insights.ipynb`

Three different time window splits:

```python
splits = {
    "6-month": (months[:6], months[-6:]),
    "8-month": (months[:8], months[-8:]),
    "thirds": (months[:len(months)//3], months[-len(months)//3:]),
}
```

Show top-10 across all splits side by side. Add markdown: "The ranking is consistent across all splits, meaning the direction is robust even if the magnitude varies."

### Step 5.4 — Seasonality check

**File:** `notebooks/07_trends_and_insights.ipynb`

Normalize by total monthly volume to get share-based growth:

```python
total_monthly = df.groupby("month").size()
normalized = monthly_clusters.div(total_monthly, axis=0) * 100

# Re-compute regression on normalized data
share_growth = []
for cluster_name in normalized.columns:
    y = normalized[cluster_name].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    share_growth.append({
        "Cluster": cluster_name,
        "Absolute Growth": growth_df[growth_df["Cluster"] == cluster_name]["Slope (papers/month)"].values[0],
        "Share Growth (%/month)": round(slope, 4),
        "Share p-value": round(p_value, 6),
    })
```

If growth disappears after normalization, the cluster was just riding overall ArXiv growth. Report which clusters grow in both absolute AND share terms.

**Expected outcome:** Recommendations backed by statistical significance, sensitivity analysis, and share-based growth.

---

## Phase 6 — Presentation (Final)

**Why this is last:** The analysis is now final. Rewriting once is better than rewriting twice.

### Step 6.1 — Rewrite ALL markdown cells across ALL notebooks

**Files:** All notebooks (01 through 07)

**Style guide:**

1. **Use "I" instead of "we":**
   - Before: "We proceed with KaLM + HDBSCAN as the primary clustering"
   - After: "I'm going with KaLM + HDBSCAN as my primary clustering"

2. **Show your thinking, not just conclusions:**
   - Before: "GMM underperforms across all embeddings"
   - After: "I was hoping GMM would handle the non-spherical cluster shapes better than K-Means, but it actually performed worse everywhere. The Gaussian assumption just doesn't fit how research topics are distributed — they have irregular, elongated shapes in embedding space."

3. **Be honest about surprises and mistakes:**
   - Before: "PCA on TF-IDF retains only 11.6% variance"
   - After: "My first attempt was PCA to 50 components, and the results were terrible — only 11.6% variance retained for TF-IDF. I initially thought this was a bug, but it makes sense: TF-IDF spreads information across 27k sparse dimensions. No small linear subspace can capture that. So I switched to SVD with 500 components for TF-IDF and skipped PCA entirely for the neural embeddings."

4. **Add brief "why I did this" before each section:**
   - Before: "## K-Means"
   - After: "## K-Means — Starting Simple\n\nI'm starting with K-Means because it's the simplest baseline. If K-Means already gives clean clusters, there's no need for fancier methods."

5. **Comment on unexpected results:**
   - "I didn't expect KaLM to only marginally beat MiniLM — KaLM is a much larger model (896d vs 384d) and took 2 hours to embed vs 5 minutes."

6. **Keep it concise.** Personal doesn't mean verbose. Short, opinionated sentences.

7. **Don't over-explain obvious things.** Skip textbook definitions.

8. **End each notebook with a 3-5 line "What I learned" section:**
   > ## What I Learned
   > The biggest surprise was how badly GMM performed — I expected soft assignments to help with overlapping topics, but the Gaussian assumption was just wrong for this data.

9. **State alternatives you considered and why you rejected them:**
   > I considered Spectral Clustering but ruled it out — with 181k papers, the affinity matrix would be 181k x 181k, which doesn't fit in memory.

### Step 6.2 — Clean up duplicated/empty cells

**Files:** All notebooks

- Remove the duplicated "Step 5 — Select Best Combinations" markdown cell in the archived notebook 05 (if any cells were copied to new 05)
- Remove empty cells at the end of notebooks
- Remove any `import importlib; importlib.reload(...)` cells left from development
- Remove `sqarify` imports if they exist anywhere

### Step 6.3 — Finalize README

**File:** `README.md`

Fill in the "What This Project Does" and "Key Findings" sections that were left as TODOs in Phase 0. Now you have actual results to report.

### Step 6.4 — Final rerun of ALL notebooks

**What to do:**

Rerun all notebooks 01-07 in order. This ensures all cell outputs match the final rewritten markdown. You don't want stale outputs from earlier runs sitting next to new commentary.

This is the last step. After this, the project is complete.

---

## Checklist

Use this to track progress:

```
Phase 0 — Build src/ Modules + Reproducibility
  [ ] 0.1  Fill in src/preprocess.py
  [ ] 0.2  Fill in src/features.py (including compute_coherence)
  [ ] 0.3  Fill in src/visualize.py
  [ ] 0.4  Pin requirements.txt (add bertopic, gensim; remove sqarify)
  [ ] 0.5  Add Makefile
  [ ] 0.6  Write README skeleton (TODOs for findings)

Phase 1 — Preprocessing Improvement
  [ ] 1.1  Update notebook 03 with src/ imports + embed_text + is_survey → rerun

Phase 2 — UMAP Hyperparameter & Dimension Experiment
  [ ] 2.1  Update notebook 04 with src/ imports + 30d/40d UMAP
  [ ] 2.2  Sweep n_neighbors (15, 30, 50) and min_dist (0.0, 0.05)
  [ ] 2.3  Compare clustering quality across dimensions, pick winner → rerun
  ⚠ CHECKPOINT: verify new silhouette >= old 0.5073 before continuing

Phase 3 — Final Clustering (12-Combination Comparison)
  [ ] 3.1  Archive old 05, create new 05_clustering.ipynb with src/ imports
  [ ] 3.2  Add BERTopic as fourth algorithm
  [ ] 3.3  Full 12-combination comparison table (with topic coherence c_v)
  [ ] 3.4  Select best combination, re-label with Claude Opus
  [ ] 3.5  Build hierarchical topic dendrogram → run

Phase 4 — Noise Investigation & External Validation
  [ ] 4.1  Create 06_validation.ipynb with src/ imports
  [ ] 4.2  Noise composition analysis (categories, time, terms, surveys)
  [ ] 4.3  Sub-clustering the noise (min_cluster_size 200/300/500)
  [ ] 4.4  ARI/NMI vs ArXiv categories + confusion heatmap
  [ ] 4.5  Cross-embedding agreement matrix → run

Phase 5 — Stress-Test Recommendations
  [ ] 5.1  Create 07_trends_and_insights.ipynb with src/ imports
  [ ] 5.2  Linear regression growth with p-values and R-squared
  [ ] 5.3  Sensitivity analysis (3 different time window splits)
  [ ] 5.4  Seasonality check (absolute vs share-of-total growth) → run

Phase 6 — Presentation (Final)
  [ ] 6.1  Rewrite all markdown cells (personal voice, show thinking)
  [ ] 6.2  Clean up duplicated/empty cells
  [ ] 6.3  Finalize README (fill in findings)
  [ ] 6.4  Final rerun of ALL notebooks (01-07)
```
