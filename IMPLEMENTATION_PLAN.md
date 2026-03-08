# Implementation Plan — ArXiv AI Trends Project

This document is a step-by-step guide for completing the project. Each phase builds on the previous one. Do them in order. Each step includes exactly what to do, what files to touch, and what the expected outcome is.

---

## Phase 1 — UMAP Dimension Experiment

**Why this is first:** Changing UMAP dimensions is the most upstream change. If 30d or 40d gives better clusters, it changes ALL downstream analysis. Do this before anything else so you only cluster once.

### Step 1.1 — Add 30d and 40d UMAP to notebook 04

**File:** `notebooks/04_dimensionality_reduction.ipynb`

**What to do:**

After the existing UMAP 20d cell (the one that produces `tfidf_umap20`, `minilm_umap20`, `kalm_umap20`), add new cells that run UMAP at 30d and 40d for all three embeddings. Use the same parameters (`n_neighbors=15`, `min_dist=0.0`, `random_state=42`).

The inputs are:
- `tfidf_svd` (the 500d SVD-reduced TF-IDF matrix)
- `minilm_emb` (raw 384d MiniLM embeddings)
- `kalm_emb` (raw 896d KaLM embeddings)

Generate and save:
- `tfidf_umap30.npy`, `minilm_umap30.npy`, `kalm_umap30.npy`
- `tfidf_umap40.npy`, `minilm_umap40.npy`, `kalm_umap40.npy`

Save all to `data/processed/`.

**Important:** Keep the existing 20d, 2d, and 3d UMAP outputs unchanged. The 2d and 3d are for visualization only and don't need 30d/40d variants.

### Step 1.2 — Compare clustering quality across 20d, 30d, 40d

**File:** `notebooks/05_clustering_and_results.ipynb` (or create a temporary comparison notebook)

**What to do:**

Run HDBSCAN (`min_cluster_size=1000`, `min_samples=10`) on KaLM embeddings at all three dimensionalities (20d, 30d, 40d). For each, compute:
- Number of clusters
- Noise percentage
- Silhouette score (on non-noise points, `sample_size=10000`)
- Davies-Bouldin index (on non-noise points)

Print a comparison table like:

```
Dims  Clusters  Noise%  Silhouette  Davies-Bouldin
20    41        36.9%   0.5073      0.7358
30    ??        ??      ??          ??
40    ??        ??      ??          ??
```

Also run K-Means (k=24) on each dimensionality and compare. This gives you a 2x3 grid (2 algorithms x 3 dimensionalities).

**Decision:** Pick the dimensionality that gives the best silhouette/DB tradeoff. If 20d was already best, great — you've now proven it. If 30d wins, use 30d for all subsequent work.

Add a markdown cell documenting your decision and reasoning.

**Expected outcome:** You know the optimal UMAP dimensionality and have the data to justify it.

---

## Phase 2 — Re-cluster With Final Settings

### Step 2.1 — Final clustering with the winning dimensionality

**File:** `notebooks/05_clustering_and_results.ipynb`

**What to do:**

If a new dimensionality won in Step 1.2, update notebook 05 to load the winning UMAP files instead of the 20d ones. Re-run the full 9-combination comparison (3 embeddings x 3 algorithms) with the new dimensionality.

If 20d was still best, no changes needed here — just add a markdown cell referencing the experiment from Phase 1 and explaining why 20d was kept.

### Step 2.2 — Re-label clusters if they changed

**File:** `notebooks/05_clustering_and_results.ipynb`, `src/label.py`

**What to do:**

If the clusters changed (different number of clusters, or significantly different composition), re-run the Claude Opus labeling. Use the existing `src/label.py` module — it already works.

Compare old labels vs new labels in a markdown cell. Note which clusters merged, split, or shifted.

If clusters didn't change meaningfully (same count, similar top terms), skip re-labeling and keep existing labels.

**Expected outcome:** You have final, stable clusters that all subsequent analysis will use.

---

## Phase 3 — Noise Investigation

**Why:** 37% of papers are labeled as noise. Right now you just assert "it's a feature." This phase turns that into a demonstrated finding.

### Step 3.1 — Create notebook 06_validation.ipynb

**File:** `notebooks/06_validation.ipynb` (NEW)

**What to do:**

Create a new notebook. Start with a markdown cell:

> # 06 — Validation & Robustness
>
> I need to prove that my clusters are trustworthy. This notebook does three things: investigates the 37% noise rate, validates clusters against external labels, and checks whether different embeddings agree with each other.

Load the cleaned dataset, the final HDBSCAN labels (KaLM), and the ArXiv primary categories.

### Step 3.2 — Noise composition analysis

**File:** `notebooks/06_validation.ipynb`

**What to do:**

Take all papers where `hdbscan_label == -1` (the noise papers). Analyze them:

1. **Category distribution of noise vs non-noise:**
   ```python
   noise_cats = df[df["cluster"] == -1]["primary_category"].value_counts(normalize=True)
   clean_cats = df[df["cluster"] != -1]["primary_category"].value_counts(normalize=True)
   ```
   Plot these side by side as horizontal bar charts. Are certain categories (like cs.AI, which is broad) over-represented in noise?

2. **Noise rate over time:**
   ```python
   monthly_noise_rate = df.groupby("month").apply(lambda x: (x["cluster"] == -1).mean())
   ```
   Plot this as a line chart. Is noise growing (meaning new topics are emerging that don't fit existing clusters) or stable?

3. **Top TF-IDF terms in noise papers:**
   Run TF-IDF on just the noise papers' abstracts. What are the top 30 terms? Are they generic (suggesting truly diffuse papers) or do they hint at hidden topics?

4. **Key term frequency in noise vs clustered:**
   For each of the key terms from notebook 02 (LLM, diffusion, transformer, agent, etc.), compute what fraction of papers mentioning that term ended up as noise. Some terms might have disproportionately high noise rates, revealing which topics HDBSCAN struggles with.

### Step 3.3 — Sub-clustering the noise

**File:** `notebooks/06_validation.ipynb`

**What to do:**

Run HDBSCAN with `min_cluster_size=500` (and also try 300) on ONLY the noise papers' KaLM UMAP embeddings (use the same dimensionality from Phase 1). This tests whether the noise contains coherent sub-groups that were just below the `min_cluster_size=1000` threshold.

```python
noise_mask = hdbscan_labels == -1
noise_embeddings = kalm_umap_final[noise_mask]  # use whatever dim won

for ms in [300, 500]:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=ms, min_samples=10)
    sub_labels = clusterer.fit_predict(noise_embeddings)
    n_sub = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
    still_noise = (sub_labels == -1).mean() * 100
    print(f"min_size={ms}: {n_sub} sub-clusters, {still_noise:.1f}% still noise")
```

If coherent sub-clusters emerge, extract their top TF-IDF terms. This shows whether `min_cluster_size=1000` was too aggressive or whether the noise is genuinely diffuse.

Write a markdown conclusion: "X% of noise papers form coherent sub-topics below my size threshold. The remaining Y% are genuinely interdisciplinary."

**Expected outcome:** You can now make an evidence-based statement about your noise rate instead of just asserting it's fine.

---

## Phase 4 — External Validation

### Step 4.1 — Cluster agreement with ArXiv categories (ARI/NMI)

**File:** `notebooks/06_validation.ipynb`

**What to do:**

Compute Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) between your HDBSCAN cluster labels and ArXiv's `primary_category`. Only use non-noise papers for this.

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

mask = hdbscan_labels != -1
ari = adjusted_rand_score(df.loc[mask, "primary_category"], hdbscan_labels[mask])
nmi = normalized_mutual_info_score(df.loc[mask, "primary_category"], hdbscan_labels[mask])
print(f"ARI: {ari:.4f}")
print(f"NMI: {nmi:.4f}")
```

Do the same for K-Means labels (which have no noise, so use all papers).

**Interpretation guidance for the markdown:**
- ARI ranges from -1 to 1. Values around 0.05-0.15 are expected because your clusters are FINER-GRAINED than ArXiv categories (41 clusters vs 8 categories). Low ARI doesn't mean bad clusters — it means you discovered sub-structure within categories.
- NMI ranges from 0 to 1. Values around 0.2-0.4 would suggest meaningful alignment. If NMI is very low (<0.1), investigate why.

Also create a **confusion-style heatmap**: for each cluster, show the distribution of ArXiv categories. This visualizes how your unsupervised clusters relate to human-assigned categories.

```python
# Crosstab: cluster vs primary_category
ct = pd.crosstab(
    df.loc[mask, "cluster_name"],
    df.loc[mask, "primary_category"],
    normalize="index"  # row-normalize so each cluster sums to 1
)

# Plot as heatmap
fig, ax = plt.subplots(figsize=(12, 16))
sns.heatmap(ct, cmap="Blues", ax=ax, annot=True, fmt=".2f")
ax.set_title("Cluster Composition by ArXiv Category")
```

This shows, for example, that your "Robot Learning" cluster is 70% cs.RO + 20% cs.AI, which validates it. If a cluster is evenly spread across all 8 categories, it's suspicious.

### Step 4.2 — Cross-embedding agreement (ARI/NMI between embeddings)

**File:** `notebooks/06_validation.ipynb`

**What to do:**

Compare whether different embeddings discover the same structure. Compute ARI and NMI between:
- KaLM-HDBSCAN vs MiniLM-HDBSCAN (both non-noise only, on their intersection)
- KaLM-HDBSCAN vs TF-IDF-HDBSCAN
- MiniLM-HDBSCAN vs TF-IDF-HDBSCAN
- KaLM-KMeans vs MiniLM-KMeans
- KaLM-KMeans vs TF-IDF-KMeans
- MiniLM-KMeans vs TF-IDF-KMeans

For HDBSCAN comparisons, only include papers that are non-noise in BOTH clusterings:

```python
both_non_noise = (hdbscan_kalm != -1) & (hdbscan_minilm != -1)
ari = adjusted_rand_score(hdbscan_kalm[both_non_noise], hdbscan_minilm[both_non_noise])
```

Present as a matrix:

```
           KaLM    MiniLM   TF-IDF
KaLM       1.000   ?.???    ?.???
MiniLM     ?.???   1.000    ?.???
TF-IDF     ?.???   ?.???    1.000
```

**What to expect and how to interpret:**
- High ARI (>0.3) between KaLM and MiniLM = neural embeddings agree, which means the cluster structure is real and not an artifact of one model.
- Lower ARI with TF-IDF = expected, because TF-IDF captures lexical patterns while neural models capture semantics.
- If KaLM and MiniLM disagree strongly (ARI < 0.1), that's a red flag — it means cluster assignments depend heavily on the embedding choice, and your conclusions are less robust.

Write a markdown interpretation of the results.

**Expected outcome:** You can now say "my clusters are validated against external labels and robust across embedding methods" (or honestly report where they're not).

---

## Phase 5 — Stress-Test Business Recommendations

### Step 5.1 — Replace naive growth calculation with regression

**File:** `notebooks/07_trends_and_insights.ipynb` (will be split from current 05)

**What to do:**

The current growth calculation compares first-6-months average to last-6-months average. This is sensitive to the exact split point and doesn't give confidence intervals. Replace it with linear regression per cluster.

```python
from scipy import stats

growth_stats = []
months = sorted(monthly_clusters.index)
x = np.arange(len(months))  # 0, 1, 2, ... (month index)

for cluster_name in monthly_clusters.columns:
    y = monthly_clusters[cluster_name].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Growth rate: slope relative to mean
    mean_val = y.mean()
    monthly_growth_pct = (slope / mean_val) * 100 if mean_val > 0 else 0

    growth_stats.append({
        "Cluster": cluster_name,
        "Slope (papers/month)": round(slope, 2),
        "Monthly Growth %": round(monthly_growth_pct, 2),
        "R-squared": round(r_value**2, 4),
        "p-value": round(p_value, 6),
        "Significant": p_value < 0.05,
    })

growth_df = pd.DataFrame(growth_stats).sort_values("Slope (papers/month)", ascending=False)
```

This gives you:
- A slope with a p-value (is the growth statistically significant?)
- R-squared (is the trend linear, or is it noisy?)
- Monthly growth percentage that doesn't depend on an arbitrary split

### Step 5.2 — Sensitivity analysis with different time windows

**File:** `notebooks/07_trends_and_insights.ipynb`

**What to do:**

Run the old first-half vs last-half comparison with THREE different splits:
1. First 6 months vs last 6 months (your current approach)
2. First 8 months vs last 8 months
3. First third vs last third of the dataset

```python
splits = {
    "6-month": (months[:6], months[-6:]),
    "8-month": (months[:8], months[-8:]),
    "thirds": (months[:len(months)//3], months[-len(months)//3:]),
}

sensitivity = {}
for split_name, (early, late) in splits.items():
    early_avg = monthly_clusters.loc[early].mean()
    late_avg = monthly_clusters.loc[late].mean()
    growth = ((late_avg - early_avg) / early_avg * 100)
    sensitivity[split_name] = growth
```

Present top-5 clusters across all splits. If "LLM Reasoning" shows +415% with one split but +180% with another, REPORT BOTH. The conclusion ("LLM Reasoning is growing fast") is still valid, but the exact number is less meaningful.

Add a markdown cell: "The exact growth percentages vary by 50-200% depending on the time window, but the ranking of fastest-growing areas is consistent across all splits: [list]. This means the direction is robust even if the magnitude is approximate."

### Step 5.3 — Seasonality check

**File:** `notebooks/07_trends_and_insights.ipynb`

**What to do:**

ArXiv submissions spike around major conference deadlines (NeurIPS ~May, ICML ~Jan, CVPR ~Nov, etc.). Check if your growth trends are confounded by seasonal patterns.

```python
# Total monthly submissions
total_monthly = df.groupby("month").size()

# Normalize each cluster by total monthly volume
normalized = monthly_clusters.div(total_monthly, axis=0)

# Re-compute growth on normalized data
```

If a cluster's growth disappears after normalization, it was just riding the overall ArXiv growth wave, not actually gaining share. Report which clusters are growing in absolute terms vs growing in share.

**Expected outcome:** Your business recommendations now have error bars, sensitivity analysis, and share-based growth. Much harder to criticize.

---

## Phase 6 — Refactor src/ Modules

**Why this is Phase 6:** You need stable analysis code before extracting it into modules. Refactoring code that's still changing is wasted effort.

### Step 6.1 — Fill in src/preprocess.py

**File:** `src/preprocess.py`

**What to extract from notebooks:**

From `notebooks/03_preprocessing.ipynb`, extract:

```python
"""Preprocessing pipeline for ArXiv papers."""

import logging
import spacy
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

TARGET_CATEGORIES = {"cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.RO", "cs.NE", "cs.MA", "cs.IR"}
MIN_ABSTRACT_WORDS = 30


def filter_papers(df, categories=TARGET_CATEGORIES, min_words=MIN_ABSTRACT_WORDS):
    """Filter to target categories and drop short abstracts."""
    df = df[df["primary_category"].isin(categories)].copy()
    df = df[df["abstract"].str.split().str.len() >= min_words].copy()
    df = df.reset_index(drop=True)
    logger.info(f"Filtered to {len(df):,} papers")
    return df


def clean_abstracts(df):
    """Add abstract_clean (whitespace normalized) column."""
    df["abstract_clean"] = df["abstract"].str.replace(r"\s+", " ", regex=True).str.strip()
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
    """Full preprocessing pipeline: load, filter, clean, lemmatize, save."""
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df):,} papers from {raw_path}")
    df = filter_papers(df)
    df = clean_abstracts(df)
    df = lemmatize_abstracts(df)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df):,} papers to {output_path}")
    return df
```

### Step 6.2 — Fill in src/features.py

**File:** `src/features.py`

**What to extract from notebooks 03 and 04:**

```python
"""Feature extraction: TF-IDF, embeddings, dimensionality reduction."""

import logging
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

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
    logger.info(f"SVD: {sparse_matrix.shape[1]} -> {n_components}, "
                f"variance explained: {svd.explained_variance_ratio_.sum():.1%}")
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
```

Note: `get_cluster_top_terms` is currently defined inline in notebook 05. Move it here.

### Step 6.3 — Fill in src/visualize.py

**File:** `src/visualize.py`

**What to extract:** The repeated plotting patterns from notebooks 02 and 05.

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
    axes = axes.flatten()
    for ax, name in zip(axes, columns):
        data = monthly_data[name]
        data.plot(kind="bar", ax=ax, color="steelblue", width=0.8)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_visible(i % 4 == 0)
    # Hide unused axes
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
```

### Step 6.4 — Update notebooks to import from src/

**Files:** All notebooks (02 through 07)

**What to do:**

Replace inline code with imports. For example, in notebook 05, replace the inline `get_cluster_top_terms` definition with:

```python
from src.features import get_cluster_top_terms
```

In notebook 03, replace the inline preprocessing with:

```python
from src.preprocess import filter_papers, clean_abstracts, lemmatize_abstracts
```

Keep the notebooks as thin orchestration: load data, call functions from `src/`, display results, write markdown commentary.

**Do not over-refactor.** Notebooks should still be readable top-to-bottom. If a piece of code is only used once and is short (< 10 lines), leave it inline. Only extract things that are:
- Used in multiple notebooks, OR
- Long enough (> 15 lines) that they clutter the notebook's narrative

**Expected outcome:** `src/` has four real modules (collect, preprocess, features, visualize, label). Notebooks are shorter and focused on narrative + results.

---

## Phase 7 — Reproducibility

### Step 7.1 — Pin requirements.txt

**File:** `requirements.txt`

**What to do:**

Replace the current bare package names with pinned versions. Run this in your conda/venv:

```bash
pip freeze > requirements_full.txt
```

Then manually curate `requirements.txt` to include only the packages you actually use, with pinned versions:

```
anthropic==0.XX.X
arxiv==2.X.X
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
squarify==0.4.X
umap-learn==0.5.X
```

Also add a note at the top of requirements.txt:

```
# Pin exact versions for reproducibility
# Python 3.11 tested
# GPU required for sentence-transformers (CUDA)
```

### Step 7.2 — Add a Makefile or run script

**File:** `Makefile` (NEW)

```makefile
.PHONY: setup collect preprocess reduce cluster validate trends all

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
```

### Step 7.3 — Write a real README

**File:** `README.md`

**What to include:**

```markdown
# ArXiv AI Research Trends (2024-2026)

Unsupervised clustering analysis of 226k+ AI research papers from ArXiv
to discover and track emerging research trends.

## What This Project Does

[2-3 sentences: collects papers, embeds them, clusters, identifies trends]

## Key Findings

[3-5 bullet points of the most interesting results]

## Project Structure

notebooks/
  01_data_collection.ipynb
  02_exploratory_analysis.ipynb
  03_preprocessing.ipynb
  04_dimensionality_reduction.ipynb
  05_clustering.ipynb
  06_validation.ipynb
  07_trends_and_insights.ipynb
src/
  collect.py       — ArXiv API data collection
  preprocess.py    — text cleaning and filtering
  features.py      — embeddings, TF-IDF, UMAP
  visualize.py     — reusable plotting functions
  label.py         — cluster labeling via Claude API
data/              — gitignored, see Reproducibility
report/            — final report / figures

## Reproducibility

1. Clone this repo
2. Create a conda/venv with Python 3.11
3. `make setup`
4. Set `ANTHROPIC_API_KEY` env var (for cluster labeling)
5. `make all` (or run notebooks in order)

Note: Full pipeline takes ~4 hours (2h for KaLM embeddings, 1h for UMAP, 1h for collection).
Data collection requires internet. Embedding requires a CUDA GPU.

## Method

[Brief: 3 embeddings x 3 algorithms, UMAP reduction, HDBSCAN selected as winner, validated with ARI/NMI]
```

**Expected outcome:** Someone can clone the repo and understand what it is, what it found, and how to reproduce it.

---

## Phase 8 — Split Notebook 05 and Rewrite All Markdown

**Why this is last:** The analysis is now final. Rewriting once is better than rewriting twice.

### Step 8.1 — Split notebook 05 into 05, 06, 07

**Current state:** Notebook 05 contains clustering + evaluation + labeling + trend analysis + recommendations.

**Target state:**

- **`05_clustering.ipynb`** — Steps 1-4 of the current notebook 05:
  - K-Means (elbow, silhouette, fine-grained search, final fit, 2D/3D viz)
  - GMM (BIC, silhouette, fine-grained search, final fit, 2D/3D viz)
  - HDBSCAN (min_cluster_size sweep, final fit, 2D/3D viz)
  - Evaluation comparison table (the 9-combination table)
  - Best combination selection + Claude labeling
  - Top TF-IDF terms per cluster

- **`06_validation.ipynb`** — The new notebook from Phase 3-4:
  - Noise investigation
  - ARI/NMI vs ArXiv categories
  - Cross-embedding agreement
  - UMAP dimensionality experiment results (reference Phase 1)

- **`07_trends_and_insights.ipynb`** — Steps 7-8 of current notebook 05:
  - Growth analysis (with regression instead of naive split)
  - Sensitivity analysis
  - Seasonality check
  - Growth bar charts and bubble plot
  - Strategic recommendations (now evidence-backed)
  - Final narrative summary

### Step 8.2 — Rewrite ALL markdown cells across ALL notebooks

**Files:** All notebooks (01 through 07)

**Style guide for the rewrite:**

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
   - After: "## K-Means — Starting Simple\n\nI'm starting with K-Means because it's the simplest baseline and everyone understands it. If K-Means already gives clean clusters, there's no need for fancier methods."

5. **Comment on unexpected results:**
   - "I didn't expect KaLM to only marginally beat MiniLM here — KaLM is a much larger model (896d vs 384d) and took 2 hours to embed vs 5 minutes. The small gap suggests that for broad topic clustering, you don't necessarily need the biggest model."

6. **Keep it concise.** Personal doesn't mean verbose. Short, opinionated sentences are better than long hedged paragraphs.

7. **Don't over-explain obvious things.** Skip generic textbook definitions ("K-Means is an algorithm that..."). Your reader knows what K-Means is.

8. **End each notebook with a 3-5 line "What I learned" section** instead of a dry summary table.

**Go through each notebook one by one.** For each markdown cell, ask: "Would I actually say this to a person?" If not, rewrite it.

### Step 8.3 — Clean up duplicated/empty cells

**Files:** All notebooks

- Remove the duplicated "Step 5 — Select Best Combinations" cell in notebook 05
- Remove empty cells at the end of notebooks
- Remove the duplicated "## Summary" heading at the end of notebook 05
- Check for any `import importlib; importlib.reload(...)` cells left from development — remove them or add a comment explaining they're for development only

---

## Checklist

Use this to track progress:

```
Phase 1 — UMAP Dimension Experiment
  [ ] 1.1  Add 30d and 40d UMAP to notebook 04
  [ ] 1.2  Compare clustering quality across 20d, 30d, 40d

Phase 2 — Re-cluster
  [ ] 2.1  Final clustering with winning dimensionality
  [ ] 2.2  Re-label clusters if changed

Phase 3 — Noise Investigation
  [ ] 3.1  Create notebook 06_validation.ipynb
  [ ] 3.2  Noise composition analysis (categories, time, terms)
  [ ] 3.3  Sub-clustering the noise

Phase 4 — External Validation
  [ ] 4.1  ARI/NMI vs ArXiv categories + heatmap
  [ ] 4.2  Cross-embedding agreement matrix

Phase 5 — Stress-Test Recommendations
  [ ] 5.1  Linear regression growth with p-values
  [ ] 5.2  Sensitivity analysis (3 different time windows)
  [ ] 5.3  Seasonality check (normalize by total volume)

Phase 6 — Refactor src/
  [ ] 6.1  Fill in src/preprocess.py
  [ ] 6.2  Fill in src/features.py
  [ ] 6.3  Fill in src/visualize.py
  [ ] 6.4  Update notebooks to import from src/

Phase 7 — Reproducibility
  [ ] 7.1  Pin requirements.txt
  [ ] 7.2  Add Makefile
  [ ] 7.3  Write real README

Phase 8 — Presentation
  [ ] 8.1  Split notebook 05 into 05, 06, 07
  [ ] 8.2  Rewrite all markdown cells (personal voice)
  [ ] 8.3  Clean up duplicated/empty cells
```
