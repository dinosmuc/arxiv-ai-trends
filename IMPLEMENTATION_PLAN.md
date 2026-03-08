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
```

---

## Phase 0 — Preprocessing Improvement (Title + Abstract Embeddings)

**Why this is Phase 0:** This changes the INPUT to all embeddings, which is the most upstream change possible. Titles are often more discriminative than abstracts alone — a title like "DPO-Augmented RLHF for Code Generation" carries dense topic signal that might be diluted in a 200-word abstract. This is a small change that could meaningfully improve every downstream result.

### Step 0.1 — Add title+abstract embedding input to notebook 03

**File:** `notebooks/03_preprocessing.ipynb`

**What to do:**

After the existing `abstract_clean` column is created, add a new column that concatenates title and abstract:

```python
df["embed_text"] = df["title"] + ". " + df["abstract_clean"]
print(f"Sample:\n{df['embed_text'].iloc[0][:200]}...")
```

Save this column in the output CSV (`arxiv_clean.csv`). The `abstract_lemma` column stays unchanged — TF-IDF still uses the lemmatized abstract only, since titles are short and would add noise to bag-of-words.

### Step 0.2 — Re-embed with title+abstract

**File:** `notebooks/03_preprocessing.ipynb`

**What to do:**

Change the MiniLM and KaLM embedding cells to use `embed_text` instead of `abstract_clean`:

```python
# MiniLM
minilm_embeddings = model_mini.encode(
    df["embed_text"].tolist(),  # was abstract_clean
    batch_size=256,
    show_progress_bar=True,
)

# KaLM
kalm_embeddings = model_kalm.encode(
    df["embed_text"].tolist(),  # was abstract_clean
    batch_size=64,
    show_progress_bar=True,
)
```

Save as the same filenames (`minilm_embeddings.npy`, `kalm_embeddings.npy`). The old files get overwritten.

**Important:** TF-IDF still uses `abstract_lemma`, NOT `embed_text`. TF-IDF on raw titles would just add noise (short text, no lemmatization). Only neural embeddings benefit from title+abstract.

**Note:** KaLM embedding takes ~2 hours on GPU. MiniLM takes ~5 minutes. Plan accordingly.

### Step 0.3 — Filter out survey papers (flag, don't remove)

**File:** `notebooks/03_preprocessing.ipynb`

**What to do:**

Survey/review papers span multiple topics by definition and confuse clustering. Detect them:

```python
survey_pattern = r"\b(survey|review|overview|tutorial|systematic review|literature review|comprehensive review|state of the art|sota review)\b"
df["is_survey"] = df["title"].str.contains(survey_pattern, case=False, na=False)
print(f"Survey papers: {df['is_survey'].sum():,} ({df['is_survey'].mean():.1%})")
```

**Do NOT remove them** from the dataset. Instead, flag them so you can:
1. In notebook 05: compare clustering quality with and without surveys (run HDBSCAN on `df[~df['is_survey']]` and see if noise drops)
2. In notebook 06: check if surveys are disproportionately in the noise category
3. In notebook 07: exclude surveys from trend counts if they distort growth rates

Save the `is_survey` column in `arxiv_clean.csv`.

**Expected outcome:** Embeddings now use richer input (title+abstract), and surveys are flagged for later analysis.

---

## Phase 1 — UMAP Hyperparameter & Dimension Experiment

**Why this is Phase 1:** UMAP settings are the most upstream change after embeddings. If different parameters give better clusters, it changes ALL downstream analysis. Do this before clustering so you only cluster once with the best settings.

### Step 1.1 — Add 30d and 40d UMAP to notebook 04

**File:** `notebooks/04_dimensionality_reduction.ipynb`

**What to do:**

After the existing UMAP 20d cell (the one that produces `tfidf_umap20`, `minilm_umap20`, `kalm_umap20`), add new cells that run UMAP at 30d and 40d for all three embeddings. Use the same parameters as existing 20d (`n_neighbors=15`, `min_dist=0.0`, `random_state=42`).

The inputs are:
- `tfidf_svd` (the 500d SVD-reduced TF-IDF matrix)
- `minilm_emb` (raw 384d MiniLM embeddings — now re-embedded with title+abstract from Phase 0)
- `kalm_emb` (raw 896d KaLM embeddings — now re-embedded with title+abstract from Phase 0)

Generate and save to `data/processed/`:
- `tfidf_umap30.npy`, `minilm_umap30.npy`, `kalm_umap30.npy`
- `tfidf_umap40.npy`, `minilm_umap40.npy`, `kalm_umap40.npy`

**Important:** Keep the existing 20d, 2d, and 3d UMAP outputs. Regenerate the 20d ones too since the embeddings changed in Phase 0. The 2d and 3d are for visualization only and don't need 30d/40d variants.

### Step 1.2 — UMAP n_neighbors sweep

**File:** `notebooks/04_dimensionality_reduction.ipynb`

**What to do:**

You currently fixed `n_neighbors=15` everywhere. With 181k papers, this is quite local — UMAP only looks at each point's 15 nearest neighbors to build the graph. Higher values capture more global structure.

Run a sweep on KaLM embeddings only (to save time), at the winning dimensionality from Step 1.1 (or just use 20d for now):

```python
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

results = []
for n_neighbors in [15, 30, 50]:
    for min_dist in [0.0, 0.05]:
        reducer = umap.UMAP(
            n_components=20,  # or whatever dim you're testing
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
        )
        reduced = reducer.fit_transform(kalm_emb)

        # Quick K-Means eval to compare
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

This is 6 runs. Pick the best `n_neighbors`/`min_dist` combo. If `n_neighbors=30` or `50` wins, regenerate ALL UMAP outputs (20d, 30d, 40d, 2d, 3d) with the winning parameters.

**Note:** Each UMAP run on 181k x 896d takes ~15-30 minutes. The full sweep is ~2-3 hours. If time is tight, just test `n_neighbors` in [15, 30] with `min_dist=0.0` (2 runs).

### Step 1.3 — Compare clustering quality across dimensions

**File:** `notebooks/04_dimensionality_reduction.ipynb` (add at the end)

**What to do:**

Run HDBSCAN (`min_cluster_size=1000`, `min_samples=10`) on KaLM embeddings at all three dimensionalities (20d, 30d, 40d), using the best `n_neighbors`/`min_dist` from Step 1.2. For each, compute:
- Number of clusters
- Noise percentage
- Silhouette score (on non-noise points, `sample_size=10000`)
- Davies-Bouldin index (on non-noise points)

Also run K-Means (k=24) on each dimensionality and compare. This gives you a 2x3 grid (2 algorithms x 3 dimensionalities).

Print a comparison table:

```
Dims  Algorithm  Clusters  Noise%  Silhouette  Davies-Bouldin
20    HDBSCAN    41        36.9%   0.5073      0.7358
30    HDBSCAN    ??        ??      ??          ??
40    HDBSCAN    ??        ??      ??          ??
20    K-Means    24        0%      0.3936      0.9448
30    K-Means    24        0%      ??          ??
40    K-Means    24        0%      ??          ??
```

**Decision:** Pick the dimensionality that gives the best silhouette/DB tradeoff. Add a markdown cell documenting your decision and reasoning. If 20d was already best, great — you've now proven it empirically rather than assumed it.

**Expected outcome:** You know the optimal UMAP dimensionality AND the optimal `n_neighbors`/`min_dist`, and have the data to justify both choices.

---

## Phase 2 — Final Clustering (Full 12-Combination Comparison)

**Why 12 now instead of 9:** You're adding BERTopic as a fourth algorithm. BERTopic is the state-of-the-art for exactly this task — neural topic modeling. It uses HDBSCAN internally but adds class-based TF-IDF (c-TF-IDF) for automatic topic labeling and has built-in topic evolution tracking.

### Step 2.1 — Add BERTopic to notebook 05

**File:** `notebooks/05_clustering_and_results.ipynb`

**What to do:**

First, add `bertopic` to `requirements.txt`.

After the existing HDBSCAN section, add a new section for BERTopic. BERTopic wraps UMAP + HDBSCAN + c-TF-IDF into one pipeline, but you can pass your own pre-computed components:

```python
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

# Use your pre-computed UMAP and let BERTopic do HDBSCAN + c-TF-IDF
for name in ["TF-IDF", "MiniLM", "KaLM"]:
    if name == "TF-IDF":
        emb = tfidf_svd  # 500d SVD-reduced
    elif name == "MiniLM":
        emb = minilm_emb
    else:
        emb = kalm_emb

    # Let BERTopic use your UMAP settings but handle clustering internally
    umap_model = UMAP(
        n_components=WINNING_DIM,  # from Phase 1
        n_neighbors=WINNING_NEIGHBORS,  # from Phase 1
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

    # BERTopic gives you automatic topic labels via c-TF-IDF
    print(topic_model.get_topic_info().head(15))
```

**Key BERTopic features to show:**
- `topic_model.get_topic_info()` — automatic topic labels with representative words
- `topic_model.visualize_topics()` — interactive topic similarity map
- `topic_model.visualize_hierarchy()` — hierarchical topic structure (dendrogram)
- `topic_model.topics_over_time(df["abstract_clean"].tolist(), df["published"].tolist())` — topic evolution chart

These are built-in and require zero extra code. They give you automatic topic labels (no Claude API needed for BERTopic clusters), topic evolution visualization, and hierarchical topic structure — all for free.

### Step 2.2 — Full comparison table (12 combinations)

**File:** `notebooks/05_clustering_and_results.ipynb`

**What to do:**

Update the evaluation comparison to include BERTopic. You now have 3 embeddings x 4 algorithms = 12 combinations. Compute silhouette, Davies-Bouldin, and Calinski-Harabasz for all 12. For HDBSCAN and BERTopic, compute metrics on non-noise points only.

Also add **Topic Coherence (c_v score)** as a fourth metric. Silhouette measures geometric cluster quality; topic coherence measures whether the top words in each cluster actually co-occur in real text — it's more meaningful for topic models.

```python
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

def compute_coherence(df, labels, text_col="abstract", n_terms=10):
    """Compute c_v topic coherence for a clustering."""
    # Tokenize
    texts = [doc.split() for doc in df[text_col]]
    dictionary = Dictionary(texts)

    # Get top terms per cluster
    topics = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        mask = labels == label
        cluster_text = " ".join(df.loc[mask, text_col].tolist())
        # Use TF-IDF to get top terms
        from sklearn.feature_extraction.text import TfidfVectorizer
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

Add `gensim` to `requirements.txt`.

The final comparison table should look like:

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

### Step 2.3 — Select best combination and re-label

**File:** `notebooks/05_clustering_and_results.ipynb`, `src/label.py`

**What to do:**

Based on the 12-combination table, select the best primary and secondary clustering. Write a markdown cell explaining the decision.

If BERTopic wins, note that it provides its own automatic labels via c-TF-IDF — you still run Claude Opus labeling for richer descriptions, but you can compare BERTopic's auto-labels vs Claude's labels as a sanity check.

If HDBSCAN is still the winner, re-run Claude Opus labeling (the clusters likely changed because embeddings changed in Phase 0). Use the existing `src/label.py` module.

Compare old labels vs new labels in a markdown cell. Note which clusters merged, split, or shifted.

### Step 2.4 — Hierarchical topic structure

**File:** `notebooks/05_clustering_and_results.ipynb`

**What to do:**

After final clustering, build a dendrogram showing how topics relate to each other. This turns your flat list of 41 topics into a taxonomy of AI research.

```python
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

# Compute centroid of each cluster in the UMAP embedding space
centroids = []
cluster_names = []
for label in sorted(set(final_labels)):
    if label == -1:
        continue
    mask = final_labels == label
    centroids.append(kalm_umap_final[mask].mean(axis=0))
    cluster_names.append(label_info[label]["label"])

centroids = np.array(centroids)

# Hierarchical clustering on centroids
Z = linkage(centroids, method="ward")

fig, ax = plt.subplots(figsize=(16, 10))
dendrogram(
    Z,
    labels=cluster_names,
    leaf_rotation=90,
    leaf_font_size=8,
    ax=ax,
)
ax.set_title("Hierarchical Structure of AI Research Topics")
ax.set_ylabel("Ward Distance")
plt.tight_layout()
plt.show()
```

This shows, for example, that "LLM Reasoning" and "LLM Agents" are siblings, which validates the clustering semantically. If unrelated topics are siblings, something is wrong.

If you used BERTopic, you get this for free with `topic_model.visualize_hierarchy()`.

**Expected outcome:** You have a complete, well-justified clustering with 4 algorithms compared, topic coherence measured, and a hierarchical topic structure visualized.

---

## Phase 3 — Noise Investigation

**Why:** 37% of papers are labeled as noise. Right now you just assert "it's a feature." This phase turns that into a demonstrated finding.

### Step 3.1 — Create notebook 06_validation.ipynb

**File:** `notebooks/06_validation.ipynb` (NEW)

**What to do:**

Create a new notebook. Start with a markdown cell:

> # 06 — Validation & Robustness
>
> I need to prove that my clusters are trustworthy. This notebook investigates the noise rate, validates clusters against external labels, and checks whether different embeddings agree.

Load the cleaned dataset (`arxiv_clean.csv`), the final HDBSCAN labels (KaLM), the K-Means labels, the BERTopic labels, and the ArXiv primary categories. Also load all the label JSONs.

### Step 3.2 — Noise composition analysis

**File:** `notebooks/06_validation.ipynb`

**What to do:**

Take all papers where `hdbscan_label == -1` (the noise papers). Analyze them:

1. **Category distribution of noise vs non-noise:**
   ```python
   noise_cats = df[df["cluster"] == -1]["primary_category"].value_counts(normalize=True)
   clean_cats = df[df["cluster"] != -1]["primary_category"].value_counts(normalize=True)
   ```
   Plot these side by side as horizontal bar charts. Are certain categories (like cs.AI, which is broad and interdisciplinary) over-represented in noise?

2. **Noise rate over time:**
   ```python
   monthly_noise_rate = df.groupby("month").apply(lambda x: (x["cluster"] == -1).mean())
   ```
   Plot this as a line chart. Is noise growing (meaning new topics are emerging that don't fit existing clusters) or stable?

3. **Top TF-IDF terms in noise papers:**
   Run TF-IDF on just the noise papers' abstracts. What are the top 30 terms? Are they generic (suggesting truly diffuse papers) or do they hint at hidden topics?

4. **Key term frequency in noise vs clustered:**
   For each of the key terms from notebook 02 (LLM, diffusion, transformer, agent, multimodal, RAG, fine-tuning, reasoning, reinforcement learning, robot), compute what fraction of papers mentioning that term ended up as noise. Some terms might have disproportionately high noise rates, revealing which topics HDBSCAN struggles with.

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
   Using the `is_survey` flag from Phase 0:
   ```python
   survey_noise_rate = df[df["is_survey"]]["cluster"].eq(-1).mean()
   non_survey_noise_rate = df[~df["is_survey"]]["cluster"].eq(-1).mean()
   print(f"Survey papers noise rate: {survey_noise_rate:.1%}")
   print(f"Non-survey papers noise rate: {non_survey_noise_rate:.1%}")
   ```
   If surveys have a much higher noise rate, that explains part of the 37%.

### Step 3.3 — Sub-clustering the noise

**File:** `notebooks/06_validation.ipynb`

**What to do:**

Run HDBSCAN with smaller `min_cluster_size` values on ONLY the noise papers' KaLM UMAP embeddings. This tests whether the noise contains coherent sub-groups that were just below the `min_cluster_size=1000` threshold.

```python
noise_mask = hdbscan_labels == -1
noise_embeddings = kalm_umap_final[noise_mask]  # use whatever dim won in Phase 1

for ms in [200, 300, 500]:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=ms, min_samples=10)
    sub_labels = clusterer.fit_predict(noise_embeddings)
    n_sub = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
    still_noise = (sub_labels == -1).mean() * 100
    print(f"min_size={ms}: {n_sub} sub-clusters, {still_noise:.1f}% still noise")
```

If coherent sub-clusters emerge, extract their top TF-IDF terms using `get_cluster_top_terms` from `src/features.py`. Show the top 5-7 terms for each sub-cluster. This tells you whether `min_cluster_size=1000` was too aggressive or whether the noise is genuinely diffuse.

Write a markdown conclusion: "X% of noise papers form coherent sub-topics below my size threshold (topics like [list examples]). The remaining Y% are genuinely interdisciplinary papers that span multiple research areas."

**Expected outcome:** You can now make an evidence-based statement about your noise rate.

---

## Phase 4 — External Validation

### Step 4.1 — Cluster agreement with ArXiv categories (ARI/NMI)

**File:** `notebooks/06_validation.ipynb`

**What to do:**

Compute Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) between your cluster labels and ArXiv's `primary_category`. Do this for all four algorithms.

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# HDBSCAN (non-noise only)
mask = hdbscan_labels != -1
ari_hdbscan = adjusted_rand_score(df.loc[mask, "primary_category"], hdbscan_labels[mask])
nmi_hdbscan = normalized_mutual_info_score(df.loc[mask, "primary_category"], hdbscan_labels[mask])

# K-Means (all papers)
ari_kmeans = adjusted_rand_score(df["primary_category"], kmeans_labels)
nmi_kmeans = normalized_mutual_info_score(df["primary_category"], kmeans_labels)

# GMM (all papers)
ari_gmm = adjusted_rand_score(df["primary_category"], gmm_labels)
nmi_gmm = normalized_mutual_info_score(df["primary_category"], gmm_labels)

# BERTopic (non-noise only)
bt_mask = np.array(bertopic_labels) != -1
ari_bt = adjusted_rand_score(df.loc[bt_mask, "primary_category"], np.array(bertopic_labels)[bt_mask])
nmi_bt = normalized_mutual_info_score(df.loc[bt_mask, "primary_category"], np.array(bertopic_labels)[bt_mask])

print(f"{'Algorithm':<12} {'ARI':>8} {'NMI':>8}")
print("-" * 30)
print(f"{'HDBSCAN':<12} {ari_hdbscan:>8.4f} {nmi_hdbscan:>8.4f}")
print(f"{'K-Means':<12} {ari_kmeans:>8.4f} {nmi_kmeans:>8.4f}")
print(f"{'GMM':<12} {ari_gmm:>8.4f} {nmi_gmm:>8.4f}")
print(f"{'BERTopic':<12} {ari_bt:>8.4f} {nmi_bt:>8.4f}")
```

**Interpretation guidance for the markdown:**
- ARI ranges from -1 to 1. Values around 0.05-0.15 are expected because your clusters are FINER-GRAINED than ArXiv categories (41 clusters vs 8 categories). Low ARI doesn't mean bad clusters — it means you discovered sub-structure within categories. Say this explicitly.
- NMI ranges from 0 to 1. Values around 0.2-0.4 would suggest meaningful alignment. If NMI is very low (<0.1), investigate why.
- Higher NMI for your chosen method vs alternatives = additional evidence for your choice.

Also create a **confusion-style heatmap**: for each cluster, show the distribution of ArXiv categories. This visualizes how your unsupervised clusters relate to human-assigned categories.

```python
import seaborn as sns

ct = pd.crosstab(
    df.loc[mask, "cluster_name"],
    df.loc[mask, "primary_category"],
    normalize="index"  # row-normalize so each cluster sums to 1
)

fig, ax = plt.subplots(figsize=(12, 16))
sns.heatmap(ct, cmap="Blues", ax=ax, annot=True, fmt=".2f")
ax.set_title("Cluster Composition by ArXiv Category")
plt.tight_layout()
plt.show()
```

This shows, for example, that your "Robot Learning" cluster is 70% cs.RO + 20% cs.AI, which validates it. If a cluster is evenly spread across all 8 categories, it's suspicious — flag it.

### Step 4.2 — Cross-embedding agreement (ARI/NMI between embeddings)

**File:** `notebooks/06_validation.ipynb`

**What to do:**

This is the missing piece in your embedding comparison. Instead of just comparing silhouette scores, check whether different embeddings discover the SAME structure.

Compute ARI and NMI between:
- KaLM-HDBSCAN vs MiniLM-HDBSCAN (both non-noise only, on their intersection)
- KaLM-HDBSCAN vs TF-IDF-HDBSCAN
- MiniLM-HDBSCAN vs TF-IDF-HDBSCAN
- KaLM-KMeans vs MiniLM-KMeans
- KaLM-KMeans vs TF-IDF-KMeans
- MiniLM-KMeans vs TF-IDF-KMeans

For HDBSCAN comparisons, only include papers that are non-noise in BOTH clusterings:

```python
# Example for HDBSCAN
both_non_noise = (hdbscan_kalm != -1) & (hdbscan_minilm != -1)
ari = adjusted_rand_score(hdbscan_kalm[both_non_noise], hdbscan_minilm[both_non_noise])
nmi = normalized_mutual_info_score(hdbscan_kalm[both_non_noise], hdbscan_minilm[both_non_noise])
```

Present as two matrices (one for ARI, one for NMI), one per algorithm:

```
HDBSCAN ARI:
           KaLM    MiniLM   TF-IDF
KaLM       1.000   ?.???    ?.???
MiniLM     ?.???   1.000    ?.???
TF-IDF     ?.???   ?.???    1.000

K-Means ARI:
           KaLM    MiniLM   TF-IDF
KaLM       1.000   ?.???    ?.???
MiniLM     ?.???   1.000    ?.???
TF-IDF     ?.???   ?.???    1.000
```

**What to expect and how to interpret:**
- High ARI (>0.3) between KaLM and MiniLM = neural embeddings agree, cluster structure is real and not an artifact of one model.
- Lower ARI with TF-IDF = expected, because TF-IDF captures lexical patterns while neural models capture semantics.
- If KaLM and MiniLM disagree strongly (ARI < 0.1), that's a red flag — cluster assignments depend heavily on embedding choice, and conclusions are less robust. Report this honestly.

**Expected outcome:** You can now say "my clusters are validated against external labels and robust across embedding methods" (or honestly report where they're not).

---

## Phase 5 — Stress-Test Business Recommendations

### Step 5.1 — Replace naive growth calculation with regression

**File:** `notebooks/07_trends_and_insights.ipynb` (will be created by splitting current notebook 05)

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
        "Significant (p<0.05)": p_value < 0.05,
    })

growth_df = pd.DataFrame(growth_stats).sort_values("Slope (papers/month)", ascending=False)
print(growth_df.to_string(index=False))
```

This gives you:
- A slope with a p-value (is the growth statistically significant?)
- R-squared (is the trend linear, or is it noisy?)
- Monthly growth percentage that doesn't depend on an arbitrary split

**Important:** Keep the old first-half vs last-half calculation too, for comparison. Show both in the notebook and discuss which is more reliable.

### Step 5.2 — Sensitivity analysis with different time windows

**File:** `notebooks/07_trends_and_insights.ipynb`

**What to do:**

Run the old first-half vs last-half comparison with THREE different splits:

```python
splits = {
    "6-month": (months[:6], months[-6:]),
    "8-month": (months[:8], months[-8:]),
    "thirds": (months[:len(months)//3], months[-len(months)//3:]),
}

sensitivity_results = {}
for split_name, (early, late) in splits.items():
    early_avg = monthly_clusters.loc[early].mean()
    late_avg = monthly_clusters.loc[late].mean()
    growth = ((late_avg - early_avg) / early_avg * 100)
    sensitivity_results[split_name] = growth

# Show top 10 across all splits
sensitivity_df = pd.DataFrame(sensitivity_results)
print(sensitivity_df.sort_values("6-month", ascending=False).head(10))
```

Present top-10 clusters across all splits side by side. If "LLM Reasoning" shows +415% with one split but +180% with another, REPORT BOTH. The conclusion ("LLM Reasoning is growing fast") is still valid, but the exact number is less meaningful.

Add a markdown cell: "The exact growth percentages vary depending on the time window, but the ranking of fastest-growing areas is consistent across all three splits: [list]. This means the direction is robust even if the magnitude is approximate."

### Step 5.3 — Seasonality check

**File:** `notebooks/07_trends_and_insights.ipynb`

**What to do:**

ArXiv submissions spike around major conference deadlines (NeurIPS ~May, ICML ~Jan, CVPR ~Nov, etc.). Check if your growth trends are confounded by seasonal patterns.

```python
# Total monthly submissions
total_monthly = df.groupby("month").size()

# Normalize each cluster by total monthly volume (= share of total)
normalized = monthly_clusters.div(total_monthly, axis=0) * 100  # as percentage

# Re-compute regression on normalized (share-based) data
share_growth = []
for cluster_name in normalized.columns:
    y = normalized[cluster_name].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    share_growth.append({
        "Cluster": cluster_name,
        "Absolute Growth (papers/month)": growth_df[growth_df["Cluster"] == cluster_name]["Slope (papers/month)"].values[0],
        "Share Growth (%/month)": round(slope, 4),
        "Share p-value": round(p_value, 6),
    })

share_df = pd.DataFrame(share_growth).sort_values("Share Growth (%/month)", ascending=False)
```

If a cluster's growth disappears after normalization (positive absolute slope but near-zero or negative share slope), it was just riding the overall ArXiv growth wave, not actually gaining share. Report which clusters are growing in BOTH absolute terms AND share.

Plot a side-by-side bar chart: absolute growth rank vs share growth rank.

**Expected outcome:** Your business recommendations now have statistical significance, sensitivity analysis, and share-based growth. Much harder to criticize.

---

## Phase 6 — Refactor src/ Modules

**Why this is Phase 6:** You need stable analysis code before extracting it into modules. Refactoring code that's still changing is wasted effort.

### Step 6.1 — Fill in src/preprocess.py

**File:** `src/preprocess.py`

**What to extract from notebook 03:**

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

### Step 6.2 — Fill in src/features.py

**File:** `src/features.py`

**What to extract from notebooks 03 and 04:**

```python
"""Feature extraction: TF-IDF, embeddings, dimensionality reduction."""

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

### Step 6.4 — Update notebooks to import from src/

**Files:** All notebooks (02 through 07)

**What to do:**

Replace inline code with imports. For example, in notebook 05, replace the inline `get_cluster_top_terms` definition with:

```python
from src.features import get_cluster_top_terms, compute_coherence
```

In notebook 03, replace the inline preprocessing with:

```python
from src.preprocess import filter_papers, clean_abstracts, flag_surveys, lemmatize_abstracts
```

In notebooks 05 and 07, replace repeated plotting code with:

```python
from src.visualize import plot_cluster_scatter_2d, plot_growth_bars, plot_monthly_trends
```

Keep the notebooks as thin orchestration: load data, call functions from `src/`, display results, write markdown commentary.

**Do not over-refactor.** Notebooks should still be readable top-to-bottom. If a piece of code is only used once and is short (< 10 lines), leave it inline. Only extract things that are:
- Used in multiple notebooks, OR
- Long enough (> 15 lines) that they clutter the notebook's narrative

**Expected outcome:** `src/` has five real modules (collect, preprocess, features, visualize, label). Notebooks are shorter and focused on narrative + results.

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

**Note:** Remove `sqarify` from requirements — it's unused. Remove `python-dotenv` if not used either. Add `bertopic` and `gensim` (new dependencies from Phase 2).

### Step 7.2 — Add a Makefile

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

### Step 7.3 — Write a real README

**File:** `README.md`

**What to include:**

```markdown
# ArXiv AI Research Trends (2024-2026)

Unsupervised clustering analysis of 226k+ AI research papers from ArXiv
to discover and track emerging research trends.

## What This Project Does

[2-3 sentences: collects papers, embeds them with neural models, clusters with
4 algorithms, validates clusters, and identifies which research areas are
growing or declining]

## Key Findings

[5-7 bullet points of the most interesting results — growth areas, declining
areas, the contrarian opportunity, what the noise tells us]

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

**Expected outcome:** Someone can clone the repo and understand what it is, what it found, and how to reproduce it.

---

## Phase 8 — Split Notebook 05 and Rewrite All Markdown

**Why this is last:** The analysis is now final. Rewriting once is better than rewriting twice.

### Step 8.1 — Split notebook 05 into 05, 06, 07

**Current state:** One monolith notebook `05_clustering_and_results.ipynb` containing clustering + evaluation + labeling + trend analysis + recommendations. Plus the new `06_validation.ipynb` from Phase 3-4.

**Target state:**

- **`05_clustering.ipynb`** — Method comparison only:
  - K-Means (elbow, silhouette, fine-grained search, final fit, 2D/3D viz)
  - GMM (BIC, silhouette, fine-grained search, final fit, 2D/3D viz)
  - HDBSCAN (min_cluster_size sweep, final fit, 2D/3D viz)
  - BERTopic (fit, automatic labels, hierarchy viz)
  - Full 12-combination evaluation table (with topic coherence)
  - Best combination selection + Claude labeling
  - Hierarchical topic dendrogram
  - Top TF-IDF terms per cluster

- **`06_validation.ipynb`** — Already created in Phase 3-4:
  - Noise investigation (composition, time trend, sub-clustering)
  - Survey papers in noise analysis
  - ARI/NMI vs ArXiv categories + heatmap
  - Cross-embedding agreement matrix

- **`07_trends_and_insights.ipynb`** — Findings and recommendations:
  - Growth analysis (linear regression with p-values)
  - Sensitivity analysis (3 time window splits)
  - Seasonality check (absolute vs share growth)
  - Growth bar charts, bubble plot
  - Strategic recommendations (now evidence-backed with statistical significance)
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

8. **End each notebook with a 3-5 line "What I learned" section** instead of a dry summary table. Example:
   > ## What I Learned
   >
   > The biggest surprise was how badly GMM performed — I expected the soft assignments to help with overlapping topics, but the Gaussian assumption was just wrong for this data. The other surprise was that title+abstract embeddings noticeably improved cluster separation over abstract-only, which makes intuitive sense but I hadn't seen quantified before.

9. **For each major decision, briefly state what alternatives you considered and why you rejected them.** This shows critical thinking:
   > I considered using Spectral Clustering but ruled it out — with 181k papers, the affinity matrix would be 181k x 181k, which doesn't fit in memory. HDBSCAN and BERTopic handle large datasets natively.

**Go through each notebook one by one.** For each markdown cell, ask: "Would I actually say this to a person?" If not, rewrite it.

### Step 8.3 — Clean up duplicated/empty cells

**Files:** All notebooks

- Remove the duplicated "Step 5 — Select Best Combinations" markdown cell in notebook 05
- Remove empty cells at the end of notebooks
- Remove the duplicated "## Summary" heading at the end of notebook 05
- Check for any `import importlib; importlib.reload(...)` cells left from development — remove them or add a comment explaining they're for development only
- Remove the `sqarify` import if it exists anywhere (unused dependency)

---

## Checklist

Use this to track progress:

```
Phase 0 — Preprocessing Improvement
  [ ] 0.1  Add title+abstract embedding input (embed_text column)
  [ ] 0.2  Re-embed MiniLM and KaLM with title+abstract
  [ ] 0.3  Flag survey papers (is_survey column)

Phase 1 — UMAP Hyperparameter & Dimension Experiment
  [ ] 1.1  Add 30d and 40d UMAP to notebook 04
  [ ] 1.2  Sweep n_neighbors (15, 30, 50) and min_dist (0.0, 0.05)
  [ ] 1.3  Compare clustering quality across dimensions, pick winner

Phase 2 — Final Clustering (12-Combination Comparison)
  [ ] 2.1  Add BERTopic as fourth algorithm
  [ ] 2.2  Full 12-combination comparison table (with topic coherence c_v)
  [ ] 2.3  Select best combination, re-label with Claude Opus
  [ ] 2.4  Build hierarchical topic dendrogram

Phase 3 — Noise Investigation
  [ ] 3.1  Create notebook 06_validation.ipynb
  [ ] 3.2  Noise composition analysis (categories, time, terms, surveys)
  [ ] 3.3  Sub-clustering the noise (min_cluster_size 200/300/500)

Phase 4 — External Validation
  [ ] 4.1  ARI/NMI vs ArXiv categories + confusion heatmap
  [ ] 4.2  Cross-embedding agreement matrix (ARI/NMI between embeddings)

Phase 5 — Stress-Test Recommendations
  [ ] 5.1  Linear regression growth with p-values and R-squared
  [ ] 5.2  Sensitivity analysis (3 different time window splits)
  [ ] 5.3  Seasonality check (absolute vs share-of-total growth)

Phase 6 — Refactor src/
  [ ] 6.1  Fill in src/preprocess.py
  [ ] 6.2  Fill in src/features.py (including compute_coherence)
  [ ] 6.3  Fill in src/visualize.py
  [ ] 6.4  Update notebooks to import from src/

Phase 7 — Reproducibility
  [ ] 7.1  Pin requirements.txt (add bertopic, gensim; remove sqarify)
  [ ] 7.2  Add Makefile
  [ ] 7.3  Write real README

Phase 8 — Presentation
  [ ] 8.1  Split notebook 05 into 05, 06, 07
  [ ] 8.2  Rewrite all markdown cells (personal voice, show thinking)
  [ ] 8.3  Clean up duplicated/empty cells
```
