"""Feature extraction: TF-IDF, embeddings, UMAP, and cluster evaluation."""

import logging

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
    """Reduce sparse TF-IDF matrix with TruncatedSVD."""
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    reduced = svd.fit_transform(sparse_matrix)
    variance = svd.explained_variance_ratio_.sum()
    logger.info(
        f"SVD: {sparse_matrix.shape[1]}d -> {n_components}d, variance: {variance:.1%}"
    )
    return reduced, svd


def compute_embeddings(texts, model_name, device="cuda", batch_size=256):
    """Compute sentence embeddings with a SentenceTransformer model."""
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
    """Extract top TF-IDF terms per cluster (skips noise label -1)."""
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
        cluster_terms[label] = [(terms[j], float(scores[j])) for j in top_idx]

    return cluster_terms


def compute_coherence(df, labels, text_col="abstract", n_terms=10):
    """
    Compute c_v topic coherence for a clustering.

    Uses gensim's CoherenceModel. Each cluster's top TF-IDF terms
    are treated as a "topic", and coherence measures how often
    those terms co-occur in the corpus.

    Returns a single float (higher = more coherent topics).
    """
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel

    # Build corpus for gensim
    texts_tokenized = [doc.split() for doc in df[text_col]]
    dictionary = Dictionary(texts_tokenized)

    # Get top terms per cluster as "topics"
    topics = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        mask = labels == label
        cluster_text = " ".join(df.loc[mask, text_col].tolist())

        vec = TfidfVectorizer(max_features=10000, stop_words="english")
        matrix = vec.fit_transform([cluster_text])
        terms = vec.get_feature_names_out()
        scores = matrix.toarray().flatten()

        top_idx = scores.argsort()[-n_terms:][::-1]
        topics.append([terms[j] for j in top_idx])

    cm = CoherenceModel(
        topics=topics,
        texts=texts_tokenized,
        dictionary=dictionary,
        coherence="c_v",
    )
    score = cm.get_coherence()
    logger.info(f"Topic coherence (c_v): {score:.4f} ({len(topics)} topics)")
    return score
