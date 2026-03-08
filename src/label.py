"""
Label clusters using Claude Opus 4.6 API with structured outputs.
"""

import json
import logging
import time

import anthropic
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-6"
SAMPLE_SIZE = 20


class ClusterLabel(BaseModel):
    """Structured output schema for cluster labeling."""

    label: str
    description: str
    subtopics: list[str]


def build_prompt(abstracts):
    """Build the labeling prompt for a cluster."""
    abstracts_text = "\n\n---\n\n".join(
        f"Abstract {i + 1}: {a}" for i, a in enumerate(abstracts)
    )

    return (
        f"You are an expert in AI research. Below are {len(abstracts)} "
        f"abstracts from a cluster of AI research papers discovered by "
        f"unsupervised clustering. All papers in this cluster were grouped "
        f"together because they are semantically similar.\n\n"
        f"{abstracts_text}\n\n"
        f"Based on these abstracts, provide:\n"
        f"1. A short label (3-6 words) that captures the core research topic\n"
        f"2. A one-sentence description of the research theme\n"
        f"3. A list of 3-5 key sub-topics within this cluster"
    )


def label_one_cluster(client, abstracts):
    """Send abstracts to Claude and get a structured label."""
    prompt = build_prompt(abstracts)

    response = client.messages.parse(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        output_format=ClusterLabel,
    )

    return response.parsed_output


def label_clusters(df, labels, n_samples=SAMPLE_SIZE, api_key=None):
    """
    Label all clusters using Claude Opus.

    Args:
        df: DataFrame with 'abstract' column
        labels: array of cluster labels (-1 = noise, skipped)
        n_samples: number of abstracts to sample per cluster
        api_key: Anthropic API key

    Returns:
        dict mapping cluster_id to {label, description, subtopics}
    """
    client = anthropic.Anthropic(api_key=api_key)

    unique_labels = sorted(set(labels))
    unique_labels = [lbl for lbl in unique_labels if lbl != -1]

    logger.info(f"Labeling {len(unique_labels)} clusters with Claude Opus")

    results = {}
    for i, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        cluster_df = df[mask]

        # Sample abstracts
        if len(cluster_df) > n_samples:
            sample = cluster_df.sample(n_samples, random_state=42)
        else:
            sample = cluster_df

        abstracts = sample["abstract"].tolist()

        try:
            result = label_one_cluster(client, abstracts)
            results[cluster_id] = {
                "label": result.label,
                "description": result.description,
                "subtopics": result.subtopics,
            }
            logger.info(
                f"  [{i + 1}/{len(unique_labels)}] "
                f"Cluster {cluster_id} ({mask.sum():,} papers): "
                f"{result.label}"
            )
        except Exception as e:
            logger.error(f"  Cluster {cluster_id}: failed — {e}")
            results[cluster_id] = {
                "label": f"Cluster {cluster_id}",
                "description": "Labeling failed",
                "subtopics": [],
            }

        # Rate limiting
        time.sleep(1)

    return results


def save_labels(results, path):
    """Save labels to JSON."""
    # Convert numpy int keys to Python int
    clean = {int(k): v for k, v in results.items()}
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    logger.info(f"Saved labels to {path}")


def load_labels(path):
    """Load labels from JSON."""
    with open(path) as f:
        raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
