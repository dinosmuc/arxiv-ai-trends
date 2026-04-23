"""Smoke tests for feature extraction utilities."""

import numpy as np
import pandas as pd

from src.features import get_cluster_top_terms


def test_top_terms_returns_one_entry_per_cluster():
    df = pd.DataFrame(
        {
            "abstract": [
                "cat dog cat dog cat dog",
                "cat dog cat dog cat dog",
                "bird fish bird fish bird fish",
                "bird fish bird fish bird fish",
            ]
        }
    )
    labels = np.array([0, 0, 1, 1])
    out = get_cluster_top_terms(df, labels, n_terms=2)
    assert set(out.keys()) == {0, 1}


def test_top_terms_skips_noise_label():
    df = pd.DataFrame(
        {
            "abstract": [
                "cat dog cat dog cat dog",
                "bird fish bird fish bird fish",
                "noise word noise word",
            ]
        }
    )
    labels = np.array([0, 1, -1])
    out = get_cluster_top_terms(df, labels, n_terms=2)
    assert -1 not in out
    assert set(out.keys()) == {0, 1}
