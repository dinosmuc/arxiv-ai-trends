"""Smoke tests for ArXiv collection utilities."""

from datetime import datetime, timezone

from src.collect import CATEGORIES, get_monthly_windows, make_query


def test_half_month_windows_for_one_month():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 2, 1, tzinfo=timezone.utc)
    assert len(get_monthly_windows(start, end)) == 2


def test_half_month_windows_for_full_year():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert len(get_monthly_windows(start, end)) == 24


def test_half_month_windows_match_dossier_count():
    """Project dossier claims 52 windows for Jan 2024 - Feb 27 2026."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 2, 27, tzinfo=timezone.utc)
    assert len(get_monthly_windows(start, end)) == 52


def test_query_contains_all_categories():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 16, tzinfo=timezone.utc)
    query = make_query(start, end)
    for cat in CATEGORIES:
        assert f"cat:{cat}" in query


def test_query_contains_date_range():
    start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 16, 0, 0, tzinfo=timezone.utc)
    query = make_query(start, end)
    assert "202401010000" in query
    assert "202401160000" in query
