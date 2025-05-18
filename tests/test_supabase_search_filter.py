# tests/test_supabase_search_filter.py

# Was dieser Test macht:
# Simuliert ein Suchergebnis mit drei Dokumenten
# Nur ein Dokument (das zur Frage passt) hat similarity >= 0.90
# Der Filter soll alle darunterliegenden Einträge entfernen
# Am Ende wird geprüft, ob nur das relevante Dokument übrig bleibt

# pip install pytest installieren
# PYTHONPATH=. pytest tests/test_supabase_search_filter.py


import pytest
from database.setup import SupabaseClient


@pytest.fixture
def mock_results():
    return [
        {
            "content": "irrelevanter Text über Motoröl",
            "similarity": 0.81,
            "metadata": {"original_filename": "irrelevant.pdf", "page": 1},
        },
        {
            "content": "2-Takt-Motoröl: Wunsch BOAT SYNTH 2-T",
            "similarity": 0.93,
            "metadata": {"original_filename": "boat_synth_2-t.pdf", "page": 1},
        },
        {
            "content": "Diesel-Motoröl",
            "similarity": 0.89,
            "metadata": {"original_filename": "diesel.pdf", "page": 1},
        },
    ]


def filter_results_by_score(results, min_score=0.90):
    return [r for r in results if r["similarity"] >= min_score]


def test_similarity_filter_applies_correctly(mock_results):
    filtered = filter_results_by_score(mock_results, min_score=0.90)

    print("\nGefilterte Ergebnisse:")
    for r in filtered:
        print(f"- {r['metadata']['original_filename']} (similarity: {r['similarity']})")

    assert len(filtered) == 1
    assert filtered[0]["metadata"]["original_filename"] == "boat_synth_2-t.pdf"
    assert filtered[0]["similarity"] >= 0.90


def test_similarity_filter_excludes_low_scores():
    mock_results = [
        {
            "content": "irgendwas",
            "similarity": 0.75,
            "metadata": {"original_filename": "fake1.pdf", "page": 1},
        },
        {
            "content": "noch ein Versuch",
            "similarity": 0.65,
            "metadata": {"original_filename": "fake2.pdf", "page": 1},
        },
    ]

    filtered = [r for r in mock_results if r["similarity"] >= 0.90]

    print("\nErgebnisse bei nur niedrigen Scores:")
    for r in filtered:
        print(f"- {r['metadata']['original_filename']} (similarity: {r['similarity']})")

    assert len(filtered) == 0
