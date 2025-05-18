import pytest
from document_processing.embeddings import EmbeddingGenerator
from database.setup import SupabaseClient


def test_prompt_retrieves_correct_documents():
    query = "Welche Öle sind für 2-Takt-Motoren geeignet?"
    embedding_generator = EmbeddingGenerator()
    supabase_client = SupabaseClient()

    # Embedding aus echter Frage erzeugen
    embedding = embedding_generator.embed_text(query)

    # Dokumente mit hoher Ähnlichkeit abrufen
    results = supabase_client.search_documents(query_embedding=embedding, match_count=5)

    print("\nErgebnisse für Prompt-Abfrage:")
    for r in results:
        filename = r["metadata"].get("original_filename", "unbekannt")
        score = r["similarity"]
        content = r["content"][:500].replace("\n", " ")
        if "2" in content or "Takt" in content:
            print("⚠️ Teils Treffer gefunden:")
        print(repr(content))

    # Sicherstellen, dass nur sehr relevante Ergebnisse zurückkommen
    assert all(
        r["similarity"] >= 0.90 for r in results
    ), "Ein oder mehrere Ergebnisse liegen unter dem Threshold"
    assert any(
        "2-Takt" in r["content"] or "2‑Takt" in r["content"]  # Achtung: das ist \u2011
        for r in results
    ), "Keine Inhalte mit Bezug zu 2-Takt Ölen gefunden"
