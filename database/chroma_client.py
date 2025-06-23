from chromadb import Client
from chromadb.config import Settings


class ChromaClient:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = Client(
            Settings(
                chroma_db_impl="duckdb+parquet", persist_directory=persist_directory
            )
        )
        self.collection_name = "rag_collection"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def store_document_chunk(
        self, doc_id: str, content: str, embedding: list, metadata: dict
    ):
        """
        Speichert ein einzelnes Chunk in Chroma.
        """
        self.collection.add(
            ids=[doc_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def query_similar_chunks(
        self, embedding: list, n_results: int = 5, min_score: float = 0.3
    ):
        """
        Sucht ähnliche Dokument-Chunks anhand des Embeddings.
        """
        results = self.collection.query(
            query_embeddings=[embedding], n_results=n_results
        )
        # Filter anhand min_score (cosine similarity)
        if "distances" in results and results["distances"]:
            filtered = []
            for i, score in enumerate(results["distances"][0]):
                if score >= min_score:
                    chunk = {
                        "id": results["ids"][0][i],
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": score,
                    }
                    filtered.append(chunk)
            return filtered
        return []

    def delete_document_chunks_by_id_prefix(self, doc_id_prefix: str):
        """
        Löscht alle Chunks mit ID-Präfix (z. B. ein gesamtes Dokument mit mehreren Chunks).
        """
        # Achtung: get_all_documents ist in Chroma derzeit nicht öffentlich dokumentiert!
        # Workaround: lokale ID-Liste pflegen oder Metadaten-Feld mit doc_id speichern und dann filtern
        raise NotImplementedError(
            "Chroma unterstützt keine direkte ID-Filterabfrage. Alternative: Collection neu bauen."
        )

    def get_all_documents(self) -> List[Dict[str, Any]]:
        results = self.collection.get(include=["metadatas"])
        docs = []
        for i in range(len(results["ids"])):
            docs.append({"id": results["ids"][i], "metadata": results["metadatas"][i]})
        return docs

    def list_all_documents(self):
        """
        Gibt alle gespeicherten Dokument-Metadaten zurück.
        """
        raise NotImplementedError(
            "Chroma bietet kein vollständiges Listing – du musst selbst mitdokumentieren."
        )

    def delete_documents_by_filename(self, filename: str) -> int:
        # Hole alle Einträge mit passendem original_filename
        results = self.collection.get(include=["metadatas", "ids"])
        deleted_count = 0

        for i, meta in enumerate(results["metadatas"]):
            if meta.get("original_filename") == filename:
                doc_id = results["ids"][i]
                self.collection.delete([doc_id])
                deleted_count += 1

        return deleted_count
