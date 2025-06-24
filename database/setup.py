"""
Database setup and connection utilities for Supabase with pgvector.
"""

import os
from typing import Dict, List, Optional, Any
from postgrest import ReturnMethod
from dotenv import load_dotenv
from pathlib import Path
from supabase import create_client

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)


class SupabaseClient:
    """
    Client for interacting with Supabase and pgvector.

    Args:
        supabase_url: URL for Supabase instance. Defaults to SUPABASE_URL env var.
        supabase_key: API key for Supabase. Defaults to SUPABASE_KEY env var.
    """

    def __init__(
        self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None
    ):
        """
        Initialize the Supabase client.
        """
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Supabase URL and key must be provided either as arguments or environment variables."
            )

        self.client = create_client(self.supabase_url, self.supabase_key)

    def store_document_chunk(
        self,
        url: str,
        chunk_number: int,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Store a document chunk with its embedding in Supabase.

        Args:
            url: Source URL or identifier for the document
            chunk_number: Chunk number within the document
            content: Text content of the chunk
            embedding: Vector embedding of the chunk
            metadata: Additional metadata about the chunk

        Returns:
            Dictionary containing the inserted record
        """
        if metadata is None:
            metadata = {}

        data = {
            "url": url,
            "chunk_number": chunk_number,
            "content": content,
            "embedding": embedding,
            "metadata": metadata,
        }

        result = self.client.table("rag_pages").insert(data).execute()
        return result.data[0] if result.data else {}

    def search_documents(
        self,
        query_embedding: List[float],
        match_threshold: float = 0.5,
        match_count: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for document chunks by vector similarity using Supabase RPC.

        Args:
            query_embedding: OpenAI-compatible query embedding (1536 dims)
            match_threshold: Minimum similarity score (0.0 – 1.0)
            match_count: Maximum number of returned matches

        Returns:
            List of matching document chunks with similarity scores
        """

        match_threshold = float(os.getenv("MIN_SIMILARITY_SCORE", "0.5"))

        # RPC-Aufruf vorbereiten
        params = {
            "query_embedding": query_embedding,
            "match_threshold": match_threshold,
            "match_count": match_count,
        }

        if filter_metadata:
            params["filter"] = filter_metadata

        try:
            result = self.client.rpc("match_rag_pages", params).execute()

            if not result.data:
                print("⚠️ Keine Dokument-Treffer für die Anfrage gefunden.")
                return []

            print(f"\n🔍 Top {len(result.data)} RAG-Matches:")
            for r in result.data:
                score = r.get("similarity", 0.0)
                preview = r["content"][:120].replace("\n", " ")
                print(f"  • Score: {score:.3f} → {preview}...")

            return result.data

        except Exception as e:
            print("❌ Fehler bei Supabase-RPC:", str(e))
            return []

    def keyword_search_documents(
        self,
        query: str,
        match_count: int = 20,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Simple keyword search as fallback for hybrid retrieval."""
        try:
            qb = self.client.table("rag_pages").select(
                "id,url,chunk_number,content,metadata"
            )
            if filter_metadata:
                for key, value in filter_metadata.items():
                    qb = qb.contains("metadata", {key: value})
            qb = qb.ilike("content", f"%{query}%").limit(match_count)
            result = qb.execute()
            return result.data or []
        except Exception as e:
            print("❌ Fehler bei Keyword-Suche:", e)
            return []

    def get_document_by_id(self, doc_id: int) -> Dict[str, Any]:
        """
        Get a document chunk by its ID.

        Args:
            doc_id: ID of the document chunk

        Returns:
            Document chunk data
        """
        result = self.client.table("rag_pages").select("*").eq("id", doc_id).execute()
        return result.data[0] if result.data else {}

    def get_all_document_sources(self) -> List[str]:
        """
        Get a list of all unique document sources.

        Returns:
            List of unique source URLs/identifiers
        """
        result = self.client.table("rag_pages").select("url").execute()
        urls = set(item["url"] for item in result.data if result.data)
        return list(urls)

    def count_documents(self) -> int:
        """
        Count the total number of unique documents in the database.

        Returns:
            Number of unique documents (based on unique URLs)
        """
        return len(self.get_all_document_sources())

    def delete_documents_by_filename(self, filename: str) -> int:
        """Delete all document chunks belonging to a given filename.

        Args:
            filename: Name stored in ``metadata['original_filename']``.

        Returns:
            Number of deleted rows.
        """
        try:
            # First try direct match on the ``url`` column.  Older entries use
            # the original filename there, which is what the UI displays.
            resp = (
                self.client.table("rag_pages")
                .delete(returning=ReturnMethod.representation)
                .eq("url", filename)
                .execute()
            )
            deleted = len(resp.data or [])

            # Fall back to matching the JSON metadata field for robustness
            if deleted == 0:
                resp = (
                    self.client.table("rag_pages")
                    .delete(returning=ReturnMethod.representation)
                    .contains("metadata", {"original_filename": filename})
                    .execute()
                )
                deleted = len(resp.data or [])

            return deleted
        except Exception as e:
            print(f"Fehler beim Löschen von {filename}: {e}")
            return 0


def setup_database_tables() -> None:
    """
    Set up the necessary database tables and functions for the RAG system.
    This should be run once to initialize the database.

    Note: This is typically done through the Supabase MCP server in production.
    """
    # This is a placeholder for the actual implementation
    # In a real application, you would use the Supabase MCP server to run the SQL
    pass
