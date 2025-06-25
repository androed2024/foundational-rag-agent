"""
Knowledge base search tool for the RAG AI agent.
"""

import os
import sys
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# Add parent directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.setup import SupabaseClient
from document_processing.embeddings import EmbeddingGenerator
from document_processing.reranker import CrossEncoderReranker


class KnowledgeBaseSearchParams(BaseModel):
    """
    Parameters for the knowledge base search tool.
    """

    query: str = Field(
        ...,
        description="The search query to find relevant information in the knowledge base",
    )
    max_results: int = Field(
        15, description="Maximum number of results to return (default: 15)"
    )
    source_filter: Optional[str] = Field(
        None, description="Optional filter to search only within a specific source"
    )


class KnowledgeBaseSearchResult(BaseModel):
    """
    Result from the knowledge base search.
    """

    content: str = Field(..., description="Content of the document chunk")
    source: str = Field(..., description="Source of the document chunk")
    source_type: str = Field(..., description="Type of source (e.g., 'pdf', 'txt')")
    similarity: float = Field(
        ..., description="Similarity score between the query and the document"
    )
    metadata: Dict[str, Any] = Field(
        ..., description="Additional metadata about the document"
    )


class KnowledgeBaseSearch:
    """
    Tool for searching the knowledge base using vector similarity.
    """

    def __init__(
        self,
        supabase_client: Optional[SupabaseClient] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        owner_agent: Optional[
            Any
        ] = None,  # Referenz zum Agenten, um Treffer dort abzulegen
    ):
        """
        Initialize the knowledge base search tool.

        Args:
            supabase_client: SupabaseClient instance for database operations
            embedding_generator: EmbeddingGenerator instance for creating embeddings
            owner_agent: Optional reference to the RAGAgent to store last_match results
        """
        self.supabase_client = supabase_client or SupabaseClient()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.owner_agent = owner_agent
        self.reranker = CrossEncoderReranker()

    async def search(
        self, params: KnowledgeBaseSearchParams
    ) -> List[KnowledgeBaseSearchResult]:
        """
        Search the knowledge base for relevant information.

        Args:
            params: Search parameters

        Returns:
            List of search results
        """
        # Generate embedding for the query
        query_embedding = self.embedding_generator.embed_text(params.query)

        # Prepare filter metadata if source filter is provided
        filter_metadata = None
        if params.source_filter:
            filter_metadata = {"source": params.source_filter}

        candidate_count = max(params.max_results, 50)
        vector_results = self.supabase_client.search_documents(
            query_embedding=query_embedding,
            match_count=candidate_count,
            filter_metadata=filter_metadata,
        )
        keyword_results = self.supabase_client.keyword_search_documents(
            params.query,
            match_count=candidate_count,
            filter_metadata=filter_metadata,
        )
        results = vector_results + [r for r in keyword_results if r not in vector_results]
        results = self.reranker.rerank(params.query, results)[: params.max_results]

        # Convert results to KnowledgeBaseSearchResult objects
        search_results = []
        for result in results:
            search_results.append(
                KnowledgeBaseSearchResult(
                    content=result.get("content", ""),
                    source=result.get("metadata", {}).get("source", "Unknown"),
                    source_type=result.get("metadata", {}).get("source_type", "Unknown"),
                    similarity=float(result.get("similarity", 0.0)),
                    metadata=result.get("metadata", {}),
                )
            )

        # Optional: speichere Treffer im Agenten für UI-Anzeige
        if self.owner_agent is not None:
            self.owner_agent.last_match = (
                results  # <- das sind die rohen Supabase-Treffer, nicht die gewrappten
            )

        # for debugging
        for res in results:
            score = float(res.get("similarity", res.get("rerank_score", 0.0)))
            print(
                f"[DEBUG] Score: {score:.4f} – Datei: {res.get('metadata', {}).get('original_filename')}"
            )

        return search_results

    async def get_available_sources(self) -> List[str]:
        """
        Get a list of all available sources in the knowledge base.

        Returns:
            List of source identifiers
        """
        return self.supabase_client.get_all_document_sources()
