import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from document_processing.chunker import TextChunker
from document_processing.embeddings import EmbeddingGenerator
from document_processing.processors import get_document_processor
from document_processing.chroma_client import ChromaClient  # ⚠️ Neue Datei

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    def __init__(self, chroma_client: Optional[ChromaClient] = None):
        self.chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        self.embedding_generator = EmbeddingGenerator()
        self.max_file_size_mb = 10
        self.chroma_client = chroma_client or ChromaClient()
        logger.info("DocumentIngestionPipeline mit Chroma initialisiert.")

    def _check_file(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            logger.error(f"Datei nicht gefunden: {file_path}")
            return False
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            logger.error(f"Datei zu groß: {size_mb:.2f} MB")
            return False
        return True

    def process_file(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if not self._check_file(file_path):
            return []

        try:
            processor = get_document_processor(file_path)
            if not processor:
                logger.error(f"Kein passender Prozessor für: {file_path}")
                return []
        except Exception as e:
            logger.error(f"Fehler beim Prozessor: {str(e)}")
            return []

        try:
            chunks = processor.extract_text(file_path)
            if not chunks:
                logger.warning(
                    f"Keine Chunks extrahiert aus {os.path.basename(file_path)}"
                )
                return []
            logger.info(
                f"{len(chunks)} Chunks extrahiert aus {os.path.basename(file_path)}"
            )
        except Exception as e:
            logger.error(f"Fehler beim Extrahieren: {str(e)}")
            return []

        try:
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_generator.embed_batch(chunk_texts, batch_size=5)

            if len(embeddings) != len(chunks):
                logger.warning("Mismatch zwischen Chunks und Embeddings")
                chunks = chunks[: len(embeddings)]
                chunk_texts = chunk_texts[: len(embeddings)]
        except Exception as e:
            logger.error(f"Fehler beim Embedding: {str(e)}")
            return []

        try:
            document_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            metadata = metadata.copy() if metadata else {}
            metadata.update(
                {
                    "filename": os.path.basename(file_path),
                    "original_filename": metadata.get("original_filename", ""),
                    "file_path": file_path,
                    "file_size_bytes": os.path.getsize(file_path),
                    "processed_at": timestamp,
                    "chunk_count": len(chunks),
                }
            )

            stored_records = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_metadata = metadata.copy()
                chunk_metadata["page"] = chunk.get("page", 1)
                chunk_id = f"{document_id}_{i}"

                self.chroma_client.store_document_chunk(
                    doc_id=chunk_id,
                    content=chunk["text"],
                    embedding=embedding,
                    metadata=chunk_metadata,
                )
                stored_records.append(
                    {"id": chunk_id, "text": chunk["text"], "metadata": chunk_metadata}
                )

            logger.info(f"{len(stored_records)} Chunks in Chroma gespeichert.")
            return stored_records

        except Exception as e:
            logger.error(f"Fehler beim Speichern in Chroma: {str(e)}")
            return []
