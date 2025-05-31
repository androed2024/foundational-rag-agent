"""
Streamlit application for the RAG AI agent.
"""

# Aufruf: streamlit run ui/app.py

# Add parent directory to path to allow relative imports
import sys
import os

# Logging
import logging

logging.basicConfig(level=logging.DEBUG)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import format_source_reference

import asyncio
from typing import List, Dict, Any
import streamlit as st
from pathlib import Path
import tempfile
from datetime import datetime

from datetime import datetime

import unicodedata
import re


import unicodedata
import re


def sanitize_filename(filename: str) -> str:
    filename = filename.strip()
    filename = filename.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    filename = filename.replace("Ä", "Ae").replace("Ö", "Oe").replace("Ü", "Ue")
    filename = filename.replace("ß", "ss")
    filename = (
        unicodedata.normalize("NFKD", filename)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
    return filename


from dotenv import load_dotenv

load_dotenv()
from supabase import create_client, Client

# Initialisierung (am Anfang der Datei)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

from collections import OrderedDict

# Add parent directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_processing.ingestion import DocumentIngestionPipeline
from document_processing.chunker import TextChunker
from document_processing.embeddings import EmbeddingGenerator
from database.setup import SupabaseClient
from agent.agent import RAGAgent, agent as rag_agent, format_source_reference
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
)

# Set page configuration
st.set_page_config(
    page_title="Wunsch Öl Wissens-Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database client
supabase_client = SupabaseClient()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

if "document_count" not in st.session_state:
    # Initialize document count from database
    try:
        st.session_state.document_count = supabase_client.count_documents()
    except Exception as e:
        print(f"Fehler beim Abrufen der Dokumentenzahl: {e}")
        st.session_state.document_count = 0

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()  # Track already processed files


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.

    Args:
        part: Message part to display
    """
    # User messages
    if part.part_kind == "user-prompt" and part.content:
        with st.chat_message("user"):
            st.markdown(part.content)
    # AI messages
    elif part.part_kind == "text" and part.content:
        with st.chat_message("assistant"):
            st.markdown(part.content)


async def process_document(
    file_path: str, original_filename: str, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    pipeline = DocumentIngestionPipeline()

    try:

        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            None, lambda: pipeline.process_file(file_path, metadata)
        )

        if not chunks:
            return {
                "success": False,
                "file_path": file_path,
                "error": "Es wurden keine gültigen Textabschnitte aus dem Dokument erzeugt",
            }

        return {"success": True, "file_path": file_path, "chunk_count": len(chunks)}
    except Exception as e:
        import traceback

        print(f"Fehler bei der Bearbeitung des Dokuments: {str(e)}")
        print(traceback.format_exc())
        return {"success": False, "file_path": file_path, "error": str(e)}


async def run_agent_with_streaming(user_input: str):
    """
    Run the RAG agent with streaming response.

    Args:
        user_input: User query

    Yields:
        Streamed response chunks
    """
    # Run the agent with the user input
    async with rag_agent.agent.iter(
        user_input,
        deps={"kb_search": rag_agent.kb_search},
        message_history=st.session_state.messages,
    ) as run:
        async for node in run:
            # Check if this is a model request node
            if hasattr(node, "request") and isinstance(node.request, ModelRequest):
                # Stream tokens from the model's request
                async with node.stream(run.ctx) as request_stream:
                    async for event in request_stream:
                        if (
                            isinstance(event, PartStartEvent)
                            and event.part.part_kind == "text"
                        ):
                            yield event.part.content
                        elif isinstance(event, PartDeltaEvent) and isinstance(
                            event.delta, TextPartDelta
                        ):
                            delta = event.delta.content_delta
                            yield delta

    # Add the new messages to the chat history
    st.session_state.messages.extend(run.result.new_messages())


async def update_available_sources():
    """
    Aktualisiert die Liste verfügbarer Quellen und die Anzahl der Dokumente.
    """
    try:
        # Alle Metadaten abrufen
        response = supabase.table("rag_pages").select("id, metadata").execute()

        file_set = set()

        for row in response.data:
            metadata = row.get("metadata", {})
            filename = metadata.get("original_filename")
            if filename:
                file_set.add(filename)

        # Update Session State
        st.session_state.sources = sorted(file_set)
        st.session_state.document_count = len(file_set)

    except Exception as e:
        print(f"Fehler beim Aktualisieren der Dokumentenliste: {e}")
        for key in ["sources", "document_count", "processed_files", "messages"]:
            if key in st.session_state:
                del st.session_state[key]


async def main():
    # Testweise alles löschen
    await update_available_sources()

    # 💡 Nach dem Wipe Session State neu initialisieren
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "sources" not in st.session_state:
        st.session_state.sources = []

    if "document_count" not in st.session_state:
        st.session_state.document_count = 0

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    # Liste nochmal laden (optional)
    await update_available_sources()

    # st.sidebar.write("Debug-Quellen:", st.session_state.sources)
    try:
        await update_available_sources()
    except Exception as e:
        print(f"Failed to update available sources: {e}")
        st.session_state.sources = []
        st.session_state.document_count = 0

    st.title("🔍 Wunsch Öl Wissens-Agent")
    st.markdown(
        """Diese Anwendung ermöglicht es, PDF- oder TXT-Datenblätter hochzuladen und anschließend Fragen dazu zu stellen. 
        Die Antworten stammen ausschließlich aus den hochgeladenen Dokumenten."""
    )

    # Sidebar
    with st.sidebar:
        st.header("📄 Dokumente hochladen (txt oder pdf)")

        uploaded_files = st.file_uploader(
            "Hochladen von Dokumenten in die Wissensdatenbank",
            type=["txt", "pdf"],
            accept_multiple_files=True,
        )

        # Display document count
        st.metric("Dokumente in der Wissensdatenbank", st.session_state.document_count)

        # Delete documents section
        st.header("🗑️ Datei löschen")

        if st.session_state.sources:
            delete_filename = st.selectbox(
                "Ausgewählte Datei löschen", st.session_state.sources
            )

            if st.button("Datei zum Löschen auswählen"):
                storage_deleted = False
                db_deleted = False

                # 1. Datei im Supabase Storage löschen
                try:
                    supabase.storage.from_("privatedocs").remove([delete_filename])
                    storage_deleted = True
                except Exception as e:
                    st.error(f"Löschen aus dem Speicher fehlgeschlagen: {e}")

                # 2. Alle zugehörigen Chunks aus der Datenbank löschen (über ID)
                try:
                    st.write("DEBUG DELETE: Trying to delete", delete_filename)

                    # Alle Einträge holen
                    response = (
                        supabase.table("rag_pages").select("id, metadata").execute()
                    )

                    # IDs herausfiltern, die zum Dateinamen gehören
                    ids_to_delete = [
                        row["id"]
                        for row in response.data
                        if row.get("metadata", {}).get("original_filename")
                        == delete_filename
                    ]

                    # Jetzt alle einzeln löschen (alternativ: .in_() statt Schleife, wenn du willst)
                    for id_ in ids_to_delete:
                        supabase.table("rag_pages").delete().eq("id", id_).execute()

                    db_deleted = True

                except Exception as e:
                    st.error(f"Datenbank-Löschung fehlgeschlagen: {e}")

                # 3. Erfolgs-/Fehlermeldung anzeigen
                if storage_deleted and db_deleted:
                    st.success(
                        f"Erfolgreich gelöscht: {delete_filename} aus Speicher und Datenbank."
                    )
                elif storage_deleted:
                    st.warning(
                        f"Aus Speicher gelöscht, aber Datenbank-Einträge konnten nicht entfernt werden."
                    )
                elif db_deleted:
                    st.warning(
                        f"Aus Datenbank gelöscht, aber Datei im Speicher konnte nicht entfernt werden."
                    )

                # 4. Dokumentliste aktualisieren
                await update_available_sources()

        else:
            st.info("Keine Dateien zur Löschung verfügbar.")

        # Handle uploads
        if uploaded_files:
            new_files = []
            for uploaded_file in uploaded_files:
                file_id = f"{uploaded_file.name}_{hash(uploaded_file.getvalue().hex())}"
                if file_id not in st.session_state.processed_files:
                    new_files.append((uploaded_file, file_id))

            if new_files:
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_files = len(new_files)

                for i, (uploaded_file, file_id) in enumerate(new_files):
                    safe_filename = sanitize_filename(uploaded_file.name)
                    print("Sanitized filename:", safe_filename)

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(uploaded_file.name).suffix
                    ) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name

                    bucket_name = "privatedocs"
                    with open(temp_file_path, "rb") as f:
                        try:
                            supabase.storage.from_(bucket_name).upload(
                                safe_filename,
                                f,
                                {"cacheControl": "3600", "x-upsert": "true"},
                            )
                        except Exception as upload_error:
                            st.error(f"Fehler beim Upload in Supabase: {upload_error}")
                            print(f"[ERROR] Supabase Upload Failed: {upload_error}")
                            continue  # Skip to next file

                        signed_url_resp = supabase.storage.from_(
                            bucket_name
                        ).create_signed_url(safe_filename, expires_in=3600)

                        public_url = signed_url_resp["signedURL"]

                        metadata = {
                            "source": "ui_upload",
                            "upload_time": str(datetime.now()),
                            "original_filename": safe_filename,
                            "signed_url": public_url,
                        }

                    try:
                        result = await process_document(
                            temp_file_path, safe_filename, metadata
                        )

                        if result["success"]:
                            st.success(
                                f"Verarbeitete {uploaded_file.name}: {result['chunk_count']} Textabschnitte"
                            )
                            st.session_state.document_count += 1
                            st.session_state.processed_files.add(file_id)
                        else:
                            st.error(
                                f"Fehler beim Verarbeiten {uploaded_file.name}: {result['error']}"
                            )
                    finally:
                        os.unlink(temp_file_path)

                progress_bar.progress(1.0)
                status_text.text("Alle Dateien bearbeitet")
                await update_available_sources()
                st.rerun()
            else:
                st.info("Alle Dateien wurden bereits verarbeitet")

    # Main Chat Area
    st.header("💬 Spreche mit der KI")

    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    if user_input := st.chat_input("Stelle eine Frage zu den Dokumenten..."):
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            generator = run_agent_with_streaming(user_input)
            async for chunk in generator:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            if hasattr(rag_agent, "last_match") and rag_agent.last_match:
                source_map = OrderedDict()
                for match in rag_agent.last_match:
                    meta = match.get("metadata", {})
                    filename = meta.get("original_filename")
                    if filename and filename not in source_map:
                        source_map[filename] = meta

                full_response += "\n\n### 📄 Verwendete Dokumente:\n"
                for meta in source_map.values():
                    # full_response += f"\n- {format_source_reference(meta)}"
                    full_response += f"\n- {format_source_reference(meta)}"

            message_placeholder.markdown(full_response)


if __name__ == "__main__":
    asyncio.run(main())
