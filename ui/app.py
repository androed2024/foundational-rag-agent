"""
Streamlit application for the RAG AI agent.
Aufruf: streamlit run ui/app.py
"""

# Add parent directory to path to allow relative imports
import sys
import os

# Projektbasisverzeichnis zum Pfad hinzufügen (eine Ebene über 'ui')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Logging
import logging

# Reduce verbosity by logging only informational messages and above
logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import format_source_reference

import asyncio
from typing import List, Dict, Any
import streamlit as st
from pathlib import Path
import tempfile
from datetime import datetime

import unicodedata
import re

from collections import defaultdict


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
from utils.supabase_client import client
from utils.delete_helper import delete_file_and_records

from document_processing.ingestion import DocumentIngestionPipeline
from database.setup import SupabaseClient
from agent.agent import RAGAgent, agent as rag_agent, format_source_reference
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
)

st.set_page_config(
    page_title="Wissens-Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

supabase_client = SupabaseClient()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()


def display_message_part(part):
    if part.part_kind == "user-prompt" and part.content:
        with st.chat_message("user"):
            st.markdown(part.content)
    elif part.part_kind == "text" and part.content:
        with st.chat_message("assistant"):
            st.markdown(part.content)


async def process_document(
    file_path: str, original_filename: str, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    pipeline = DocumentIngestionPipeline()

    loop = asyncio.get_event_loop()

    try:
        chunks = await loop.run_in_executor(
            None,
            lambda: pipeline.process_file(
                file_path, metadata  # , on_progress=streamlit_progress
            ),
        )
        if not chunks:
            return {
                "success": False,
                "file_path": file_path,
                "error": "Keine gültigen Textabschnitte gefunden",
            }
        return {"success": True, "file_path": file_path, "chunk_count": len(chunks)}
    except Exception as e:
        import traceback

        print(f"Fehler bei der Bearbeitung des Dokuments: {str(e)}")
        print(traceback.format_exc())
        return {"success": False, "file_path": file_path, "error": str(e)}


async def run_agent_with_streaming(user_input: str):
    async with rag_agent.agent.iter(
        user_input,
        deps={"kb_search": rag_agent.kb_search},
        message_history=st.session_state.messages,
    ) as run:
        async for node in run:
            if hasattr(node, "request") and isinstance(node.request, ModelRequest):
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
                            yield event.delta.content_delta

    st.session_state.messages.extend(run.result.new_messages())


async def update_available_sources():
    try:
        response = client.table("rag_pages").select("id, metadata, url").execute()
        file_set = set()
        for row in response.data:
            print("DEBUG ROW:", row)
            metadata = row.get("metadata", {})
            print("DEBUG METADATA:", metadata)
            filename = metadata.get("original_filename")
            print("DEBUG FILENAME:", filename)
            if filename:
                file_set.add(filename)
        st.session_state.sources = sorted(file_set)
        st.session_state.document_count = len(file_set)
    except Exception as e:
        print(f"Fehler beim Aktualisieren der Dokumentenliste: {e}")
        for key in ["sources", "document_count", "processed_files", "messages"]:
            if key in st.session_state:
                del st.session_state[key]


async def main():
    await update_available_sources()

    # Initialisierung des Flags
    if "just_uploaded" not in st.session_state:
        st.session_state.just_uploaded = False

    # Robuste Initialisierung aller benötigten session_state Variablen
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "sources" not in st.session_state:
        st.session_state.sources = []

    if "document_count" not in st.session_state:
        try:
            st.session_state.document_count = supabase_client.count_documents()
        except Exception as e:
            print(f"Fehler beim Abrufen der Dokumentenzahl: {e}")
            st.session_state.document_count = 0

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    st.title("🔍 Wissens-Agent")
    st.markdown(
        """Diese Anwendung ermöglicht es, PDF- oder TXT-Datenblätter hochzuladen und anschließend Fragen dazu zu stellen. 
        Die Antworten stammen ausschließlich aus den hochgeladenen Dokumenten."""
    )

    with st.sidebar:
        st.header("📄 Dokumente hochladen")
        uploaded_files = st.file_uploader(
            "Hochladen von Dokumenten in die Wissensdatenbank",
            type=["txt", "pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files and not st.session_state.just_uploaded:
            new_files = [
                (f, f"{f.name}_{hash(f.getvalue().hex())}")
                for f in uploaded_files
                if f"{f.name}_{hash(f.getvalue().hex())}"
                not in st.session_state.processed_files
            ]

            if new_files:
                st.subheader("⏳ Upload-Fortschritt")
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, (uploaded_file, file_id) in enumerate(new_files):
                    safe_filename = sanitize_filename(uploaded_file.name)
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(uploaded_file.name).suffix
                    ) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name

                    try:
                        progress_bar.progress(0.05)
                        status_text.markdown(
                            f"🟡 **{safe_filename}**: 📥 *Upload startet...*"
                        )

                        with open(temp_file_path, "rb") as f:
                            client.storage.from_("privatedocs").upload(
                                safe_filename,
                                f,
                                {
                                    "cacheControl": "3600",
                                    "x-upsert": "true",
                                    "content-type": "application/pdf",
                                },
                            )

                        progress_bar.progress(0.3)
                        status_text.markdown(
                            f"🟠 **{safe_filename}**: 📤 *Dateiübertragung abgeschlossen*"
                        )

                        metadata = {
                            "source": "ui_upload",
                            "upload_time": str(datetime.now()),
                            "original_filename": safe_filename,
                        }

                        status_text.markdown(
                            f"🔵 **{safe_filename}**: 🧠 *Verarbeitung läuft...*"
                        )

                        result = await process_document(
                            temp_file_path, safe_filename, metadata
                        )

                        progress_bar.progress(0.8)

                        if result["success"]:
                            st.success(
                                f"✅ {uploaded_file.name} verarbeitet: {result['chunk_count']} Textabschnitte"
                            )
                            st.session_state.document_count += 1
                            st.session_state.processed_files.add(file_id)
                        else:
                            st.error(
                                f"❌ Fehler beim Verarbeiten {uploaded_file.name}: {result['error']}"
                            )

                        progress_bar.progress(1.0)
                        status_text.markdown(
                            f"🟢 **{safe_filename}**: ✅ *Verarbeitung abgeschlossen*"
                        )

                    finally:
                        os.unlink(temp_file_path)

                st.session_state.just_uploaded = True
                await update_available_sources()
                st.rerun()

            else:
                st.info("Alle Dateien wurden bereits verarbeitet")

        st.metric("Dokumente in der Wissensdatenbank", st.session_state.document_count)

        st.header("🗑️ Datei löschen")
        if st.session_state.sources:
            delete_filename = st.selectbox(
                "Ausgewählte Datei löschen", st.session_state.sources
            )
            if st.button("Ausgewählte Datei Löschen"):
                st.write("Dateiname zur Löschung:", delete_filename)
                result_log = delete_file_and_records(delete_filename)
                st.code(result_log)
                await update_available_sources()
                storage_deleted = db_deleted = False
                try:
                    st.write("Dateiname zur Löschung:", delete_filename)
                    print("Lösche:", delete_filename)
                    client.storage.from_("privatedocs").remove([delete_filename])
                    storage_deleted = True
                except Exception as e:
                    st.error(f"Löschen aus dem Speicher fehlgeschlagen: {e}")
                try:
                    deleted_count = supabase_client.delete_documents_by_filename(
                        delete_filename
                    )
                    st.code(
                        f"🧨 SQL-Delete für '{delete_filename}' – {deleted_count} Einträge entfernt."
                    )
                    db_deleted = True
                except Exception as e:
                    st.error(f"Datenbank-Löschung fehlgeschlagen: {e}")
                    db_deleted = False
                if storage_deleted and db_deleted:
                    st.success("✅ Vollständig gelöscht.")
                elif storage_deleted and not db_deleted:
                    st.warning(
                        "⚠️ Datei im Storage gelöscht, aber kein Eintrag in der Datenbank gefunden."
                    )
                elif not storage_deleted and db_deleted:
                    st.warning(
                        "⚠️ Datenbankeinträge gelöscht, aber Datei im Storage konnte nicht entfernt werden."
                    )
                else:
                    st.error(
                        "❌ Weder Datei noch Datenbankeinträge konnten gelöscht werden."
                    )
                await update_available_sources()
                st.rerun()
        else:
            st.info("Keine Dateien zur Löschung verfügbar.")

    st.header("💬 Spreche mit der KI")
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    user_input = st.chat_input("Stelle eine Frage zu den Dokumenten...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            async for chunk in run_agent_with_streaming(user_input):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            if hasattr(rag_agent, "last_match") and rag_agent.last_match:
                source_pages = defaultdict(set)

                for match in rag_agent.last_match:
                    sim = match.get("similarity", 0)
                    if sim < 0.7:
                        continue
                    meta = match.get("metadata", {})
                    fn = meta.get("original_filename")
                    pg = meta.get("page", 1)
                    if fn:
                        source_pages[fn].add(pg)

                if source_pages:
                    # 🧼 Veraltete PDF-Links entfernen
                    full_response = re.sub(
                        r"\[PDF öffnen\]\([^)]+\)", "", full_response
                    )

                    # 🧩 Neue Quellen-Liste einfügen
                    full_response += "\n\n### 📄 Verwendete Dokumente:\n"
                    for fn, pages in source_pages.items():
                        for pg in sorted(pages):
                            meta = {
                                "original_filename": fn,
                                "page": pg,
                                "source_filter": "privatedocs",
                            }
                            print("Format Link für:", meta)
                            full_response += f"\n- {format_source_reference(meta)}"

            # ✅ Endgültige Antwort anzeigen
            message_placeholder.markdown(full_response)


if __name__ == "__main__":
    asyncio.run(main())
