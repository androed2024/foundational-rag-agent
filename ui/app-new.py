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

# Date+Time for post knowledge in db
from datetime import datetime
import pytz

# Reduce verbosity by logging only informational messages and above
logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import format_source_reference

import asyncio
from typing import List, Dict, Any
from pathlib import Path
import tempfile
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
import base64

import unicodedata
import re

import hashlib


def compute_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


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
    initial_sidebar_state="collapsed",
)

supabase_client = SupabaseClient()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()


def custom_file_uploader(label="Datei auswählen", file_types="pdf,txt"):
    custom_uploader = f"""
    <style>
      .upload-btn-wrapper {{
        position: relative;
        overflow: hidden;
        display: inline-block;
        margin-bottom: 18px;
      }}
      .btn {{
        border: 1px solid #ccc;
        color: #333;
        background-color: #f6f7fa;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.1rem;
        font-family: inherit;
        font-weight: 500;
        cursor: pointer;
      }}
      .upload-btn-wrapper input[type=file] {{
        font-size: 100px;
        position: absolute;
        left: 0;
        top: 0;
        opacity: 0;
      }}
    </style>
    <div class="upload-btn-wrapper">
      <button class="btn">{label}</button>
      <input type="file" id="file-upload" accept="{file_types}" />
      <span id="file-selected"></span>
    </div>
    <script>
      const input = window.parent.document.getElementById("file-upload");
      if (input) {{
        input.onchange = function(event) {{
          const file = event.target.files[0];
          if (file) {{
            var reader = new FileReader();
            reader.onload = function(e) {{
              var dataUrl = e.target.result;
              window.parent.postMessage({{
                isStreamlitMessage: true,
                type: "streamlit:setComponentValue",
                value: dataUrl
              }}, "*");
              var el = window.parent.document.getElementById("file-selected");
              if (el) el.innerText = file.name + " ausgewählt";
            }};
            reader.readAsDataURL(file);
          }}
        }};
      }}
    </script>
    """
    result = components.html(custom_uploader, height=100)
    return result


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
        response = client.table("rag_pages").select("url, metadata").execute()

        file_set = set()
        knowledge_set = set()

        for row in response.data:
            metadata = row.get("metadata", {})
            url = row.get("url", "")
            if not url:
                continue

            if metadata.get("source") == "ui_upload":
                file_set.add(url)
            elif metadata.get("source") == "manuell":
                knowledge_set.add(url)

        # 👇 Kombinieren und sortieren
        all_sources = sorted(file_set.union(knowledge_set))

        st.session_state.sources = all_sources
        st.session_state.document_count = len(file_set)
        st.session_state.knowledge_count = len(knowledge_set)

    except Exception as e:
        print(f"Fehler beim Aktualisieren der Dokumentenliste: {e}")
        for key in [
            "sources",
            "document_count",
            "knowledge_count",
            "processed_files",
            "messages",
        ]:
            if key in st.session_state:
                del st.session_state[key]


async def main():
    await update_available_sources()

    doc_count = st.session_state.get("document_count", 0)
    note_count = st.session_state.get("knowledge_count", 0)

    st.markdown(
        f"""
        <div style='text-align: right; margin-top: -40px; margin-bottom: 10px; font-size: 14px;'>
            📄 Dokumente: {doc_count} &nbsp;&nbsp;&nbsp; 🧠 Notizen: {note_count}
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "💬 Wunsch-Öl KI Assistent",
            "➕ Wissen hinzufügen",
            "📄 Dokumente hochladen",
            "🗑️ Dokument / Notiz löschen",
        ]
    )

    with tab1:
        st.markdown(
            "<h4>💬 Spreche mit dem Wunsch-Öl KI Assistenten</h4>",
            unsafe_allow_html=True,
        )

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

                # Chatbot Interface
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

    with tab2:
        st.markdown("<h4>➕ Wissen hinzufügen</h4>", unsafe_allow_html=True)
        st.markdown(
            "Du kannst hier eigene Notizen, Feedback oder Empfehlungen eintragen, die sofort durchsuchbar sind."
        )

        # Initialisierung der Eingabefelder in session_state
        for key in ["manual_title", "manual_text", "manual_source"]:
            if key not in st.session_state:
                st.session_state[key] = ""

        # Eingabefelder mit session_state
        manual_title = st.text_input(
            "🏷️ Überschrift",
            key="manual_title_input",
        )
        manual_text = st.text_area("✍️ Dein Wissen", key="manual_text_input")

        # Handle manuelle Quelle sicher
        source_options = ["Beratung", "Meeting", "Feedback", "Sonstiges"]
        try:
            source_index = source_options.index(st.session_state.manual_source)
        except ValueError:
            source_index = 0

        source_type = st.selectbox(
            "Quelle des Wissens",
            source_options,
            index=source_index,
            key="manual_source_input",
        )

        # Button-Reihe nebeneinander mit Columns
        col1, col2 = st.columns([3, 2])
        with col1:
            if st.button("✅ Wissen / Notiz speichern", key="save_button"):
                if not manual_title.strip() or not manual_text.strip():
                    st.warning(
                        "⚠️ Bitte gib sowohl eine Überschrift als auch einen Text ein."
                    )
                else:
                    existing = (
                        client.table("rag_pages")
                        .select("url")
                        .ilike("url", f"{manual_title.strip()}%")
                        .execute()
                    )
                    if existing.data:
                        st.warning(
                            f"⚠️ Ein Eintrag mit der Überschrift '{manual_title.strip()}' existiert bereits."
                        )
                    else:
                        try:
                            pipeline = DocumentIngestionPipeline()
                            tz_berlin = pytz.timezone("Europe/Berlin")
                            now_berlin = datetime.now(tz_berlin)
                            timestamp = now_berlin.strftime("%Y-%m-%d %H:%M")
                            full_title = f"{manual_title.strip()} ({timestamp})"
                            metadata = {
                                "source": "manuell",
                                "quelle": source_type,
                                "title": manual_title.strip(),
                                "upload_time": now_berlin.isoformat(),
                                "original_filename": manual_title.strip(),
                                "source_filter": "notes",
                            }
                            result = pipeline.process_text(
                                content=manual_text,
                                metadata=metadata,
                                url=full_title,
                            )
                            st.toast(
                                "🧠 Wissen/Notizen erfolgreich gespeichert", icon="✅"
                            )
                            await update_available_sources()
                            st.session_state.manual_title = ""
                            st.session_state.manual_text = ""
                            st.session_state.manual_source = "Beratung"
                            st.rerun()
                        except Exception as e:
                            st.error(
                                f"❌ Fehler beim Speichern des Wissens/der Notiz: {e}"
                            )

        with col2:
            if st.button("🧹 Eingaben leeren", key="clear_button"):
                st.session_state.manual_title = ""
                st.session_state.manual_text = ""
                st.session_state.manual_source = "Beratung"
                st.rerun()

    import base64

    with tab3:
        st.markdown(
            """
            <div style="padding:1rem;background:#f6f7fa;border-radius:8px;font-size:16px;">
                📎 Ziehe deine Dateien hier rein oder nutze den Button
                <strong>„Dateien hochladen“</strong><br>
                <small>(max. 200 MB pro Datei • PDF oder TXT)</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_data = custom_file_uploader(
            "📂 Dateien hochladen", "application/pdf,.txt"
        )

        # Achtung: uploaded_data ist ein DataURL (base64)
        if (
            uploaded_data is not None
            and isinstance(uploaded_data, str)
            and uploaded_data.startswith("data:")
        ):
            header, encoded = uploaded_data.split(",", 1)
            file_bytes = base64.b64decode(encoded)
            st.success("Datei erfolgreich empfangen! Größe: %d Bytes" % len(file_bytes))
            # Hier deine eigene Upload-Weiterverarbeitung:
            # - Dateityp prüfen
            # - In temp_file schreiben
            # - Supabase Upload, etc.

        st.markdown(
            "<hr style='margin-top: 6px; margin-bottom: 6px;'>", unsafe_allow_html=True
        )

    with tab4:
        st.markdown("<h4>🗑️ Dokument / Notiz löschen</h4>", unsafe_allow_html=True)

        if st.session_state.sources:
            delete_filename = st.selectbox(
                "Dokument/Notiz selektieren", st.session_state.sources
            )
            if st.button("Ausgewählte Dokument/Notiz löschen"):
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
                        "⚠️ Dokument/Notiz im Storage gelöscht, aber kein Eintrag in der Datenbank gefunden."
                    )
                elif not storage_deleted and db_deleted:
                    st.warning(
                        "⚠️ Datenbankeinträge gelöscht, aber Dokument/Notiz im Storage konnte nicht entfernt werden."
                    )
                else:
                    st.error(
                        "❌ Weder Dokument/Notiz noch Datenbankeinträge konnten gelöscht werden."
                    )
                await update_available_sources()
                st.rerun()
        else:
            st.info("Keine Dokumente/Notizen zur Löschung verfügbar.")


if __name__ == "__main__":
    asyncio.run(main())
