"""
Streamlit application for the RAG AI agent.
"""

import os
import sys
import asyncio
from typing import List, Dict, Any
import streamlit as st
from pathlib import Path
import tempfile
from datetime import datetime

from supabase_py import create_client

# Initialisierung (am Anfang der Datei)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Add parent directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_processing.ingestion import DocumentIngestionPipeline
from document_processing.chunker import TextChunker
from document_processing.embeddings import EmbeddingGenerator
from database.setup import SupabaseClient
from agent.agent import RAGAgent, agent as rag_agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
)

# Set page configuration
st.set_page_config(
    page_title="RAG AI Agent",
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
        print(f"Error getting document count: {e}")
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

result = await process_document(temp_file_path, metadata)
    pipeline = DocumentIngestionPipeline()

    try:
        metadata = {
            "source": "ui_upload",
            "upload_time": str(datetime.now()),
            "original_filename": original_filename,
        }

        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            None, lambda: pipeline.process_file(file_path, metadata)
        )

        if not chunks:
            return {
                "success": False,
                "file_path": file_path,
                "error": "No valid chunks were generated from the document",
            }

        return {"success": True, "file_path": file_path, "chunk_count": len(chunks)}
    except Exception as e:
        import traceback

        print(f"Error processing document: {str(e)}")
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
    Update the list of available sources in the knowledge base and refresh document count.
    """
    # Update sources list
    sources = await rag_agent.get_available_sources()
    st.session_state.sources = sources

    # Refresh document count from database
    try:
        st.session_state.document_count = supabase_client.count_documents()
    except Exception as e:
        print(f"Error updating document count: {e}")


async def main():
    """
    Main function for the Streamlit application.
    """
    # Display header
    st.title("🔍 RAG AI Agent")
    st.markdown(
        """
        This application allows you to upload documents (TXT and PDF) to a knowledge base 
        and ask questions that will be answered using the information in those documents.
        """
    )

    # Sidebar for document upload
    with st.sidebar:
        st.header("📄 Document Upload")

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload documents to the knowledge base",
            type=["txt", "pdf"],
            accept_multiple_files=True,
        )

        # Process only new uploaded files
        if uploaded_files:
            # Get list of files that haven't been processed yet
            new_files = []
            for uploaded_file in uploaded_files:
                # Create a unique identifier for the file based on name and content hash
                file_id = f"{uploaded_file.name}_{hash(uploaded_file.getvalue().hex())}"

                # Check if this file has already been processed
                if file_id not in st.session_state.processed_files:
                    new_files.append((uploaded_file, file_id))

            # Only show progress bar if there are new files to process
            if new_files:
                progress_bar = st.progress(0)
                status_text = st.empty()

                total_files = len(new_files)
                for i, (uploaded_file, file_id) in enumerate(new_files):
                    # Update progress
                    progress = i / total_files
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Processing {uploaded_file.name}... ({i+1}/{total_files})"
                    )

                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(uploaded_file.name).suffix
                    ) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name

                        # Datei nach Supabase hochladen
                        bucket_name = "privatedocs"
                        with open(temp_file_path, "rb") as f:
                        supabase.storage.from_(bucket_name).upload(
                        uploaded_file.name, f, {"cacheControl": "3600", "upsert": True}
                        )           

                        # Signierte URL generieren (z. B. gültig für 1 Stunde)
                        signed_url_resp = supabase.storage.from_(bucket_name).create_signed_url(
                            uploaded_file.name, expires_in=3600
                        )
                        public_url = signed_url_resp["signedURL"]

# Schreibe sie in die Metadaten
metadata = {
    "source": "ui_upload",
    "upload_time": str(datetime.now()),
    "original_filename": uploaded_file.name,
    "signed_url": public_url,
}

    
                    try:
                        # Process the document
                        result = await process_document(
                            temp_file_path, uploaded_file.name
                        )

                        # Display result
                        if result["success"]:
                            st.success(
                                f"Processed {uploaded_file.name}: {result['chunk_count']} chunks"
                            )
                            st.session_state.document_count += 1
                            # Mark this file as processed
                            st.session_state.processed_files.add(file_id)
                        else:
                            st.error(
                                f"Error processing {uploaded_file.name}: {result['error']}"
                            )
                    finally:
                        # Remove temporary file
                        os.unlink(temp_file_path)

                # Complete progress bar
                progress_bar.progress(1.0)
                status_text.text("All documents processed!")

                # Update available sources
                await update_available_sources()
            elif uploaded_files:  # If we have files but none are new
                st.info("All files have already been processed.")

        # Display document count
        st.metric("Documents in Knowledge Base", st.session_state.document_count)

    # Display available sources
    if st.session_state.sources:
        st.subheader("Available Sources")
        for source in st.session_state.sources:
            st.write(f"- {source}")

        # Dokument löschen Funktion
        st.subheader("🗑️ Delete Document")
        delete_filename = st.selectbox("Select a document to delete:", st.session_state.sources)
        if st.button("Delete selected document"):
            # 1. Lösche Datei aus Supabase Storage
            try:
                supabase.storage.from_("privatedocs").remove([delete_filename])
                st.success(f"Deleted from storage: {delete_filename}")
            except Exception as e:
                st.error(f"Storage delete failed: {e}")

            # 2. Lösche zugehörige Chunks aus Datenbank
            try:
                supabase.table("rag_pages").delete().eq("metadata->>original_filename", delete_filename).execute()
                st.success(f"Deleted from database: {delete_filename}")

                # Aktualisiere Quellenliste
                await update_available_sources()
            except Exception as e:
                st.error(f"DB delete failed: {e}")

    # Main chat interface
    st.header("💬 Chat with the AI")

    # Display all messages from the conversation so far
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input
    if user_input := st.chat_input("Ask a question about your documents..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display assistant response with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Stream the response
            generator = run_agent_with_streaming(user_input)
            async for chunk in generator:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            # Ergänzung: signierte URL anzeigen, falls verfügbar
            if hasattr(rag_agent, "last_match") and rag_agent.last_match:
                match = rag_agent.last_match[0]
                meta = match.get("metadata", {})
                signed_url = meta.get("signed_url")
                if signed_url:
                    full_response += f"\n\n🔗 [PDF öffnen]({signed_url})"

            # Final response ohne Cursor
            message_placeholder.markdown(full_response)


if __name__ == "__main__":
    asyncio.run(main())
