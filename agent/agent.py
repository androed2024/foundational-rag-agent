"""
Main agent definition for the RAG AI agent.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, TypedDict

from pydantic_ai import Agent
from pydantic_ai.tools import Tool
from supabase import create_client, SupabaseException

from dotenv import load_dotenv
from pathlib import Path

# Add parent directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tools import (
    KnowledgeBaseSearch,
    KnowledgeBaseSearchParams,
    KnowledgeBaseSearchResult,
)
from agent.prompts import RAG_SYSTEM_PROMPT

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path, override=True)

# Supabase-Konfiguration (Service-Role-Key!)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # angepasst an .env

# Lazy-init placeholder
global_supabase = None


def get_supabase_client():
    global global_supabase
    if global_supabase is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise SupabaseException(
                "SUPABASE_URL und SUPABASE_KEY m\u00fcssen gesetzt sein"
            )
        global_supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return global_supabase


class AgentDeps(TypedDict, total=False):
    kb_search: KnowledgeBaseSearch


class RAGAgent:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        kb_search: Optional[KnowledgeBaseSearch] = None,
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided either as an argument or environment variable."
            )

        self.kb_search = kb_search or KnowledgeBaseSearch(owner_agent=self)
        self.search_tool = Tool(self.kb_search.search)

        self.agent = Agent(
            f"openai:{self.model}",
            system_prompt=RAG_SYSTEM_PROMPT,
            tools=[self.search_tool],
        )

    async def query(
        self, question: str, max_results: int = 5, source_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        deps = AgentDeps(kb_search=self.kb_search)
        result = await self.agent.run(question, deps=deps)
        response = result.output
        kb_results = []
        for tool_call in getattr(result, "tool_calls", []):
            if tool_call.tool.name == "search":
                kb_results = tool_call.result or []
        return {"response": response, "kb_results": kb_results}

    async def get_available_sources(self) -> List[str]:
        return await self.kb_search.get_available_sources()


# Funktion aus Klasse herausgel\u00f6st


def format_source_reference(metadata: Dict[str, Any], short: bool = False) -> str:
    """
    Erzeugt on-demand eine signierte URL f\u00fcr private Supabase-Dokumente.
    Ignoriert vorhandene (veraltete) signed_url-Eintr\u00e4ge aus dem Upload.
    """
    filename = metadata.get("original_filename", "Unbekanntes Dokument")
    page = metadata.get("page") or "?"
    bucket = metadata.get("source_filter", "privatedocs")

    logging.debug(f"Erzeuge Signed URL f\u00fcr Datei: {filename} im Bucket: {bucket}")

    if short:
        return filename

    client = get_supabase_client()
    try:
        res = client.storage.from_(bucket).create_signed_url(filename, 3600)
        signed = res.get("signedURL")
        if not signed:
            logging.error(f"Keine signed URL in Response: {res}")
            return f"**Quelle:** {filename}, Seite {page} (kein Link verf\u00fcgbar)"
    except Exception as e:
        logging.error(f"Fehler beim Erstellen der signierten URL: {e}")
        return f"**Quelle:** {filename}, Seite {page} (Fehler beim Link-Aufbau)"

    if page:
        signed += f"#page={page}"
    page_info = f"Seite {page}" if page else "ohne Seitenangabe"
    return f"**Quelle:** {filename}, {page_info}\n\n[PDF \u00f6ffnen]({signed})"


# Singleton-Instanz f\u00fcr einfachen Import
agent = RAGAgent()
