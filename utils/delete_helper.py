from supabase import Client
from typing import Optional
from supabase.lib.client_options import ClientOptions
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def delete_file_and_records(filename: str, bucket: str = "privatedocs") -> str:
    log = []

    # 1. SQL-basierter Delete (per RPC)
    try:
        print(f"[DEBUG] Sende SQL DELETE für URL = {filename}")
        response = client.postgrest.rpc(
            "execute_sql", {"query": f"DELETE FROM rag_pages WHERE url = '{filename}'"}
        ).execute()
        log.append(f"🧨 SQL-Delete für '{filename}' ausgeführt.")
        db_deleted = True
    except Exception as e:
        log.append(f"❌ Fehler bei SQL-Delete: {e}")
        db_deleted = False

    # 2. Datei im Storage löschen
    try:
        client.storage.from_(bucket).remove([filename])
        log.append(f"🗑️ Storage-Datei gelöscht: {filename}")
        storage_deleted = True
    except Exception as e:
        log.append(f"❌ Fehler beim Löschen im Storage: {e}")
        storage_deleted = False

    # Zusammenfassung
    if storage_deleted and db_deleted:
        log.append("✅ Vollständig gelöscht.")
    elif storage_deleted:
        log.append("⚠️ Nur aus dem Storage gelöscht.")
    elif db_deleted:
        log.append("⚠️ Nur aus der Datenbank gelöscht.")
    else:
        log.append("🚫 Nichts wurde gelöscht.")

    return "\n".join(log)
