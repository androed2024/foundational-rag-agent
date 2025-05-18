# schnelltest.py

import asyncio
from agent import agent  # nutzt den Singleton RAGAgent


async def test_question(question: str):
    result = await agent.query(question)

    print("\n💬 Antwort:")
    print(result["response"])

    print("\n📄 Verwendete Chunks:")
    for r in result["kb_results"]:
        meta = r.get("metadata", {})
        filename = meta.get("original_filename", "Unbekannt")
        print(f"• {filename} → {r['content'][:100].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    frage = "Was ist der Brugger-Test?"  # Hier beliebige Testfrage setzen
    asyncio.run(test_question(frage))
