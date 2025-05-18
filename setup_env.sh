#!/bin/zsh

# 🔹 Wechsel in dein Projektverzeichnis
cd ~/__Projects__/ottomator-agents/WissensDB-Agent

# 🔹 Lösche altes venv, falls vorhanden
rm -rf .venv

# 🔹 Installiere Python 3.10 über pyenv, falls noch nicht installiert
if ! pyenv versions | grep -q "3.10."; then
  echo "🔧 Installing Python 3.10 via pyenv..."
  pyenv install 3.10.13
fi

# 🔹 Setze lokal auf Python 3.10 um
pyenv local 3.10.13

# 🔹 Erstelle virtuelles Environment
python -m venv .venv

# 🔹 Aktiviere venv
source .venv/bin/activate

# 🔹 Upgrade pip & wheel
pip install --upgrade pip wheel setuptools

# 🔹 Installiere benötigte Pakete
pip install supabase-py streamlit pydantic openai python-dotenv PyPDF2 unstructured[pdf,ocr]

# 🔹 Optional: installiere dev dependencies
pip install pytest pytest-asyncio

echo "✅ Setup abgeschlossen. Jetzt kannst du deine App starten mit:"
echo "source .venv/bin/activate && streamlit run ui/app.py"
