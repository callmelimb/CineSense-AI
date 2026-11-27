#!/usr/bin/env bash
# One-step launcher for macOS / Linux.
set -e
cd "$(dirname "$0")"

# Rebuild a broken or Windows-made .venv
if [ -d .venv ] && [ ! -x .venv/bin/python ]; then rm -rf .venv; fi
if [ -x .venv/bin/python ] && ! .venv/bin/python -c "import sys" 2>/dev/null; then rm -rf .venv; fi
[ -x .venv/bin/python ] || python3 -m venv .venv

echo "Installing requirements (first run takes a few minutes)..."
.venv/bin/python -m pip install -q -r requirements.txt
echo "Starting CineSense AI - press Ctrl+C to stop."
.venv/bin/python -m streamlit run app.py
