#!/usr/bin/env bash
# Satellite NITF — sets up its virtual environment (numpy + rasterio) and then
# prints usage so you can run it with your own coordinates. This is a
# command-line tool: it does not open a window, it writes imagery files.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "First-time setup: creating virtual environment…"
  python3 -m venv .venv
fi
echo "Installing numpy + rasterio (first run only)…"
.venv/bin/python -m pip install -q --upgrade pip numpy rasterio
echo
echo "=== Satellite NITF ready. Run it with your coordinates, e.g.: ==="
echo "   .venv/bin/python get_nitf.py --lat 34.15 --lon 77.57 --name leh"
echo
.venv/bin/python get_nitf.py --help || true
echo
echo "(Type your command above when ready — the venv is already set up.)"
