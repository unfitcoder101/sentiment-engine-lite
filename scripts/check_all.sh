#!/bin/sh
# Full local check. Run from the project folder: sh scripts/check_all.sh
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo "== pytest (tests marked 'models' need real local weights and are skipped) =="
$PY -m pytest -q -m "not models"
echo "== compileall =="
$PY -m compileall -q se run_demo.py serve.py
echo "== node --check on the dashboard script =="
if command -v node >/dev/null 2>&1; then
  TMP=$(mktemp -d)
  sed -n '/^<script>$/,/^<\/script>$/p' web/index.html | sed '1d;$d' > "$TMP/app.js"
  node --check "$TMP/app.js" && echo "script parses"
  rm -f "$TMP/app.js"; rmdir "$TMP"
else
  echo "node not installed: skipped"
fi
echo "== verify =="
$PY -m se.cli verify
