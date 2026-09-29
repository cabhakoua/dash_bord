#!/usr/bin/env bash
set -e
for s in src/0*.py; do
  echo "=== $s ==="
  python "$s"
done
echo "Termine. Lancer le tableau de bord :  streamlit run app.py"
