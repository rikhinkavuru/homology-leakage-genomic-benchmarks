#!/usr/bin/env bash
# Build the manuscript. Run from anywhere: paper/build.sh
# TinyTeX is a user-local install, so put it on PATH explicitly rather than relying
# on the caller's environment.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"
cd "$HERE"

LOG=/tmp/paper_build.log
: > "$LOG"

# Regenerate the numeric table bodies from the committed CSVs before compiling, so a
# build can never ship a table that disagrees with its source. tab_reportcard.tex and
# tab_roster.tex are generated files -- edit the CSV, not the .tex.
PYTHONPATH="$HERE/.." "$HERE/../venv/bin/python" -m audit.pipeline.emit_tables >>"$LOG" 2>&1 \
  || { echo "emit_tables FAILED -- see $LOG"; tail -20 "$LOG"; exit 1; }
pdflatex -interaction=nonstopmode -file-line-error main.tex >>"$LOG" 2>&1 || true
bibtex main   >>"$LOG" 2>&1 || true
pdflatex -interaction=nonstopmode -file-line-error main.tex >>"$LOG" 2>&1 || true
pdflatex -interaction=nonstopmode -file-line-error main.tex >>"$LOG" 2>&1 || true

echo "pages:    $(pdfinfo main.pdf | awk '/^Pages/{print $2}')"
echo "errors:   $(grep -cE '^\./main\.tex:[0-9]+:|^! ' "$LOG" || true)"
echo "undef:    $(grep -cE 'Reference .* undefined|Citation .* undefined' "$LOG" || true)"
echo "overfull: $(grep -c 'Overfull \\hbox' "$LOG" || true)"
grep -E '^\./main\.tex:[0-9]+:|^! |Reference .* undefined|Citation .* undefined' "$LOG" | head -8 || true
