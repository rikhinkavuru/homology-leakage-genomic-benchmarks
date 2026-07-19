#!/usr/bin/env bash
# Rebuild the cover letter and response-to-reviewers PDFs and .docx from their
# markdown sources. Run from anywhere: paper/build_letters.sh
#
# xelatex rather than pdflatex because the letters contain Unicode math symbols
# (in, approx, <=, >=, pi, tau, ->); letter_header.tex maps those onto real math
# glyphs so the text font does not have to cover them.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"
cd "$HERE"

for f in cover_letter response_to_reviewers; do
  LOG="/tmp/build_${f}.log"
  pandoc "$f.md" -o "$f.docx" 2>"$LOG"
  pandoc "$f.md" -o "$f.pdf" --pdf-engine=xelatex \
    -V geometry:margin=1in -V fontsize=11pt -V colorlinks=true \
    -V mainfont="Times New Roman" -H letter_header.tex >>"$LOG" 2>&1
  echo "$f: missing-glyphs=$(grep -c 'Missing character' "$LOG" || true) errors=$(grep -c '^!' "$LOG" || true)"
done
