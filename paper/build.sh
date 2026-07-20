#!/usr/bin/env bash
# Build the submission bundle. Run from anywhere: paper/build.sh
# TinyTeX is a user-local install, so put it on PATH explicitly rather than relying
# on the caller's environment.
#
# Builds BOTH documents that are actually submitted: main_resubmission.tex and
# supplementary.tex. It used to build main.tex, which is the FROZEN PRIOR SUBMISSION
# and no longer the artifact a referee reads -- so a clean build reported nothing
# about the manuscript being edited. Set DOC to build something else.
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

build_one() {
  local doc="$1"
  pdflatex -interaction=nonstopmode -file-line-error "$doc.tex" >>"$LOG" 2>&1 || true
  bibtex "$doc" >>"$LOG" 2>&1 || true
  pdflatex -interaction=nonstopmode -file-line-error "$doc.tex" >>"$LOG" 2>&1 || true
  pdflatex -interaction=nonstopmode -file-line-error "$doc.tex" >>"$LOG" 2>&1 || true

  # Per-document counts. Grep the document's own .log, not the shared build log, so
  # one document's warnings are never attributed to the other.
  echo "$doc:"
  echo "  pages:    $(pdfinfo "$doc.pdf" | awk '/^Pages/{print $2}')"
  echo "  errors:   $(grep -cE "^\./$doc\.tex:[0-9]+:|^! " "$doc.log" || true)"
  echo "  undef:    $(grep -cE 'Reference .* undefined|Citation .* undefined' "$doc.log" || true)"
  echo "  overfull: $(grep -c 'Overfull \\hbox' "$doc.log" || true)"
  grep -E "^\./$doc\.tex:[0-9]+:|^! |Reference .* undefined|Citation .* undefined" "$doc.log" \
    | head -8 || true
}

for doc in ${DOC:-main_resubmission supplementary}; do
  build_one "$doc"
done
