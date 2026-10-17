#!/bin/sh
set -eu
# Edit full-draft.tex, numerical-main.tex, and numerical-supplement.tex.
# The split fscb-*.tex documents are regenerated below.
cd "$(dirname "$0")"
python3 code/split_manuscript.py
SCB_LATEXMK=${LATEXMK:-latexmk}
"$SCB_LATEXMK" -pdf -interaction=nonstopmode -halt-on-error fscb-draft.tex
"$SCB_LATEXMK" -pdf -interaction=nonstopmode -halt-on-error fscb-supplement.tex
"$SCB_LATEXMK" -g -pdf -interaction=nonstopmode -halt-on-error fscb-draft.tex
"$SCB_LATEXMK" -g -pdf -interaction=nonstopmode -halt-on-error fscb-supplement.tex
"$SCB_LATEXMK" -pdf -interaction=nonstopmode -halt-on-error full-draft.tex
