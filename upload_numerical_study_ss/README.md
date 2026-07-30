# Numerical Study SS-template Package

This directory is a self-contained Statistica Sinica-template package for the numerical study section.

Main file:

- `SS-template.tex`

Local dependencies:

- `references.bib`
- `chicago.bst`
- `results/figures/*.png`
- `results/realdata/*.png`

Compile with:

```bash
pdflatex SS-template.tex
bibtex SS-template
pdflatex SS-template.tex
pdflatex SS-template.tex
```

The figure paths and bibliography path are local to this directory, so the folder can be uploaded as a unit.
