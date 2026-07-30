# Numerical Study SS-template Package

This directory is a self-contained Statistica Sinica-template package for the numerical study section.

Main file:

- `SS-template.tex`

Local dependencies:

- `references.bib`
- `chicago.bst`
- `results/figures/*.png`
- `results/realdata/*.png`

The same curated package is also archived at repository root as
`upload_numerical_study_ss.zip`. The archive intentionally includes only the
TeX source, compiled PDF, bibliography files, style file, and local figures;
LaTeX temporary files are omitted.

Compile with:

```bash
pdflatex SS-template.tex
bibtex SS-template
pdflatex SS-template.tex
pdflatex SS-template.tex
```

The figure paths and bibliography path are local to this directory, so the folder can be uploaded as a unit.
