#!/bin/bash

# Compile LaTeX report to PDF
echo "Compiling Assignment 3 Report..."

# Run pdflatex twice to resolve references
pdflatex Assignment_3_report.tex
pdflatex Assignment_3_report.tex

# Clean up auxiliary files
rm -f *.aux *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz

echo "Report compiled successfully!"
echo "Generated: Assignment_3_report.pdf"
