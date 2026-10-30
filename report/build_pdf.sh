#!/bin/zsh
# Build the technical report PDF from report/paper_final.md with the official NeurIPS 2026 style (neurips_2026.sty).
# Requires pandoc and a TeX distribution with pdflatex. Run from anywhere: it works next to its own files and writes
# report/build/ (paper_final.pdf, .tex, pdflatex logs). Figures resolve relative to report/ (../swarm/...).
set -e
HERE=${0:A:h}
OUT=$HERE/build
mkdir -p $OUT
cd $HERE
python3 pandoc_pre.py paper_final.md $OUT/paper_final_pre.md
pandoc $OUT/paper_final_pre.md -f markdown -t latex -s --resource-path=. -V documentclass=article -V indent=true \
  -V linkcolor=blue -V urlcolor=blue -V citecolor=blue -H pandoc_header.tex -o $OUT/paper_final.tex
# command-line flags such as --reps inside the \texttt{} paths of Appendix B would print as en dashes (the "--"
# ligature); break the ligature there so the PDF shows two hyphens
perl -pi -e 's/--/-{}-/g if /\\texttt\{/' $OUT/paper_final.tex
for pass in 1 2; do
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$OUT $OUT/paper_final.tex \
    > $OUT/pdflatex_pass$pass.out 2>&1 || { echo "pdflatex pass $pass failed; see $OUT/pdflatex_pass$pass.out"; exit 1; }
done
echo "built $OUT/paper_final.pdf"
