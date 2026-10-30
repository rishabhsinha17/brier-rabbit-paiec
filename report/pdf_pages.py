"""Report the first PDF page on which each given heading appears (pdftotext per page). Usage: pdf_pages.py file.pdf"""
import subprocess
import sys

pdf = sys.argv[1]
n = int([l for l in subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout.split("\n")
         if l.startswith("Pages")][0].split()[-1])
heads = ["1 Introduction", "2 Task and protocol", "3 Data", "4 Model", "5 Local evaluation",
         "6 Live evaluation methodology", "7 Evidence about the hidden pool", "7.5 Headroom", "8 Cross-system item pooling",
         "9 Negative results", "10 Final submission", "11 Limitations", "12 Reproducibility",
         "13 Disclosures and acknowledgments", "References", "Appendix A.", "Appendix B.", "Appendix C.", "Appendix D.",
         "Appendix E.", "Appendix F.", "Appendix G.", "Appendix H."]
found = {}
texts = []
for p in range(1, n + 1):
    t = subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), "-layout", pdf, "-"],
                       capture_output=True, text=True).stdout
    texts.append(t)
    for h in heads:
        if h not in found:
            for line in t.split("\n"):
                s = line.strip()
                if s.startswith(h) and (h != "References" or s == "References"):
                    found[h] = p
                    break
print("pages:", n)
for h in heads:
    print("%-36s page %s" % (h, found.get(h, "?")))
# where does Section 11 end: the last page before "12 Reproducibility" heading, plus how full that page is
if "12 Reproducibility" in found:
    p = found["12 Reproducibility"]
    t = texts[p - 1]
    lines = [l for l in t.split("\n") if l.strip()]
    idx = [i for i, l in enumerate(lines) if l.strip().startswith("12 Reproducibility")][0]
    print("Section 12 heading at line %d of %d non-empty lines on page %d" % (idx + 1, len(lines), p))
