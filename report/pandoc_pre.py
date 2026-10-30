"""Round-10 PDF build of v4_1, step 1 (copy of pandoc_pre.py with three additions from factcheck_v4.md section 8):
title, author line and abstract move into pandoc metadata; the author block now carries the "Code: ..." clause (item
11); the keywords line becomes pandoc `keywords` metadata (PDF metadata; item 11); and the inline code spans of
Appendix B become raw LaTeX \\texttt{} with legal line breaks after "/", "_" and "." so that long paths wrap inside
their column instead of overflowing it (item 4c). The report file itself is never modified. No dependencies.

Usage: python3 pandoc_pre_final.py report/paper_final.md out.md
(final version: the author block takes its Code clause from the markdown)
"""
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
lines = open(src, encoding="utf-8").read().split("\n")
assert lines[0].startswith("# "), lines[0]
title = lines[0][2:].strip()
ia = lines.index("# Abstract")
ii = lines.index("# 1 Introduction")
abstract = "\n".join(lines[ia + 1:ii]).strip()

# drop the author paragraph (it goes into the metadata); keep the draft note, keywords and TL;DR in the body
pre, skip, keywords = [], False, []
for line in lines[1:ia]:
    if line.startswith("*Rishabh Sinha"):
        skip = True
    if skip:
        if line.strip() == "":
            skip = False
        continue
    pre.append(line)
m = re.search(r"\*Keywords:(.*?)\*", "\n".join(lines[1:ia]), re.S)
if m:
    keywords = [k.strip() for k in m.group(1).replace("\n", " ").split(";") if k.strip()]
    keywords = [k.rstrip(".") for k in keywords]


def tt(path: str) -> str:
    """Raw LaTeX monospace with break points after / _ . (only used inside Appendix B)."""
    s = path.replace("\\", "\\textbackslash{}")
    for ch in "#%&$":
        s = s.replace(ch, "\\" + ch)
    s = s.replace("_", "\\_\\allowbreak{}").replace("/", "/\\allowbreak{}").replace(".", ".\\allowbreak{}")
    return "\\texttt{" + s + "}"


body_lines = pre + lines[ii:]
in_b = False
for i, line in enumerate(body_lines):
    if line.startswith("# Appendix B."):
        in_b = True
    elif line.startswith("# Appendix C."):
        in_b = False
    if in_b and line.startswith("|"):
        body_lines[i] = re.sub(r"`([^`]+)`", lambda m: tt(m.group(1)), line)
body = "\n".join(body_lines)

# the Code clause (URL, license, release commit) comes from the markdown's author paragraph
mc = re.search(r"Code: (https://\S+ \([^)]*\))", "\n".join(lines[1:ia]).replace("\n", " "))
code = mc.group(1) if mc else "https://github.com/rishabhsinha17/brier-rabbit-paiec (MIT license)"
author = ("Rishabh Sinha (Team Brier Rabbit) `\\\\`{=latex} "
          "Independent researcher, personal capacity; rsinha17@terpmail.umd.edu `\\\\`{=latex} "
          "Code: " + code.replace(" (", " `\\\\`{=latex} (", 1))  # license and commit on their own line
yaml = ["---",
        'title: "%s"' % title.replace('"', "'"),
        "author: '%s'" % author,
        "keywords: [%s]" % ", ".join('"%s"' % k for k in keywords),
        "abstract: |"] + ["  " + l for l in abstract.split("\n")] + ["---", ""]
open(dst, "w", encoding="utf-8").write("\n".join(yaml) + body)
print("title:", title[:60], "| abstract chars:", len(abstract), "| keywords:", len(keywords), "| body lines:",
      body.count("\n"))
