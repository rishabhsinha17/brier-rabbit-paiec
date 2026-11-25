"""Audit the saved official XLSX, including empty worksheets and every nonempty cell."""
from pathlib import Path
import csv
import hashlib
import json
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "camera_ready"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE = "http://schemas.openxmlformats.org/package/2006/relationships"


def main():
    path = HERE / "sources/inventory.xlsx"
    cells, slugs, sheets = [], [], []
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            tree = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(t.text or "" for t in si.findall(".//m:t", NS)) for si in tree.findall("m:si", NS)]
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        paths = {r.attrib["Id"]: r.attrib["Target"] for r in rels.findall(f"{{{PACKAGE}}}Relationship")}
        for sheet in ET.fromstring(archive.read("xl/workbook.xml")).findall("m:sheets/m:sheet", NS):
            target = paths[sheet.attrib[f"{{{REL}}}id"]]
            target = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/"+target)
            records = []
            for cell in ET.fromstring(archive.read(target)).findall(".//m:sheetData/m:row/m:c", NS):
                value = cell.find("m:v", NS)
                text = (value.text or "") if value is not None else "".join(t.text or "" for t in cell.findall(".//m:t", NS))
                if cell.get("t") == "s":
                    text = strings[int(text)]
                if not text.strip():
                    continue
                address = cell.get("r")
                records.append({"sheet": sheet.attrib["name"], "cell": address, "text": text})
                if re.fullmatch(r"A\d+", address) and address != "A1":
                    slugs.append(text)
            cells.extend(records)
            sheets.append({"name": sheet.attrib["name"], "nonempty_cells": len(records)})
    norm = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
    aliases = {"real_webagents": "real"}
    known = set(map(norm, slugs))
    kept = [r["bench"] for r in csv.DictReader((ROOT / "swarm/round12/bench_table.csv").open()) if r["kept"] == "True"]
    originals = ["matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
    result = {
        "source_url": "https://docs.google.com/spreadsheets/d/1T69QyIZJRWX3d-Tmuh_lcFwtpqv9ytcLxfLWKn6gopI/export?format=xlsx",
        "retrieved_date": "2026-11-24", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "worksheets": sheets, "first_column_nonempty_data_cells": len(slugs),
        "explicit_aliases": aliases, "original_source_matches": {b: norm(aliases.get(b,b)) in known for b in originals},
        "financebench_or_legalbench_matches_any_cell": [c for c in cells if re.search(r"(?<![A-Za-z0-9])(?:finance[\s_-]*bench|legal[\s_-]*bench)(?![A-Za-z0-9])", c["text"], re.I)],
        "historical_retained_rows": len(kept),
        "unmatched_normalized_names_not_an_eligibility_decision": [b for b in kept if norm(aliases.get(b,b)) not in known],
        "limitation": "Current snapshot, not proof of historical inventory contents; aliases and source-family coverage require organizer interpretation. Retained table entries do not establish which survived every later fitting filter.",
    }
    (HERE / "results/inventory_audit.json").write_text(json.dumps(result, indent=2))
    with (HERE / "sources/inventory_cells.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["sheet", "cell", "text"])
        writer.writeheader(); writer.writerows(cells)
    print(json.dumps({k:v for k,v in result.items() if not k.startswith("unmatched")}, indent=2))


if __name__ == "__main__":
    main()
