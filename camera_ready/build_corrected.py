"""Build a separately named correction; never overwrite the selected submission."""
from pathlib import Path
import hashlib
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_src/v6_combo_w20_c1"
DEST = ROOT / "camera_ready/corrected_w20"


def build():
    source = (SOURCE / "model.py").read_text()
    old_item = '''    # Head + tail + length: cheap for 100k-character items, collision-safe in practice.
    text = str(item.get("item_content", ""))
    return hashlib.sha1(f"{len(text)}|{text[:2048]}|{text[-2048:]}".encode()).hexdigest()'''
    new_item = '''    # Full text: distinct middle sections must not share an item effect.
    text = str(item.get("item_content", ""))
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()'''
    old_key = '''        h = hashlib.sha1()
        for subject, item, y in _entries(labeled):
            h.update(_sk(subject).encode())
            h.update(str(item.get("benchmark_id", "")).encode())
            h.update(_item_key(item).encode())
            h.update(b"1" if y else b"0")
        return h.hexdigest()'''
    new_key = '''        h = hashlib.sha256()
        for subject, item, y in _entries(labeled):
            # Frame every record and include the features used by the posterior.
            record = [_sk(subject), str(item.get("benchmark_id", "")),
                      _item_key(item), str(item.get("item_features", "") or ""), int(y)]
            encoded = json.dumps(record, ensure_ascii=True, separators=(",", ":")).encode()
            h.update(len(encoded).to_bytes(8, "big"))
            h.update(encoded)
        return h.hexdigest()'''
    for old, new in ((old_item, new_item), (old_key, new_key)):
        assert source.count(old) == 1, "Frozen source differs from the reviewed release"
        source = source.replace(old, new)
    DEST.mkdir(exist_ok=True)
    (DEST / "model.py").write_text(source)
    for name in ("params.json", "subject_prior.json", "prediction_config.json"):
        shutil.copyfile(SOURCE / name, DEST / name)
    return {"original_model_sha256": hashlib.sha256((SOURCE / "model.py").read_bytes()).hexdigest(),
            "corrected_model_sha256": hashlib.sha256((DEST / "model.py").read_bytes()).hexdigest()}


if __name__ == "__main__":
    print(build())
