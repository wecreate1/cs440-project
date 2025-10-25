from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

# Set this to the top of your extracted dataset
DATA_ROOT = ROOT / "data" / "raw" / "ts"
CLASSES_FILE_CANDIDATES = [
  ROOT / "data" / "raw" / "classes.names",
  ROOT / "data" / "raw" / "classes.txt",
  DATA_ROOT / "classes.names",
  DATA_ROOT / "classes.txt",
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LBL_EXT = ".txt"

def is_image(p: Path) -> bool:
  return p.suffix.lower() in IMG_EXTS

def is_label(p: Path) -> bool:
  if p.suffix.lower() != LBL_EXT:
    return False
  # ignore known non-label text files by name
  if p.name.lower() in {"train.txt", "test.txt", "valid.txt", "val.txt", "classes.txt"}:
    return False
  return True

def read_classes():
  for cf in CLASSES_FILE_CANDIDATES:
    if cf.exists():
      return [s.strip() for s in cf.read_text().splitlines() if s.strip()]
  return None

def parse_label_lines(txt: str):
  rows = []
  for line in txt.splitlines():
    line = line.strip()
    if not line:
      continue
    parts = line.split()
    if len(parts) < 5:
      # malformed line (not YOLO format)
      rows.append(("MALFORMED", line))
      continue
    try:
      cid = int(parts[0])
      x, y, w, h = map(float, parts[1:5])
    except Exception:
      rows.append(("MALFORMED", line))
      continue
    rows.append((cid, (x, y, w, h)))
  return rows

def main():
  if not DATA_ROOT.exists():
    print(f"DATA_ROOT does not exist: {DATA_ROOT}")
    return

  # Collect recursively, with mixed-case extensions
  all_files = list(DATA_ROOT.rglob("*"))
  images = [p for p in all_files if p.is_file() and is_image(p)]
  labels = [p for p in all_files if p.is_file() and is_label(p)]

  # Stem-based matching across the whole tree
  img_stems = {p.stem for p in images}
  lbl_stems = {p.stem for p in labels}

  matched_label_files = [p for p in labels if p.stem in img_stems]
  labels_without_images = sorted([p for p in labels if p.stem not in img_stems])
  images_without_labels = sorted([p for p in images if p.stem not in lbl_stems])

  classes = read_classes()
  if classes:
    print("classes file:")
    for i, c in enumerate(classes):
      print(f"  {i}: {c}")
  else:
    print("No classes file found (classes.names / classes.txt)")

  print(f"\nFound {len(images)} image files")
  print(f"Found {len(labels)} .txt files (candidate labels)")
  print(f"Matched {len(matched_label_files)} label files to existing images")

  # Count instances + track malformed
  cnt = Counter()
  malformed = []
  empty_labels = []

  for p in matched_label_files:
    txt = p.read_text().strip()
    if not txt:
      empty_labels.append(p)
      continue
    rows = parse_label_lines(txt)
    for r in rows:
      if r[0] == "MALFORMED":
        malformed.append((p, r[1]))
      else:
        cid, _ = r
        cnt[cid] += 1

  if cnt:
    print("\ninstance counts:")
    for cid, n in sorted(cnt.items()):
      name = classes[cid] if classes and cid < len(classes) else str(cid)
      print(f"  {cid:2d} ({name}): {n}")
    print(f"\nTotal labeled boxes: {sum(cnt.values())}")
  else:
    print("\nNo labeled boxes counted (all empty or malformed?)")

  # Diagnostics
  if empty_labels:
    print(f"\nEmpty label files (no objects): {len(empty_labels)} (showing up to 10)")
    for p in empty_labels[:10]:
      print(" ", p.relative_to(DATA_ROOT))

  if malformed:
    print(f"\nMalformed label lines: {len(malformed)} (showing up to 10)")
    for p, line in malformed[:10]:
      print(" ", p.relative_to(DATA_ROOT), "->", line)

  if labels_without_images:
    print(f"\nLabels without matching images: {len(labels_without_images)} (up to 10)")
    for p in labels_without_images[:10]:
      print(" ", p.relative_to(DATA_ROOT))

  if images_without_labels:
    print(f"\nImages without matching labels: {len(images_without_labels)} (up to 10)")
    for p in images_without_labels[:10]:
      print(" ", p.relative_to(DATA_ROOT))

if __name__ == "__main__":
  main()
