import re
import csv
import argparse
from pathlib import Path
import PyPDF2

# --- Tunables ----------------------------------------------------
HEADERS = [
    r"^\s*references\s*$",
    r"^\s*suggested\s+reading\s*$",
    r"^\s*bibliography\s*$",
]
HEADER_RX = re.compile("|".join(HEADERS), re.IGNORECASE | re.MULTILINE)

MIN_NUMBERED_CITATIONS = 6  # lines like "1. ..." or "[1] ..."
CITATION_LINE_RX = re.compile(r"^\s*(\[\d+\]|\d{1,3}\.)\s+", re.MULTILINE)

TOP_PORTION_FRACTION = 0.35  # top 35% of page text
# ----------------------------------------------------------------

def extract_text(reader: PyPDF2.PdfReader, i: int) -> str:
    try:
        return reader.pages[i].extract_text() or ""
    except Exception:
        return ""

def is_references_page(text: str) -> bool:
    if not text:
        return False
    top_len = max(500, int(len(text) * TOP_PORTION_FRACTION))
    top_text = text[:top_len]

    if HEADER_RX.search(top_text):
        return True

    if len(CITATION_LINE_RX.findall(text)) >= MIN_NUMBERED_CITATIONS:
        return True

    return False

def first_references_page(reader: PyPDF2.PdfReader) -> int | None:
    n = len(reader.pages)
    # scan from end; refs are usually last
    for i in range(n - 1, -1, -1):
        if is_references_page(extract_text(reader, i)):
            # backtrack to the first page of the references block
            start = i
            while start - 1 >= 0 and is_references_page(extract_text(reader, start - 1)):
                start -= 1
            return start
    return None

def strip_references_from_pdf(src_pdf: Path, dst_pdf: Path) -> tuple[int, list[int]]:
    reader = PyPDF2.PdfReader(str(src_pdf))
    writer = PyPDF2.PdfWriter()
    n = len(reader.pages)

    start_ref_idx = first_references_page(reader)  # 0-based
    last_keep = n - 1 if start_ref_idx is None else max(0, start_ref_idx - 1)

    removed_pages_1based: list[int] = []
    kept_count = 0
    for i in range(n):
        if i <= last_keep:
            writer.add_page(reader.pages[i])
            kept_count += 1
        else:
            removed_pages_1based.append(i + 1)  # report as 1-based

    # Edge case: if nothing kept (over-detection), keep original
    if kept_count == 0:
        writer = PyPDF2.PdfWriter()
        for i in range(n):
            writer.add_page(reader.pages[i])
        removed_pages_1based = []
        kept_count = n

    dst_pdf.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_pdf, "wb") as f:
        writer.write(f)

    return kept_count, removed_pages_1based

def main():
    ap = argparse.ArgumentParser(
        description="Remove References/Suggested Reading/Bibliography pages from each chapter PDF in a folder."
    )
    ap.add_argument("in_dir", help="Folder with original chapter PDFs (e.g., out/)")
    ap.add_argument("-o", "--out_dir", default="out_no_refs", help="Destination folder (default: out_no_refs)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted([p for p in in_dir.iterdir() if p.suffix.lower() == ".pdf"])
    if not pdfs:
        print(f"No PDFs found in {in_dir}")
        return

    manifest_rows = []
    for pdf in pdfs:
        dst = out_dir / pdf.name
        try:
            kept_pages, removed_pages = strip_references_from_pdf(pdf, dst)
            status = f"removed {len(removed_pages)} page(s)" if removed_pages else "no references found"
            print(f"[OK] {pdf.name} → {dst.name}  ({status})")

            manifest_rows.append({
                "filename": pdf.name,
                "removed_pages_1based": ",".join(map(str, removed_pages)) if removed_pages else "",
                "kept_pages_count": kept_pages,
            })
        except Exception as e:
            print(f"[ERR] {pdf.name}: {e}")
            manifest_rows.append({
                "filename": pdf.name,
                "removed_pages_1based": "ERROR",
                "kept_pages_count": 0,
            })

    # Save manifest
    mf = out_dir / "manifest.csv"
    with open(mf, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "removed_pages_1based", "kept_pages_count"])
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"\nDone. Cleaned files + manifest saved to: {out_dir}")

if __name__ == "__main__":
    main()