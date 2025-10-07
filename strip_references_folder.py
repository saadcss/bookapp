import re
import csv
import argparse
import logging
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
    logger = logging.getLogger(__name__)
    logger.debug(f"Opening PDF for reference stripping: {src_pdf}")
    reader = PyPDF2.PdfReader(str(src_pdf))
    writer = PyPDF2.PdfWriter()
    n = len(reader.pages)

    logger.debug(f"Pages detected: {n}")
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

    logger.debug(
        "Reference detection: start_ref_idx=%s, last_keep=%s, kept=%s, removed=%s",
        start_ref_idx,
        last_keep,
        kept_count,
        len(removed_pages_1based),
    )

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
    ap.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting reference stripping")
    logger.info("Input folder: %s", in_dir)
    logger.info("Output folder: %s", out_dir)

    pdfs = sorted([p for p in in_dir.iterdir() if p.suffix.lower() == ".pdf"])
    if not pdfs:
        logger.warning("No PDFs found in %s", in_dir)
        return

    total = len(pdfs)
    ok = 0
    no_refs = 0
    failed = 0

    manifest_rows = []
    for i, pdf in enumerate(pdfs, 1):
        logger.info("(%d/%d) Processing: %s", i, total, pdf.name)
        dst = out_dir / pdf.name
        try:
            kept_pages, removed_pages = strip_references_from_pdf(pdf, dst)
            if removed_pages:
                logger.info("→ Wrote %s (removed %d page(s))", dst.name, len(removed_pages))
            else:
                logger.info("→ Wrote %s (no references detected)", dst.name)
                no_refs += 1

            manifest_rows.append({
                "filename": pdf.name,
                "removed_pages_1based": ",".join(map(str, removed_pages)) if removed_pages else "",
                "kept_pages_count": kept_pages,
            })
            ok += 1
        except Exception as e:
            failed += 1
            logger.error("Failed: %s (%s)", pdf.name, e)
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
    logger.info("\nSummary: processed=%d, success=%d, no_refs=%d, failed=%d", total, ok, no_refs, failed)
    logger.info("Manifest saved: %s", mf)
    logger.info("Done. Cleaned files saved to: %s", out_dir)

if __name__ == "__main__":
    main()
