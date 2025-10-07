import argparse
import csv
import re
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import PyPDF2
import fitz  # PyMuPDF


# ---------------------------- Utilities ----------------------------
def sanitize(name: str) -> str:
    """Make a filename safe across Windows/macOS/Linux."""
    name = re.sub(r'[\\/:*?"<>|]', ' ', name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def load_reader(pdf_path: Path) -> PyPDF2.PdfReader:
    return PyPDF2.PdfReader(str(pdf_path))


# ---------------------------- Bookmark split ----------------------------
@dataclass
class OutlineNode:
    depth: int
    title: str
    page_index: int  # 0-based


def flatten_outline(reader: PyPDF2.PdfReader) -> List[OutlineNode]:
    try:
        outline = reader.outline  # PyPDF2 >= 3
    except Exception:
        try:
            outline = reader.getOutlines()  # older
        except Exception:
            outline = []

    def walk(items, depth=0):
        for it in items or []:
            if isinstance(it, list):
                yield from walk(it, depth + 1)
            else:
                title = getattr(it, "title", None) or getattr(it, "/Title", None) or "Untitled"
                # Page index resolution varies by PyPDF2 version
                page: Optional[int] = None
                try:
                    page = reader.get_destination_page_number(it)
                except Exception:
                    try:
                        page = reader.getDestinationPageNumber(it)  # type: ignore[attr-defined]
                    except Exception:
                        page = None
                if page is not None:
                    yield OutlineNode(depth=depth, title=str(title), page_index=int(page))

    flat = list(walk(outline, 0))
    flat.sort(key=lambda x: x.page_index)
    return flat


def page_ranges(chapter_nodes: List[OutlineNode], last_idx: int) -> List[Tuple[int, int]]:
    starts = [n.page_index for n in chapter_nodes]
    out: List[Tuple[int, int]] = []
    for i, s in enumerate(starts):
        e = starts[i + 1] - 1 if i < len(starts) - 1 else last_idx
        if e < s:
            e = s
        out.append((s, e))
    return out


def choose_levels(flat: List[OutlineNode], mode: str) -> Tuple[Optional[int], int]:
    """Return (section_depth or None, chapter_depth)."""
    depths = sorted({n.depth for n in flat})
    if not depths:
        raise ValueError("No outline depths available.")

    if mode == "chapter_only" or len(depths) == 1:
        # pick the most common depth as chapter level
        from collections import Counter

        counts = Counter(n.depth for n in flat)
        chapter_depth = max(counts, key=lambda d: counts[d])
        return None, chapter_depth
    else:
        # assume the 0th is section, 1st is chapter
        return depths[0], depths[1]


@dataclass
class ChapterInfo:
    index: int
    section_title: str
    chapter_title: str
    start_page_1based: int
    end_page_1based: int
    base_name: str  # e.g., "01 - Section … - 1. Chapter …"
    pdf_path: Path


def split_into_chapters(
    src_pdf: Path, out_root: Path, mode: str = "auto"
) -> List[ChapterInfo]:
    logger = logging.getLogger(__name__)
    logger.info("Splitting into chapters from bookmarks …")
    logger.info("Source PDF: %s", src_pdf)
    reader = load_reader(src_pdf)
    last_idx = len(reader.pages) - 1
    flat = flatten_outline(reader)
    if not flat:
        raise SystemExit("No bookmarks found in this PDF.")

    section_depth, chapter_depth = choose_levels(flat, mode)
    sections = [n for n in flat if section_depth is not None and n.depth == section_depth]
    chapters = [n for n in flat if n.depth == chapter_depth]
    ranges = page_ranges(chapters, last_idx)

    def section_for(page_idx: int) -> Optional[str]:
        if not sections:
            return None
        prev = [n.title for n in sections if n.page_index <= page_idx]
        return prev[-1] if prev else None

    out_root.mkdir(parents=True, exist_ok=True)
    digits = max(2, len(str(len(chapters))))

    chapter_infos: List[ChapterInfo] = []
    logger.info("Detected %d chapters", len(chapters))
    for idx, (chap, (s, e)) in enumerate(zip(chapters, ranges), start=1):
        sect_title = section_for(chap.page_index) or ""
        composed = f"{sect_title} - {chap.title}" if sect_title else chap.title
        fname_base = f"{idx:0{digits}d} - {sanitize(composed)}"
        pdf_name = f"{fname_base}.pdf"

        writer = PyPDF2.PdfWriter()
        for i in range(s, e + 1):
            writer.add_page(reader.pages[i])

        out_path = out_root / pdf_name
        with open(out_path, "wb") as f:
            writer.write(f)
        logger.info("[%02d] Wrote chapter PDF: %s (pages %d-%d)", idx, out_path.name, s + 1, e + 1)

        chapter_infos.append(
            ChapterInfo(
                index=idx,
                section_title=sect_title,
                chapter_title=chap.title,
                start_page_1based=s + 1,
                end_page_1based=e + 1,
                base_name=fname_base,
                pdf_path=out_path,
            )
        )

    return chapter_infos


# ---------------------------- Strip references ----------------------------
HEADERS = [
    r"^\s*references\s*$",
    r"^\s*suggested\s+reading\s*$",
    r"^\s*bibliography\s*$",
]
HEADER_RX = re.compile("|".join(HEADERS), re.IGNORECASE | re.MULTILINE)

MIN_NUMBERED_CITATIONS = 6  # lines like "1. ..." or "[1] ..."
CITATION_LINE_RX = re.compile(r"^\s*(\[\d+\]|\d{1,3}\.)\s+", re.MULTILINE)

TOP_PORTION_FRACTION = 0.35  # top 35% of page text


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


def first_references_page(reader: PyPDF2.PdfReader) -> Optional[int]:
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


def strip_references_from_pdf(src_pdf: Path, dst_pdf: Path) -> Tuple[int, List[int]]:
    logger = logging.getLogger(__name__)
    logger.debug("Opening PDF for reference stripping: %s", src_pdf)
    reader = load_reader(src_pdf)
    writer = PyPDF2.PdfWriter()
    n = len(reader.pages)

    start_ref_idx = first_references_page(reader)  # 0-based
    last_keep = n - 1 if start_ref_idx is None else max(0, start_ref_idx - 1)

    removed_pages_1based: List[int] = []
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


# ---------------------------- Figures & tables extraction ----------------------------
FIG_PAGE_RX = re.compile(r"(?:^|\n)(?:•\s*[Ff]ig|^\s*[Ff]ig)(?:ure)?\.?\s+\d+\.?\d*", re.MULTILINE)
FIG_LABEL_RX = re.compile(r"(?:^|\n)(?:•\s*)?([Ff]ig(?:ure)?\.?\s+\d+\.?\d*)", re.MULTILINE)
TABLE_PAGE_RX = re.compile(r"(?:^|\n)\s*TABLE\s+\d+\.?\d*", re.MULTILINE)
TABLE_LABEL_RX = re.compile(r"(?:^|\n)\s*(TABLE\s+\d+\.?\d*)", re.MULTILINE)


@dataclass
class ExtractSummary:
    total_pages_with_content: int
    items_by_page: List[Tuple[int, List[str]]]
    pdf_path: Optional[Path]


def extract_figures_and_tables(
    pdf_path: Path,
    out_dir: Path,
    base_name: str,
    zoom: float = 3.0,
    image_quality: int = 90,
) -> ExtractSummary:
    logger = logging.getLogger(__name__)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    out_pdf = fitz.open()

    items_by_page: List[Tuple[int, List[str]]] = []
    extracted_count = 0

    logger.info("Scanning for figures/tables in: %s", pdf_path.name)
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text() or ""

        has_figure = bool(FIG_PAGE_RX.search(text))
        has_table = bool(TABLE_PAGE_RX.search(text))
        if not (has_figure or has_table):
            continue

        content_items: List[str] = []
        if has_figure:
            for fig in FIG_LABEL_RX.findall(text):
                fig_clean = re.sub(r"[•\s]", "", fig).strip()  # e.g., "Fig.1.1"
                if fig_clean:
                    content_items.append(fig_clean)
        if has_table:
            for tab in TABLE_LABEL_RX.findall(text):
                tab_clean = tab.strip()  # keep space: "TABLE 1.1"
                if tab_clean:
                    content_items.append(tab_clean)

        if not content_items:
            # if page had markers but we failed to extract labels, still capture
            content_items = ["content"]

        # Render page to image
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("jpeg", jpg_quality=image_quality)

        # Compose filename — include first few labels to keep names manageable
        label = "_".join(content_items[:3])
        if len(content_items) > 3:
            label += "_plus"

        img_name = f"{base_name}_page{page_idx + 1:03d}_{label}.jpg"
        img_path = out_dir / img_name
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        # Add this image as a page to the figures/tables PDF
        tmp = fitz.open()
        p = tmp.new_page(width=pix.width, height=pix.height)
        p.insert_image(p.rect, stream=img_bytes)
        out_pdf.insert_pdf(tmp)
        tmp.close()

        logger.debug("  page %03d → %s", page_idx + 1, img_name)
        items_by_page.append((page_idx + 1, content_items))
        extracted_count += 1

    pdf_out_path: Optional[Path] = None
    if extracted_count > 0:
        pdf_out_path = out_dir / f"{base_name}_figures_and_tables.pdf"
        out_pdf.save(pdf_out_path)
        logger.info("Figures/tables PDF: %s (pages: %d)", pdf_out_path.name, extracted_count)

    out_pdf.close()
    doc.close()

    return ExtractSummary(
        total_pages_with_content=extracted_count,
        items_by_page=items_by_page,
        pdf_path=pdf_out_path,
    )


# ---------------------------- Orchestrator ----------------------------
def build_book_assets(
    src_pdf: Path,
    out_root: Path,
    mode: str = "auto",
    keep_raw: bool = False,
) -> None:
    logger = logging.getLogger(__name__)
    out_root.mkdir(parents=True, exist_ok=True)

    # Step 1: Split book into chapter PDFs with sequential prefixes
    raw_chapters_dir = out_root / "_chapters_raw"
    chapters = split_into_chapters(src_pdf, out_root=raw_chapters_dir, mode=mode)

    # Step 2: Per chapter — create output folder, strip references, then extract figures/tables
    manifest_rows = []
    total = len(chapters)
    ok = 0
    failed = 0
    chapters_with_no_figs = 0
    total_fig_pages = 0

    for idx, ch in enumerate(chapters, 1):
        logger.info("Processing chapter %d/%d: %s", idx, total, ch.base_name)
        chapter_folder = out_root / ch.base_name
        chapter_folder.mkdir(parents=True, exist_ok=True)

        # 2.1 Strip references from chapter PDF
        cleaned_pdf = chapter_folder / f"{ch.base_name}.pdf"
        try:
            kept_pages, removed_pages = strip_references_from_pdf(ch.pdf_path, cleaned_pdf)
            # Optionally remove the intermediate raw chapter to avoid duplicates
            if not keep_raw:
                try:
                    ch.pdf_path.unlink(missing_ok=True)
                except Exception as del_err:
                    logger.debug("  (raw cleanup skipped) %s", del_err)
            if removed_pages:
                logger.info("  → Cleaned references: removed %d page(s)", len(removed_pages))
            else:
                logger.info("  → No references detected")

        # 2.2 Extract figures/tables and produce images + compiled PDF
            extract_summary = extract_figures_and_tables(
                pdf_path=cleaned_pdf,
                out_dir=chapter_folder,
                base_name=ch.base_name,
            )
            if extract_summary.total_pages_with_content == 0:
                chapters_with_no_figs += 1
            total_fig_pages += extract_summary.total_pages_with_content

            # Manifest entry
            manifest_rows.append(
                {
                    "index": ch.index,
                    "section": ch.section_title,
                    "chapter_title": ch.chapter_title,
                    "start_page_1based": ch.start_page_1based,
                    "end_page_1based": ch.end_page_1based,
                    "chapter_pdf": str(cleaned_pdf.name),
                    "removed_ref_pages_1based": ",".join(map(str, removed_pages)) if removed_pages else "",
                    "figtab_pdf": extract_summary.pdf_path.name if extract_summary.pdf_path else "",
                    "figtab_pages": extract_summary.total_pages_with_content,
                }
            )
            ok += 1
        except Exception as e:
            failed += 1
            logger.error("Chapter failed: %s (%s)", ch.base_name, e)
            manifest_rows.append(
                {
                    "index": ch.index,
                    "section": ch.section_title,
                    "chapter_title": ch.chapter_title,
                    "start_page_1based": ch.start_page_1based,
                    "end_page_1based": ch.end_page_1based,
                    "chapter_pdf": "ERROR",
                    "removed_ref_pages_1based": "ERROR",
                    "figtab_pdf": "ERROR",
                    "figtab_pages": 0,
                }
            )

    # Step 3: Write top-level manifest
    manifest_path = out_root / "manifest.csv"
    if manifest_rows:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "index",
                    "section",
                    "chapter_title",
                    "start_page_1based",
                    "end_page_1based",
                    "chapter_pdf",
                    "removed_ref_pages_1based",
                    "figtab_pdf",
                    "figtab_pages",
                ],
            )
            w.writeheader()
            w.writerows(manifest_rows)
    logger.info(
        "\nSummary: chapters=%d, success=%d, failed=%d, chapters_no_figs=%d, figure_pages=%d",
        total,
        ok,
        failed,
        chapters_with_no_figs,
        total_fig_pages,
    )
    # Optionally cleanup the raw chapters folder entirely if we removed files
    if not keep_raw and raw_chapters_dir.exists():
        try:
            # Remove the folder tree; safe as it's inside output/
            shutil.rmtree(raw_chapters_dir)
            logger.info("Removed intermediate folder: %s", raw_chapters_dir)
        except Exception as e:
            logger.debug("Could not remove %s: %s", raw_chapters_dir, e)
    logger.info("Manifest saved: %s", manifest_path)
    logger.info("Done. Outputs in: %s", out_root)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Split a book PDF into chapter folders, strip references, and extract figures/tables "
            "(PDF + images) in a Windows-friendly structure."
        )
    )
    ap.add_argument(
        "pdf",
        nargs="?",
        default=None,
        help="Source PDF (default: tries 'book.pdf' then a single *.pdf in cwd)",
    )
    ap.add_argument("-o", "--out", default="output", help="Output folder (default: output)")
    ap.add_argument(
        "--mode",
        choices=["auto", "chapter_only", "section_chapter"],
        default="auto",
        help="Bookmark levels: auto detect, or force chapter-only or section+chapter",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    ap.add_argument("-k", "--keep-raw", action="store_true", help="Keep intermediate _chapters_raw PDFs (default: delete)")
    return ap.parse_args()


def resolve_default_pdf() -> Optional[Path]:
    cwd = Path.cwd()
    # Prefer explicit 'book.pdf' if present
    cand = cwd / "book.pdf"
    if cand.exists():
        return cand
    # Otherwise choose the single .pdf if only one exists
    pdfs = [p for p in cwd.iterdir() if p.suffix.lower() == ".pdf"]
    if len(pdfs) == 1:
        return pdfs[0]
    return None


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    src = Path(args.pdf) if args.pdf else resolve_default_pdf()
    if not src or not src.exists():
        raise SystemExit(
            "Provide a source PDF path, or place a single 'book.pdf' (or only one *.pdf) in the root."
        )

    out_root = Path(args.out)
    build_book_assets(src_pdf=src, out_root=out_root, mode=args.mode, keep_raw=bool(args.keep_raw))


if __name__ == "__main__":
    main()
