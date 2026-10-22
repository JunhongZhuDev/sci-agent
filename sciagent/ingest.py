"""PDF -> structured chunks (text, figure captions, tables) + figure images.

Design choices
- Section-aware chunking: detect headings, chunk within a section (~220 words,
  40-word overlap, sentence boundaries). Section names are kept as metadata and
  prepended to the chunk text ("contextual chunking"), which helps BM25 and
  dense retrieval alike.
- References/bibliography are dropped: they match many queries lexically but
  never contain evidence.
- Figures: scientific CFD figures are mostly *vector graphics*, so extracting
  embedded bitmaps misses them. Instead we locate each "Figure N" caption and
  render the page region above it to PNG (PyMuPDF). The caption becomes a
  searchable `figure` chunk that points at the PNG, so the agent can find a
  figure through text search and then *look at it* with a vision model.
- Tables: "Table N" captions + pdfplumber table extraction rendered as
  pipe-separated rows -> `table` chunks (numbers become searchable/parsable).
- The arXiv abstract is always added as an `abstract` chunk, so a paper whose
  PDF failed to download/parse is still represented (graceful degradation).

Usage:
    python -m sciagent.ingest
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import config

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24 (new import name)
except Exception:  # pragma: no cover
    try:
        import fitz  # older PyMuPDF
    except Exception:
        fitz = None

HEADING_RE = re.compile(
    r"^\s*(?:(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+)?"
    r"(Abstract|Introduction|Related Work|Background|Method(?:s|ology)?|Approach|"
    r"Model|Problem (?:Setup|Formulation)|Experiments?|Numerical (?:Experiments|Results|Examples)|"
    r"Results(?: and Discussion)?|Discussion|Limitations|Conclusions?|Summary|"
    r"[A-Z][A-Za-z\- ]{2,50})\s*$"
)
CAPTION_RE = re.compile(r"^\s*(Fig\.?|Figure|Table)\s*(\d+)\s*[:.|]\s*(.*)", re.I)
REFS_RE = re.compile(r"^\s*(\d+\.?\s+)?(References|Bibliography|REFERENCES)\s*$")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


# ---------------------------------------------------------------- utilities
def clean_text(t: str) -> str:
    t = t.replace("­", "")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)        # de-hyphenate line breaks
    t = re.sub(r"[ \t]+", " ", t)
    return t


def is_heading(line: str) -> str | None:
    line = line.strip()
    if not (3 <= len(line) <= 60) or line.endswith((".", ",", ";")):
        return None
    m = HEADING_RE.match(line)
    if not m:
        return None
    # unnumbered generic Title Case lines are only headings if short
    if m.group(1) is None and m.group(2) not in (
        "Abstract", "Introduction", "Related Work", "Background", "Conclusion",
        "Conclusions", "Discussion", "Results", "Experiments", "Limitations", "Summary"):
        return None
    return m.group(2).strip()


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    sents = [s.strip() for s in SENT_SPLIT.split(" ".join(text.split())) if s.strip()]
    chunks, cur, n = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and n + w > size:
            chunks.append(" ".join(cur))
            # overlap: keep trailing sentences up to `overlap` words
            keep, k = [], 0
            for prev in reversed(cur):
                k += len(prev.split())
                if k > overlap:
                    break
                keep.insert(0, prev)
            cur, n = keep, sum(len(x.split()) for x in keep)
        cur.append(s); n += w
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if len(c.split()) >= 15]


# ---------------------------------------------------------------- PDF reading
def read_pages(pdf: Path) -> list[str]:
    if fitz is not None:
        with fitz.open(pdf) as doc:
            return [p.get_text("text") for p in doc]
    from pypdf import PdfReader
    return [(p.extract_text() or "") for p in PdfReader(str(pdf)).pages]


def render_figures(pdf: Path, paper_id: str, outdir: Path, dpi: int = 110) -> dict[tuple[int, int], str]:
    """Render the region above each 'Figure N' caption. Returns {(page, num): png}."""
    if fitz is None:
        return {}
    out = {}
    with fitz.open(pdf) as doc:
        for pno, page in enumerate(doc):
            blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, ...)
            for b in blocks:
                m = CAPTION_RE.match(b[4])
                if not m or m.group(1).lower().startswith("tab"):
                    continue
                num = int(m.group(2))
                if (pno, num) in out:
                    continue
                H = page.rect.height
                top = max(0.0, b[1] - 0.55 * H)
                # stop at the nearest preceding long text block (body text)
                for o in blocks:
                    if o[3] <= b[1] - 5 and o[3] > top and len(o[4].split()) > 40:
                        top = o[3]
                clip = fitz.Rect(page.rect.x0, top, page.rect.x1, b[3])
                if clip.height < 40:
                    clip = page.rect
                png = outdir / f"{paper_id.replace('/', '_')}_p{pno + 1}_fig{num}.png"
                page.get_pixmap(dpi=dpi, clip=clip).save(png)
                out[(pno, num)] = str(png)
    return out


def extract_tables(pdf: Path, max_pages: int = 30, budget_s: float = 25.0) -> dict[int, list[str]]:
    """pdfplumber tables per page as pipe-separated text.
    pdfplumber can be very slow on figure-heavy pages, so we cap pages, skip pages with huge
    numbers of vector objects, and stop after a time budget (keeping tables found so far)."""
    import time
    try:
        import pdfplumber
    except Exception:
        return {}
    res: dict[int, list[str]] = {}
    t0 = time.perf_counter()
    try:
        with pdfplumber.open(pdf) as doc:
            for pno, page in enumerate(doc.pages[:max_pages]):
                if time.perf_counter() - t0 > budget_s:
                    print(f"  ~ table extraction budget reached for {pdf.name} (page {pno})")
                    break
                if len(page.objects.get("curve", [])) + len(page.objects.get("line", [])) > 3000:
                    continue  # dense vector plot, not a table; pdfplumber would crawl
                for tb in page.extract_tables() or []:
                    rows = [" | ".join((c or "").replace("\n", " ").strip() for c in r) for r in tb if r]
                    rows = [r for r in rows if r.strip(" |")]
                    if len(rows) >= 2:
                        res.setdefault(pno, []).append("\n".join(rows[:30]))
    except Exception as e:
        print(f"  ! table extraction failed for {pdf.name}: {e}")
    return res


# ---------------------------------------------------------------- main logic
def process_paper(p: dict, fig_dir: Path, do_tables: bool = True) -> list[dict]:
    base = {"paper_id": p["paper_id"], "title": p["title"], "year": p.get("year"),
            "authors": p.get("authors", [])[:3]}
    chunks = [{**base, "chunk_id": f"{p['paper_id']}#abs", "type": "abstract",
               "section": "Abstract", "page": 1, "text": p.get("abstract", "")}]
    pdf = Path(p["pdf_path"]) if p.get("pdf_path") else None
    if not pdf or not pdf.exists():
        return chunks  # degraded: abstract only

    try:
        pages = [clean_text(t) for t in read_pages(pdf)]
    except Exception as e:
        print(f"  ! cannot parse {pdf}: {e}")
        return chunks
    figs = render_figures(pdf, p["paper_id"], fig_dir)
    tables = extract_tables(pdf) if do_tables else {}

    section, buf, buf_page = "Body", [], 1
    seen_caps: set[tuple[str, int]] = set()
    text_chunks, n = [], 0

    def flush():
        nonlocal n
        for c in chunk_words("\n".join(buf), config.CHUNK_WORDS, config.CHUNK_OVERLAP):
            text_chunks.append({**base, "chunk_id": f"{p['paper_id']}#t{n}", "type": "text",
                                "section": section, "page": buf_page,
                                "text": f"[{section}] {c}"})
            n += 1

    stop = False
    for pno, page_text in enumerate(pages):
        lines = page_text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            if REFS_RE.match(line):
                stop = True
                break
            cap = CAPTION_RE.match(line)
            if cap:
                kind = "table" if cap.group(1).lower().startswith("tab") else "figure"
                num = int(cap.group(2))
                # caption continues until a blank line or ~5 lines
                cap_lines = [cap.group(3)]
                j = i + 1
                while (j < len(lines) and lines[j].strip() and j - i < 6
                       and not REFS_RE.match(lines[j]) and not CAPTION_RE.match(lines[j])
                       and not is_heading(lines[j])):
                    cap_lines.append(lines[j]); j += 1
                key = (kind, num)
                if key not in seen_caps:
                    seen_caps.add(key)
                    label = "Figure" if kind == "figure" else "Table"
                    c = {**base, "chunk_id": f"{p['paper_id']}#{kind[:3]}{num}", "type": kind,
                         "section": section, "page": pno + 1,
                         "text": f"{label} {num}: " + " ".join(" ".join(cap_lines).split())}
                    if kind == "figure":
                        c["image_path"] = figs.get((pno, num))
                    else:
                        tb = tables.get(pno) or []
                        if tb:
                            c["table_text"] = tb.pop(0)
                            c["text"] += "\n" + c["table_text"]
                    chunks.append(c)
                i = j
                continue
            h = is_heading(line)
            if h:
                flush()
                section, buf, buf_page = h, [], pno + 1
            else:
                if not buf:
                    buf_page = pno + 1
                buf.append(line)
            i += 1
        if stop:
            break
    flush()
    return chunks + text_chunks


class time_limit:
    """Per-paper wall-clock limit (SIGALRM; Linux/macOS main thread, e.g. Colab)."""
    def __init__(self, seconds: int):
        self.seconds = seconds

    def __enter__(self):
        import signal
        self.ok = hasattr(signal, "SIGALRM")
        if self.ok and self.seconds > 0:
            def handler(*_):
                raise TimeoutError()
            self.old = signal.signal(signal.SIGALRM, handler)
            signal.alarm(self.seconds)

    def __exit__(self, *exc):
        import signal
        if self.ok and self.seconds > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.old)
        return False


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tables", action="store_true", help="skip pdfplumber (faster)")
    ap.add_argument("--timeout", type=int, default=120, help="seconds per paper before giving up")
    ap.add_argument("--force", action="store_true", help="ignore the per-paper cache")
    a = ap.parse_args(argv)
    config.ensure_dirs()
    if not config.PAPERS_FILE.exists():
        raise SystemExit(f"{config.PAPERS_FILE} not found - run `python -m sciagent.collect` first "
                         "and check that it printed 'wrote .../papers.jsonl'.")
    papers = [json.loads(l) for l in open(config.PAPERS_FILE)]
    if not papers:
        raise SystemExit("papers.jsonl is empty - arXiv returned nothing (rate limit?). Re-run collect.")
    # per-paper cache -> the run is resumable: re-running skips papers already processed
    cache_dir = config.DATA_DIR / "chunk_cache"
    cache_dir.mkdir(exist_ok=True)
    all_chunks = []
    for i, p in enumerate(papers, 1):
        cpath = cache_dir / f"{p['paper_id'].replace('/', '_')}.json"
        if cpath.exists() and not a.force:
            cs = json.load(open(cpath))
            print(f"[{i}/{len(papers)}] {p['paper_id']}: cached ({len(cs)} chunks)", flush=True)
            all_chunks += cs
            continue
        print(f"[{i}/{len(papers)}] {p['paper_id']} {p['title'][:60]} ...", flush=True)
        cs = None
        for tables in ([not a.no_tables, False] if not a.no_tables else [False]):
            try:
                with time_limit(a.timeout):
                    cs = process_paper(p, config.FIG_DIR, do_tables=tables)
                break
            except TimeoutError:
                print(f"  ! timeout after {a.timeout}s" + (" - retrying without tables" if tables else ""), flush=True)
            except Exception as e:
                print(f"  ! failed: {type(e).__name__}: {e}", flush=True)
                break
        if cs is None:  # degrade gracefully: keep the abstract
            cs = process_paper({**p, "pdf_path": None}, config.FIG_DIR)
        kinds = {k: sum(c["type"] == k for c in cs) for k in ("text", "figure", "table")}
        print(f"      -> {len(cs)} chunks {kinds}", flush=True)
        json.dump(cs, open(cpath, "w"))
        all_chunks += cs
    with open(config.CHUNKS_FILE, "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(all_chunks)} chunks -> {config.CHUNKS_FILE}")


if __name__ == "__main__":
    main()
