from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Union, Dict, Any
from collections import Counter, defaultdict
from statistics import mean
import json, re, unicodedata, os, csv
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextContainer, LTTextLine, LTChar, LTLine

# ================================================================
# CONFIG RESTRAINTS
# ================================================================
TARGET_SUBJECTS = ["경찰학개론","헌법", "형사법", "경찰학", "형법", "형사소송법", "경찰학개론"]
CIRCLED_CHOICE_CHARS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳ⓛ"
CIRCLED_CHOICE_RE = re.compile(rf'^[{CIRCLED_CHOICE_CHARS}]')
PLAIN_CHOICE_RE = re.compile(r'^\s*(\(?[1-9]\d?\)|[1-9]\d?\.)\s*')
QNUM_RE = re.compile(r'^\s*(\d{1,3})\.(?!\d)\s*(.*)')
HANGUL_RUN_RE = re.compile(r"[가-힣]{2,}(?:\s*[가-힣]{2,})*")
PAGE_NO_RE = re.compile(r"^\s*-\s*\d+\s*-\s*$")
SUBJECT_HEADER_RE = re.compile(
    r"(?P<header>[【\[]\s*(?P<subject>[^】\]]+?)\s*[】\]](?:\s*\([^)]*\))?)"
)
DISPUTE_RE = re.compile(
    r"다툼이\s*있는\s*경우\s*(?P<site>.*?)\s*(?:판례|결정)에\s*의함",
    re.IGNORECASE,
)

MATRIX_LABEL_RE = re.compile(r'^[㉠-㉷]$')
MATRIX_LABEL_PREFIX_RE = re.compile(r'^([㉠-㉷])')
PAREN_MATRIX_LABEL_RE = re.compile(r'^\(?[가-힣]\)?$')
PAREN_MATRIX_PREFIX_RE = re.compile(r'^\(?([가-힣])\)')

LABEL_SEQUENCE = "㉠㉡㉢㉣㉤㉥㉦㉧㉨㉩㉪㉫㉬㉭㉮㉯㉰㉱㉲㉳㉴㉵㉶㉷"
LABEL_ORDER = {ch: idx for idx, ch in enumerate(LABEL_SEQUENCE)}
LABEL_PATTERN = re.compile(f"([{LABEL_SEQUENCE}])")

# Exact boilerplate line (no middle text)
DISPUTE_LINE_RE = re.compile(
    r"^\s*다툼이\s*있는\s*경우\s*(?:판례|결정)에\s*의함\s*$",
    re.IGNORECASE,
)
ORDER_MODE = "smart"


CIRCLED_TO_INT = {ch: idx + 1 for idx, ch in enumerate(CIRCLED_CHOICE_CHARS)}
CIRCLED_TO_INT['ⓛ'] = 1

# ================================================================
# DATA STRUCTURES
# ================================================================
@dataclass
class Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    font: str
    column: str  # "left" or "right"
    page_index: int = -1

# ================================================================
# UTILITIES
# ================================================================
def _normalize_header_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[\u00A0\u2000-\u200B]", " ", s)
    s = re.sub(r"[【】()\[\]{}·･•※〈〉《》『』—–\-:·•\s]+", "", s)
    return s.strip()

TRAILING_HEADER_SUBJECTS = {_normalize_header_text(s) for s in TARGET_SUBJECTS if s}

def _extract_subject_and_target(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts both subject and target from headers like:
    【헌 법】(일 반 공 채･101경 비 단)
    """
    if not raw:
        return None, None
    subj, target = None, None
    m1 = re.search(r"[【\[]\s*([^】\]]+?)\s*[】\]]", raw)
    if m1:
        subj = _normalize_header_text(m1.group(1))
    m2 = re.search(r"\(([^)]+)\)", raw)
    if m2:
        target = _normalize_header_text(m2.group(1))
    return subj, target


def _strip_trailing_subject_header(text: str) -> str:
    if not text:
        return ""
    s = text.rstrip()
    while s.endswith("】") or s.endswith("]"):
        close_char = s[-1]
        open_char = "【" if close_char == "】" else "["
        start = s.rfind(open_char)
        if start == -1:
            break
        inner = s[start + 1 : -1]
        normalized = _normalize_header_text(inner)
        if normalized and normalized in TRAILING_HEADER_SUBJECTS:
            s = s[:start].rstrip()
            continue
        break
    return s


def _remove_subject_headers(text: str) -> str:
    if not text:
        return ""

    def _repl(match: re.Match[str]) -> str:
        subject = _normalize_header_text(match.group("subject"))
        if subject and subject in TRAILING_HEADER_SUBJECTS:
            return " "
        return match.group(0)

    cleaned = SUBJECT_HEADER_RE.sub(_repl, text)
    return cleaned


def _clean_option_text(text: Optional[str]) -> str:
    cleaned = (text or "").replace("\u00A0", " ").strip()
    cleaned = _remove_subject_headers(cleaned)
    cleaned = _strip_trailing_subject_header(cleaned)
    if "/" in cleaned:
        cleaned = re.sub(r"([㉠-㉷])\s*이다\.(?=\s*/)", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _order_labelled_clauses(text: str) -> str:
    if not text:
        return text

    def _extract_disputes(segment: str) -> Tuple[str, List[str]]:
        """Remove dispute boilerplate and return cleaned text plus removed snippets."""
        disputes: List[str] = []

        def _repl(match: re.Match[str]) -> str:
            snippet = match.group(0).strip()
            if snippet:
                start, end = match.span()
                left = segment[:start]
                right = segment[end:]
                wrap_left = left.rstrip().endswith("(")
                wrap_right = right.lstrip().startswith(")")
                if wrap_left and wrap_right and not snippet.startswith("("):
                    snippet = f"({snippet})"
            if snippet:
                disputes.append(snippet)
            return " "

        cleaned = DISPUTE_LINE_RE.sub(_repl, segment)
        cleaned = DISPUTE_RE.sub(_repl, cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\(\s*\)", " ", cleaned)
        return cleaned.strip(), disputes

    positions: List[Tuple[int, str]] = []
    for match in LABEL_PATTERN.finditer(text):
        label = match.group(1)
        start = match.start()
        prev_idx = start - 1
        while prev_idx >= 0 and text[prev_idx].isspace():
            prev_idx -= 1
        prev_char = text[prev_idx] if prev_idx >= 0 else ""
        is_valid = prev_idx < 0 or prev_char in {".", "?", "!", "(", ")", "“", "”", '"', "'", "·", ":", ";"}
        if not is_valid and prev_char:
            if prev_char in {"-", "–", "—", "〔", "[", "〈"}:
                is_valid = True
        if not is_valid:
            continue
        positions.append((start, label))

    if len(positions) < 2:
        return text

    label_sequence = [LABEL_ORDER.get(label, 0) for _, label in positions]
    if label_sequence == sorted(label_sequence):
        return text

    segments: List[Tuple[str, str, int, int]] = []
    disputes_accum: List[str] = []
    for idx, (start, label) in enumerate(positions):
        seg_start = start
        prev_idx = seg_start - 1
        while prev_idx >= 0 and text[prev_idx].isspace():
            prev_idx -= 1
        if prev_idx >= 0 and text[prev_idx] == "(":
            seg_start = prev_idx
        next_start = positions[idx + 1][0] if idx + 1 < len(positions) else len(text)
        segment_text = text[seg_start:next_start]
        cleaned_segment, seg_disputes = _extract_disputes(segment_text)
        segments.append((label, cleaned_segment, seg_start, next_start))
        disputes_accum.extend(seg_disputes)

    earliest_start = min(seg_start for _, _, seg_start, _ in segments)
    latest_end = max(seg_end for _, _, _, seg_end in segments)

    prefix_text = text[:earliest_start]
    prefix_clean, prefix_disputes = _extract_disputes(prefix_text)
    disputes_accum = prefix_disputes + disputes_accum

    suffix_text = text[latest_end:]
    suffix_clean, suffix_disputes = _extract_disputes(suffix_text)
    disputes_accum.extend(suffix_disputes)

    seen_disputes: set[str] = set()
    ordered_disputes: List[str] = []
    for snippet in disputes_accum:
        if snippet and snippet not in seen_disputes:
            seen_disputes.add(snippet)
            ordered_disputes.append(snippet)

    segments.sort(key=lambda item: (LABEL_ORDER.get(item[0], 0), item[2]))

    rebuilt: List[str] = []
    if prefix_clean:
        rebuilt.append(prefix_clean)
    rebuilt.extend(ordered_disputes)
    for _, seg_text, _, _ in segments:
        if seg_text:
            rebuilt.append(seg_text.strip())
    if suffix_clean:
        rebuilt.append(suffix_clean)

    return " ".join(part for part in rebuilt if part).strip()

def _line_font_stats(ln: LTTextLine) -> Tuple[float, str]:
    sizes, fonts = [], []
    for obj in ln:
        if isinstance(obj, LTChar):
            sizes.append(obj.size)
            fonts.append(obj.fontname)
    if not sizes:
        return (10.0, "UNKNOWN")
    return (mean(sizes), max(set(fonts), key=fonts.count))

def _is_choice_anchor(text: str, allow_plain: bool = True) -> bool:
    t = text.strip()
    if CIRCLED_CHOICE_RE.match(t): return True
    if allow_plain and PLAIN_CHOICE_RE.match(t): return True
    return False

def _auto_x_threshold(x_centers: List[float]) -> Optional[float]:
    if not x_centers: return None
    c1, c2 = min(x_centers), max(x_centers)
    for _ in range(15):
        left = [x for x in x_centers if abs(x - c1) <= abs(x - c2)]
        right = [x for x in x_centers if abs(x - c2) < abs(x - c1)]
        if left: c1 = sum(left) / len(left)
        if right: c2 = sum(right) / len(right)
    return (c1 + c2) / 2.0


def _block_line_sort_key(ln: Line, anchor_column: str):
    text = (ln.text if isinstance(ln, Line) else str(ln) or "").strip()
    if QNUM_RE.match(text):
        priority = 0
    else:
        priority = 1
    return (priority, -ln.y1, ln.x0)


def _choice_label_to_int(label: str) -> Optional[int]:
    if not label:
        return None
    if label in CIRCLED_TO_INT:
        return CIRCLED_TO_INT[label]
    digits = re.sub(r'\D', '', label)
    if digits:
        try:
            return int(digits)
        except ValueError:
            return None
    return None


def _collect_choice_ordinals(lines: Iterable[Line]) -> List[int]:
    ordinals: List[int] = []
    for ln in lines:
        text = (ln.text if isinstance(ln, Line) else str(ln) or "").strip()
        if not text:
            continue
        if QNUM_RE.match(text):
            continue
        if CIRCLED_CHOICE_RE.match(text):
            idx = _choice_label_to_int(text[0])
            if idx is not None:
                ordinals.append(idx)
            continue
        m = PLAIN_CHOICE_RE.match(text)
        if m:
            idx = _choice_label_to_int(m.group(1))
            if idx is not None:
                ordinals.append(idx)
    return ordinals


def _extract_leading_option_lines(lines: List[Line]) -> List[Line]:
    """Return consecutive leading lines that look like option continuations."""
    leading: List[Line] = []
    seen_option = False
    for ln in lines:
        text = (ln.text if isinstance(ln, Line) else str(ln) or "").strip()
        if not text:
            if seen_option:
                leading.append(ln)
                continue
            break
        if QNUM_RE.match(text):
            break
        parts = split_inline_options(text)
        if parts:
            leading.append(ln)
            seen_option = True
            continue
        if seen_option:
            # treat any non-question text following an option anchor as continuation
            leading.append(ln)
            continue
        break
    return leading

# ================================================================
# FOOTER REMOVAL
# ================================================================
def _get_bbox_and_text(line):
    # Works with your Line dataclass OR (x0,y0,x1,y1,text) tuples
    if hasattr(line, "x0") and hasattr(line, "text"):
        x0, y0, x1, y1, txt = line.x0, line.y0, line.x1, line.y1, line.text
    else:
        x0, y0, x1, y1, txt = line
    return x0, y0, x1, y1, txt


def filter_page_numbers(lines, page_w: float, page_h: float):
    """
    Drop lines that look like centered page numbers ('- 5 -') that appear near the bottom.
    """
    out = []
    page_center_x = page_w * 0.5
    for line in lines:
        x0, y0, x1, y1, txt = _get_bbox_and_text(line)
        t = (txt or "").replace("\u00A0", " ").strip()

        # only consider lower 25% of the page as "footer zone"
        in_footer_zone = y1 < page_h * 0.04
        looks_like_page_no = bool(PAGE_NO_RE.match(t))
        roughly_centered = abs(((x0 + x1) * 0.5) - page_center_x) < page_w * 0.25

        if in_footer_zone and looks_like_page_no and roughly_centered:
            continue  # drop it

        out.append(line)
    return out

# ================================================================
# DISPUTE TRIGGER
# ================================================================

def _norm_space(s: str) -> str:
    return (s or "").replace("\u00A0", " ").strip()

def _clean_dispute_site(s: str) -> str:
    s = _norm_space(s)
    # strip most common surrounding brackets/punct
    s = s.strip("()[]{}〈〉《》『』“”\"' ")
    return s


def _extract_dispute_flags(text: str) -> Tuple[bool, Optional[str]]:
    """Detect dispute boilerplate and return (bool, cleaned_site)."""
    normalized = _norm_space(text)
    if not normalized:
        return False, None

    match = DISPUTE_RE.search(normalized)
    if match:
        site_raw = match.groupdict().get("site", "")
        site_clean = _clean_dispute_site(site_raw)
        return True, site_clean or None

    if DISPUTE_LINE_RE.search(normalized):
        return True, None

    return False, None


# ================================================================
# SUBJECT DETECTION
# ================================================================
def _detect_subject_for_page(all_elements: List[Any], page_width: float, page_height: float):
    """Detect subject & target if wide horizontal line (>=70% width) exists near top."""
    candidate_lines = []
    for el in all_elements:
        if isinstance(el, LTLine):
            width, height = abs(el.x1 - el.x0), abs(el.y1 - el.y0)
            if width >= page_width * 0.7 and height < 5:
                candidate_lines.append(el.y1)
    if not candidate_lines:
        return None, None
    top_line_y = max(candidate_lines)
    text_lines = [el for el in all_elements if isinstance(el, Line) and el.y0 > top_line_y]
    if not text_lines:
        return None, None
    merged = " ".join(l.text for l in sorted(text_lines, key=lambda L: (-L.y1, L.x0)))
    subj, target = _extract_subject_and_target(merged)
    return subj, target

# ================================================================
# PAGE SPLITTING
# ================================================================
def _split_block_on_gaps(block: Dict[str, Any], keep_first_qnum: Optional[str]) -> List[Dict[str, Any]]:
    """Split a block whenever vertical gaps suggest a new section."""
    lines: List[Line] = block["lines"]
    if len(lines) <= 1:
        clone = dict(block)
        clone["qnum"] = keep_first_qnum
        return [clone]

    # Questions with an explicit number can legitimately span columns, so allow
    # tighter splitting to peel off follow-up fragments (e.g., next question's
    # options) while keeping unnumbered continuations intact.
    threshold = 45.0 if keep_first_qnum else 80.0

    chunks: List[List[Line]] = []
    current: List[Line] = [lines[0]]
    for prev, cur in zip(lines, lines[1:]):
        prev_has_ord = bool(_collect_choice_ordinals([prev]))
        cur_has_ord = bool(_collect_choice_ordinals([cur]))
        dynamic_threshold = threshold
        if prev_has_ord and not cur_has_ord:
            dynamic_threshold = min(dynamic_threshold, 35.0)
        if abs(prev.y1 - cur.y1) > dynamic_threshold:
            chunks.append(current[:])
            current = [cur]
        else:
            current.append(cur)
    if current:
        chunks.append(current)

    out: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        q = keep_first_qnum if idx == 0 else None
        out.append(
            {
                "lines": chunk,
                "top": max(ln.y1 for ln in chunk),
                "qnum": q,
                "column": block["column"],
            }
        )
    return out


def _group_column_blocks(lines: List[Line], column: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    current: List[Line] = []
    current_q: Optional[str] = None

    for ln in lines:
        text = ln.text.strip()
        m = QNUM_RE.match(text)
        if m and current:
            blocks.extend(
                _split_block_on_gaps(
                    {"lines": current, "top": max(l.y1 for l in current), "qnum": current_q, "column": column},
                    current_q,
                )
            )
            current = [ln]
            current_q = m.group(1)
        else:
            if not current:
                current = [ln]
                current_q = m.group(1) if m else None
            else:
                current.append(ln)
    if current:
        blocks.extend(
            _split_block_on_gaps(
                {"lines": current, "top": max(l.y1 for l in current), "qnum": current_q, "column": column},
                current_q,
            )
        )
    return blocks


def _first_qnum(blocks: List[Dict[str, Any]]) -> Optional[int]:
    for blk in blocks:
        q = blk.get("qnum")
        if q is None:
            continue
        try:
            return int(q)
        except ValueError:
            continue
    return None


def _merge_column_blocks(left_blocks: List[Dict[str, Any]], right_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    left_fixed = [b for b in left_blocks if b.get("qnum")]
    left_float = [b for b in left_blocks if not b.get("qnum")]
    right_fixed = [b for b in right_blocks if b.get("qnum")]
    right_float = [b for b in right_blocks if not b.get("qnum")]

    lf_q, rf_q = _first_qnum(left_blocks), _first_qnum(right_blocks)
    if lf_q is None and rf_q is None:
        left_first = True
    elif lf_q is None:
        left_first = False
    elif rf_q is None:
        left_first = True
    else:
        left_first = lf_q <= rf_q

    def _attach(source: List[Dict[str, Any]], anchors: List[Dict[str, Any]], prefer_last: bool = False) -> List[Dict[str, Any]]:
        leftovers: List[Dict[str, Any]] = []
        for blk in source:
            if not anchors:
                leftovers.append(blk)
                continue
            blk_ordinals = _collect_choice_ordinals(blk["lines"])
            anchor = None
            if blk_ordinals:
                blk_min = min(blk_ordinals)
                sequential_candidates = []
                blank_candidates = []
                for cand in anchors:
                    cand_ordinals = _collect_choice_ordinals(cand["lines"])
                    if cand_ordinals and max(cand_ordinals) < blk_min:
                        sequential_candidates.append((abs(cand["top"] - blk["top"]), cand["top"], cand))
                    elif cand_ordinals == [] and cand.get("qnum"):
                        blank_candidates.append((cand["top"], cand))
                    elif cand_ordinals:
                        union = set(cand_ordinals) | set(blk_ordinals)
                        if union and union == set(range(1, max(union) + 1)) and not (set(cand_ordinals) & set(blk_ordinals)):
                            # Candidate + block form a contiguous 1..N range with no overlap -> likely split options
                            sequential_candidates.append((abs(cand["top"] - blk["top"]), cand["top"], cand))
                if sequential_candidates:
                    # Prefer spatially closest sequential candidate.
                    anchor = min(sequential_candidates, key=lambda item: (item[0], item[1]))[2]
                elif blank_candidates:
                    anchor = min(blank_candidates, key=lambda item: item[0])[1]
            if anchor is None:
                anchor = min(anchors, key=lambda b: abs(b["top"] - blk["top"]))
            gap = abs(anchor["top"] - blk["top"])
            if prefer_last and gap > 220.0:
                anchor = min(anchors, key=lambda b: b["top"])
                gap = abs(anchor["top"] - blk["top"])
            elif gap > 220.0 and not blk_ordinals:
                leftovers.append(blk)
                continue
            merged = anchor["lines"] + blk["lines"]
            anchor_col = anchor.get("column", getattr(anchor["lines"][0], "column", "left"))
            anchor["lines"] = sorted(merged, key=lambda ln: _block_line_sort_key(ln, anchor_col))
            anchor["top"] = max(ln.y1 for ln in anchor["lines"])
        return leftovers

    loose_left = _attach(left_float, right_fixed, prefer_last=not left_first)
    loose_right = _attach(right_float, left_fixed, prefer_last=left_first)

    ordered: List[Dict[str, Any]] = []

    def _append(block: Dict[str, Any]) -> None:
        ordered.append(block)

    if left_first:
        for blk in left_fixed:
            _append(blk)
        for blk in right_fixed:
            _append(blk)
    else:
        for blk in right_fixed:
            _append(blk)
        for blk in left_fixed:
            _append(blk)

    if left_first:
        ordered.extend(loose_left)
        ordered.extend(loose_right)
    else:
        ordered.extend(loose_right)
        ordered.extend(loose_left)

    return ordered


def _stitch_smart(left: List[Line], right: List[Line]) -> List[Line]:
    left_blocks = _group_column_blocks(left, "left")
    right_blocks = _group_column_blocks(right, "right")
    merged_blocks = _merge_column_blocks(left_blocks, right_blocks)
    ordered: List[Line] = []
    for blk in merged_blocks:
        ordered.extend(blk["lines"])
    return ordered


def extract_lines_by_side(pdf_path: str) -> List[Dict[str, Any]]:
    laparams = LAParams(char_margin=3.0, word_margin=0.2, line_margin=0.3)
    out = []
    for idx, layout in enumerate(extract_pages(pdf_path, laparams=laparams)):
        W, H = layout.bbox[2], layout.bbox[3]
        raw_lines, all_elements = [], []
        for el in layout:
            all_elements.append(el)
            if isinstance(el, LTTextContainer):
                for ln in el:
                    if isinstance(ln, LTTextLine):
                        t = ln.get_text().strip()
                        if not t: continue
                        size, font = _line_font_stats(ln)
                        raw_lines.append(Line(t, ln.x0, ln.y0, ln.x1, ln.y1, size, font, "?", page_index=idx))
        raw_lines = filter_page_numbers(raw_lines, W, H)
        subj, target = _detect_subject_for_page(raw_lines + all_elements, W, H)
        usable_lines = [
            l
            for l in raw_lines
            if not (
                re.fullmatch(r"[【\[]\s*[^】\]]+\s*[】\]]", l.text)
                and _normalize_header_text(l.text) in TRAILING_HEADER_SUBJECTS
            )
        ]
        centers = [(l.x0 + l.x1) / 2 for l in usable_lines]
        x_thr = _auto_x_threshold(centers) or (W / 2)
        for L in usable_lines:
            L.column = "left" if ((L.x0 + L.x1) / 2) < x_thr else "right"
        left = sorted([l for l in usable_lines if l.column == "left"], key=lambda L: (-L.y1, L.x0))
        right = sorted([l for l in usable_lines if l.column == "right"], key=lambda L: (-L.y1, L.x0))
        if ORDER_MODE == "pdfminer":
            ordered = raw_lines[:]
        elif ORDER_MODE == "natural":
            ordered = sorted(raw_lines, key=lambda L: (-L.y1, L.x0))
        elif ORDER_MODE == "column_first":
            ordered = left + right
        else:  # smart column stitching with floating block support
            ordered = _stitch_smart(left, right)
        out.append(
            {
                "page_index": idx,
                "subject": subj,
                "target": target,
                "left": left,
                "right": right,
                "ordered": ordered,
            }
        )
    return out

# ================================================================
# QA PARSER
# ================================================================
OPT_RE_CIRCLED = re.compile(r'^\s*([①②③④⑤⑥⑦⑧⑨⑩])\s*(.*)')
OPT_RE_PLAIN = re.compile(r'^\s*(\(?[1-5]\)|[1-5]\.)\s*(.*)')

# --- Added: unified anchor regex + inline splitter (supports multiple anchors per line) ---
OPT_SPLIT_RE = re.compile(
    r"""
    (?P<circ>[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳ⓛ])
    |
    \(\s*(?P<num_paren>[1-9]|1[0-9]|20)\s*\)
    |
    (?P<num_rparen>[1-9]|1[0-9]|20)\)
    |
    (?P<num_dot>[1-9]|1[0-9]|20)\.
    """,
    re.VERBOSE,
)

def split_inline_options(line_text: str):
    s = (line_text or '').replace('\u00A0', ' ').strip()
    raw_matches = list(OPT_SPLIT_RE.finditer(s))
    matches: List[re.Match[str]] = []
    for m in raw_matches:
        prefix = s[:m.start()]
        prefix = prefix.rstrip()
        prev_char = prefix[-1] if prefix else ''
        if prev_char and (prev_char.isdigit() or prev_char == '.'):
            continue
        matches.append(m)
    if not matches:
        return []
    parts = []
    for i, m in enumerate(matches):
        if m.group('circ'):
            idx = m.group('circ')
        elif m.group('num_paren'):
            idx = m.group('num_paren')
        elif m.group('num_rparen'):
            idx = m.group('num_rparen')
        elif m.group('num_dot'):
            idx = m.group('num_dot')
        else:
            idx = m.group(0).strip()
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(s)
        body = s[start:end].strip()
        parts.append((idx, body))
    return parts
# --- end added ---


def _reshape_matrix_options(opts: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Detects simple row/column matrices flattened into a single option and spreads values across options."""
    if not opts:
        return opts
    nonempty = [opt for opt in opts if (opt.get("text") or "").strip()]
    empty = [opt for opt in opts if not (opt.get("text") or "").strip()]
    if len(nonempty) != 1 or not empty:
        return opts
    combined = nonempty[0]["text"].strip()
    tokens = combined.split()
    if not tokens:
        return opts
    rows: List[Tuple[str, List[str]]] = []
    current_label = None
    current_values: List[str] = []
    for token in tokens:
        if token.startswith("(") and token.endswith(")"):
            if current_label is not None:
                rows.append((current_label, current_values))
                current_values = []
            current_label = token
        else:
            current_values.append(token)
    if current_label is not None:
        rows.append((current_label, current_values))
    option_count = len(opts)
    if not rows or any(len(values) != option_count for _, values in rows):
        return opts
    rebuilt: Dict[str, List[str]] = {opt["index"]: [] for opt in opts}
    order = [opt["index"] for opt in opts]
    for label, values in rows:
        for idx, value in enumerate(values):
            rebuilt[order[idx]].append(f"{label} {value}".strip())
    out = []
    for opt in opts:
        pieces = rebuilt.get(opt["index"], [])
        text = " ".join(pieces).strip()
        out.append({"index": opt["index"], "text": text})
    return out


def _collect_matrix_block(lines: List[Line], start: int) -> Tuple[List[Line], int]:
    block: List[Line] = []
    j = start
    while j < len(lines):
        text = (lines[j].text or '').strip()
        if j > start and QNUM_RE.match(text):
            break
        block.append(lines[j])
        j += 1
    return block, j


def _parse_matrix_block(block: List[Line]) -> Optional[List[Dict[str, str]]]:
    if not block:
        return None

    paren_columns: List[Tuple[str, float]] = []
    circ_columns: List[Tuple[str, float]] = []
    for ln in block:
        label = (ln.text or '').strip()
        if PAREN_MATRIX_LABEL_RE.match(label):
            center = (ln.x0 + ln.x1) / 2.0
            paren_columns.append((label, center))
        elif MATRIX_LABEL_RE.match(label):
            center = (ln.x0 + ln.x1) / 2.0
            circ_columns.append((label, center))
    columns = paren_columns if len(paren_columns) >= 2 else circ_columns
    use_paren_labels = len(paren_columns) >= 2
    if len(columns) < 2:
        return None
    columns.sort(key=lambda item: item[1])
    column_labels = [label for label, _ in columns]
    column_centers = [center for _, center in columns]

    rows: List[Dict[str, Any]] = []
    anchor_lines: set[int] = set()
    for ln in block:
        text = (ln.text or '').strip()
        if not text:
            continue
        m_circ = CIRCLED_CHOICE_RE.match(text)
        m_plain = PLAIN_CHOICE_RE.match(text)
        anchor = None
        remainder = ''
        if m_circ:
            anchor = m_circ.group(0)
            remainder = text[m_circ.end():].strip()
        elif m_plain:
            anchor = m_plain.group(1)
            remainder = text[m_plain.end():].strip()
        if anchor:
            row = {
                'index': anchor,
                'y': ln.y1,
                'cols': [[] for _ in column_labels],
            }
            if remainder:
                row['cols'][0].append(remainder)
            rows.append(row)
            anchor_lines.add(id(ln))
    if len(rows) < 2:
        return None

    rows.sort(key=lambda r: -r['y'])

    for ln in block:
        if id(ln) in anchor_lines:
            continue
        text = (ln.text or '').strip()
        if not text:
            continue
        if use_paren_labels and PAREN_MATRIX_LABEL_RE.match(text):
            continue
        if not use_paren_labels and MATRIX_LABEL_RE.match(text):
            continue
        if CIRCLED_CHOICE_RE.match(text) or PLAIN_CHOICE_RE.match(text):
            continue
        center = (ln.x0 + ln.x1) / 2.0
        col_idx = min(range(len(column_centers)), key=lambda idx: abs(column_centers[idx] - center))
        row = min(rows, key=lambda r: abs(r['y'] - ln.y1))
        row['cols'][col_idx].append(text)

    options: List[Dict[str, str]] = []
    for row in rows:
        pieces = []
        for label, texts in zip(column_labels, row['cols']):
            cell = ' '.join(txt for txt in texts if txt).strip()
            if not cell:
                continue
            pieces.append(f"{label} {cell}")
        options.append({'index': row['index'], 'text': ' / '.join(pieces)})
    return options


def _parse_qas_from_lines(lines: List[Line], subject: str, year: Optional[int], target: Optional[str]) -> List[Dict[str, Any]]:
    qas: List[Dict[str, Any]] = []
    qnum, qtxt, opts, cur_opt, cur_txt = None, [], [], None, []
    cur_opt_column: Optional[str] = None
    cur_opt_y: Optional[float] = None
    opts_meta: List[Dict[str, Any]] = []
    CROSS_COLUMN_VERTICAL_SLOP = 24.0
    question_last_page_index: Optional[int] = None
    incomplete_qas: List[Dict[str, Any]] = []
    backfill_state: Optional[Dict[str, Any]] = None

    def flush_opt():
        nonlocal cur_opt, cur_txt, opts, cur_opt_column, cur_opt_y, opts_meta
        if cur_opt:
            text = _clean_option_text(" ".join(cur_txt))
            needs_more = _needs_continuation(text)
            opts.append({"index": cur_opt, "text": text})
            opts_meta.append(
                {
                    "index": cur_opt,
                    "column": cur_opt_column,
                    "y": cur_opt_y,
                    "needs_continuation": needs_more,
                }
            )
        cur_opt, cur_txt, cur_opt_column, cur_opt_y = None, [], None, None

    def _max_option_ordinal(options: List[Dict[str, Any]]) -> int:
        max_ord = 0
        for opt in options:
            idx = _choice_label_to_int(opt.get("index"))
            if idx and idx > max_ord:
                max_ord = idx
        return max_ord

    def _needs_continuation(text: str) -> bool:
        if not text:
            return True
        stripped = text.strip()
        if not stripped:
            return True
        last = stripped[-1]
        terminators = {'.', '?', '!', ')', ']', '】', '〉', '』', '”', '’', '"', "'"}
        if last in terminators:
            return False
        return True

    def flush_q():
        nonlocal qnum, qtxt, opts, qas, incomplete_qas, backfill_state, question_last_page_index, opts_meta
        if qnum:
            normalized_opts = [
                {"index": opt.get("index"), "text": _clean_option_text(opt.get("text"))}
                for opt in _reshape_matrix_options(opts[:])
            ]
            raw_question_text = " ".join(qtxt)
            ordered_question_text = _order_labelled_clauses(raw_question_text)
            dispute_bool, dispute_site = _extract_dispute_flags(ordered_question_text)
            qa_entry = {
                "subject": subject,
                "year": year,
                "target": target,
                "content": {
                    "question_number": qnum,
                    "question_text": _clean_option_text(ordered_question_text),
                    "dispute_bool": dispute_bool,
                    "dispute_site": dispute_site,
                    "options": normalized_opts,
                },
            }
            qas.append(qa_entry)
            ordinals = []
            for opt in normalized_opts:
                ord_val = _choice_label_to_int(opt.get("index"))
                if ord_val:
                    ordinals.append(ord_val)
            missing = [n for n in range(1, 5) if n not in ordinals]
            if missing:
                incomplete_qas.append({
                    "qa": qa_entry,
                    "remaining": missing[:],
                    "page_index": question_last_page_index,
                })
        qnum, qtxt, opts = None, [], []
        opts_meta = []
        backfill_state = None
        question_last_page_index = None

    i = 0
    while i < len(lines):
        raw = lines[i]
        line_page = getattr(raw, "page_index", None)
        text = (raw.text if isinstance(raw, Line) else str(raw)) or ''
        stripped = text.strip()

        m_q = QNUM_RE.match(stripped)
        parts = split_inline_options(text)

        if backfill_state:
            if m_q or parts:
                backfill_state = None
            else:
                if stripped:
                    option = backfill_state["option"]
                    if option["text"]:
                        option["text"] = f"{option['text']} {stripped}".strip()
                    else:
                        option["text"] = stripped
                    option["text"] = _clean_option_text(option["text"])
                i += 1
                continue

        # 1) Detect start of a new question
        if m_q:
            flush_opt(); flush_q()
            qnum = int(m_q.group(1))
            qtxt = [m_q.group(2).strip()]
            question_last_page_index = line_page
            i += 1
            continue

        if qnum and (MATRIX_LABEL_RE.match(stripped) or PAREN_MATRIX_LABEL_RE.match(stripped)):
            block, next_idx = _collect_matrix_block(lines, i)
            matrix_opts = _parse_matrix_block(block)
            if matrix_opts:
                flush_opt()
                opts.extend(matrix_opts)
                question_last_page_index = line_page
                i = next_idx
                continue

        if qnum and re.match(r"\s*[㉠-㉷]", stripped):
            qtxt.append(stripped)
            if line_page is not None:
                question_last_page_index = line_page
            i += 1
            continue

        consumed = 0
        if parts:
            current_ordinals: set[int] = set()
            if qnum:
                for opt in opts:
                    val = _choice_label_to_int(opt.get("index"))
                    if val:
                        current_ordinals.add(val)
                if cur_opt:
                    val = _choice_label_to_int(cur_opt)
                    if val:
                        current_ordinals.add(val)
            remaining_parts: List[Tuple[str, str]] = []
            for idx_label, body in parts:
                ord_val = _choice_label_to_int(idx_label)
                entry = None
                if ord_val is not None:
                    for candidate in incomplete_qas:
                        remaining = candidate.get("remaining") or []
                        if ord_val in remaining:
                            entry = candidate
                            break
                if entry is None:
                    remaining_parts.append((idx_label, body))
                    continue
                allow_backfill = False
                entry_qnum = entry["qa"]["content"].get("question_number")
                entry_page = entry.get("page_index")
                if qnum is None:
                    allow_backfill = True
                elif qnum is not None and entry_qnum == qnum:
                    allow_backfill = True
                elif isinstance(entry_qnum, int) and entry_qnum < (qnum or entry_qnum):
                    if entry_page is not None and line_page is not None and line_page > entry_page:
                        allow_backfill = True
                if not allow_backfill:
                    remaining_parts.append((idx_label, body))
                    continue
                option_text = _clean_option_text(body)
                option = {"index": idx_label, "text": option_text}
                entry["qa"]["content"]["options"].append(option)
                remaining = entry.get("remaining")
                if remaining and ord_val in remaining:
                    remaining.remove(ord_val)
                if remaining:
                    remaining.sort()
                if not remaining:
                    incomplete_qas.remove(entry)
                if _needs_continuation(option_text):
                    backfill_state = {"entry": entry, "option": option}
                else:
                    backfill_state = None
                consumed += 1
                question_last_page_index = line_page if (qnum is not None and entry_qnum == qnum) else question_last_page_index
            if consumed:
                if not remaining_parts:
                    i += 1
                    continue
                parts = remaining_parts
                flush_opt()

        if parts and qnum:
            flush_opt()
            if len(parts) > 1:
                for idx_label, body in parts[:-1]:
                    cleaned_body = _clean_option_text(body)
                    opts.append({"index": idx_label, "text": cleaned_body})
                    opts_meta.append(
                        {
                            "index": idx_label,
                            "column": getattr(raw, "column", None),
                            "y": getattr(raw, "y0", None),
                            "needs_continuation": _needs_continuation(cleaned_body),
                        }
                    )
                    question_last_page_index = line_page
            last_idx, last_body = parts[-1]
            cur_opt = last_idx
            cur_txt = [last_body.strip()] if last_body else []
            cur_opt_column = getattr(raw, "column", None)
            cur_opt_y = getattr(raw, "y0", None)
            question_last_page_index = line_page
            i += 1
            continue

        if cur_opt:
            line_column = getattr(raw, "column", None)
            if cur_opt_column and line_column and line_column != cur_opt_column:
                current_opt_text = _clean_option_text(" ".join(cur_txt))
                current_labels_only = bool(
                    current_opt_text
                    and re.fullmatch(r"[㉠-㉷0-9\s,./·ㆍ]+", current_opt_text)
                )
                line_y = getattr(raw, "y0", None)
                within_vertical_window = False
                if line_y is not None and cur_opt_y is not None:
                    if (
                        line_y <= cur_opt_y + 2.0
                        and line_y >= cur_opt_y - CROSS_COLUMN_VERTICAL_SLOP
                    ):
                        within_vertical_window = True
                if current_labels_only or not within_vertical_window:
                    flush_opt()
                    if stripped:
                        qtxt.append(stripped)
                        if line_page is not None:
                            question_last_page_index = line_page
                else:
                    cur_txt.append(stripped)
                    if line_y is not None:
                        cur_opt_y = line_y
                    if line_page is not None:
                        question_last_page_index = line_page
                i += 1
                continue
            cur_txt.append(stripped)
            line_y = getattr(raw, "y0", None)
            if line_y is not None:
                cur_opt_y = line_y
            if line_page is not None:
                question_last_page_index = line_page
            i += 1
            continue
        elif opts_meta:
            if re.search(r"경우\s*판례에\s*의함", stripped):
                qtxt.append(stripped)
                if line_page is not None:
                    question_last_page_index = line_page
                i += 1
                continue
            line_y = getattr(raw, "y0", None)
            chosen_idx: Optional[int] = None
            if line_y is not None:
                candidates: List[Tuple[float, int]] = []
                for idx_meta, meta in enumerate(opts_meta):
                    opt_text = opts[idx_meta]["text"]
                    needs_more = meta.get("needs_continuation")
                    if needs_more is False:
                        continue
                    meta_y = meta.get("y")
                    if meta_y is None:
                        continue
                    if line_y > meta_y + 2.0:
                        continue
                    if line_y < meta_y - CROSS_COLUMN_VERTICAL_SLOP:
                        continue
                    candidates.append((abs(meta_y - line_y), idx_meta))
                if not candidates:
                    for idx_meta, meta in enumerate(opts_meta):
                        needs_more = meta.get("needs_continuation")
                        if needs_more is False:
                            continue
                        meta_y = meta.get("y")
                        if meta_y is None:
                            continue
                        distance = abs(meta_y - line_y)
                        if (
                            line_y <= meta_y + 2.0
                            and line_y >= meta_y - CROSS_COLUMN_VERTICAL_SLOP
                            and distance <= CROSS_COLUMN_VERTICAL_SLOP
                        ):
                            candidates.append((distance, idx_meta))
                if candidates:
                    _, chosen_idx = min(candidates, key=lambda t: t[0])
            if chosen_idx is not None and stripped and re.fullmatch(r"[㉠-㉷\s]+", stripped):
                chosen_idx = None
            if chosen_idx is not None:
                existing = opts[chosen_idx]["text"]
                new_text = f"{existing} {stripped}".strip()
                cleaned = _clean_option_text(new_text)
                opts[chosen_idx]["text"] = cleaned
                if line_y is not None:
                    opts_meta[chosen_idx]["y"] = line_y
                opts_meta[chosen_idx]["needs_continuation"] = _needs_continuation(cleaned)
                if line_page is not None:
                    question_last_page_index = line_page
                i += 1
                continue
            if stripped:
                qtxt.append(stripped)
                if line_page is not None:
                    question_last_page_index = line_page
            i += 1
            continue
        elif qnum:
            qtxt.append(stripped)
            if line_page is not None:
                question_last_page_index = line_page
        i += 1

    flush_opt(); flush_q()
    return qas

# ================================================================
# MAIN EXTRACTION
# ================================================================
def _summarize_subject_qas(
    qas: List[Dict[str, Any]],
    *,
    expected_choice_count: int = 4,
    include_subjects: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for qa in qas:
        grouped[qa.get("subject")].append(qa)

    subjects: Iterable[str]
    if include_subjects is None:
        subjects = grouped.keys()
    else:
        subjects = include_subjects

    for subj in subjects:
        items = grouped.get(subj, [])
        question_numbers = [
            qa.get("content", {}).get("question_number")
            for qa in items
            if isinstance(qa.get("content"), dict)
        ]
        number_counter = Counter(n for n in question_numbers if isinstance(n, int))
        sorted_numbers = sorted(number_counter.keys())
        max_number = max(sorted_numbers) if sorted_numbers else 0
        expected_range = set(range(1, max_number + 1)) if max_number else set()
        present_numbers = set(sorted_numbers)
        missing_numbers = sorted(expected_range - present_numbers)
        duplicate_numbers = sorted(n for n, c in number_counter.items() if c > 1)

        option_count_issues: List[int] = []
        blank_option_entries: List[int] = []
        option_index_gaps: Dict[int, List[int]] = {}
        duplicate_questions: List[int] = []

        normalized_text_counter: Counter[str] = Counter()
        normalized_text_to_numbers: Dict[str, List[int]] = defaultdict(list)

        for qa in items:
            content = qa.get("content") or {}
            qnum = content.get("question_number")
            options = content.get("options") or []
            if len(options) != expected_choice_count and isinstance(qnum, int):
                option_count_issues.append(qnum)

            option_ints = [
                _choice_label_to_int(opt.get("index"))
                for opt in options
                if isinstance(opt, dict)
            ]
            if isinstance(qnum, int):
                missing_indices = sorted(
                    n
                    for n in range(1, expected_choice_count + 1)
                    if n not in option_ints
                )
                if missing_indices:
                    option_index_gaps[qnum] = missing_indices

            if any(not (opt.get("text") or "").strip() for opt in options) and isinstance(qnum, int):
                blank_option_entries.append(qnum)

            question_text = (content.get("question_text") or "").strip()
            if question_text:
                normalized_text_counter[question_text] += 1
                normalized_text_to_numbers[question_text].append(qnum)

        for text, count in normalized_text_counter.items():
            if count > 1:
                duplicate_questions.extend(
                    sorted(n for n in normalized_text_to_numbers.get(text, []) if isinstance(n, int))
                )

        summary[subj] = {
            "question_count": len(items),
            "question_numbers": sorted_numbers,
            "missing_question_numbers": missing_numbers,
            "duplicate_question_numbers": duplicate_numbers,
            "option_count_mismatches": sorted(option_count_issues),
            "option_index_gaps": option_index_gaps,
            "blank_option_entries": sorted(blank_option_entries),
            "duplicate_question_text_numbers": sorted(set(duplicate_questions)),
        }

    return summary


def extract_all_subjects_qa(
    pdf_path: str,
    json_out_combined: Optional[str] = None,
    json_out_per_subject_dir: Optional[str] = None,
    audit: bool = True,
    audit_csv_out: Optional[str] = None,
    audit_json_out: Optional[str] = None,
    summary_out: Optional[str] = None,
):
    year = infer_year_from_filename(pdf_path)
    pages = extract_lines_by_side(pdf_path)
    current_subj, current_target, skip = None, None, False
    per_subject = {s: [] for s in TARGET_SUBJECTS}
    audit_rows = []

    for p in pages:
        pg = p["page_index"] + 1
        subj, target = p["subject"], p["target"]
        is_target = subj in TARGET_SUBJECTS
        if is_target:
            current_subj, current_target, skip = subj, target, False
            action = f"ENTER {subj}"
        elif subj is None:
            if skip or not current_subj:
                action = "SKIP"
                audit_rows.append((pg, subj, current_subj, skip, action))
                continue
            action = f"INHERIT {current_subj}"
        else:
            current_subj, current_target, skip = None, None, True
            action = f"RESET (non-target)"
            audit_rows.append((pg, subj, current_subj, skip, action))
            continue
        if audit: print(f"[page {pg}] header={subj} target={target} -> {action}")
        audit_rows.append((pg, subj, current_subj, skip, action))
        if current_subj not in TARGET_SUBJECTS:
            continue
        leading_orphans = _extract_leading_option_lines(list(p.get("left") or []))
        orphan_ids: set[int] = set()
        if leading_orphans:
            key_target = current_target or "default"
            for prev_target, prev_lines in reversed(per_subject[current_subj]):
                if prev_target == key_target:
                    prev_lines.extend(leading_orphans)
                    orphan_ids = {id(ln) for ln in leading_orphans}
                    break
        page_lines = [ln for ln in (p.get("ordered") or []) if id(ln) not in orphan_ids]
        ordered_lines = list(page_lines)
        if not ordered_lines:
            left_blocks = list(p.get("left") or [])
            right_blocks = list(p.get("right") or [])
            ordered_lines = left_blocks + right_blocks
        per_subject[current_subj].append((current_target or "default", ordered_lines))

    combined = []
    for subj in dict.fromkeys(TARGET_SUBJECTS):
        groups = {}
        for targ, lines in per_subject[subj]:
            groups.setdefault(targ, []).extend(lines)
        for targ, lines in groups.items():
            qas = _parse_qas_from_lines(lines, subj, year, targ)
            combined.extend(qas)
            if json_out_per_subject_dir:
                os.makedirs(json_out_per_subject_dir, exist_ok=True)
                fname = f"{subj}_{targ}_QA.json".replace("/", "_")
                with open(os.path.join(json_out_per_subject_dir, fname), "w", encoding="utf-8") as f:
                    json.dump(qas, f, ensure_ascii=False, indent=2)

    summary = _summarize_subject_qas(
        combined,
        include_subjects=dict.fromkeys(TARGET_SUBJECTS).keys(),
    )

    if json_out_combined:
        with open(json_out_combined, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)

    if summary_out:
        with open(summary_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    if audit_csv_out:
        with open(audit_csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["page", "detected_subject", "current_subject", "skip", "action"])
            for r in audit_rows: w.writerow(r)

    if audit_json_out:
        with open(audit_json_out, "w", encoding="utf-8") as f:
            json.dump(
                [{"page": r[0], "detected_subject": r[1], "current_subject": r[2], "skip": r[3], "action": r[4]}
                 for r in audit_rows],
                f, ensure_ascii=False, indent=2)

    return combined

# ================================================================
# YEAR INFERENCE
# ================================================================
def infer_year_from_filename(path: str) -> Optional[int]:
    fname = os.path.basename(path)
    m = re.search(r"(\d{2})년", fname)
    if m:
        yy = int(m.group(1))
        return 2000 + yy
    m = re.search(r"(20\d{2}|19\d{2})", fname)
    if m:
        return int(m.group(1))
    return None


# ================================================================
# RUN
# ================================================================
# =========================
# Error reporting helper
# =========================
ERROR_FIELDS = [
    "missing_question_numbers",
    "duplicate_question_numbers",
    "option_count_mismatches",
    "option_index_gaps",
    "blank_option_entries",
    "duplicate_question_text_numbers",
]

from pathlib import Path as _Path
from typing import Dict as _Dict, Any as _Any

def write_error_report_if_any(summary_json_path: Path, error_txt_path: Path) -> bool:
    """
    Reads summary.json and writes error.txt if any of the target fields are populated.
    Supports two layouts:
      1) flat: fields at the root
      2) nested: fields under each subject key (e.g., '형사법', '경찰학')
    """
    TARGET_FIELDS = [
        "missing_question_numbers",
        "duplicate_question_numbers",
        "option_count_mismatches",
        "option_index_gaps",
        "blank_option_entries",
        "duplicate_question_text_numbers",
    ]

    def _is_populated(val):
        if isinstance(val, list):
            return len(val) > 0
        if isinstance(val, dict):
            return len(val.keys()) > 0
        return bool(val)

    if not summary_json_path.exists():
        error_txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(error_txt_path, "w", encoding="utf-8") as ef:
            ef.write("[summary.json missing]\n")
            ef.write(f"Not found: {summary_json_path}\n")
        return True

    try:
        import json
        with open(summary_json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as e:
        error_txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(error_txt_path, "w", encoding="utf-8") as ef:
            ef.write("[summary.json read error]\n")
            ef.write(str(e) + "\n")
        return True

    populated_report = {}

    # Case A: flat schema (fields at root)
    flat_hits = {k: summary.get(k) for k in TARGET_FIELDS if _is_populated(summary.get(k))}
    if flat_hits:
        populated_report["_overall"] = flat_hits

    # Case B: nested schema (each top-level key is a subject dict)
    # Heuristic: if a top-level value is a dict that contains ANY of the target fields (present or empty),
    # treat it as a subject block.
    for subject, block in (summary.items() if isinstance(summary, dict) else []):
        if not isinstance(block, dict):
            continue
        if any(k in block for k in TARGET_FIELDS):
            hits = {k: block.get(k, {} if k == "option_index_gaps" else []) for k in TARGET_FIELDS}
            hits = {k: v for k, v in hits.items() if _is_populated(v)}
            if hits:
                populated_report[subject] = hits

    # Nothing to report
    if not populated_report:
        if error_txt_path.exists():
            try:
                error_txt_path.unlink()
            except Exception:
                pass
        return False

    # Write grouped, human-readable report
    lines = ["[Validation Issues Detected]\n"]
    import json as _json
    for subject, issues in populated_report.items():
        header = "Overall" if subject == "_overall" else f"Subject: {subject}"
        lines.append(header)
        for k, v in issues.items():
            lines.append(f"  {k}:")
            lines.append(_json.dumps(v, ensure_ascii=False, indent=2))
        lines.append("")  # blank line between subjects

    error_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(error_txt_path, "w", encoding="utf-8") as ef:
        ef.write("\n".join(lines))

    return True


def _collect_pdfs(in_path, recursive: bool):
    """Return a list of .pdf files (case-insensitive) from a file/dir input."""
    from pathlib import Path
    in_path = Path(in_path)
    if in_path.is_file():
        return [in_path] if in_path.suffix.lower() == ".pdf" else []
    if in_path.is_dir():
        iterfunc = in_path.rglob if recursive else in_path.glob
        return [p for p in iterfunc("*.pdf")] + [p for p in iterfunc("*.PDF")]
    return []

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Batch extract QA from PDFs or a single PDF; write error.txt if summary shows issues."
    )
    parser.add_argument("input", help="Path to a folder (batch) or a single PDF")
    parser.add_argument("--out", default="outputs", help="Base output directory (default: ./outputs)")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders when input is a folder")
    parser.add_argument("--audit", action="store_true", help="Write audit CSV/JSON per PDF")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_base = Path(args.out).expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    pdfs = _collect_pdfs(in_path, args.recursive)
    if not pdfs:
        raise SystemExit(f"❌ No PDF files found under: {in_path}")

    print(f"🔎 Found {len(pdfs)} PDF(s). Starting extraction...")

    for pdf_path in sorted(pdfs):
        stem = pdf_path.stem
        pdf_out_dir = out_base / stem
        pdf_out_dir.mkdir(parents=True, exist_ok=True)

        combined_out = pdf_out_dir / f"{stem}_ALL_QA.json"
        per_subject_dir = pdf_out_dir / "per_subject_QA"
        audit_csv = pdf_out_dir / f"{stem}_audit_pages.csv"
        audit_json = pdf_out_dir / f"{stem}_audit_pages.json"
        summary_out = pdf_out_dir / f"{stem}_summary.json"
        error_txt = pdf_out_dir / "error.txt"

        per_subject_dir.mkdir(parents=True, exist_ok=True)

        print(f"📄 Processing: {pdf_path}")
        try:
            results = extract_all_subjects_qa(
                pdf_path=str(pdf_path),
                json_out_combined=str(combined_out),
                json_out_per_subject_dir=str(per_subject_dir),
                audit=args.audit,
                audit_csv_out=str(audit_csv) if args.audit else None,
                audit_json_out=str(audit_json) if args.audit else None,
                summary_out=str(summary_out),
            )

            # Write error.txt if summary has issues
            had_errors = write_error_report_if_any(summary_out, error_txt)
            if had_errors:
                print(f"⚠️  Issues found → {error_txt.name} written.")
            else:
                print("✅ No validation issues.")

            try:
                n_items = len(results) if results is not None else "?"
                print(f"✅ {stem}: extracted {n_items} QA item(s)")
            except Exception:
                print(f"✅ {stem}: extraction complete")

        except Exception as e:
            with open(error_txt, "w", encoding="utf-8") as ef:
                ef.write("[extraction error]")
                ef.write(str(e) + "")
            print(f"❌ {stem}: failed with error -> {e}")

    print("🎉 Done.")
