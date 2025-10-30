
# AGENTS.md — Parsing & QA Automation Plan (Improved)

> Drop-in replacement for your current `AGENTS.md`. It formalizes objectives, I/O schemas, the loop that fixes `error.txt` issues, and a robust parsing plan using **pdfminer.six**, while leaving room to propose better methods when accuracy gains are provable.

---

## 1) Objective

Given **one hour of wall-clock time per run**, automatically parse all PDFs in the repo into **validated JSON QA items**, fix known issues, and keep re-testing until the test suite is green. When `error.txt` or validator failures are present, the agent must **repair the root cause**, re-run parsing, and confirm the fixed outputs.

> If there is a better way to parse it, the agent **must briefly justify and document** the approach and proceed if accuracy improves measurably without breaking constraints.

---

## 2) Scope & Constraints

- **Parser library**: Primary is **pdfminer.six** (text and layout via LT* objects).
- **Page numbers**: Must be **omitted** from extracted text.
- **Two-column PDFs**: Correctly recover reading order for left→right, top→bottom in each column.
- **Headers/Footers**: Detect and exclude (course titles, exam names, running headers).
- **Korean text**: Preserve Unicode; avoid normalization that removes Hangul Jamo.
- **Performance bound**: Keep within ~1 hour per full repo run.
- **Idempotence**: Re-running must not duplicate items.
- **Determinism**: Same input ⇒ same JSON output.
- **Security**: No network calls; filesystem-only.

---

## 3) Inputs, Outputs, and File Layout

### Input
- PDFs in: `PDF_Racer/docs/exam/pdf` (and subfolders when `--recursive` is used).

### Output
- JSON results to: `PDF_Racer/docs/exam/pdf/results/{exam_slug}/{subject}.json`
- Logs: `PDF_Racer/docs/exam/pdf/results/{exam_slug}/parse.log`
- Errors (machine-readable): `PDF_Racer/docs/exam/pdf/results/{exam_slug}/error.txt`

### JSON QA Schema
```jsonc
{
  "exam": "2024-1차",
  "subject": "형법",
  "question_id": 19,
  "question_text": "…",
  "options": [
    "…", "…", "…", "…"
  ],
  "answer_key": "3",                 // if available; else null
  "dispute_bool": false,             // derived by regex
  "dispute_site": null,              // site captured if dispute_bool==true
  "metadata": {
    "source_pdf": "24년 1차.pdf",
    "page_span": [12,13],
    "parser": "pdfminer.six",
    "version": "v2.0",
    "hash": "sha256:…"
  }
}
```

---

## 4) CLI & Default Workflow

### Canonical command
```bash
python PDF_Parser_V2_patched_full.py PDF_Racer/docs/exam/pdf   --recursive   --out PDF_Racer/docs/exam/pdf/results
```

### Agent loop
1. **Discover** PDFs (respecting `--recursive`).
2. **Parse** with pdfminer.six into structured blocks, pages → columns → lines → tokens.
3. **Classify** blocks into: `{header, footer, page_number, section_header, question, option, annotation}`.
4. **Assemble** QAs per subject, preserving order.
5. **Detect disputes** with the provided regexes (see §6).
6. **Validate** with unit + golden tests; write `error.txt` if failures.
7. **Repair** parsing rules based on failures; **re-parse** only the affected PDFs.
8. **Re-validate**; stop when green or time budget exhausted.
9. **Write report** summary to `parse.log` (fix counts, remaining anomalies).

---

## 5) Parsing Strategy (pdfminer.six)

### 5.1 Page → Column segmentation
- Use `LAParams` and object positions (x0,x1,y0,y1).
- Estimate **median gutter** by clustering x-centers of text boxes; if two dense clusters exist, treat as **two columns**.
- Within each column, sort by `(-y, x)` for natural reading order.

### 5.2 Header/Footer/Page number removal
- Learn **per-document templates**:
  - Elements repeating with high frequency at same y-band ⇒ header/footer.
  - Numeric-only lines near bottom/top with small variance ⇒ page numbers.
- Maintain an **exclusion mask** of y-ranges per page for extraction.

### 5.3 Section & subject header recognition
- Heuristics:
  - Font size jump vs running text.
  - Bracketed tokens like `[헌법]`, `[형법]`, `[형사소송법]` at line start.
  - Centered text with large width coverage.
- When uncertain, fall back to **lexicon** of known subjects.

### 5.4 Question / Option extraction
- **Question start**: “제?\s*\d+\s*문” or `\bQ\s*\d+\b` or numeric followed by punctuation, **and** high-probability markers (“다음”, “옳은 것”, etc.).
- **Option markers**:
  - Numeric: `① ② ③ ④` (and ASCII `(1) (2) (3) (4)`).
  - Hangul bullets: `ㄱ, ㄴ, ㄷ, ㄹ` and combinations.
- **Carry-over guard**: If an option’s tail is spatially below the next question header band, **truncate** at the band to avoid tail-join bugs.
- **Join rules**: Merge wrapped lines if vertical gap < threshold and same left margin cluster.

### 5.5 Multi-line options & “tail-join” prevention
- Create **question bands**: from question header y to next header y (same column). Limit options to the current band only.
- Use **x-margin clusters** to keep option bodies aligned with their label.

### 5.6 Jumble detection & auto-repair
- If consecutive questions have:
  - Option-count != 4, or
  - Cross-bleed (option text contains the next question header), then:
    - **Backtrack**: re-split bands using stricter header threshold and larger vertical gap.
    - **Re-check** option labels; if missing, infer by position order.

---

## 6) Dispute Detection

Use your supplied regexes:

```python
DISPUTE_RE = re.compile(
    r"다툼이\s*있는\s*경우\s*(?P<site>.*?)\s*(?:판례|결정)에\s*의함",
    re.IGNORECASE,
)
DISPUTE_LINE_RE = re.compile(
    r"^\s*다툼이\s*있는\s*경우\s*(?:판",
    re.IGNORECASE,
)
```

- Set `dispute_bool = True` when matched.
- Set `dispute_site = site.group('site')` (trim brackets/quotes).

---

## 7) Validation & Tests

### 7.1 Unit tests
- Parsing primitives: column detection, header/footer strip, question banding, option alignment, dispute extraction.
- Korean Unicode handling (round-trip).

### 7.2 Golden tests
- Build **gold files** for problematic pages/questions listed in §10 (Acceptance Criteria). Each run must match gold text exactly (ignoring whitespace normalization).

### 7.3 Schema tests
- JSON shape matches schema; exactly 4 options per MCQ unless explicitly annotated.

### 7.4 Invariants
- Questions strictly increasing per subject.
- No option text appears in `question_text` field.
- No cross-page bleed unless `page_span` shows contiguous continuation and banding allows it.

---

## 8) Repair Loop

When `error.txt` or validator flags:
1. Locate affected `{exam, subject, qid}`.
2. Load its page range; narrow to bands.
3. Tighten header threshold or band split rules.
4. Re-run parsing for only those pages.
5. Confirm the fix by **local golden** + unit tests.
6. Update **gold files** if the PDF genuinely changed (with reviewer note).

---

## 9) Implementation Tips (pdfminer.six)

- Configure:
  ```python
  from pdfminer.high_level import extract_pages
  from pdfminer.layout import LTTextContainer, LTChar, LAParams

  laparams = LAParams(line_margin=0.12, word_margin=0.08, char_margin=2.0,
                      detect_vertical=False, boxes_flow=None)
  for page_layout in extract_pages(pdf_path, laparams=laparams):
      # collect LTTextContainer objects with x0,x1,y0,y1, text
  ```
- For fonts & sizes: scan `LTChar.size` to detect headers.
- Build a **spatial index** (interval tree) for band checks.
- Normalize whitespace; keep Hangul intact.
- Do **hashing** of source PDF (sha256) into metadata for traceability.

---

## 10) Acceptance Criteria (must pass)

The following issues **must be fixed** and verified by tests (from your notes):

- **20년 2차**
  - 경찰학개론 q17, q19 — jumbles resolved.
- **22년 1차**
  - 형법 q15 — “option is in the question text” fixed.
  - 형법 q18, q20 — **de-jumbled**.
  - 형사소송법 q19, q20 — **de-jumbled**.
- **22년 2차**
  - 헌법 q18–q20 — **de-jumbled**.
  - 경찰학 q7 — option leakage fixed.
  - 경찰학 q29 — option leakage fixed.
- **23년 1차**
  - 헌법_일반공채101경비단 q20 — “다툼이 있는 경우 판례에 의함” not lost; not merged into q19.
  - 형사법_일반공채101경비단경행경채 q39–q40 — **de-jumbled**.
  - 경찰학_일반공채101경비단경행경채 q13 — option labels ㄴ,ㄷ,ㄹ present.
  - 형사소송법_전의경경채 q18–q20 — tails not joined across questions.
  - 경찰학_전의경경채 q7 — option labels ㄴ,ㄷ,ㄹ present.
  - 경찰학_전의경경채 q19 — tail not joined into q20.
- **23년 2차**
  - 헌법 q19 — tail not joined into q20.
  - 형사법 q3 — dispute site not misrecognized as header.
- **24년 1차**
  - 헌법 q18 — option 3 tail not joined to q19.
  - 경찰학 q2 — options not jumbled into question.
  - 경찰학 q33 — option tails not joined to question.
  - 형법 q19–q20 — **de-jumbled**.
  - 경찰학_default_variant2 q2 — options not jumbled into question.
  - 경찰학_default_variant2 q19–q20 — not cross-matched.
- **24년 2차**
  - 경찰학 q38 — q38 not joined to q40.
- **25년 1차**
  - 형사법 q27 — option 3 tail not joined to question.
  - 형사법 q31 — q33 opt4 tail not joined to q31.
  - 형사법 q36 — opt4 tail not joined to q39.
  - 경찰학 q26 — labels ㄴ,ㄷ,ㄹ present.
  - 경찰학 q33 — circled ㄱ,ㄴ not joined to question text.
  - 경찰학 q38 — labels ㄴ,ㄷ present.
- **25년 2차**
  - 헌법 q13 — opt2 tail not joined to question.
  - 헌법 q15 — opt4 tail not joined to q18.
  - 헌법 q20 — opt4 tail not joined to 형사법 q6.
  - 형사법 q4 — opt2 tail not joined to question.
  - 형사법 q6 — opt4 tail not joined to q13.
  - 형사법 q10 — opt1 tail not joined to question.
  - 형사법 q13 — opt4 tail not joined to q17.
  - 형사법 q16 — opt4 tail not joined to q14.
  - 형사법 q19, q22, q25 — wrong matching fixed.
  - 형사법 q36 — truncated question fixed.
  - 형사법 q37 — not mixed with q38 opt4 or other text.
  - 경찰학 q4 — opt1 tail not joined to question.
  - 경찰학 q18 — end not joined to q26; options accurate.
  - 경찰학 q18, q22, q24 — fully reparsed; accurate.
  - 경찰학 q26 — opt4 tail not joined to q32.
  - 경찰학 q36 — options not mixed.
  - 경찰학 q38 — options not mixed.

All the above must be **unit-tested** with gold references pulled from the exact PDF pages.

---

## 11) Test Harness & Commands

- **Run parser**: (see §4)
- **Run tests**:
  ```bash
  pytest -q
  ```
- **Golden update** (only with reviewer ack):
  ```bash
  pytest -q --update-gold
  ```
- **Validate JSON**:
  ```bash
  python -m scripts.validate_results PDF_Racer/docs/exam/pdf/results
  ```

---

## 12) Better Parsing (Optional, with Justification)

If pdfminer.six yields edge errors, the agent may propose and (if approved by config flag) switch to:
- **PyMuPDF (fitz)** for page-level textblocks (more stable bounding boxes).
- **Hybrid order check**: compare block order from pdfminer vs PyMuPDF; choose the ordering with **fewer band violations**.
- **Layout post-check** using character-size histograms to detect dropped superscripts (e.g., circled numerals).

Document any switch in `parse.log` with **before/after diffs** for the affected questions.

---

## 13) Logging

- `parse.log` includes: runtime, PDFs processed, anomalies/fixes, banding thresholds, column decisions, and a per-issue checklist from §10 (pass/fail).

---

## 14) Definition of Done

- All §10 acceptance items pass.
- `pytest` is green.
- No new `error.txt` lines created.
- JSON schema validated for all subjects.
- Reruns produce identical outputs (hash-stable).

---

## 15) Quick Start

```bash
# 1) Install
pip install pdfminer.six pytest

# 2) Parse
python PDF_Parser_V2_patched_full.py PDF_Racer/docs/exam/pdf --recursive --out PDF_Racer/docs/exam/pdf/results

# 3) Test
pytest -q

# 4) Inspect reports
cat PDF_Racer/docs/exam/pdf/results/*/parse.log
```

---

## 16) Maintainers’ Notes

- Encapsulate heuristics behind a `ParserPolicy` class so fixes are toggled by exam-year/subject if needed.
- Keep a tiny **fixtures/** dir with cropped problem pages to speed local tests.
- Normalize only whitespace; **do not** strip Hangul punctuation.

