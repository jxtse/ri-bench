"""Symmetric de-identification of ICLR paper full texts.

Removes author, venue and version cues from every paper in the same way, regardless of its outcome:

- venue running headers ("Published as a conference paper at ICLR 202x", "Under review as ...");
- the front-matter block between the title and the abstract (author names, affiliations, emails or the
  "Anonymous authors / Paper under double-blind review" placeholder), replaced by one neutral line;
- footnote lines with correspondence, emails, equal-contribution or internship/affiliation notes;
- acknowledgement / funding / author-contribution sections (heading through the next heading);
- submission line numbers;
- URLs (code, project pages, anonymous repositories) replaced by "[URL removed]".

Nothing else is edited. `sanitize` is deterministic and returns the text plus a per-rule removal count.
"""

import re

NEUTRAL_AUTHORS = "Authors withheld for review"

HEADER = re.compile(r"^\s*(?:\*\*)?(?:published as a conference paper at|under review as a conference paper at|"
                    r"accepted (?:as a|at|to) |workshop paper at |preprint\. under review)[^\n]*$", re.I | re.M)
ABSTRACT = re.compile(r"^#{1,6}\s*\**\s*abstract\b.*$", re.I | re.M)
ABSTRACT_INLINE = re.compile(r"\bABSTRACT\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"(?:https?://\s?|www\.)[^\s)\]>\"'}]+|(?:github\.com|gitlab\.com|huggingface\.co|anonymous\.4open\.science)"
                 r"/[^\s)\]>\"'}]*", re.I)
FOOTNOTE = re.compile(r"^[ \t>_*\[\]0-9∗†‡§¶#,.:-]*(?:\S+\s*)?(?:correspond|equal contribution|contributed equally|"
                      r"work (?:was )?done (?:while|during|as)|internship|project lead|joint first|"
                      r"these authors|first authors?)[^\n]*$", re.I | re.M)
BACKMATTER = re.compile(r"^#{1,6}\s*\**\s*(?:[0-9A-Z]+(?:\.[0-9]+)*\.?\s+)?(?:acknowledg(?:e)?ments?|funding|"
                        r"author contributions?|contributions? of authors)\b.*$", re.I | re.M)
BOLD_BACKMATTER = re.compile(r"^\s*\*\*\s*(?:acknowledg(?:e)?ments?|funding)\s*[.:]?\s*\*\*.*$", re.I | re.M)
HEADING = re.compile(r"^#{1,6}\s", re.M)


def _front_matter(text, counts):
    """Replace everything between the first heading (title) and the Abstract heading."""
    abstract = ABSTRACT.search(text)
    if not abstract:
        # line-numbered submission PDFs render the abstract label inline ("... review ABSTRACT The k-means ...")
        abstract = ABSTRACT_INLINE.search(text, 0, 6000)
    if not abstract:
        counts["front_matter_missing_abstract"] += 1
        return text
    head = text[:abstract.start()]
    title = HEADING.search(head) or re.search(r"^(?!\s*$).+$", head, re.M)
    if not title:
        counts["front_matter_missing_title"] += 1
        return text
    title_end = head.find("\n", title.start())
    title_end = len(head) if title_end < 0 else title_end
    # titles sometimes wrap onto a second heading line
    rest = head[title_end:]
    while True:
        nxt = re.match(r"\s*\n(#{1,6}\s[^\n]*)", rest)
        if not nxt or re.search(r"anonymous|\*\*|@|universit|institute|lab\b", nxt.group(1), re.I):
            break
        title_end += nxt.end()
        rest = head[title_end:]
    removed = head[title_end:]
    if removed.strip():
        counts["front_matter"] += 1
    return head[:title_end] + f"\n\n{NEUTRAL_AUTHORS}\n\n" + text[abstract.start():]


def _strip_sections(text, pattern, counts, key):
    while True:
        match = pattern.search(text)
        if not match:
            return text
        nxt = HEADING.search(text, match.end())
        end = nxt.start() if nxt else len(text)
        # never swallow references: stop at a references heading even if it is not a markdown heading
        refs = re.search(r"^\s*(?:#{1,6}\s*)?\**\s*references\b", text[match.end():end], re.I | re.M)
        if refs:
            end = match.end() + refs.start()
        text = text[:match.start()] + text[end:]
        counts[key] += 1


# ICLR 2025/2026 submission templates print margin line numbers, which the PDF->markdown step rendered as
# "**012**", "**026 027**" or "> **030** ..."; final-version PDFs have none, so they are a version cue too.
LINE_NUMBERS = re.compile(r"(?:^>\s*)?\*\*\d{3}(?:\s+\d{3})*\*\*\s?", re.M)
ANON_BLOCK = re.compile(r"(?:\*\*\s*)?anonymous authors?(?:\s*\*\*)?|paper under double-blind review", re.I)


def sanitize(text):
    from collections import Counter
    counts = Counter()
    text_ln, n = LINE_NUMBERS.subn("", text)
    # bold 3-digit numbers also occur in result tables; only line-numbered submission PDFs have dozens of them
    if n >= 30:
        text = text_ln
        counts["line_numbers"] += n
    text, n = HEADER.subn("", text)
    counts["venue_header"] += n
    text = _front_matter(text, counts)
    text, n = ANON_BLOCK.subn("", text)
    counts["anonymous_block"] += n
    text, n = FOOTNOTE.subn("", text)
    counts["footnote"] += n
    text = _strip_sections(text, BACKMATTER, counts, "backmatter_section")
    text = _strip_sections(text, BOLD_BACKMATTER, counts, "backmatter_bold")
    text, n = EMAIL.subn("[email removed]", text)
    counts["email"] += n
    text, n = URL.subn("[URL removed]", text)
    counts["url"] += n
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, dict(counts)


RESIDUAL = {
    "venue_header": re.compile(r"published as a conference|under review as a conference|double-blind review|anonymous author", re.I),
    "camera_ready": re.compile(r"camera[- ]ready", re.I),
    "acknowledg": re.compile(r"acknowledg", re.I),
    "email": EMAIL,
    "url": re.compile(r"https?://|github\.com|anonymous\.4open", re.I),
    "iclr_year": re.compile(r"\bICLR\s*20(?:2[4-9])\b"),
    "line_numbers": re.compile(r"(?:^>\s*)?\*\*\d{3}(?:\s+\d{3})*\*\*", re.M),
    "acknowledgement_heading": re.compile(r"^\s*(?:#{1,6}\s*)?\**\s*(?:[0-9A-Z]+\.?\s+)?acknowledg(?:e)?ments?\b", re.I | re.M),
}


def residual_cues(text):
    return {key: len(pattern.findall(text)) for key, pattern in RESIDUAL.items()}
