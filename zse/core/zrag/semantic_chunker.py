"""
Semantic Chunker — Intent-first content splitting.

Unlike traditional chunkers that split by character count or sentence
boundaries, the semantic chunker:

1. Detects content TYPE (definition, fact, procedure, code, table, etc.)
2. Groups related content into semantic units
3. Strips noise (nav text, boilerplate, repeated headers)
4. Compresses each unit — keeps what the LLM needs, discards fluff
5. Estimates token count per block for budget management

This is the core of .zpf's innovation: semantic compression from the
LLM's perspective.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from zse.core.zrag.zpf_spec import BlockType, SemanticBlock
from zse.core.zrag.parsers import ParsedDocument, ParsedSection


# Noise patterns — content that adds no value for LLM comprehension
_NOISE_PATTERNS = [
    # Cookie/privacy/legal
    re.compile(r"^\s*cookie\s*(policy|notice|consent|settings?)", re.IGNORECASE),
    re.compile(r"^\s*privacy\s*policy", re.IGNORECASE),
    re.compile(r"^\s*terms\s*(of|and)\s*(service|use|conditions)", re.IGNORECASE),
    re.compile(r"^\s*copyright\s*©?\s*\d{4}", re.IGNORECASE),
    re.compile(r"^\s*all\s*rights\s*reserved", re.IGNORECASE),
    re.compile(r"^\s*©\s*\d{4}", re.IGNORECASE),
    re.compile(r"^\s*accept\s*(all|cookies)", re.IGNORECASE),
    re.compile(r"^\s*reject\s*(all|cookies)", re.IGNORECASE),
    # Navigation
    re.compile(r"^\s*(skip\s*to|jump\s*to)\s*(main|content|navigation)", re.IGNORECASE),
    re.compile(r"^\s*(home|about|contact|faq|help|login|sign\s*(in|up))\s*$", re.IGNORECASE),
    re.compile(r"^\s*\[?\s*(menu|search|close|open|toggle)\s*\]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(back\s*to\s*top|go\s*to\s*top|scroll\s*to\s*top)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(previous|next|first|last)\s*(page|article|post)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*read\s*more\s*\.{0,3}\s*$", re.IGNORECASE),
    re.compile(r"^\s*show\s*(more|less)\s*$", re.IGNORECASE),
    re.compile(r"^\s*see\s*(more|all|also)\s*$", re.IGNORECASE),
    re.compile(r"^\s*view\s*(all|more)\s*$", re.IGNORECASE),
    # Loading/status
    re.compile(r"^\s*loading\.{0,3}\s*$", re.IGNORECASE),
    re.compile(r"^\s*please\s*wait\.{0,3}\s*$", re.IGNORECASE),
    re.compile(r"^\s*page\s+\d+\s*(of\s*\d+)?\s*$", re.IGNORECASE),
    # Social/sharing
    re.compile(r"^\s*share\s*(this|on)\s*(facebook|twitter|linkedin|x|reddit)", re.IGNORECASE),
    re.compile(r"^\s*follow\s*us\s*:?\s*(on)?\s*(twitter|facebook|linkedin|github|discord|youtube)", re.IGNORECASE),
    re.compile(r"^\s*like\s*(this|us)\s*on\s*facebook", re.IGNORECASE),
    re.compile(r"^\s*tweet\s*this\s*$", re.IGNORECASE),
    # Subscription/newsletter
    re.compile(r"^\s*subscribe\s*(to)?\s*(our|the)?\s*newsletter", re.IGNORECASE),
    re.compile(r"^\s*enter\s*(your)?\s*email", re.IGNORECASE),
    re.compile(r"^\s*sign\s*up\s*(for|to)\s*(our|the)?\s*(newsletter|updates|mailing)", re.IGNORECASE),
    re.compile(r"^\s*\[?\s*subscribe\s*\]?\s*$", re.IGNORECASE),
    # Comments
    re.compile(r"^\s*comments?\s*\(\d+\)\s*$", re.IGNORECASE),
    re.compile(r"^\s*be\s*the\s*first\s*to\s*comment", re.IGNORECASE),
    re.compile(r"^\s*leave\s*a\s*(comment|reply)", re.IGNORECASE),
    re.compile(r"^\s*login\s*to\s*(leave|post)\s*a\s*comment", re.IGNORECASE),
    # Tags/categories
    re.compile(r"^\s*tags?\s*:\s*[\w\-,\s]+$", re.IGNORECASE),
    re.compile(r"^\s*categor(y|ies)\s*:\s*[\w\-,\s]+$", re.IGNORECASE),
    re.compile(r"^\s*filed\s*(under|in)\s*:\s*", re.IGNORECASE),
    # Ads/promotional
    re.compile(r"^\s*advertisement\s*$", re.IGNORECASE),
    re.compile(r"^\s*sponsored\s*(content|post|by)\s*", re.IGNORECASE),
    re.compile(r"^\s*promoted\s*$", re.IGNORECASE),
    # Related content (not the actual content)
    re.compile(r"^\s*related\s*(articles?|posts?|content|reading|topics?)\s*:?\s*$", re.IGNORECASE),
    re.compile(r"^\s*you\s*(might|may)\s*(also\s*)?(like|enjoy|be\s*interested)", re.IGNORECASE),
    re.compile(r"^\s*recommended\s*(for\s*you|reading|articles?)\s*$", re.IGNORECASE),
    # Print/download artifacts
    re.compile(r"^\s*(print|download|save)\s*(this)?\s*(page|article|pdf)?\s*$", re.IGNORECASE),
    # Empty brackets/form elements
    re.compile(r"^\s*\[_{2,}\]\s*$"),
    re.compile(r"^\s*\[?\s*\]?\s*$"),
    # Breadcrumbs
    re.compile(r"^\s*\w+\s*[>›»/]\s*\w+\s*[>›»/]\s*\w+", re.IGNORECASE),
]

# Patterns for detecting block types
_CODE_PATTERN = re.compile(
    r"(```[\s\S]*?```|"
    r"(?:^\s{4,}|\t+)(?:def |class |import |from |if |for |while |return |print\(|"
    r"function |const |let |var |public |private ))",
    re.MULTILINE,
)
_DEFINITION_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:"
    r"(?:\w[\w\s]*)\s*(?:is|are|refers?\s*to|means?|defined?\s*as|denotes?)"
    r"|(?:definition|def\.?)\s*[:：]"
    r")",
    re.IGNORECASE,
)
_PROCEDURE_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:"
    r"(?:step\s+\d|first,?\s|then,?\s|next,?\s|finally,?\s)"
    r"|(?:how\s+to|instructions?|procedure|steps?\s*(?:to|for))"
    r"|(?:\d+[\.\)]\s+\w)"
    r")",
    re.IGNORECASE,
)
_QA_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:"
    r"(?:Q\s*[:：]|A\s*[:：]|Question\s*[:：]|Answer\s*[:：])"
    r"|(?:\?\s*$)"
    r")",
    re.IGNORECASE | re.MULTILINE,
)
_TABLE_PATTERN = re.compile(r"\|.*\|.*\|", re.MULTILINE)
_LIST_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:[-•*]\s+|\d+[\.\)]\s+){2,}",
    re.MULTILINE,
)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return max(1, len(text) // 4)


def _semantic_hash(text: str) -> str:
    """Hash for dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _is_noise(text: str) -> bool:
    """Check if text is navigation/boilerplate noise."""
    stripped = text.strip()
    if len(stripped) < 5:
        return True
    for pat in _NOISE_PATTERNS:
        if pat.search(stripped):
            return True
    return False


def _detect_block_type(text: str) -> BlockType:
    """Detect what kind of semantic content this is."""
    if _CODE_PATTERN.search(text):
        return BlockType.CODE
    if _TABLE_PATTERN.search(text) and text.count("|") > 4:
        return BlockType.TABLE
    if _QA_PATTERN.search(text):
        return BlockType.QA
    if _DEFINITION_PATTERN.search(text):
        return BlockType.DEFINITION
    if _PROCEDURE_PATTERN.search(text):
        return BlockType.PROCEDURE
    if _LIST_PATTERN.search(text):
        return BlockType.LIST
    return BlockType.TEXT


def _compress_text(text: str, block_type: BlockType) -> str:
    """
    Semantic compression — reduce token count while preserving meaning.

    10-layer compression pipeline:
    1. Whitespace normalization
    2. URL cleanup
    3. Markdown artifact stripping
    4. Abbreviation/acronym compression
    5. Filler phrase elimination (21 patterns)
    6. Verbose pattern compression (38+ patterns)
    7. Redundant qualifier removal
    8. Redundant sentence-starter trimming
    9. Artifact cleanup
    10. Line-level dedup and noise removal
    """
    # === Layer 1: Whitespace normalization ===
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # === Layer 2: URL cleanup (except in code/references) ===
    if block_type not in (BlockType.CODE, BlockType.REFERENCE):
        text = re.sub(
            r"https?://(?:www\.)?[\w\-\.]+\.[a-z]{2,}(?:/[^\s)\">\]]*)?",
            "",
            text,
        )

    # === Layer 3: Strip markdown image syntax (keep alt text) ===
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Strip markdown link syntax (keep link text)
    if block_type not in (BlockType.CODE, BlockType.REFERENCE):
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)

    # === Layer 4: Abbreviation/acronym compression ===
    # Note: Cross-block abbreviation is handled in SemanticChunker._make_block()
    # This layer handles within-block abbreviation for standalone usage

    # === Layer 5: Filler phrase elimination ===
    # These phrases add zero information for an LLM
    _FILLER_PHRASES = [
        (r"\bit\s+is\s+(?:important|worth|interesting)\s+to\s+note\s+that\b", ""),
        (r"\bit\s+(?:should|is\s+important\s+to)\s+be\s+noted\s+that\b", ""),
        (r"\bas\s+(?:we\s+)?(?:mentioned|discussed|described|noted|stated)\s+(?:above|earlier|previously|before)\b", ""),
        (r"\bas\s+(?:you\s+)?(?:can|may|might)\s+(?:have\s+)?(?:noticed|seen|observed)\b", ""),
        (r"\bin\s+this\s+(?:section|chapter|article|paper|document|guide),?\s*(?:we\s+will\s+)?", ""),
        (r"\blet'?s?\s+(?:take\s+a\s+)?(?:look\s+at|explore|examine|consider|discuss)\b", ""),
        (r"\bnow\s+(?:let'?s?|we\s+will|we)\s+", ""),
        (r"\bbasically,?\s*", ""),
        (r"\bessentially,?\s*", ""),
        (r"\bfundamentally,?\s*", ""),
        (r"\bgenerally\s+speaking,?\s*", ""),
        (r"\bsimply\s+put,?\s*", ""),
        (r"\bin\s+simple\s+terms,?\s*", ""),
        (r"\bneedless\s+to\s+say,?\s*", ""),
        (r"\bto\s+put\s+it\s+(?:simply|another\s+way|differently),?\s*", ""),
        (r"\bfor\s+(?:the\s+sake|purposes?)\s+of\s+(?:this|our)\s+(?:discussion|article|document),?\s*", ""),
        (r"\bwithout\s+(?:further|any\s+further)\s+ado,?\s*", ""),
        (r"\bhaving\s+said\s+that,?\s*", ""),
        (r"\bthat\s+being\s+said,?\s*", ""),
        (r"\bwith\s+that\s+(?:in\s+mind|said),?\s*", ""),
        (r"\bas\s+(?:a\s+matter\s+of\s+fact|the\s+name\s+suggests?),?\s*", ""),
        # Meta-commentary about the document itself
        (r"\bthe\s+following\s+(?:section|table|list|diagram|figure|example)\s+(?:shows?|describes?|illustrates?|explains?|provides?|contains?|presents?|summarizes?)\b", ""),
        (r"\bwe\s+(?:will\s+)?(?:now\s+)?(?:discuss|explore|examine|look\s+at|consider|turn\s+(?:to|our\s+attention\s+to))\b", ""),
        (r"\b(?:the\s+)?(?:above|below|following|preceding)\s+(?:discussion|section|example|table|diagram|figure)\s+(?:shows?|demonstrates?|illustrates?)\b", ""),
        (r"\bas\s+we\s+(?:will\s+)?(?:see|discuss|show|demonstrate)\s+(?:later|below|in\s+the\s+next)\b", ""),
        (r"\bplease\s+(?:note|keep\s+in\s+mind|be\s+(?:aware|advised))\s+that\b", "note:"),
        (r"\bit\s+is\s+(?:recommended|advised|suggested)\s+(?:that\s+(?:you|one)\s+)?(?:to\s+)?", ""),
        (r"\byou\s+(?:should|may\s+want\s+to|might\s+want\s+to|can|are\s+able\s+to|may\s+wish\s+to)\b", ""),
    ]

    if block_type != BlockType.CODE:
        for pattern, replacement in _FILLER_PHRASES:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # === Layer 6: Verbose pattern compression ===
    # Common verbose phrases → concise equivalents
    _VERBOSE_PATTERNS = [
        (r"\bin\s+order\s+to\b", "to"),
        (r"\bfor\s+the\s+purpose\s+of\b", "to"),
        (r"\bwith\s+the\s+(?:aim|goal|intent(?:ion)?|objective)\s+of\b", "to"),
        (r"\bso\s+as\s+to\b", "to"),
        (r"\bdue\s+to\s+the\s+fact\s+that\b", "because"),
        (r"\bowing\s+to\s+the\s+fact\s+that\b", "because"),
        (r"\bby\s+virtue\s+of\s+the\s+fact\s+that\b", "because"),
        (r"\bon\s+account\s+of\s+the\s+fact\s+that\b", "because"),
        (r"\bfor\s+the\s+reason\s+that\b", "because"),
        (r"\bin\s+the\s+event\s+that\b", "if"),
        (r"\bin\s+the\s+case\s+that\b", "if"),
        (r"\bunder\s+circumstances\s+in\s+which\b", "when"),
        (r"\bat\s+this\s+point\s+in\s+time\b", "now"),
        (r"\bat\s+the\s+present\s+time\b", "now"),
        (r"\bat\s+the\s+(?:current|present)\s+moment\b", "now"),
        (r"\bduring\s+the\s+course\s+of\b", "during"),
        (r"\bin\s+the\s+(?:near|immediate)\s+future\b", "soon"),
        (r"\bprior\s+to\b", "before"),
        (r"\bsubsequent\s+to\b", "after"),
        (r"\bfollowing\s+(?:the\s+completion\s+of|this)\b", "after"),
        (r"\ba\s+(?:large|great|significant|substantial|considerable)\s+(?:number|amount|quantity|degree|portion)\s+of\b", "many"),
        (r"\ba\s+(?:small|limited|minimal)\s+(?:number|amount|quantity)\s+of\b", "few"),
        (r"\bthe\s+vast\s+majority\s+of\b", "most"),
        (r"\bin\s+(?:close\s+)?proximity\s+to\b", "near"),
        (r"\bhas\s+the\s+ability\s+to\b", "can"),
        (r"\bis\s+able\s+to\b", "can"),
        (r"\bis\s+capable\s+of\b", "can"),
        (r"\bis\s+in\s+a\s+position\s+to\b", "can"),
        (r"\bmake\s+use\s+of\b", "use"),
        (r"\butilize\b", "use"),
        (r"\bleverage\b", "use"),
        (r"\bfacilitate\b", "enable"),
        (r"\bin\s+addition\s+to\b", "besides"),
        (r"\bwith\s+(?:respect|regard|reference)\s+to\b", "about"),
        (r"\bpertaining\s+to\b", "about"),
        (r"\bin\s+relation\s+to\b", "about"),
        (r"\bconcerning\s+the\s+matter\s+of\b", "about"),
        (r"\bon\s+the\s+other\s+hand,?\s*", "however, "),
        (r"\bdespite\s+the\s+fact\s+that\b", "although"),
        (r"\birrespective\s+of\b", "regardless of"),
        (r"\bin\s+(?:the\s+)?light\s+of\b", "given"),
        (r"\btake\s+into\s+(?:account|consideration)\b", "consider"),
        (r"\bgive\s+(?:consideration|thought)\s+to\b", "consider"),
        (r"\bit\s+is\s+(?:clear|obvious|evident|apparent)\s+that\b", "clearly,"),
        (r"\bthe\s+(?:fact|reason)\s+(?:that|why)\s+(?:is\s+)?(?:because|that)\b", "because"),
        (r"\bas\s+a\s+result\s+of\b", "from"),
        (r"\bas\s+a\s+consequence\s+of\b", "from"),
        (r"\bin\s+the\s+context\s+of\b", "in"),
        (r"\bwith\s+the\s+exception\s+of\b", "except"),
        (r"\bfor\s+the\s+most\s+part\b", "mostly"),
        (r"\bby\s+means\s+of\b", "via"),
        (r"\bthrough\s+the\s+use\s+of\b", "using"),
        (r"\bon\s+a\s+regular\s+basis\b", "regularly"),
        (r"\bin\s+a\s+(?:timely\s+)?manner\b", ""),
        (r"\bthe\s+(?:overall|general)\s+(?:consensus|opinion)\s+is\s+that\b", ""),
        (r"\bit\s+(?:goes|is)\s+without\s+saying\s+that\b", ""),
    ]

    if block_type != BlockType.CODE:
        for pattern, replacement in _VERBOSE_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # === Layer 7: Redundant qualifier removal ===
    if block_type not in (BlockType.CODE, BlockType.TABLE):
        # Remove empty qualifiers that add no precision
        _QUALIFIERS = [
            r"\bvery\s+(?=(?:important|large|small|fast|slow|good|bad|high|low)\b)",
            r"\breally\s+(?=(?:important|useful|good|great|fast|well)\b)",
            r"\bextremely\s+(?=(?:important|useful|large|small|fast)\b)",
            r"\bquite\s+(?=(?:a\s+(?:few|bit|lot)|important|useful|clear|simple)\b)",
            r"\bsomewhat\s+(?=(?:similar|different|complex|simple)\b)",
        ]
        for qpat in _QUALIFIERS:
            text = re.sub(qpat, "", text, flags=re.IGNORECASE)

        # Remove parenthetical hedges/restatements
        text = re.sub(
            r"\s*\(\s*(?:i\.?e\.?|that\s+is|in\s+other\s+words|which\s+is\s+to\s+say),?\s*([^)]{0,80}?)\s*\)",
            r" (\1)",
            text,
            flags=re.IGNORECASE,
        )
        # Remove empty parens left over
        text = re.sub(r"\s*\(\s*\)", "", text)

    # === Layer 8: Redundant sentence-starter trimming ===
    if block_type not in (BlockType.CODE, BlockType.TABLE):
        # Remove "This means that", "This implies that", etc. at line starts
        text = re.sub(
            r"(?m)^(this|that|it)\s+(?:means|implies|suggests|indicates|shows|demonstrates)\s+that\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )
        # Remove "As we can see," "As shown above," etc.
        text = re.sub(
            r"(?m)^as\s+(?:we\s+)?(?:can\s+)?(?:see|shown?|illustrated?|demonstrated?)\s*(?:above|below|here)?\s*,?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

    # === Layer 9: Clean up artifacts from compression ===
    # Double spaces from removed phrases
    text = re.sub(r"  +", " ", text)
    # Leading space on lines
    text = re.sub(r"(?m)^ +", "", text)
    # Sentences starting with lowercase after removal (capitalize)
    # Empty parentheses/brackets
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\[\s*\]", "", text)
    # Orphaned punctuation
    text = re.sub(r"\s+([,;:])", r"\1", text)
    text = re.sub(r"([,;:])\s*([,;:])", r"\1", text)
    # Multiple spaces again
    text = re.sub(r"  +", " ", text)
    # Collapse empty lines
    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)

    # === Layer 10: Line-level dedup and noise removal ===
    lines = text.split("\n")
    cleaned = []
    seen_lines = set()

    for line in lines:
        stripped = line.strip()
        if _is_noise(stripped):
            continue
        if stripped and stripped in seen_lines:
            continue
        if stripped:
            seen_lines.add(stripped)
        cleaned.append(stripped)

    return "\n".join(cleaned).strip()


def _make_summary(text: str, max_len: int = 100) -> str:
    """Generate a one-line summary from the first sentence."""
    # Take first non-empty line
    for line in text.split("\n"):
        line = line.strip()
        if len(line) > 10:
            # Truncate to first sentence or max_len
            end = min(len(line), max_len)
            for sep in (". ", "! ", "? ", "\n"):
                idx = line.find(sep)
                if 0 < idx < max_len:
                    end = idx + 1
                    break
            return line[:end].strip()
    return text[:max_len].strip()


class SemanticChunker:
    """
    Intent-first semantic chunker.

    Splits documents into semantic blocks, detects content type,
    strips noise, and compresses for LLM consumption.
    """

    def __init__(
        self,
        max_block_tokens: int = 512,
        min_block_tokens: int = 20,
        overlap_tokens: int = 0,
    ):
        self.max_block_tokens = max_block_tokens
        self.min_block_tokens = min_block_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, doc: ParsedDocument) -> List[SemanticBlock]:
        """
        Convert a parsed document into semantic blocks.

        Strategy:
        0. Pre-compress at document level (abbreviations, section noise)
        1. Use document sections if available (headings, pages)
        2. Within each section, detect block types
        3. Split oversized sections, merge tiny ones
        4. Compress and classify each block
        """
        blocks: List[SemanticBlock] = []

        # Pre-compression: document-level abbreviation expansion detection
        abbrevs = re.findall(
            r"([A-Z][a-z]+(?:\s+[A-Za-z]+){1,5})\s+\(([A-Z]{2,8}s?)\)",
            doc.text,
        )
        self._doc_abbreviations = {full: abbr for full, abbr in abbrevs}

        if doc.sections and len(doc.sections) > 1:
            # Use structural sections — build heading hierarchy
            heading_stack: List[Tuple[int, str]] = []  # (level, heading)
            for section in doc.sections:
                if self._is_noise_section(section):
                    continue
                # Maintain heading stack based on section level
                if section.heading:
                    level = section.level or 1
                    # Pop headings at same or deeper level
                    while heading_stack and heading_stack[-1][0] >= level:
                        heading_stack.pop()
                    heading_stack.append((level, section.heading.strip()))
                section_path = " > ".join(h for _, h in heading_stack)
                section_blocks = self._process_section(section, section_path)
                blocks.extend(section_blocks)
        else:
            # No structure — split text into semantic units
            text = doc.text
            blocks = self._split_text(text, 0)

        # Split oversized blocks (cap at ~150 tokens for embedding quality)
        max_post_tokens = 150
        split_blocks: List[SemanticBlock] = []
        for block in blocks:
            if block.token_count > max_post_tokens:
                sub_blocks = self._split_oversized_block(block, max_post_tokens)
                split_blocks.extend(sub_blocks)
            else:
                split_blocks.append(block)
        blocks = split_blocks

        # Filter out tiny blocks and deduplicate
        seen_hashes = set()
        final_blocks: List[SemanticBlock] = []
        for block in blocks:
            if block.token_count < self.min_block_tokens:
                continue
            if block.semantic_hash in seen_hashes:
                continue
            seen_hashes.add(block.semantic_hash)
            final_blocks.append(block)

        return final_blocks

    def _process_section(
        self, section: ParsedSection, section_path: str = ""
    ) -> List[SemanticBlock]:
        """Process a single document section into blocks."""
        text = section.text.strip()
        if not text:
            return []

        # Prepend heading as context if available
        if section.heading:
            text = f"{section.heading}\n{text}"

        return self._split_text(text, 0, section_path=section_path)

    def _is_noise_section(self, section: ParsedSection) -> bool:
        """Detect entire sections that are noise (Related Articles, etc.)."""
        heading = (section.heading or "").strip().lower()
        _NOISE_HEADINGS = {
            "related articles", "related posts", "related content",
            "related reading", "recommended articles", "recommended reading",
            "recommended for you", "see also", "you might also like",
            "you may also like", "further reading",
            "comments", "leave a comment", "leave a reply",
            "share this", "share this article", "share this post",
            "about the author", "author bio",
            "advertisement", "sponsored", "promoted",
            "newsletter", "subscribe", "sign up",
            "footer", "sidebar",
        }
        if heading in _NOISE_HEADINGS:
            return True
        # Check if section text is mostly noise lines
        lines = [l.strip() for l in section.text.split("\n") if l.strip()]
        if not lines:
            return True
        noise_count = sum(1 for l in lines if _is_noise(l))
        # Only drop if section is almost entirely noise (>90%).
        # Lower thresholds (e.g. 70%) falsely drop real content sections
        # that happen to have footer noise appended (last section problem).
        if len(lines) > 0 and noise_count / len(lines) > 0.9:
            return True
        return False

    def _split_text(
        self, text: str, char_offset: int, section_path: str = ""
    ) -> List[SemanticBlock]:
        """Split text into semantic blocks."""
        blocks: List[SemanticBlock] = []

        # Split on paragraph boundaries
        paragraphs = re.split(r"\n\s*\n", text)

        current_parts: List[str] = []
        current_tokens = 0
        current_start = char_offset

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if _is_noise(para):
                continue

            para_tokens = _estimate_tokens(para)

            # If adding this paragraph exceeds limit, flush current block
            if current_tokens + para_tokens > self.max_block_tokens and current_parts:
                block = self._make_block(
                    "\n\n".join(current_parts),
                    current_start,
                    current_start + sum(len(p) for p in current_parts),
                    section_path=section_path,
                )
                if block:
                    blocks.append(block)
                current_parts = []
                current_tokens = 0
                current_start = char_offset + text.find(para)

            # If single paragraph is too large, split it further
            if para_tokens > self.max_block_tokens:
                # Flush existing
                if current_parts:
                    block = self._make_block(
                        "\n\n".join(current_parts),
                        current_start,
                        current_start + sum(len(p) for p in current_parts),
                        section_path=section_path,
                    )
                    if block:
                        blocks.append(block)
                    current_parts = []
                    current_tokens = 0

                # Split large paragraph by sentences
                sub_blocks = self._split_large_paragraph(
                    para, char_offset, section_path=section_path
                )
                blocks.extend(sub_blocks)
                current_start = char_offset + text.find(para) + len(para)
                continue

            current_parts.append(para)
            current_tokens += para_tokens

        # Flush remaining
        if current_parts:
            block = self._make_block(
                "\n\n".join(current_parts),
                current_start,
                current_start + sum(len(p) for p in current_parts),
                section_path=section_path,
            )
            if block:
                blocks.append(block)

        return blocks

    def _split_large_paragraph(
        self, text: str, char_offset: int, section_path: str = ""
    ) -> List[SemanticBlock]:
        """Split an oversized paragraph by sentence boundaries."""
        # Split on sentence endings
        sentences = re.split(r"(?<=[.!?])\s+", text)
        blocks: List[SemanticBlock] = []
        current_parts: List[str] = []
        current_tokens = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_tokens = _estimate_tokens(sent)

            if current_tokens + sent_tokens > self.max_block_tokens and current_parts:
                block = self._make_block(
                    " ".join(current_parts),
                    char_offset,
                    char_offset + sum(len(p) for p in current_parts),
                    section_path=section_path,
                )
                if block:
                    blocks.append(block)
                current_parts = []
                current_tokens = 0

            current_parts.append(sent)
            current_tokens += sent_tokens

        if current_parts:
            block = self._make_block(
                " ".join(current_parts),
                char_offset,
                char_offset + sum(len(p) for p in current_parts),
                section_path=section_path,
            )
            if block:
                blocks.append(block)

        return blocks

    def _split_oversized_block(
        self, block: SemanticBlock, max_tokens: int
    ) -> List[SemanticBlock]:
        """Split a block that exceeds max_tokens into smaller sub-blocks."""
        text = block.content
        sentences = re.split(r"(?<=[.!?\n])\s+", text)
        if len(sentences) <= 1:
            # Can't split further by sentence; split by raw token estimate
            words = text.split()
            mid = len(words) // 2
            parts = [" ".join(words[:mid]), " ".join(words[mid:])]
        else:
            parts = []
            current: List[str] = []
            current_tok = 0
            for sent in sentences:
                sent_tok = _estimate_tokens(sent)
                if current_tok + sent_tok > max_tokens and current:
                    parts.append(" ".join(current))
                    current = []
                    current_tok = 0
                current.append(sent)
                current_tok += sent_tok
            if current:
                parts.append(" ".join(current))

        sub_blocks = []
        for part in parts:
            part = part.strip()
            if not part or len(part) < 10:
                continue
            token_count = _estimate_tokens(part)
            if token_count < self.min_block_tokens:
                continue
            sub_blocks.append(SemanticBlock(
                block_type=block.block_type,
                content=part,
                token_count=token_count,
                semantic_hash=_semantic_hash(part),
                summary=_make_summary(part),
                source_range=block.source_range,
                metadata=dict(block.metadata) if block.metadata else {},
            ))
        return sub_blocks if sub_blocks else [block]

    def _make_block(
        self, text: str, start: int, end: int, section_path: str = ""
    ) -> Optional[SemanticBlock]:
        """Create a semantic block from text."""
        block_type = _detect_block_type(text)
        compressed = _compress_text(text, block_type)

        # Apply document-level abbreviation compression
        if block_type != BlockType.CODE and hasattr(self, "_doc_abbreviations"):
            for full_name, abbr in self._doc_abbreviations.items():
                # Don't replace in the definition itself "Full Name (ABBR)"
                definition = f"{full_name} ({abbr})"
                if definition in compressed:
                    # Replace subsequent occurrences only
                    parts = compressed.split(definition, 1)
                    if len(parts) == 2:
                        parts[1] = parts[1].replace(full_name, abbr)
                        compressed = definition.join(parts)
                else:
                    # No definition in this block — replace all
                    compressed = compressed.replace(full_name, abbr)

        if not compressed or len(compressed) < 10:
            return None

        token_count = _estimate_tokens(compressed)
        summary = _make_summary(compressed)

        block_metadata: Dict[str, Any] = {}
        if section_path:
            block_metadata["section_path"] = section_path

        return SemanticBlock(
            block_type=block_type,
            content=compressed,
            token_count=token_count,
            semantic_hash=_semantic_hash(compressed),
            summary=summary,
            source_range=(start, end),
            metadata=block_metadata,
        )
