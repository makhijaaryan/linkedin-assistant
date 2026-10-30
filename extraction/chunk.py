"""Split a document into 200-500-word chunks with ~10% overlap, keeping char offsets.

Chunks break at line ends where possible (section cues), so a requirement or role
header is rarely cut in half. Short documents stay as one chunk.
"""

import re
from dataclasses import dataclass

WORD = re.compile(r"\S+")


@dataclass
class Chunk:
    index: int
    start: int          # char offset of the chunk in the document
    text: str


def chunk_text(text: str, min_words: int = 200, max_words: int = 500, overlap: float = 0.10) -> list[Chunk]:
    words = list(WORD.finditer(text))
    if len(words) <= max_words:
        return [Chunk(0, 0, text)]
    # word indices that end a line: preferred cut points
    line_end = {i for i, w in enumerate(words[:-1]) if "\n" in text[w.end():words[i + 1].start()]}
    chunks, first = [], 0
    while first < len(words):
        last = min(first + max_words, len(words)) - 1
        if last < len(words) - 1:
            cuts = [i for i in range(first + min_words - 1, last + 1) if i in line_end]
            last = cuts[-1] if cuts else last
        start, end = words[first].start(), words[last].end()
        chunks.append(Chunk(len(chunks), start, text[start:end]))
        if last == len(words) - 1:
            break
        first = max(first + 1, last + 1 - int((last - first + 1) * overlap))
    return chunks
