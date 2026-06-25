"""
Corpus loader + chunker for the unstructured (Wikipedia) side of the pipeline.

Each Wikipedia JSON (produced by wikipedia_fetcher.py) looks like:
    {title, url, category, summary, text, sections: {sec_title: sec_text}}

We turn every article into a list of overlapping passage-level chunks, each
carrying enough metadata to be traceable back to its source section. Chunk
granularity = section, with long sections sliced into word-windows so a single
retrieved chunk is focused enough to be a "right vs lucky" judgement call.
"""

import os
import json
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKI_DIR = os.path.join(ROOT, 'data/wikipedia')

# Word-window sizing for long sections
CHUNK_WORDS   = 180
CHUNK_OVERLAP = 40


def _window(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Slice text into overlapping word windows."""
    words = text.split()
    if len(words) <= size:
        return [text] if text.strip() else []
    out, start = [], 0
    step = size - overlap
    while start < len(words):
        out.append(' '.join(words[start:start + size]))
        start += step
    return out


def _load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_chunks(wiki_dir=WIKI_DIR):
    """
    Walk every category folder (seasons/teams/players) and emit chunk dicts:
        {chunk_id, doc_title, category, section, url, text}
    """
    chunks = []
    for path in sorted(glob.glob(os.path.join(wiki_dir, '*', '*.json'))):
        doc = _load_json(path)
        title    = doc.get('title', os.path.basename(path))
        category = doc.get('category', os.path.basename(os.path.dirname(path)))
        url      = doc.get('url', '')

        # 1) Summary chunk (always present, high-value)
        summary = (doc.get('summary') or '').strip()
        if summary:
            for j, w in enumerate(_window(summary)):
                chunks.append(_mk(title, category, 'Summary', url, w, j))

        # 2) Section chunks
        for sec_title, sec_text in (doc.get('sections') or {}).items():
            sec_text = (sec_text or '').strip()
            if not sec_text:
                continue
            for j, w in enumerate(_window(sec_text)):
                chunks.append(_mk(title, category, sec_title, url, w, j))

    # Assign stable ids
    for i, c in enumerate(chunks):
        c['chunk_id'] = i
    return chunks


def _mk(title, category, section, url, text, j):
    return {
        'doc_title': title,
        'category':  category,
        'section':   section,
        'url':       url,
        'text':      text,
        '_sub':      j,
    }


if __name__ == '__main__':
    cs = load_chunks()
    from collections import Counter
    by_cat = Counter(c['category'] for c in cs)
    print(f"Loaded {len(cs)} chunks from {WIKI_DIR}")
    print("By category:", dict(by_cat))
    print("\nExample chunk:")
    ex = cs[0]
    print({k: (v[:120] + '...' if k == 'text' else v) for k, v in ex.items()})
