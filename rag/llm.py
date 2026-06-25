"""
Pluggable LLM synthesis layer.

The assignment allows any LLM (or none). To guarantee the pipeline runs
end-to-end with NO API key, this module auto-selects a backend:

    1. ANTHROPIC_API_KEY present + `anthropic` installed -> Claude
    2. OPENAI_API_KEY present + `openai` installed        -> GPT
    3. otherwise                                           -> Extractive fallback

The extractive fallback is fully deterministic: it composes the structured
findings (the hard numbers) with the most relevant retrieved Wikipedia
sentences into a defensible, traceable answer — no hallucination, no network.
Swapping in a real LLM later changes only the prose, not the evidence.
"""

import os
import re

CLAUDE_MODEL = 'claude-opus-4-8'
OPENAI_MODEL = 'gpt-4o'


def get_backend():
    if os.getenv('ANTHROPIC_API_KEY'):
        try:
            import anthropic  # noqa
            return 'anthropic'
        except ImportError:
            pass
    if os.getenv('OPENAI_API_KEY'):
        try:
            import openai  # noqa
            return 'openai'
        except ImportError:
            pass
    return 'extractive'


# ── Real LLM calls ──────────────────────────────────────────────────────────
def _call_anthropic(prompt):
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        messages=[{'role': 'user', 'content': prompt}],
    )
    return msg.content[0].text.strip()


def _call_openai(prompt):
    import openai
    client = openai.OpenAI()
    r = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=600,
        messages=[{'role': 'user', 'content': prompt}],
    )
    return r.choices[0].message.content.strip()


# ── Extractive fallback ─────────────────────────────────────────────────────
def _split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _relevant_sentences(chunks, keywords, k=3):
    """Pick the sentences across retrieved chunks that best match keywords."""
    kws = [w.lower() for w in keywords if len(w) > 2]
    scored = []
    for ch in chunks:
        for sent in _split_sentences(ch['text']):
            sl = sent.lower()
            score = sum(sl.count(w) for w in kws)
            if score > 0:
                scored.append((score, sent, ch['doc_title'], ch['section']))
    scored.sort(key=lambda x: -x[0])
    seen, out = set(), []
    for _, sent, doc, sec in scored:
        if sent in seen:
            continue
        seen.add(sent)
        out.append((sent, doc, sec))
        if len(out) >= k:
            break
    return out


def _extractive(question, headline, evidence_lines, chunks, keywords):
    parts = [headline, ""]
    if evidence_lines:
        parts.append("Structured evidence (Cricsheet):")
        parts += [f"  - {e}" for e in evidence_lines]
        parts.append("")
    sents = _relevant_sentences(chunks, keywords)
    if sents:
        parts.append("Supporting Wikipedia evidence:")
        for sent, doc, sec in sents:
            parts.append(f"  - \"{sent}\"  [{doc} § {sec}]")
    elif chunks:
        parts.append("Supporting Wikipedia evidence (top retrieved passage):")
        ch = chunks[0]
        snippet = ch['text'][:300] + ('...' if len(ch['text']) > 300 else '')
        parts.append(f"  - \"{snippet}\"  [{ch['doc_title']} § {ch['section']}]")
    return "\n".join(parts).strip()


# ── Public API ──────────────────────────────────────────────────────────────
def synthesize(question, headline, evidence_lines, chunks, keywords=None):
    """
    Compose a final answer.
      question      : the user question (for the LLM prompt)
      headline      : one-line structured conclusion (the hard answer)
      evidence_lines: list of structured stat strings
      chunks        : retrieved Wikipedia chunks (dicts with text/doc_title/section)
      keywords      : terms used to pick relevant sentences in fallback mode
    """
    keywords = keywords or []
    backend = get_backend()

    if backend == 'extractive':
        return _extractive(question, headline, evidence_lines, chunks, keywords)

    # Build an evidence-grounded prompt for the real LLM
    ev = "\n".join(f"- {e}" for e in evidence_lines)
    ctx = "\n\n".join(
        f"[{c['doc_title']} § {c['section']}]\n{c['text']}" for c in chunks[:6]
    )
    prompt = (
        f"Question: {question}\n\n"
        f"Structured finding (treat as ground truth, do not contradict):\n"
        f"{headline}\n{ev}\n\n"
        f"Retrieved Wikipedia context:\n{ctx}\n\n"
        f"Write a concise, defensible answer. Use ONLY the evidence above. "
        f"Cite the Wikipedia article/section in brackets when you use it. "
        f"If the context does not support a claim, say so."
    )
    try:
        if backend == 'anthropic':
            return _call_anthropic(prompt)
        if backend == 'openai':
            return _call_openai(prompt)
    except Exception as e:
        # Never let a network/LLM error fail the pipeline
        return _extractive(question, headline, evidence_lines, chunks, keywords) \
            + f"\n\n[note: LLM backend '{backend}' failed ({e}); used extractive fallback]"


if __name__ == '__main__':
    print("Active LLM backend:", get_backend())
    demo = synthesize(
        "Demo?", "Headline: X depends on Y.",
        ["stat 1 = 0.26", "stat 2 = 25"],
        [{'text': 'Y is a death-over specialist known for yorkers. He debuted in 2013.',
          'doc_title': 'Y', 'section': 'Career'}],
        keywords=['death', 'specialist', 'yorker'],
    )
    print("\n" + demo)
