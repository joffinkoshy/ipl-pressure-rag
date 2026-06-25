"""
FAISS + sentence-transformers vector store over the Wikipedia chunks.

Embeddings: all-MiniLM-L6-v2 (384-dim, local, free).
Index: FAISS IndexFlatIP over L2-normalized vectors == cosine similarity.

The store is rebuilt in-memory each run from the on-disk chunks (fast at this
scale, ~130+ chunks). Player pages fetched on demand are picked up by calling
build() again after fetching.
"""

import os
import numpy as np

from rag.corpus import load_chunks

_MODEL = None
EMBED_MODEL = 'all-MiniLM-L6-v2'


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(EMBED_MODEL)
    return _MODEL


class VectorStore:
    def __init__(self, chunks, embeddings, index):
        self.chunks = chunks
        self.embeddings = embeddings
        self.index = index

    @classmethod
    def build(cls, wiki_dir=None):
        import faiss
        chunks = load_chunks(wiki_dir) if wiki_dir else load_chunks()
        model = _get_model()
        texts = [c['text'] for c in chunks]
        emb = model.encode(texts, normalize_embeddings=True,
                           show_progress_bar=False).astype('float32')
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        return cls(chunks, emb, index)

    def search(self, query, k=5, category=None, doc_title=None):
        """
        Return top-k chunks for a query, each annotated with similarity score.
        Optional hard filters by category ('players'/'teams'/'seasons') or an
        exact doc_title — used to keep a hop scoped to the right entity.
        """
        model = _get_model()
        q = model.encode([query], normalize_embeddings=True).astype('float32')
        # Over-fetch then filter, so filters still return k results
        scores, idx = self.index.search(q, min(len(self.chunks), k * 6))
        out = []
        for s, i in zip(scores[0], idx[0]):
            c = self.chunks[i]
            if category and c['category'] != category:
                continue
            if doc_title and c['doc_title'].lower() != doc_title.lower():
                continue
            hit = dict(c)
            hit['score'] = float(s)
            out.append(hit)
            if len(out) >= k:
                break
        return out


if __name__ == '__main__':
    vs = VectorStore.build()
    print(f"Indexed {len(vs.chunks)} chunks, dim={vs.embeddings.shape[1]}")
    for q in ["death over bowling specialist yorkers",
              "most dramatic final match result"]:
        print(f"\nQUERY: {q}")
        for h in vs.search(q, k=3):
            print(f"  {h['score']:.3f}  [{h['doc_title']} § {h['section']}]  "
                  f"{h['text'][:90]}...")
