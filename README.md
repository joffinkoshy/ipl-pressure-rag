# Multi-Hop RAG over IPL Cricket Data (2024–2026)

A multi-hop Retrieval-Augmented Generation system that answers 5 questions by
chaining retrievals across **two sources that no single document can answer
alone**:

- **Structured** — Cricsheet ball-by-ball data for IPL **2024, 2025, 2026**
  (219 matches, 51,915 deliveries), enriched with a custom **Pressure Index**.
- **Unstructured** — Wikipedia articles for the 3 seasons, all 10 franchises,
  and (on demand) every player referenced in an answer.

Everything runs **end-to-end with no API key** — retrieval is local
(FAISS + sentence-transformers) and answer synthesis falls back to a
deterministic extractive composer. A real LLM (Claude/GPT) can be plugged in by
setting one environment variable.

---

## 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Raw data prerequisite
Download the **Indian Premier League** CSV package from
[cricsheet.org/downloads](https://cricsheet.org/downloads/) and unzip it so the
per-match CSVs live in `./ipl_csv/` (e.g. `ipl_csv/1535465.csv`). The loader
auto-selects the 2024/2025/2026 seasons.

---

## 2. Run everything

```bash
python run_all.py
```

This executes all five stages in order. To run them individually:

| # | Command | Produces |
|---|---|---|
| 1 | `python main.py` | `data/ipl_2024_25_26.csv` |
| 2 | `python wikipedia_fetcher.py` | `data/wikipedia/**`, `data/outputs/match_importance.json` |
| 3 | `python pressure_index.py` | `data/outputs/ipl_with_pressure_index.csv`, phase plot |
| 4 | `python -m rag.run_questions` | `data/outputs/answers.md` (+ Q4 arc plot) |
| 5 | `python -m rag.break_pipeline` | `data/outputs/part3_break_pipeline.md` |

---

## 3. Part 1 — Pressure Index

A **per-delivery, batsman-POV** index: *how much pressure is this batsman under
on this ball?* Built multiplicatively from justified variables (see
`pressure_index.py` for the full rationale on each):

**Innings 2 (chase):**
`PI = rrr_ratio × wicket_factor × phase_weight × recent_wicket × dot_streak × settedness × runs_needed × match_importance`

| Variable | Captures |
|---|---|
| `rrr_ratio` | Required-vs-current run rate (non-linear: an impossible chase scores *low*) |
| `wicket_factor` | Batting resources lost (irreversible) |
| `phase_weight` | Powerplay 1.0 / middle 1.2 / death 1.5 |
| `recent_wicket` | New batsman / momentum shift |
| `dot_streak` | Batsman strangled (3+ dots) |
| `settedness` (IBF) | Balls the striker has faced — new batsman = more pressure |
| `runs_needed` | Absolute size of the task |
| `match_importance` | league 1.0 / playoff 1.15 / **final 1.30** (cross-checked vs Wikipedia) |

Innings 1 (no target) uses a venue **below-par factor** with the other
variables applied *conditionally* (they only bite when the team is also losing
wickets).

**Validation — top-5 highest-pressure deliveries** are all death-over chase
moments; the single highest is from the **2025 final** (Punjab chasing 191, lost
by 6 runs). **Phase means: death 1.85 > middle 1.60 > powerplay 1.45.**

---

## 4. Part 2 — Multi-Hop RAG (`rag/`)

| Module | Role |
|---|---|
| `corpus.py` | Chunk Wikipedia JSON into passages (section-level, word-windowed) |
| `vector_store.py` | `all-MiniLM-L6-v2` embeddings + FAISS cosine index; entity-scoped search |
| `structured.py` | Pandas "tools" over the pressure table — the Cricsheet hops |
| `llm.py` | Pluggable synthesis (Claude → GPT → extractive fallback) |
| `pipeline.py` | Defines each question as a ≥3-hop chain; logs every hop |
| `run_questions.py` | Runs all 5, writes the traceable report |

Each answer in `answers.md` shows **what query triggered each hop, what was
retrieved, how the answer was synthesised, and a confidence level with reason.**

### Optional: use a real LLM
```bash
pip install anthropic            # or: pip install openai
export ANTHROPIC_API_KEY=sk-...  # or: export OPENAI_API_KEY=sk-...
python -m rag.run_questions
```
Without a key it uses the extractive composer (no hallucination, fully traceable).

---

## 5. Part 3 — Break Your Own Pipeline

`python -m rag.break_pipeline` → `data/outputs/part3_break_pipeline.md`

- **Failure A (fixed + measured):** unscoped semantic retrieval pulls a
  topically-similar but *wrong-entity* chunk. **Retrieval precision 67% → 100%**
  after scoping each hop to its intended document.
- **Failure B (shown):** naive resolution of Cricsheet initials returns the
  wrong Wikipedia page (`SV Samson` → *Ajinkya Rahane*, `TH David` → a band).
  Mitigated by a curated name map + TLS-verified search fallback.

---

## 6. Reflection

See [REFLECTION.md](REFLECTION.md) (≤200 words).
