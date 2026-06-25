# Reflection

**Architecture decisions.** I split the two sources by what they're good at:
structured cricket questions are answered with *deterministic pandas tools* over
a pressure-enriched table, and only the qualitative "what does Wikipedia say"
sub-questions go through vector retrieval. This keeps the hard numbers exact and
reserves embeddings for fuzzy text, where they belong. The LLM layer is
pluggable with an extractive fallback so the pipeline runs end-to-end with no
API key — retrieval correctness never depends on a model. Every hop logs its
query, its result, and a confidence reason, so an answer can be audited rather
than trusted.

**What I'd improve with more time.** Wikipedia's `wikipedia-api` drops tables, so
season articles lack per-league-match prose — Q2/Q4's "what does the article
say" hop can only reach the season summary. I'd add an infobox/table parser or a
second source. I'd also replace heuristic confidence with retrieval-overlap
calibration.

**What surprised me.** How easily naive name resolution fails: Cricsheet
initials resolved "TH David" to a *band* and "SV Samson" to a different
cricketer. The biggest reliability wins came from entity-scoping and a curated
name map — not from the model.
