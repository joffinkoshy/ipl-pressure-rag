# Part 3 — Break Your Own Pipeline

## Retrieval precision across all 5 questions

Top-1 retrieved chunk is from the INTENDED document?

| Q | Semantic hop | Intended doc | Unscoped top-1 | Before | After |
|---|---|---|---|:---:|:---:|
| Q1 | death-specialist page for JJ Bumrah | Jasprit Bumrah | Jasprit Bumrah | ✅ | ✅ |
| Q2 | 2024 season article for momentum match | 2024 Indian Premier League | Royal Challengers Bengaluru | ❌ | ✅ |
| Q3 | characterization page for H Klaasen | Heinrich Klaasen | Heinrich Klaasen | ✅ | ✅ |
| Q4 | 2024 season article for dip match | 2024 Indian Premier League | Sunrisers Hyderabad | ❌ | ✅ |
| Q5 | fact page for opener SV Samson | Sanju Samson | Sanju Samson | ✅ | ✅ |
| Q5 | fact page for opener V Kohli | Virat Kohli | Virat Kohli | ✅ | ✅ |

**Retrieval precision BEFORE fix (unscoped): 67%**  
**Retrieval precision AFTER fix (entity-scoped): 100%**

## Failure Mode A — wrong-entity chunk (FIXED)

**Q2 hop — query:** `Punjab Kings v Royal Challengers Bengaluru Royal Challengers Bengaluru 2024 IPL close match result final over thriller`
- Intended document: **2024 Indian Premier League**
- Naively retrieved instead: **Royal Challengers Bengaluru § See also** (sim=0.669)
- Wrong chunk text: "2026 Royal Challengers Bengaluru season – Indian Premier League cricket team
Royal Challengers Bengaluru (WPL team)..."
- Why wrong: the embedding matched generic cricket terms, not the specific entity. Scoping to `doc_title=2024 Indian Premier League` fixes it.

**Q4 hop — query:** `Sunrisers Hyderabad v Mumbai Indians 2024 IPL Sunrisers Hyderabad match result`
- Intended document: **2024 Indian Premier League**
- Naively retrieved instead: **Sunrisers Hyderabad § Home ground** (sim=0.795)
- Wrong chunk text: "in Chennai after TNCA failed to secure permission to open three locked stands for the match. Hyderabad Cricket Association won the award for best ground and pitch during the IPL 2019 and IPL 2024 seas..."
- Why wrong: the embedding matched generic cricket terms, not the specific entity. Scoping to `doc_title=2024 Indian Premier League` fixes it.

## Failure Mode B — naive name resolution → wrong page (shown)

Cricsheet uses initials (`SV Samson`); Wikipedia titles are full names (`Sanju Samson`). A naive resolver that just searches Wikipedia for the abbreviated name returns a confidently WRONG entity, and every 'fact' in that hop is then about the wrong person.

Resolving the referenced players via the **search fallback only** (curated overrides disabled):

| Cricsheet name | Naive search resolves to | Correct (override) |
|---|---|---|
| H Klaasen | Virat Kohli ❌ | Heinrich Klaasen |
| JJ Bumrah | Isa Guha ❌ | Jasprit Bumrah |
| SV Samson | Ajinkya Rahane ❌ | Sanju Samson |
| TH David | Th' Dudes ❌ | Tim David |
| V Kohli | Virat Kohli  | Virat Kohli |

**4/5 referenced players resolve to the WRONG Wikipedia page under naive search** (e.g. a batsman maps to a different cricketer, or even a band). Left unfixed, the player hop would retrieve and quote facts about the wrong person — a confidently wrong answer.

**Mitigation:** the curated `NAME_OVERRIDES` map in `wikipedia_fetcher.py` pins each referenced Cricsheet name to its correct Wikipedia title (and the resolver now uses `requests` for TLS-verified search as a fallback for non-curated names).
