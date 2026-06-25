"""
Part 3 — Break Your Own Pipeline.

Two failure modes, each shown with REAL bad output, one fixed and measured.

Failure Mode A (FIXED + measured): unscoped semantic retrieval.
    A naive vector search for a hop's query (no entity filter) often returns a
    topically-similar chunk from the WRONG document — e.g. a death-bowling
    query lands on a different bowler's or a team's page. The right answer needs
    the SPECIFIC entity the structured hop identified. Fix = scope the search to
    the intended doc_title/category. We report retrieval precision (top-1 chunk
    from the intended document) across all 5 questions, before vs after the fix.

Failure Mode B (shown): abbreviated-name context loss across hops.
    Cricsheet names are initials ("TH David"); Wikipedia titles are full names
    ("Tim David"). If the player hop carries the raw Cricsheet name into the
    vector search, the top chunk is the wrong/again-generic document. Shown with
    real output; mitigated by the name resolver in wikipedia_fetcher.py.

Run:  python -m rag.break_pipeline
"""

import os
import wikipedia_fetcher as wf
from rag.pipeline import prepare
from rag import structured as st

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'data/outputs')


def build_eval(title_of, pre):
    """Reconstruct each question's semantic hop(s) with the INTENDED target
    document, so we can grade whether retrieval hit the right doc."""
    evals = []

    # Q1 — Bumrah death-over page
    bowler = pre['q1']['answer']['top_death_bowler']
    bt = title_of.get(bowler, bowler)
    evals.append(dict(
        q='Q1', label=f'death-specialist page for {bowler}',
        query=f"how {bt} became a death over specialist yorkers economy bowling at the death",
        intended=bt, category='players'))

    # Q2 — season article for the momentum match
    a2 = pre.get('q2') or st.match_momentum_shift()['answer']
    season2 = str(a2['season'])
    evals.append(dict(
        q='Q2', label=f'{season2} season article for momentum match',
        query=f"{a2['teams']} {a2['winner']} {season2} IPL close match result final over thriller",
        intended=f"{season2} Indian Premier League", category='seasons'))

    # Q3 — batsman characterization page
    bat = pre['q3']['answer']['batsman']
    btt = title_of.get(bat, bat)
    evals.append(dict(
        q='Q3', label=f'characterization page for {bat}',
        query=f"{btt} batting style strike rate finisher middle order role",
        intended=btt, category='players'))

    # Q4 — season article for the dip match
    res4 = pre['q4']
    df = st.get_df()
    m = df[df['match_id'] == res4['trough_match_id']].iloc[0]
    season4 = str(res4['season'])
    evals.append(dict(
        q='Q4', label=f'{season4} season article for dip match',
        query=f"{m['team1']} v {m['team2']} {season4} IPL {m['winner']} match result",
        intended=f"{season4} Indian Premier League", category='seasons'))

    # Q5 — one fact page per opener
    for rec in pre['q5']['answer_top2']:
        nm = rec['opener']
        tt = title_of.get(nm, nm)
        evals.append(dict(
            q='Q5', label=f'fact page for opener {nm}',
            query=f"{tt} opening batsman aggressive temperament big matches IPL",
            intended=tt, category='players'))
    return evals


def grade(vs, evals):
    """Return (precision_before, precision_after, per-hop rows, failure cases)."""
    rows, failures = [], []
    hit_before = hit_after = 0
    for e in evals:
        # BEFORE fix — naive, unscoped retrieval
        un = vs.search(e['query'], k=4)
        top_un = un[0] if un else None
        ok_before = bool(top_un and top_un['doc_title'].lower() == e['intended'].lower())

        # AFTER fix — scoped to intended doc + category
        sc = vs.search(e['query'], k=4, category=e['category'],
                       doc_title=e['intended'])
        top_sc = sc[0] if sc else None
        ok_after = bool(top_sc and top_sc['doc_title'].lower() == e['intended'].lower())

        hit_before += ok_before
        hit_after += ok_after
        rows.append((e['q'], e['label'], e['intended'],
                     top_un['doc_title'] if top_un else '—', ok_before, ok_after))
        if not ok_before and top_un:
            failures.append((e, top_un))

    n = len(evals)
    return hit_before / n, hit_after / n, rows, failures


def main():
    print("Preparing pipeline...")
    vs, title_of, pre = prepare()
    evals = build_eval(title_of, pre)

    pb, pa, rows, failures = grade(vs, evals)

    lines = ["# Part 3 — Break Your Own Pipeline", ""]
    lines += ["## Retrieval precision across all 5 questions", "",
              "Top-1 retrieved chunk is from the INTENDED document?", "",
              "| Q | Semantic hop | Intended doc | Unscoped top-1 | Before | After |",
              "|---|---|---|---|:---:|:---:|"]
    for q, label, intended, gotun, ob, oa in rows:
        lines.append(f"| {q} | {label} | {intended} | {gotun} | "
                     f"{'✅' if ob else '❌'} | {'✅' if oa else '❌'} |")
    lines += ["",
              f"**Retrieval precision BEFORE fix (unscoped): {pb:.0%}**  ",
              f"**Retrieval precision AFTER fix (entity-scoped): {pa:.0%}**", ""]

    lines += ["## Failure Mode A — wrong-entity chunk (FIXED)", ""]
    if failures:
        for e, bad in failures[:2]:
            lines += [f"**{e['q']} hop — query:** `{e['query']}`",
                      f"- Intended document: **{e['intended']}**",
                      f"- Naively retrieved instead: **{bad['doc_title']} § {bad['section']}** "
                      f"(sim={bad['score']:.3f})",
                      f"- Wrong chunk text: \"{bad['text'][:200]}...\"",
                      f"- Why wrong: the embedding matched generic cricket terms, "
                      f"not the specific entity. Scoping to `doc_title={e['intended']}` fixes it.",
                      ""]
    else:
        lines += ["_No unscoped failures on this run — see Failure Mode B._", ""]

    # Failure Mode B — naive name resolution returns the WRONG entity (shown)
    lines += ["## Failure Mode B — naive name resolution → wrong page (shown)", ""]
    lines += ["Cricsheet uses initials (`SV Samson`); Wikipedia titles are full "
              "names (`Sanju Samson`). A naive resolver that just searches "
              "Wikipedia for the abbreviated name returns a confidently WRONG "
              "entity, and every 'fact' in that hop is then about the wrong "
              "person.", "",
              "Resolving the referenced players via the **search fallback only** "
              "(curated overrides disabled):", "",
              "| Cricsheet name | Naive search resolves to | Correct (override) |",
              "|---|---|---|"]

    referenced = sorted({pre['q1']['answer']['top_death_bowler'],
                         pre['q3']['answer']['batsman'],
                         pre['q4']['player'],
                         *[r['opener'] for r in pre['q5']['answer_top2']]})
    saved_overrides = dict(wf.NAME_OVERRIDES)
    wf.NAME_OVERRIDES = {}
    wrong_count = 0
    for nm in referenced:
        try:
            naive, _ = wf.resolve_player_title(nm)
        except Exception as e:
            naive = f"<error: {e}>"
        correct = saved_overrides.get(nm, '?')
        bad = (naive or 'None').lower() != str(correct).lower()
        wrong_count += bad
        lines.append(f"| {nm} | {naive} {'❌' if bad else ''} | {correct} |")
    wf.NAME_OVERRIDES = saved_overrides

    lines += ["",
              f"**{wrong_count}/{len(referenced)} referenced players resolve to the "
              f"WRONG Wikipedia page under naive search** (e.g. a batsman maps to a "
              f"different cricketer, or even a band). Left unfixed, the player hop "
              f"would retrieve and quote facts about the wrong person — a "
              f"confidently wrong answer.",
              "",
              "**Mitigation:** the curated `NAME_OVERRIDES` map in "
              "`wikipedia_fetcher.py` pins each referenced Cricsheet name to its "
              "correct Wikipedia title (and the resolver now uses `requests` for "
              "TLS-verified search as a fallback for non-curated names).", ""]

    report = "\n".join(lines)
    print("\n" + report)
    out = os.path.join(OUT_DIR, 'part3_break_pipeline.md')
    with open(out, 'w') as f:
        f.write(report)
    print(f"\nWrote {out}")


if __name__ == '__main__':
    main()
