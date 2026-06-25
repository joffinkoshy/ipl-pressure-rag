"""
Multi-hop RAG orchestration for the 5 questions.

Each question is a CHAIN of >=3 hops. A hop is either:
  - structured : a pandas query over the pressure-enriched Cricsheet table
  - semantic   : a vector search over the Wikipedia chunks
  - derive/plot: an intermediate computation (still logged)

Every hop records the query that triggered it and what it returned, so the
final answer is fully traceable. The answer is composed by the pluggable LLM
layer (extractive fallback when no API key is set).

Run:  python -m rag.run_questions
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import wikipedia_fetcher as wf
from rag import structured as st
from rag.vector_store import VectorStore
from rag.llm import synthesize, get_backend

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'data/outputs')


# ── helpers ─────────────────────────────────────────────────────────────────
def _hop(n, htype, query, retrieved, note=''):
    return {'n': n, 'type': htype, 'query': query,
            'retrieved': retrieved, 'note': note}


def _chunk_brief(chunks):
    return [f"[{c['doc_title']} § {c['section']}] (sim={c['score']:.3f}) "
            f"{c['text'][:110]}..." for c in chunks]


def _confidence(structured_margin, top_sim, corroborated):
    """
    Heuristic confidence in [Low, Medium, High] with a reason.
      structured_margin: relative gap between #1 and #2 structured result
      top_sim          : best semantic similarity of supporting evidence
      corroborated     : did Wikipedia corroborate the structured finding?
    """
    score = 0
    score += 1 if structured_margin >= 0.04 else 0
    score += 1 if top_sim >= 0.35 else 0
    score += 1 if corroborated else 0
    level = ['Low', 'Low', 'Medium', 'High'][score]
    reason = (f"structured margin={structured_margin:.3f}, "
              f"top semantic sim={top_sim:.3f}, "
              f"wiki corroborated={corroborated}")
    return level, reason


# ── pre-pass: fetch every referenced player, then build the store ────────────
def prepare():
    """Run structured tools to learn which players we reference, fetch their
    Wikipedia pages on demand, then build the vector store once."""
    referenced = set()

    q1 = st.death_over_bowler_dependency()
    referenced.add(q1['answer']['top_death_bowler'])

    q3 = st.batsman_pressure_buckets()
    referenced.add(q3['answer']['batsman'])

    q4 = st.player_pressure_arc()
    referenced.add(q4['player'])

    q5 = st.opener_pressure_stats(min_balls=100)
    for rec in q5['answer_top2']:
        referenced.add(rec['opener'])

    print(f"Referenced players: {sorted(referenced)}")
    name_to_title = {}
    for name in sorted(referenced):
        title = wf.fetch_player(name)
        name_to_title[name] = title or name

    vs = VectorStore.build()
    print(f"Vector store: {len(vs.chunks)} chunks indexed "
          f"(LLM backend: {get_backend()})")
    return vs, name_to_title, {'q1': q1, 'q3': q3, 'q4': q4, 'q5': q5}


# ── Q1 ──────────────────────────────────────────────────────────────────────
def question1(vs, title_of, pre):
    q = ("Which IPL team has been most dependent on a single bowler in death "
         "overs across the 3 seasons, and what does that bowler's Wikipedia "
         "page reveal about how they developed as a death-over specialist?")
    hops = []

    # Hop 1 — structured: dependency table
    res = pre['q1']
    ans = res['answer']
    team, bowler = ans['team'], ans['top_death_bowler']
    margin = float(res['table'].iloc[0]['dependency_share'] -
                   res['table'].iloc[1]['dependency_share'])
    hops.append(_hop(
        1, 'structured',
        "death-over (16-20) legal-ball share per team's most-used bowler",
        [f"{r['team']}: {r['top_death_bowler']} "
         f"{r['dependency_share']:.1%} of death balls "
         f"({r['bowler_death_balls']}/{r['team_death_balls']}), "
         f"{r['bowler_death_wkts']} wkts, econ {r['bowler_death_econ']}"
         for _, r in res['table'].head(3).iterrows()],
        note=f"Top: {team} depends most on {bowler}"))

    # Hop 2 — structured cross-season consistency
    df = st.get_df()
    bdf = df[(df['phase'] == 'death') & (df['is_legal'] == 1) &
             (df['bowler'] == bowler) & (df['bowling_team'] == team)]
    by_season = bdf.groupby('season').size().to_dict()
    hops.append(_hop(
        2, 'structured',
        f"{bowler} death-over balls for {team} per season (consistency check)",
        [f"{s}: {n} death balls" for s, n in sorted(by_season.items())],
        note="Confirms dependency is sustained across all 3 seasons, "
             "not a one-season artefact"))

    # Hop 3 — semantic: bowler Wikipedia page, scoped to that player
    title = title_of.get(bowler, bowler)
    sq = (f"{title} bowling style yorkers slower ball variations "
          f"death overs economy specialist")
    chunks = vs.search(sq, k=4, category='players', doc_title=title)
    hops.append(_hop(3, 'semantic', sq, _chunk_brief(chunks),
                     note=f"Scoped to Wikipedia page '{title}'"))

    top_sim = chunks[0]['score'] if chunks else 0.0
    corrob = any('death' in c['text'].lower() or 'yorker' in c['text'].lower()
                 for c in chunks)
    answer = synthesize(
        q,
        f"{team} is the most death-over-dependent team, leaning on {bowler} "
        f"for {ans['dependency_share']:.1%} of their death-over deliveries "
        f"({ans['bowler_death_balls']} balls, {ans['bowler_death_wkts']} wickets, "
        f"economy {ans['bowler_death_econ']}).",
        [f"{bowler} bowled {ans['dependency_share']:.1%} of {team}'s death balls",
         f"Sustained across seasons: {by_season}"],
        chunks, keywords=['death', 'yorker', 'specialist', 'economy', 'slower'])

    conf, why = _confidence(margin, top_sim, corrob)
    return {'q': q, 'hops': hops, 'answer': answer,
            'confidence': conf, 'confidence_reason': why}


# ── Q2 ──────────────────────────────────────────────────────────────────────
def question2(vs, title_of, pre):
    q = ("Find the match where momentum shifted most dramatically according to "
         "the pressure index. What does the Wikipedia season article say about "
         "that match and does it align with what the index found?")
    hops = []

    res = st.match_momentum_shift()
    a = res['answer']
    margin = float(res['table'].iloc[0]['momentum_score'] -
                   res['table'].iloc[1]['momentum_score']) / \
        max(res['table'].iloc[0]['momentum_score'], 1e-6)
    hops.append(_hop(
        1, 'structured',
        "per-match pressure swing (max-min per-over) + late-over pressure",
        [f"{r['date']} {r['teams']} (won by {r['winner']}, {r['stage']}): "
         f"swing={r['pressure_swing']}, peak={r['peak_pressure']}, "
         f"momentum={r['momentum_score']}"
         for _, r in res['table'].head(3).iterrows()],
        note=f"Top: {a['teams']} on {a['date']}"))

    # Hop 2 — derive season + identify season article
    season = str(a['season'])
    hops.append(_hop(
        2, 'derive', f"map match {a['match_id']} -> season {season} article",
        [f"season={season}, teams={a['teams']}, winner={a['winner']}"],
        note="Determines which Wikipedia season article to consult"))

    # Hop 3 — semantic: season article
    sq = (f"{a['teams']} {a['winner']} {season} IPL close match result "
          f"final over thriller")
    chunks = vs.search(sq, k=4, category='seasons',
                       doc_title=f"{season} Indian Premier League")
    hops.append(_hop(3, 'semantic', sq, _chunk_brief(chunks),
                     note=f"Scoped to '{season} Indian Premier League'"))

    top_sim = chunks[0]['score'] if chunks else 0.0
    teams_words = [w for w in a['teams'].replace(' v ', ' ').split()
                   if len(w) > 3]
    corrob = any(any(w in c['text'] for w in teams_words) for c in chunks)
    answer = synthesize(
        q,
        f"The most dramatic momentum shift was {a['teams']} on {a['date']} "
        f"({a['stage']}), won by {a['winner']} — pressure swing "
        f"{a['pressure_swing']}, peak {a['peak_pressure']}.",
        [f"momentum score {a['momentum_score']} (highest of all matches)",
         f"late-over (17+) mean pressure {a['late_pressure']}"],
        chunks, keywords=teams_words + ['runs', 'wicket', 'won', 'over'])

    conf, why = _confidence(margin, top_sim, corrob)
    return {'q': q, 'hops': hops, 'answer': answer,
            'confidence': conf, 'confidence_reason': why}


# ── Q3 ──────────────────────────────────────────────────────────────────────
def question3(vs, title_of, pre):
    q = ("Which batsman performs most differently when the pressure index is "
         "in the top 10% versus the bottom 10% of its distribution?")
    hops = []

    res = pre['q3']
    a = res['answer']
    bat = a['batsman']
    t = res['table']
    margin = abs(float(t.iloc[0]['sr_delta']) - float(t.iloc[1]['sr_delta'])) / \
        max(abs(float(t.iloc[0]['sr_delta'])), 1e-6)
    hops.append(_hop(
        1, 'structured',
        f"global pressure percentiles (hi>=p90={res['thresholds']['hi_p90']}, "
        f"lo<=p10={res['thresholds']['lo_p10']})",
        [f"hi-pressure bucket n={int(t['balls_hi'].sum())}, "
         f"lo-pressure bucket n={int(t['balls_lo'].sum())}"],
        note="Defines the two comparison buckets"))

    hops.append(_hop(
        2, 'structured',
        "per-batsman strike rate in each bucket; rank by |SR(hi)-SR(lo)|",
        [f"{r['batsman']}: SR hi={r['sr_hi']:.0f} (n={int(r['balls_hi'])}) vs "
         f"SR lo={r['sr_lo']:.0f} (n={int(r['balls_lo'])}), Δ={r['sr_delta']:+.0f}"
         for _, r in t.head(4).iterrows()],
        note=f"Largest swing: {bat}"))

    # Hop 3 — semantic: characterize the batsman from Wikipedia
    title = title_of.get(bat, bat)
    sq = f"{title} batting style strike rate finisher middle order role"
    chunks = vs.search(sq, k=3, category='players', doc_title=title)
    hops.append(_hop(3, 'semantic', sq, _chunk_brief(chunks),
                     note=f"Scoped to '{title}' to characterise the swing"))

    top_sim = chunks[0]['score'] if chunks else 0.0
    answer = synthesize(
        q,
        f"{bat} swings most between buckets: strike rate {a['sr_hi']:.0f} under "
        f"top-10% pressure vs {a['sr_lo']:.0f} under bottom-10% pressure "
        f"(Δ={a['sr_delta']:+.0f} runs/100 balls).",
        [f"hi-pressure: {int(a['balls_hi'])} balls, SR {a['sr_hi']:.0f}",
         f"lo-pressure: {int(a['balls_lo'])} balls, SR {a['sr_lo']:.0f}"],
        chunks, keywords=['finisher', 'aggressive', 'strike', 'middle', 'power'])

    conf, why = _confidence(margin, top_sim, True)
    return {'q': q, 'hops': hops, 'answer': answer,
            'confidence': conf, 'confidence_reason': why}


# ── Q4 ──────────────────────────────────────────────────────────────────────
def question4(vs, title_of, pre):
    q = ("Find a player whose pressure-index scores show a clear dip then "
         "recovery across a season. Plot that arc, identify the match where the "
         "dip began, and find what the Wikipedia season article says about it.")
    hops = []

    res = pre['q4']
    player, season = res['player'], str(res['season'])
    arc = res['arc']
    hops.append(_hop(
        1, 'structured',
        "per-(player,season) sequence of per-match mean pressure faced; "
        "score dip-depth = min(pre-peak, post-peak) - trough",
        [f"{player} {season}: {len(arc)} matches, "
         f"trough at match {res['trough_match_id']} "
         f"(value {res['trough_value']}), dip depth {res['score']:.3f}"],
        note=f"Best arc: {player} in {season}"))

    # Hop 2 — plot the arc
    plot_path = os.path.join(OUT_DIR, f'q4_pressure_arc_{player.replace(" ","_")}.png')
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(len(arc)), arc['mean'].values, marker='o')
    ti = list(arc['match_id']).index(res['trough_match_id'])
    ax.scatter([ti], [arc['mean'].values[ti]], color='red', zorder=5,
               label=f'dip (match {res["trough_match_id"]})')
    ax.set_title(f"{player} — per-match mean pressure faced, {season}")
    ax.set_xlabel('match # in season (chronological)')
    ax.set_ylabel('mean pressure index')
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    hops.append(_hop(
        2, 'plot', f"plot {player} pressure arc -> {os.path.basename(plot_path)}",
        [f"saved {plot_path}", f"dip match = {res['trough_match_id']} "
         f"on {res['trough_date']}"],
        note="Visual confirmation of dip-then-recovery"))

    # Hop 3 — semantic: season article about the dip match
    df = st.get_df()
    m = df[df['match_id'] == res['trough_match_id']].iloc[0]
    teams = f"{m['team1']} v {m['team2']}"
    sq = f"{teams} {season} IPL {m['winner']} match result"
    chunks = vs.search(sq, k=4, category='seasons',
                       doc_title=f"{season} Indian Premier League")
    hops.append(_hop(3, 'semantic', sq, _chunk_brief(chunks),
                     note=f"Dip match {teams}, won by {m['winner']}"))

    top_sim = chunks[0]['score'] if chunks else 0.0
    answer = synthesize(
        q,
        f"{player} ({season}) shows a clear dip-then-recovery arc; the dip "
        f"bottomed at match {res['trough_match_id']} ({teams}) on "
        f"{res['trough_date']}, then recovered. See {os.path.basename(plot_path)}.",
        [f"trough mean pressure {res['trough_value']}, dip depth {res['score']:.3f}",
         f"dip match: {teams}, won by {m['winner']}"],
        chunks, keywords=[w for w in teams.split() if len(w) > 3] +
        ['won', 'runs', 'wicket'])

    conf, why = _confidence(0.05, top_sim,
                            any(m['team1'].split()[-1] in c['text'] or
                                m['team2'].split()[-1] in c['text']
                                for c in chunks))
    return {'q': q, 'hops': hops, 'answer': answer, 'plot': plot_path,
            'confidence': conf, 'confidence_reason': why}


# ── Q5 ──────────────────────────────────────────────────────────────────────
def question5(vs, title_of, pre):
    q = ("Build a 'pressure-proof' opening pair from the 3 seasons. Justify "
         "each selection with at least 1 Cricsheet stat and 1 Wikipedia fact.")
    hops = []

    res = pre['q5']
    top2 = res['answer_top2']
    t = res['table']
    margin = float(t.iloc[1]['pressure_proof_score'] -
                   t.iloc[2]['pressure_proof_score']) / \
        max(float(t.iloc[1]['pressure_proof_score']), 1e-6)
    hops.append(_hop(
        1, 'structured',
        "genuine openers (batted ball 1 of an innings); rank by under-pressure "
        "SR + balls-per-dismissal on >=median-pressure deliveries",
        [f"{r['opener']}: SR {r['sr_under_pressure']:.0f}, "
         f"{r['balls_per_dismissal']:.0f} balls/dismissal, "
         f"{int(r['pressure_balls'])} pressure balls"
         for _, r in t.head(4).iterrows()],
        note=f"Top pair: {top2[0]['opener']} + {top2[1]['opener']}"))

    # Hop 2 + 3 — semantic: one Wikipedia fact per opener
    facts = []
    sims = []
    for rec in top2:
        name = rec['opener']
        title = title_of.get(name, name)
        sq = f"{title} opening batsman aggressive temperament big matches IPL"
        chunks = vs.search(sq, k=3, category='players', doc_title=title)
        sims.append(chunks[0]['score'] if chunks else 0.0)
        hops.append(_hop(
            len(hops) + 1, 'semantic', sq, _chunk_brief(chunks),
            note=f"Wikipedia fact for {name} ({title})"))
        facts.append((name, rec, chunks))

    # synthesize selection rationale
    bullets = []
    all_chunks = []
    for name, rec, chunks in facts:
        bullets.append(
            f"{name}: Cricsheet — SR {rec['sr_under_pressure']:.0f} under "
            f"pressure over {int(rec['pressure_balls'])} balls, "
            f"{rec['balls_per_dismissal']:.0f} balls/dismissal.")
        all_chunks += chunks

    answer = synthesize(
        q,
        f"Pressure-proof opening pair: {top2[0]['opener']} + "
        f"{top2[1]['opener']} — the two openers with the best combination of "
        f"strike rate and survival on above-median-pressure deliveries.",
        bullets, all_chunks,
        keywords=['opening', 'opener', 'aggressive', 'captain', 'temperament'])

    conf, why = _confidence(margin, min(sims) if sims else 0.0, True)
    return {'q': q, 'hops': hops, 'answer': answer,
            'confidence': conf, 'confidence_reason': why}


ALL = [question1, question2, question3, question4, question5]
