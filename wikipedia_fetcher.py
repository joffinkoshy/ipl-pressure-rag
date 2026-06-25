import wikipediaapi
import json
import os
import time
import pandas as pd

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='IPL-RAG-Assignment/1.0'
)

base_dir   = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'data/wikipedia')
os.makedirs(output_dir, exist_ok=True)


def fetch_and_save(title, category, filename):
    filepath = os.path.join(output_dir, category, f'{filename}.json')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if os.path.exists(filepath):
        print(f"  [SKIP] Already exists: {filename}")
        return True

    page = wiki.page(title)

    if not page.exists():
        print(f"  [NOT FOUND] {title}")
        return False

    data = {
        'title':    page.title,
        'url':      page.fullurl,
        'category': category,
        'summary':  page.summary,
        'text':     page.text,
        'sections': {s.title: s.text for s in page.sections}
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  [OK] {title} ({len(page.text)} chars)")
    time.sleep(0.5)
    return True


# ---- 1. SEASONS ----
def fetch_seasons():
    print("\n=== Fetching IPL Season Articles ===")
    seasons = {
        '2024': '2024 Indian Premier League',
        '2025': '2025 Indian Premier League',
        '2026': '2026 Indian Premier League',
    }
    for year, title in seasons.items():
        fetch_and_save(title, 'seasons', f'ipl_{year}')


# ---- 2. TEAMS ----
def fetch_teams():
    print("\n=== Fetching IPL Team Articles ===")
    teams = {
        'Mumbai_Indians':              'Mumbai Indians',
        'Chennai_Super_Kings':         'Chennai Super Kings',
        'Royal_Challengers_Bengaluru': 'Royal Challengers Bengaluru',
        'Kolkata_Knight_Riders':       'Kolkata Knight Riders',
        'Delhi_Capitals':              'Delhi Capitals',
        'Punjab_Kings':                'Punjab Kings',
        'Rajasthan_Royals':            'Rajasthan Royals',
        'Sunrisers_Hyderabad':         'Sunrisers Hyderabad',
        'Gujarat_Titans':              'Gujarat Titans',
        'Lucknow_Super_Giants':        'Lucknow Super Giants',
    }
    for filename, wiki_title in teams.items():
        fetch_and_save(wiki_title, 'teams', filename)


# ---- 3. PLAYER (fetch on demand later) ----

# Cricsheet uses initials ("JJ Bumrah"); Wikipedia titles are full names.
# Curated overrides for players we actually reference guarantee correctness;
# everything else falls back to the Wikipedia search API.
NAME_OVERRIDES = {
    'JJ Bumrah':   'Jasprit Bumrah',
    'V Kohli':     'Virat Kohli',
    'SV Samson':   'Sanju Samson',
    'H Klaasen':   'Heinrich Klaasen',
    'TH David':    'Tim David',
    'Arshdeep Singh': 'Arshdeep Singh',
    'B Kumar':     'Bhuvneshwar Kumar',
    'YBK Jaiswal': 'Yashasvi Jaiswal',
    'Shubman Gill':'Shubman Gill',
    'B Sai Sudharsan': 'Sai Sudharsan',
    'MR Marsh':    'Mitchell Marsh',
    'RD Gaikwad':  'Ruturaj Gaikwad',
}


def resolve_player_title(short_name):
    """
    Map a Cricsheet short name to a Wikipedia article title.
    Order: curated override -> Wikipedia search API ('<name> cricketer').
    Returns (title, method) or (None, 'unresolved').
    """
    if short_name in NAME_OVERRIDES:
        return NAME_OVERRIDES[short_name], 'override'

    # Use `requests` (bundled with wikipedia-api) — handles TLS verification,
    # unlike raw urllib which fails SSL verification on macOS.
    import requests
    try:
        r = requests.get(
            'https://en.wikipedia.org/w/api.php',
            params={'action': 'query', 'list': 'search',
                    'srsearch': f'{short_name} cricketer',
                    'srlimit': 1, 'format': 'json'},
            headers={'User-Agent': 'IPL-RAG-Assignment/1.0'},
            timeout=10)
        hits = r.json().get('query', {}).get('search', [])
        if hits:
            return hits[0]['title'], 'search'
    except Exception as e:
        print(f"  [resolve error] {short_name}: {e}")
    return None, 'unresolved'


def fetch_player(player_name):
    """
    Resolve a Cricsheet short name to its Wikipedia page and fetch it.
    Returns the resolved title (or None). Idempotent (skips if cached).
    """
    title, method = resolve_player_title(player_name)
    if title is None:
        print(f"  [UNRESOLVED] {player_name}")
        return None
    safe_name = player_name.replace(' ', '_').replace('/', '_')
    ok = fetch_and_save(title, 'players', safe_name)
    if ok:
        print(f"    resolved '{player_name}' -> '{title}' ({method})")
    return title if ok else None


# ---- 4. MATCH IMPORTANCE (for Pressure Index) ----
def build_match_importance(
    csv_path=os.path.join(base_dir, 'data/ipl_2024_25_26.csv'),
    out_path=os.path.join(base_dir, 'data/outputs/match_importance.json'),
):
    """
    Tag every match with a 'stage' and an 'importance' multiplier used by
    the Pressure Index. A delivery in a final carries more pressure than the
    same delivery in a dead-rubber league game.

    Derivation (robust, schedule-based):
      IPL playoffs are always the LAST 4 matches of a season by date —
      Qualifier 1, Eliminator, Qualifier 2, then the Final (the very last).
      Everything before is the league stage.

        league   -> 1.00   (baseline)
        playoff  -> 1.15   (Q1 / Eliminator / Q2 — knockout)
        final    -> 1.30   (title decider)

    Cross-check (traceability): we confirm the two finalists derived from the
    schedule actually appear in that season's Wikipedia article summary. This
    is logged, not enforced — if Wikipedia text shifts, importance still works.
    """
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])

    importance = {}
    print("\n=== Building Match Importance ===")

    for season, g in df.groupby('season'):
        # One row per match, ordered by date
        matches = (g.groupby('match_id')
                     .agg(date=('date', 'first'),
                          team1=('team1', 'first'),
                          team2=('team2', 'first'),
                          winner=('winner', 'first'))
                     .sort_values('date'))

        match_ids = matches.index.tolist()
        playoff_ids = match_ids[-4:]   # last 4 = playoffs
        final_id    = match_ids[-1]    # last = final

        for mid in match_ids:
            if mid == final_id:
                stage, mult = 'final', 1.30
            elif mid in playoff_ids:
                stage, mult = 'playoff', 1.15
            else:
                stage, mult = 'league', 1.00
            importance[str(mid)] = {'stage': stage, 'importance': mult}

        # ---- Wikipedia cross-check on the final ----
        finalists = {matches.loc[final_id, 'team1'],
                     matches.loc[final_id, 'team2']}
        wiki_file = os.path.join(output_dir, 'seasons', f'ipl_{season}.json')
        confirmed = 'no wiki file'
        if os.path.exists(wiki_file):
            with open(wiki_file, encoding='utf-8') as f:
                summary = json.load(f).get('summary', '')
            hits = sum(1 for t in finalists if t.split()[-1] in summary)
            confirmed = f'{hits}/2 finalists found in article'

        print(f"  {season}: final={final_id} "
              f"({' vs '.join(finalists)}) | wiki cross-check: {confirmed}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(importance, f, indent=2)

    n_final   = sum(1 for v in importance.values() if v['stage'] == 'final')
    n_playoff = sum(1 for v in importance.values() if v['stage'] == 'playoff')
    print(f"  Tagged {len(importance)} matches "
          f"({n_final} finals, {n_playoff} playoffs). Saved to {out_path}")
    return importance


# ---- RUN ----
if __name__ == '__main__':
    fetch_seasons()
    fetch_teams()
    build_match_importance()

    print("\n=== Done ===")
    print(f"Saved to: {output_dir}")