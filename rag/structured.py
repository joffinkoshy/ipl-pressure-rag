"""
Structured-data tools = the Cricsheet hops of the multi-hop pipeline.

Each function answers ONE structured sub-question against the pressure-enriched
ball-by-ball table and returns a dict carrying both the answer AND the numbers
that justify it, so every hop is traceable (right vs lucky).

Input table: data/outputs/ipl_with_pressure_index.csv
  key cols: match_id, season, date, innings, over, batting_team, team1, team2,
            striker, non_striker, bowler, runs_off_bat, total_runs, is_wicket,
            wides, noballs, phase, pressure_index, match_stage, target
"""

import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, 'data/outputs/ipl_with_pressure_index.csv')

_DF = None


def get_df(path=CSV):
    """Load + cache the enriched table, deriving bowling_team and legal-ball flag."""
    global _DF
    if _DF is None:
        df = pd.read_csv(path)
        df['bowling_team'] = np.where(
            df['batting_team'] == df['team1'], df['team2'], df['team1']
        )
        df['is_legal'] = ((df['wides'] == 0) & (df['noballs'] == 0)).astype(int)
        _DF = df
    return _DF


# ── Q1: death-over single-bowler dependency ─────────────────────────────────
def death_over_bowler_dependency(min_team_balls=200):
    """
    For every bowling team, find the bowler who bowled the largest share of the
    team's DEATH-over (16-20) legal deliveries across the 3 seasons. The team
    with the highest single-bowler share is the most 'dependent'.

    Returns the ranked table + the top team/bowler.
    """
    df = get_df()
    death = df[(df['phase'] == 'death') & (df['is_legal'] == 1)]

    rows = []
    for team, g in death.groupby('bowling_team'):
        total = len(g)
        if total < min_team_balls:
            continue
        by_bowler = g.groupby('bowler')
        balls = by_bowler.size()
        top_bowler = balls.idxmax()
        top_balls  = int(balls.max())
        wkts = int(by_bowler.get_group(top_bowler)['is_wicket'].sum())
        runs = int(by_bowler.get_group(top_bowler)['total_runs'].sum())
        econ = round(runs / (top_balls / 6), 2) if top_balls else None
        rows.append({
            'team': team,
            'top_death_bowler': top_bowler,
            'bowler_death_balls': top_balls,
            'team_death_balls': total,
            'dependency_share': round(top_balls / total, 3),
            'bowler_death_wkts': wkts,
            'bowler_death_econ': econ,
        })

    table = pd.DataFrame(rows).sort_values('dependency_share', ascending=False)
    top = table.iloc[0].to_dict()
    return {'answer': top, 'table': table}


# ── Q2: most dramatic momentum shift ────────────────────────────────────────
def match_momentum_shift():
    """
    Define momentum shift per match using the pressure index: the largest swing
    between a smoothed pressure low and a subsequent high within the SAME match
    (pressure surges back and forth = a dramatic, fought match). We smooth over
    a 6-ball rolling window per innings, then take (max - min) of the per-over
    mean pressure within the chase (innings 2), plus the peak pressure reached.

    Returns ranked table + the top match.
    """
    df = get_df()
    rows = []
    for mid, g in df.groupby('match_id'):
        chase = g[g['innings'] == 2]
        if len(chase) < 30:
            continue
        per_over = chase.groupby('over')['pressure_index'].mean()
        swing = float(per_over.max() - per_over.min())
        peak  = float(chase['pressure_index'].max())
        # late drama: pressure still high in the final 3 overs
        late = chase[chase['over'] >= 17]['pressure_index']
        late_mean = float(late.mean()) if len(late) else 0.0
        meta = g.iloc[0]
        rows.append({
            'match_id': mid,
            'season': meta['season'],
            'date': meta['date'],
            'teams': f"{meta['team1']} v {meta['team2']}",
            'winner': meta['winner'],
            'stage': meta['match_stage'],
            'pressure_swing': round(swing, 3),
            'peak_pressure': round(peak, 3),
            'late_pressure': round(late_mean, 3),
            'momentum_score': round(swing + late_mean, 3),
        })
    table = pd.DataFrame(rows).sort_values('momentum_score', ascending=False)
    return {'answer': table.iloc[0].to_dict(), 'table': table}


# ── Q3: batsman top-10% vs bottom-10% pressure performance ──────────────────
def batsman_pressure_buckets(min_balls_each=40):
    """
    Split deliveries into top-10% and bottom-10% pressure by global percentile.
    For each batsman with enough balls in BOTH buckets, compare strike rate.
    Return the batsman whose strike rate changes most (high minus low).
    """
    df = get_df()
    df = df[df['is_legal'] == 1]
    hi_thr = df['pressure_index'].quantile(0.90)
    lo_thr = df['pressure_index'].quantile(0.10)

    hi = df[df['pressure_index'] >= hi_thr]
    lo = df[df['pressure_index'] <= lo_thr]

    def sr(group):
        return group.groupby('striker').agg(
            balls=('runs_off_bat', 'size'),
            runs=('runs_off_bat', 'sum'),
        )

    H, L = sr(hi), sr(lo)
    joined = H.join(L, lsuffix='_hi', rsuffix='_lo', how='inner')
    joined = joined[(joined['balls_hi'] >= min_balls_each) &
                    (joined['balls_lo'] >= min_balls_each)].copy()
    joined['sr_hi'] = 100 * joined['runs_hi'] / joined['balls_hi']
    joined['sr_lo'] = 100 * joined['runs_lo'] / joined['balls_lo']
    joined['sr_delta'] = joined['sr_hi'] - joined['sr_lo']
    joined = joined.sort_values('sr_delta', key=lambda s: s.abs(), ascending=False)

    table = joined.reset_index().rename(columns={'striker': 'batsman'})
    return {
        'answer': table.iloc[0].to_dict(),
        'thresholds': {'hi_p90': round(float(hi_thr), 3),
                       'lo_p10': round(float(lo_thr), 3)},
        'table': table,
    }


# ── Q4: player pressure arc (dip then recovery) across a season ──────────────
def player_pressure_arc(season=None, min_matches=6, min_balls_per_match=8):
    """
    For each (player, season), build the sequence of per-match mean pressure the
    batsman FACED, ordered by date. Score a 'dip then recovery' arc: a clear
    trough with higher values both before and after. Return the best candidate
    plus its arc series so it can be plotted.
    """
    df = get_df()
    df = df[df['is_legal'] == 1]
    if season is not None:
        df = df[df['season'].astype(str) == str(season)]

    best = None
    for (player, ssn), g in df.groupby(['striker', 'season']):
        per_match = (g.groupby(['date', 'match_id'])['pressure_index']
                       .agg(['mean', 'size'])
                       .reset_index()
                       .sort_values('date'))
        per_match = per_match[per_match['size'] >= min_balls_per_match]
        if len(per_match) < min_matches:
            continue
        vals = per_match['mean'].values
        trough_i = int(np.argmin(vals))
        # need room on both sides for a before/after
        if trough_i == 0 or trough_i == len(vals) - 1:
            continue
        before = vals[:trough_i].max()
        after  = vals[trough_i + 1:].max()
        trough = vals[trough_i]
        # dip depth = how far below surrounding peaks; recovery = rebound after
        dip_depth = min(before, after) - trough
        if dip_depth <= 0:
            continue
        score = dip_depth
        if best is None or score > best['score']:
            best = {
                'player': player,
                'season': ssn,
                'score': float(score),
                'trough_match_id': int(per_match.iloc[trough_i]['match_id']),
                'trough_date': str(per_match.iloc[trough_i]['date']),
                'trough_value': round(float(trough), 3),
                'arc': per_match.assign(
                    player=player, season=ssn
                )[['date', 'match_id', 'mean', 'size']],
            }
    return best


# ── Q5: opener pressure-proof stats ─────────────────────────────────────────
def opener_pressure_stats(min_balls=120):
    """
    Identify genuine openers (faced ball in over 0 of an innings) and rank them
    by performance under elevated pressure: strike rate AND average on
    above-median-pressure deliveries, plus dismissal rate. A 'pressure-proof'
    opener scores high while keeping pressure-ball dismissals low.
    """
    df = get_df()
    legal = df[df['is_legal'] == 1]

    # Genuine openers: the two batters on the FIRST delivery of each innings.
    # We record (match_id, innings, player) so a batter only counts in innings
    # where they actually opened — not every innings they ever batted.
    first_ball = (df.sort_values(['over', 'over_ball'])
                    .groupby(['match_id', 'innings']).first().reset_index())
    opener_keys = set()
    for _, r in first_ball.iterrows():
        opener_keys.add((r['match_id'], r['innings'], r['striker']))
        opener_keys.add((r['match_id'], r['innings'], r['non_striker']))

    legal = legal.copy()
    legal['is_opener_ball'] = [
        (m, i, s) in opener_keys
        for m, i, s in zip(legal['match_id'], legal['innings'], legal['striker'])
    ]
    med = legal['pressure_index'].median()
    sub = legal[(legal['is_opener_ball']) &
                (legal['pressure_index'] >= med)]

    g = sub.groupby('striker').agg(
        pressure_balls=('runs_off_bat', 'size'),
        runs=('runs_off_bat', 'sum'),
        dismissals=('is_wicket', 'sum'),
        avg_pressure=('pressure_index', 'mean'),
    )
    g = g[g['pressure_balls'] >= min_balls].copy()
    g['sr_under_pressure'] = 100 * g['runs'] / g['pressure_balls']
    g['balls_per_dismissal'] = g['pressure_balls'] / g['dismissals'].replace(0, np.nan)
    # composite: reward strike rate and survival
    g['pressure_proof_score'] = (
        g['sr_under_pressure']
        + g['balls_per_dismissal'].fillna(g['balls_per_dismissal'].max())
    )
    table = g.sort_values('pressure_proof_score', ascending=False).reset_index()
    table = table.rename(columns={'striker': 'opener'})
    return {'answer_top2': table.head(2).to_dict('records'), 'table': table}


if __name__ == '__main__':
    pd.set_option('display.width', 160)
    pd.set_option('display.max_columns', 20)

    print("── Q1 death-over dependency ──")
    r = death_over_bowler_dependency()
    print(r['table'].head(5).to_string(index=False))

    print("\n── Q2 momentum shift ──")
    r = match_momentum_shift()
    print(r['table'].head(5).to_string(index=False))

    print("\n── Q3 pressure buckets ──")
    r = batsman_pressure_buckets()
    print('thresholds', r['thresholds'])
    print(r['table'][['batsman', 'balls_hi', 'sr_hi', 'balls_lo', 'sr_lo', 'sr_delta']].head(5).to_string(index=False))

    print("\n── Q4 pressure arc ──")
    r = player_pressure_arc()
    print({k: v for k, v in r.items() if k != 'arc'})

    print("\n── Q5 openers ──")
    r = opener_pressure_stats()
    print(pd.DataFrame(r['answer_top2']).to_string(index=False))
