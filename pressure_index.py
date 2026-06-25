import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


# ── 1. RRR/CRR RATIO → PRESSURE MAPPING ─────────────────────────────────────
def rrr_ratio_to_pressure(rrr: float, crr: float) -> float:
    """
    Maps the chase situation to a batsman pressure score using the RATIO of
    Required Run Rate to Current Run Rate — not raw RRR.

    Why the ratio (improvement over raw RRR):
        Raw RRR = 10 tells you the task is hard in absolute terms, but not
        relative to how the batsmen are already going.
            RRR 10, CRR 9  -> ratio 1.11  (slightly behind, manageable)
            RRR 10, CRR 5  -> ratio 2.00  (scoring half the needed rate, deep trouble)
        Same RRR, very different pressure. The ratio captures "how much harder
        than current pace must I now go".

    Non-linearity preserved:
        RRR > 15 -> mentally game over, pressure DROPS (team has given up).
    """
    # Impossibility guard first — a hopeless chase is low pressure
    if rrr > 15:
        return 0.3

    crr = max(crr, 2.0)            # floor avoids div-by-near-zero blow-ups
    ratio = rrr / crr

    if ratio < 0.8:    return 0.4   # well ahead of the rate, cruising
    elif ratio < 1.0:  return 0.7   # slightly ahead, comfortable
    elif ratio < 1.25: return 1.0   # neck and neck
    elif ratio < 1.5:  return 1.3   # falling behind, pressure building
    elif ratio < 2.0:  return 1.5   # well behind but still fighting — PEAK
    else:              return 1.4   # very behind (yet RRR<=15, still gettable)


# ── 2. IBF: BATSMAN SETTEDNESS ──────────────────────────────────────────────
def settedness_factor(balls_faced: int) -> float:
    """
    Pressure from how 'set' the striker is, using balls they've already faced
    in this innings (IBF — innings balls faced).

    Rationale: a brand-new batsman hasn't read the pitch or pace and is most
    vulnerable; a set batsman who has faced 15+ balls is in rhythm and under
    less pressure facing the same delivery.
        IBF 0     -> 1.20  (just walked in, highest vulnerability)
        IBF 1-5   -> 1.10  (still settling)
        IBF 6-15  -> 1.00  (neutral)
        IBF 15+   -> 0.90  (set, in control)
    """
    if balls_faced == 0:    return 1.20
    elif balls_faced <= 5:  return 1.10
    elif balls_faced <= 15: return 1.00
    else:                   return 0.90


# ── 3. RUNS-NEEDED FACTOR (replaces blunt high-target bonus) ─────────────────
def runs_needed_factor(runs_needed: float) -> float:
    """
    Absolute size of the remaining task. Distinct from RRR/ratio: a team
    needing 100 off 60 and 100 off 12 share the runs_needed but have very
    different RRR — and a larger absolute gap is its own source of pressure.
    Smoothly scales and caps so it amplifies but never dominates.
        10 needed  -> 1.05
        50 needed  -> 1.25
        100+ needed-> 1.30 (capped)
    """
    if runs_needed <= 0:
        return 1.0
    return min(1.30, 1.0 + runs_needed / 200.0)


# ── 4. PHASE LABEL + WEIGHT ──────────────────────────────────────────────────
def get_phase(over: int):
    """
    Returns (phase_label, base_phase_weight) for a given over.

    Innings 2: weights applied fully — every over counts toward target.
    Innings 1: weights applied conditionally — see compute function.
    """
    if over <= 5:
        return 'powerplay', 1.0
    elif over <= 14:
        return 'middle', 1.2
    else:
        return 'death', 1.5


# ── 5. VENUE PAR SCORE ───────────────────────────────────────────────────────
def compute_venue_par(df: pd.DataFrame) -> dict:
    """
    Average first-innings total per venue across 3 seasons. Used as 'par' for
    innings-1 pressure: 80/4 in over 12 at Wankhede (par ~175) is more
    pressured than the same score at a low-scoring ground.
    """
    venue_par = (
        df[df['innings'] == 1]
        .groupby(['match_id', 'venue'])['total_runs']
        .sum()
        .reset_index()
        .groupby('venue')['total_runs']
        .mean()
        .to_dict()
    )
    return venue_par


# ── 6. MATCH IMPORTANCE LOADER ──────────────────────────────────────────────
def load_match_importance(path: str) -> dict:
    """
    Loads {match_id(str): importance_multiplier} produced by
    wikipedia_fetcher.build_match_importance(). league 1.0 / playoff 1.15 /
    final 1.30. Missing file → all matches treated as league (1.0).
    """
    if not os.path.exists(path):
        print(f"  [WARN] match importance file missing ({path}); "
              f"defaulting all matches to league (1.0)")
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {k: v['importance'] for k, v in raw.items()}


# ── 7. INNINGS 1 HELPER: CONDITIONAL WEIGHTS ────────────────────────────────
def innings1_conditional_weights(
    below_par: bool,
    wickets_so_far: int,
    base_phase_weight: float,
    dot_streak: bool
):
    """
    In innings 1 there is no target, so wicket / phase / dot-streak pressure
    only becomes meaningful when BAD THINGS HAPPEN TOGETHER — low score AND
    wickets falling. Each variable is weakened unless it co-occurs.

      High wickets alone     -> manageable (can still score)
      Low score alone        -> manageable (wickets in hand)
      Low score + wickets    -> real pressure, variables amplify
      Low score + dots + wkts-> peak innings-1 pressure
    """
    # Conditional wicket factor
    if wickets_so_far >= 5 and below_par:
        wicket_factor = 1.0 + (wickets_so_far / 10.0)   # full: up to 1.5
    elif wickets_so_far >= 3 and below_par:
        wicket_factor = 1.0 + (wickets_so_far / 20.0)   # half
    else:
        wicket_factor = 1.0 + (wickets_so_far / 30.0)   # minimal

    # Conditional phase weight
    if below_par and wickets_so_far >= 4:
        phase_weight = base_phase_weight
    elif below_par:
        phase_weight = 1.0 + (base_phase_weight - 1.0) * 0.5
    else:
        phase_weight = 1.0

    # Conditional dot-streak bonus
    if dot_streak and below_par and wickets_so_far >= 3:
        dot_streak_bonus = 1.15
    else:
        dot_streak_bonus = 1.0

    return wicket_factor, phase_weight, dot_streak_bonus


# ── 8. MAIN PRESSURE INDEX COMPUTATION ──────────────────────────────────────
def compute_pressure_index(
    df: pd.DataFrame,
    importance_path: str = os.path.join(ROOT, 'data/outputs/match_importance.json')
) -> pd.DataFrame:
    """
    Batsman Pressure Index for every delivery:
        How much pressure is THIS BATSMAN under on THIS delivery?

    ── INNINGS 2 FORMULA ──────────────────────────────────────────────
    PI = rrr_ratio_pressure   [V1 — scoring urgency, relative to current pace]
       × wicket_factor        [V2 — batting resources remaining]
       × phase_weight         [V3 — delivery criticality]
       × recent_wicket_bonus  [V4 — new batsman / momentum shift]
       × dot_streak_bonus     [V5 — batsman strangled]
       × settedness_factor    [V6 — how 'in' the striker is (IBF)]
       × runs_needed_factor   [V7 — absolute size of the task]
       × match_importance     [V8 — league / playoff / final stakes]

    ── INNINGS 1 FORMULA ──────────────────────────────────────────────
    PI = below_par_factor          [V1b — venue-relative urgency]
       × conditional_wicket_factor [V2 — only bites when score also low]
       × conditional_phase_weight  [V3 — only bites when already in trouble]
       × recent_wicket_bonus       [V4 — reduced base]
       × conditional_dot_streak    [V5 — only when already in trouble]
       × settedness_factor         [V6 — new batsman vulnerable regardless]
       × match_importance          [V8 — stakes apply to both innings]
    """
    df = df.copy()
    df = df.sort_values(
        ['match_id', 'innings', 'over', 'over_ball']
    ).reset_index(drop=True)

    # Phase
    phases             = df['over'].apply(get_phase)
    df['phase']        = phases.apply(lambda x: x[0])
    df['phase_weight'] = phases.apply(lambda x: x[1])

    venue_par  = compute_venue_par(df)
    importance = load_match_importance(importance_path)

    pressure_values = [None] * len(df)

    for (match_id, innings), grp_idx in df.groupby(
        ['match_id', 'innings']
    ).groups.items():

        grp = df.loc[grp_idx].sort_values(['over', 'over_ball'])

        target  = grp['target'].iloc[0]
        venue   = grp['venue'].iloc[0]
        par     = venue_par.get(venue, 160)
        mi_mult = importance.get(str(match_id), 1.0)   # V8 — per match

        # Running state
        runs_so_far    = 0
        wickets_so_far = 0
        legal_balls    = 0
        history        = []
        striker_balls  = {}   # IBF: legal balls faced per striker this innings

        for idx, row in grp.iterrows():

            balls_remaining   = max(1, 120 - legal_balls)
            base_phase_weight = row['phase_weight']
            striker           = row['striker']
            ibf               = striker_balls.get(striker, 0)   # before this ball

            # ── V6: SETTEDNESS (both innings) ────────────────────────────────
            setted = settedness_factor(ibf)

            # ── V1 / V7: CHASE VARIABLES (innings 2) ─────────────────────────
            if innings == 2 and pd.notna(target):
                runs_needed = float(target) - runs_so_far
                rrr  = (runs_needed / balls_remaining) * 6
                crr  = (runs_so_far / max(legal_balls, 1)) * 6
                rrr_p     = rrr_ratio_to_pressure(rrr, crr)
                rn_factor = runs_needed_factor(runs_needed)
            else:
                rrr_p     = 1.0
                rn_factor = 1.0

            # ── V4: RECENT WICKET (both innings, lighter in innings 1) ───────
            recent_wkt = any(h['is_wicket'] for h in history[-2:])
            if innings == 2:
                recent_wicket_bonus = 1.25 if recent_wkt else 1.0
            else:
                recent_wicket_bonus = 1.10 if recent_wkt else 1.0

            # ── V5: DOT STREAK flag ──────────────────────────────────────────
            dot_streak = (
                len(history) >= 3
                and all(h['runs'] == 0 for h in history[-3:])
            )

            # ── INNINGS-SPECIFIC ─────────────────────────────────────────────
            if innings == 2:
                wicket_factor    = 1.0 + (wickets_so_far / 10.0)   # V2 full
                phase_weight     = base_phase_weight               # V3 full
                dot_streak_bonus = 1.15 if dot_streak else 1.0     # V5 full
                below_par_factor = 1.0
            else:
                expected_runs = (par / 20.0) * (row['over'] + 1)
                below_par     = runs_so_far < expected_runs
                below_par_factor = min(1.6, 1.0 + max(
                    0.0,
                    (expected_runs - runs_so_far) / max(expected_runs, 1.0)
                ))
                wicket_factor, phase_weight, dot_streak_bonus = (
                    innings1_conditional_weights(
                        below_par, wickets_so_far,
                        base_phase_weight, dot_streak
                    )
                )

            # ── FINAL PI ─────────────────────────────────────────────────────
            if innings == 2:
                pi = (rrr_p
                      * wicket_factor
                      * phase_weight
                      * recent_wicket_bonus
                      * dot_streak_bonus
                      * setted
                      * rn_factor
                      * mi_mult)
            else:
                pi = (below_par_factor
                      * wicket_factor
                      * phase_weight
                      * recent_wicket_bonus
                      * dot_streak_bonus
                      * setted
                      * mi_mult)

            pressure_values[idx] = pi

            # ── UPDATE RUNNING STATE ─────────────────────────────────────────
            runs_so_far += row['total_runs']
            if row['is_wicket']:
                wickets_so_far += 1

            is_legal = (row['wides'] == 0 and row['noballs'] == 0)
            if is_legal:
                legal_balls += 1
                striker_balls[striker] = ibf + 1   # striker faced a legal ball

            history.append({
                'is_wicket': bool(row['is_wicket']),
                'runs':      int(row['runs_off_bat'] + row['extras'])
            })

    df['pressure_index'] = pressure_values
    # Attach stage label for inspection / downstream RAG
    stage_map = {}
    if os.path.exists(importance_path):
        with open(importance_path) as f:
            stage_map = {k: v['stage'] for k, v in json.load(f).items()}
    df['match_stage'] = df['match_id'].astype(str).map(stage_map).fillna('league')
    return df


# ── 9. VALIDATION: TOP 5 DELIVERIES ─────────────────────────────────────────
def validate_top5(df: pd.DataFrame):
    cols = [
        'match_id', 'date', 'innings', 'over', 'over_ball', 'match_stage',
        'batting_team', 'striker', 'bowler',
        'runs_off_bat', 'wicket_type', 'player_dismissed',
        'target', 'phase', 'pressure_index'
    ]
    top5 = df.nlargest(5, 'pressure_index')[cols]
    print("\n=== TOP 5 HIGHEST PRESSURE DELIVERIES (BATSMAN POV) ===")
    print(top5.to_string(index=False))
    return top5


# ── 10. PHASE DISTRIBUTION ───────────────────────────────────────────────────
def plot_phase_distribution(df: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'Batsman Pressure Index — Distribution Across Match Phases',
        fontsize=14
    )

    phase_order = ['powerplay', 'middle', 'death']
    phase_data  = [
        df[df['phase'] == p]['pressure_index'].dropna()
        for p in phase_order
    ]

    axes[0].boxplot(
        phase_data, tick_labels=phase_order, patch_artist=True,
        boxprops=dict(facecolor='steelblue', alpha=0.6)
    )
    axes[0].set_title('Distribution by Phase (Box Plot)')
    axes[0].set_ylabel('Pressure Index')
    axes[0].set_xlabel('Match Phase')

    means = (
        df.groupby('phase')['pressure_index']
        .mean()
        .reindex(phase_order)
    )
    axes[1].bar(phase_order, means.values, color='steelblue', alpha=0.7)
    axes[1].set_title('Mean Pressure Index by Phase')
    axes[1].set_ylabel('Mean Pressure Index')
    axes[1].set_xlabel('Match Phase')
    for i, v in enumerate(means.values):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'pressure_phase_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to: {path}")

    print("\n=== PHASE STATISTICS ===")
    print(df.groupby('phase')['pressure_index'].describe().round(3))


# ── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    csv_path        = os.path.join(ROOT, 'data/ipl_2024_25_26.csv')
    output_dir      = os.path.join(ROOT, 'data/outputs')
    importance_path = os.path.join(output_dir, 'match_importance.json')

    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} deliveries")

    print("\nComputing Batsman Pressure Index...")
    df = compute_pressure_index(df, importance_path)
    print("Done.")

    validate_top5(df)
    plot_phase_distribution(df, output_dir)

    out_path = os.path.join(output_dir, 'ipl_with_pressure_index.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved enriched data to: {out_path}")
