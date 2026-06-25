import os
import pandas as pd

def parse_cricsheet_file(filepath):
    meta = {}
    balls = []
    players = {}  # team -> list of players
    umpires = []

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')

            if parts[0] == 'info':
                key = parts[1]
                if key == 'team':
                    meta.setdefault('teams', []).append(parts[2])
                elif key == 'player':
                    team = parts[2]
                    player = parts[3]
                    players.setdefault(team, []).append(player)
                elif key == 'umpire':
                    umpires.append(parts[2])
                elif key in ('target_runs', 'target_overs'):
                    meta[f'{key}_inn{parts[2]}'] = int(parts[3]) if parts[3] else None
                elif key != 'registry':
                    meta[key] = parts[2]

            elif parts[0] == 'ball':
                over_ball = parts[2]
                over = int(over_ball.split('.')[0])

                # Squad lists per team
                teams = meta.get('teams', [])
                squad_team1 = players.get(teams[0], []) if len(teams) > 0 else []
                squad_team2 = players.get(teams[1], []) if len(teams) > 1 else []

                ball = {
                    # Match metadata
                    'match_id':           meta.get('match_id'),
                    'season':             meta.get('season'),
                    'date':               meta.get('date'),
                    'venue':              meta.get('venue'),
                    'city':               meta.get('city'),
                    'winner':             meta.get('winner'),
                    'winner_runs':        meta.get('winner_runs'),
                    'winner_wickets':     meta.get('winner_wickets'),
                    'toss_winner':        meta.get('toss_winner'),
                    'toss_decision':      meta.get('toss_decision'),
                    'player_of_match':    meta.get('player_of_match'),
                    'target':             meta.get('target_runs_inn2'),

                    # Teams
                    'team1':              teams[0] if len(teams) > 0 else None,
                    'team2':              teams[1] if len(teams) > 1 else None,

                    # Squads
                    'squad_team1':        '|'.join(squad_team1),
                    'squad_team2':        '|'.join(squad_team2),

                    # Umpires
                    'umpire1':            umpires[0] if len(umpires) > 0 else None,
                    'umpire2':            umpires[1] if len(umpires) > 1 else None,

                    # Ball details
                    'innings':            int(parts[1]),
                    'over':               over,
                    'over_ball':          over_ball,
                    'batting_team':       parts[3],
                    'striker':            parts[4],
                    'non_striker':        parts[5],
                    'bowler':             parts[6],
                    'runs_off_bat':       int(parts[7]),
                    'extras':             int(parts[8]),
                    'wides':              int(parts[9])  if parts[9]  else 0,
                    'noballs':            int(parts[10]) if parts[10] else 0,
                    'byes':               int(parts[11]) if parts[11] else 0,
                    'legbyes':            int(parts[12]) if parts[12] else 0,
                    'wicket_type':        parts[14] if parts[14] not in ('""', '') else None,
                    'player_dismissed':   parts[15] if parts[15] not in ('""', '') else None,
                }

                ball['total_runs'] = ball['runs_off_bat'] + ball['extras']
                ball['is_wicket']  = 1 if ball['wicket_type'] else 0
                ball['is_wide']    = 1 if ball['wides'] > 0 else 0
                ball['is_noball']  = 1 if ball['noballs'] > 0 else 0

                balls.append(ball)

    return meta.get('season'), balls


def load_three_seasons(data_dir, seasons={'2024', '2025', '2026'}):
    all_balls = []
    matched_files = 0

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.csv'):
            continue

        filepath = os.path.join(data_dir, fname)

        # Quick peek at season
        season = None
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('info,season'):
                    season = line.strip().split(',')[2]
                    break

        if season not in seasons:
            continue

        _, balls = parse_cricsheet_file(filepath)
        all_balls.extend(balls)
        matched_files += 1

    df = pd.DataFrame(all_balls)
    print(f"Found {matched_files} matches across seasons: {df['season'].unique()}")
    print(f"Total deliveries: {len(df)}")
    print(f"Matches per season:\n{df.groupby('season')['match_id'].nunique()}")
    return df


# ---- RUN ----
ROOT = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(ROOT, 'ipl_csv')
df = load_three_seasons(data_dir)

# ---- SAVE ----
output_path = os.path.join(ROOT, 'data/ipl_2024_25_26.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")
print(f"Columns: {df.columns.tolist()}")