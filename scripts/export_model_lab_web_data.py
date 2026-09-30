"""Create the permanent compact historical dataset used by the hosted Model Lab.

The historical database is downloaded/built only when this export workflow is
explicitly run. Streamlit reads these saved files and never rebuilds nflverse.
"""
from pathlib import Path
import pandas as pd

R=Path(__file__).resolve().parents[1]
CUR=R/"data/curated"
WEB=R/"data/web"
WEB.mkdir(parents=True,exist_ok=True)

tg=pd.read_parquet(CUR/"team_game_derived.parquet")
games=pd.read_parquet(CUR/"games.parquet")

# Keep every usable scalar numeric team-game feature so the website's Add Stat
# dropdown is not limited to a hand-written list. Identifiers stay for joins.
ids=["game_id","season","week","game_date","team"]
numeric=[c for c in tg.columns if pd.api.types.is_numeric_dtype(tg[c])]
team_cols=list(dict.fromkeys([c for c in ids+numeric if c in tg.columns]))

# Keep schedule/result fields needed by Week Explorer, including final scores.
game_candidates=[
    "game_id","season","week","game_date","away_team","home_team",
    "away_score","home_score","home_margin","result"
]
game_cols=[c for c in game_candidates if c in games.columns]

tg[team_cols].to_csv(WEB/"model_lab_team_games.csv.gz",index=False,compression="gzip")
games[game_cols].to_csv(WEB/"model_lab_games.csv.gz",index=False,compression="gzip")
print(f"Wrote {len(tg):,} team-game rows / {len(team_cols)} columns and {len(games):,} games / {len(game_cols)} columns to data/web")
