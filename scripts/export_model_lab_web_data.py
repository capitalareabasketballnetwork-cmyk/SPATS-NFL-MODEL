"""Create the compact historical dataset used by the hosted Model Lab."""
from pathlib import Path
import pandas as pd

R=Path(__file__).resolve().parents[1]
CUR=R/"data/curated"
WEB=R/"data/web"
WEB.mkdir(parents=True,exist_ok=True)

tg=pd.read_parquet(CUR/"team_game_derived.parquet")
games=pd.read_parquet(CUR/"games.parquet")

team_cols=[
    "game_id","season","week","game_date","team",
    "epa_per_play","def_success_rate",
    "third_down_conversion_rate","third_down_rate",
    "yards_per_play","turnover_margin_per_game",
    "yards_per_game","time_of_possession_seconds",
    "fourth_down_conversion_rate","fourth_down_rate",
    "penalty_yards_per_game","penalties_per_game",
]
team_cols=[c for c in team_cols if c in tg.columns]
game_cols=[c for c in ["game_id","season","week","game_date","away_team","home_team","home_margin"] if c in games.columns]

tg[team_cols].to_csv(WEB/"model_lab_team_games.csv.gz",index=False,compression="gzip")
games[game_cols].to_csv(WEB/"model_lab_games.csv.gz",index=False,compression="gzip")
print(f"Wrote {len(tg):,} team-game rows and {len(games):,} games to data/web")
