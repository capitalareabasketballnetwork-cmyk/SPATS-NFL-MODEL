"""Download real NFL historical closing odds from nflverse (1999-present).

Usage: python scripts/update_historical_odds.py
Outputs data/web/nfl_historical_odds.csv.gz and reports/odds_coverage.csv.
Missing odds remain missing; no synthetic lines or postgame calculations.
"""
from pathlib import Path
import io
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
COLUMNS = ["game_id","season","week","game_type","gameday","gametime",
           "away_team","home_team","away_moneyline","home_moneyline",
           "spread_line","away_spread_odds","home_spread_odds",
           "total_line","over_odds","under_odds"]
ODDS = COLUMNS[8:]

def main():
    response = requests.get(SOURCE, timeout=120)
    response.raise_for_status()
    data = pd.read_csv(io.BytesIO(response.content), low_memory=False)
    missing = set(COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Source schema changed; missing columns: {sorted(missing)}")
    data = data[COLUMNS].copy()
    data = data.drop_duplicates(subset=["game_id"], keep="last")
    data = data.sort_values(["season","week","game_id"])
    for field in ODDS:
        data[field] = pd.to_numeric(data[field], errors="coerce")
    # Preserve missing fields as null: no invented prices or odds.
    output = ROOT / "data/web/nfl_historical_odds.csv.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False, compression="gzip")
    coverage = data.groupby("season", dropna=False).agg(
        games=("game_id","size"),
        moneyline_games=("home_moneyline","count"),
        spread_games=("spread_line","count"),
        total_games=("total_line","count"),
    ).reset_index()
    report = ROOT / "reports/odds_coverage.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(report,index=False)
    print(f"Saved {len(data)} NFL games to {output}")
    print(coverage.to_string(index=False))
    print("NOTE: These are historical closing-market values; individual sportsbook")
    print("and exact pre-kickoff snapshot timestamps are not verified by this source.")

if __name__ == "__main__":
    main()
