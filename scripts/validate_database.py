from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parents[1]
def main():
    g=pd.read_parquet(R/"data/curated/games.parquet")
    t=pd.read_parquet(R/"data/curated/team_game_derived.parquet")
    m=pd.read_parquet(R/"data/features/matchups.parquet")
    checks={
      "unique_game_ids":g.game_id.is_unique,
      "teams_present":not g[["home_team","away_team"]].isna().any().any(),
      "different_teams":(g.home_team!=g.away_team).all(),
      "unique_matchups":m.game_id.is_unique,
      "pbp_starts_1999":int(g.season.min())<=1999,
      "two_team_game_rows":int(t.groupby("game_id").team.nunique().max())<=2
    }
    out=pd.DataFrame([{"check":k,"passed":bool(v)} for k,v in checks.items()])
    out.to_csv(R/"reports/validation.csv",index=False)
    print(out.to_string(index=False))
    if not out.passed.all(): raise SystemExit("Validation failed")
if __name__=="__main__": main()
