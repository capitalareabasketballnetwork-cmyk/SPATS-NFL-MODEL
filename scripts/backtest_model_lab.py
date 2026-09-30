"""Config-driven SPATS Model Lab backtester.

Edit config/model_lab.json to turn stats on/off and change weights.
All rolling features are shifted one game, so a prediction only uses prior games.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parents[1]
CFG=R/"config/model_lab.json"
OUT=R/"reports/model_lab_results.csv"
SUMMARY=R/"reports/model_lab_summary.csv"

ALIASES={
 "point_differential_per_game":["point_differential_per_game","point_differential","margin","point_diff","team_margin"],
 "offensive_epa_per_play":["epa_per_play","total_epa_per_play","off_epa_per_play"],
 "defensive_success_rate":["def_success_rate","success_rate_allowed"],
 "third_down_conversion_rate":["third_down_conversion_rate","third_down_rate","third_down_pct","third_down_conv"],
 "yards_per_play":["yards_per_play","ypp"],
 "turnover_margin_per_game":["turnover_margin","turnover_margin_per_game"],
 "yards_per_game":["yards_per_game","ypg"],
 "time_of_possession_seconds":["time_of_possession_seconds","time_of_possession","top_seconds"],
 "fourth_down_conversion_rate":["fourth_down_conversion_rate","fourth_down_rate","fourth_down_pct","fourth_down_conv"],
 "penalty_yards_per_game":["penalty_yards_per_game","penalty_yards","penalty_yards_per_play"],
 "penalties_per_game":["penalties_per_game","penalties","penalty_rate"],
 "qbr":["qbr","total_qbr"],
 "strength_of_schedule":["sos","strength_of_schedule"],
 "strength_of_record":["sor","strength_of_record"],
}

def find_col(df,key):
    for c in ALIASES.get(key,[key]):
        if c in df.columns:return c
    return None

def zscore_by_game_state(s, season, week):
    # Cross-sectional standardization within each pregame week.
    tmp=pd.DataFrame({"x":s,"season":season,"week":week})
    return tmp.groupby(["season","week"])["x"].transform(
        lambda x:(x-x.mean())/x.std(ddof=0) if x.notna().sum()>1 and x.std(ddof=0)>0 else 0.0)

def main():
    cfg=json.loads(CFG.read_text())
    tg=pd.read_parquet(R/"data/curated/team_game_derived.parquet").copy()
    games=pd.read_parquet(R/"data/curated/games.parquet").copy()

    # Derive team point differential directly from the canonical game scores.
    # This avoids depending on a duplicated score column in team_game_derived.
    home_margin=games[["game_id","home_team","home_margin"]].rename(
        columns={"home_team":"team","home_margin":"point_differential_per_game"})
    away_margin=games[["game_id","away_team","home_margin"]].rename(
        columns={"away_team":"team"})
    away_margin["point_differential_per_game"]=-away_margin["home_margin"]
    away_margin=away_margin[["game_id","team","point_differential_per_game"]]
    margins=pd.concat([home_margin,away_margin],ignore_index=True)
    tg=tg.drop(columns=["point_differential_per_game"],errors="ignore").merge(
        margins,on=["game_id","team"],how="left")

    tg["game_date"]=pd.to_datetime(tg["game_date"],errors="coerce")
    tg=tg.sort_values(["team","game_date","game_id"])

    active={k:v for k,v in cfg["stats"].items() if v.get("enabled") and float(v.get("weight",0))!=0}
    missing=[]
    feature_cols={}
    for key,spec in active.items():
        c=find_col(tg,key)
        if not c:
            missing.append(key); continue
        # Expanding pregame average: every row sees only earlier games.
        pc=f"pregame_{key}"
        tg[pc]=tg.groupby("team")[c].transform(lambda x:x.shift(1).expanding().mean())
        feature_cols[key]=pc
    if missing:
        raise RuntimeError(
            "MODEL LAB STOPPED: enabled stats are missing from team_game_derived.parquet: "
            + ", ".join(missing)
            + ". No partial-model result will be produced."
        )
    if not feature_cols: raise RuntimeError("No enabled stats are available in team_game_derived.parquet")

    usable={k:active[k] for k in feature_cols}
    expected_weight=sum(abs(float(v["weight"])) for v in active.values())
    used_weight=sum(abs(float(usable[k]["weight"])) for k in usable)
    if len(usable) != len(active) or abs(used_weight-expected_weight) > 1e-9:
        raise RuntimeError(f"MODEL LAB STOPPED: expected {len(active)} enabled stats / weight {expected_weight}, got {len(usable)} / {used_weight}.")
    denom=sum(abs(float(v["weight"])) for v in usable.values()) if cfg.get("normalize_weights",True) else 100.0
    tg["rating"]=0.0
    for key,spec in usable.items():
        z=zscore_by_game_state(tg[feature_cols[key]],tg["season"],tg["week"])
        tg["rating"] += z*float(spec["direction"])*float(spec["weight"])/denom

    ratings=tg[["game_id","team","rating"]].copy()
    h=ratings.rename(columns={"team":"home_team","rating":"home_rating"})
    a=ratings.rename(columns={"team":"away_team","rating":"away_rating"})
    d=games.merge(h,on=["game_id","home_team"],how="left").merge(a,on=["game_id","away_team"],how="left")
    start,end=cfg["seasons"]["start"],cfg["seasons"]["end"]
    d=d[(d.season>=start)&(d.season<=end)&d.home_margin.notna()&(d.home_margin!=0)].copy()
    d["model_edge"]=d.home_rating-d.away_rating
    d["pick"]=np.where(d.model_edge>=0,d.home_team,d.away_team)
    d["actual_winner"]=np.where(d.home_margin>0,d.home_team,d.away_team)
    d["correct"]=d.pick.eq(d.actual_winner)
    d["abs_edge"]=d.model_edge.abs()
    cols=["game_id","season","week","away_team","home_team","pick","actual_winner","correct","model_edge","abs_edge"]
    d[cols].to_csv(OUT,index=False)
    by=d.groupby("season").agg(games=("correct","size"),wins=("correct","sum"),accuracy=("correct","mean")).reset_index()
    total=pd.DataFrame([{"season":"TOTAL","games":len(d),"wins":int(d.correct.sum()),"accuracy":d.correct.mean()}])
    pd.concat([by,total],ignore_index=True).to_csv(SUMMARY,index=False)
    print("Model:",cfg["name"])
    print("Used:",", ".join(f"{k}={usable[k]['weight']}" for k in usable))
    print(pd.concat([by,total],ignore_index=True).to_string(index=False))

if __name__=="__main__": main()
