from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.curate import pregame_roll

R=Path(__file__).resolve().parents[1]
FEATURES=[
"total_epa_l8","def_pass_epa_l8","success_rate_ewm","sack_rate_l5",
"cpoe_l8","success_rate_l8","wp_added_l5","def_turnovers_per_drive_ewm",
"def_turnovers_per_drive_l8","explosive_pass_rate_l8","success_rate_l5"]
C=.05

def model():
    return Pipeline([("impute",SimpleImputer(strategy="median")),
                     ("scale",StandardScaler()),
                     ("lr",LogisticRegression(C=C,max_iter=5000))])

def main():
    tg=pd.read_parquet(R/"data/curated/team_game_derived.parquet")
    games=pd.read_parquet(R/"data/curated/games.parquet")
    sched=pd.read_parquet(R/"data/curated/schedules.parquet")
    # Train exactly the selected modern model on completed games through Week 2 2026.
    hist=pre_game_rows=pre_game_rows=None
    # Rebuild pregame rows for all completed team games.
    pg=pregame_roll(tg)
    home=pg.add_prefix("home_"); away=pg.add_prefix("away_")
    train=games.merge(home,left_on=["game_id","home_team"],right_on=["home_game_id","home_team"],how="left")
    train=train.merge(away,left_on=["game_id","away_team"],right_on=["away_game_id","away_team"],how="left")
    diffcols=[]
    for f in FEATURES:
        c=f+"_diff"; train[c]=train["home_"+f]-train["away_"+f]; diffcols.append(c)
    train=train[(train.home_margin.notna())&(train.home_margin!=0)&(train.season>=2015)].copy()
    m=model(); m.fit(train[diffcols],(train.home_margin>0).astype(int))

    # Identify 2026 Week 3 schedule rows and add one dummy row per team so rolling
    # features are calculated using every completed game before Week 3.
    s=sched[(sched.season==2026)&(sched.week==3)].copy()
    datecol="gameday" if "gameday" in s.columns else "game_date"
    s[datecol]=pd.to_datetime(s[datecol],errors="coerce")
    dummies=[]
    metric_cols=[c for c in tg.columns if c not in {"game_id","season","week","game_date","team","opponent","home"}]
    for _,r in s.iterrows():
        gid=str(r["game_id"])
        for team,opp,homeflag in [(r["home_team"],r["away_team"],1),(r["away_team"],r["home_team"],0)]:
            row={c:np.nan for c in tg.columns}
            row.update({"game_id":gid,"season":2026,"week":3,"game_date":r[datecol],
                        "team":team,"opponent":opp,"home":homeflag})
            dummies.append(row)
    aug=pd.concat([tg,pd.DataFrame(dummies)],ignore_index=True,sort=False)
    pga=pregame_roll(aug)
    wk=pga[(pga.season==2026)&(pga.week==3)&(pga.game_id.isin(s.game_id.astype(str)))].copy()
    wh=wk.add_prefix("home_"); wa=wk.add_prefix("away_")
    out=s.merge(wh,left_on=["game_id","home_team"],right_on=["home_game_id","home_team"],how="left")
    out=out.merge(wa,left_on=["game_id","away_team"],right_on=["away_game_id","away_team"],how="left")
    for f in FEATURES: out[f+"_diff"]=out["home_"+f]-out["away_"+f]
    p=m.predict_proba(out[diffcols])[:,1]
    res=pd.DataFrame({"game_id":out.game_id,"away_team":out.away_team,"home_team":out.home_team,
                      "home_win_prob":p,"away_win_prob":1-p,
                      "pick":np.where(p>=.5,out.home_team,out.away_team),
                      "confidence":np.maximum(p,1-p)})
    res.to_csv(R/"reports/week3_2026_predictions.csv",index=False)
    print(res.sort_values("confidence",ascending=False).to_string(index=False))

if __name__=="__main__": main()
