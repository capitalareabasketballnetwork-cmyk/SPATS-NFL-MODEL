from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

R=Path(__file__).resolve().parents[1]
LIVE=R/"live"; MODEL=LIVE/"model"; CACHE=LIVE/"cache"
FEATURES=["total_epa_l8","def_pass_epa_l8","success_rate_ewm","sack_rate_l5","cpoe_l8",
"success_rate_l8","wp_added_l5","def_turnovers_per_drive_ewm","def_turnovers_per_drive_l8",
"explosive_pass_rate_l8","success_rate_l5"]
C=.05

def main():
    MODEL.mkdir(parents=True,exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)
    d=pd.read_parquet(R/"data/features/matchups.parquet")
    diff=[f+"_diff" for f in FEATURES]
    tr=d[(d.season>=2015)&(d.season<=2025)&d.home_margin.notna()&(d.home_margin!=0)].copy()
    m=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),
                ("lr",LogisticRegression(C=C,max_iter=5000))])
    y=(tr.home_margin>0).astype(int); m.fit(tr[diff],y)
    imp=m.named_steps["impute"]; sc=m.named_steps["scale"]; lr=m.named_steps["lr"]
    obj={"model_version":"SPATS_V1","frozen":True,"trained_through":2025,"C":C,
         "features":FEATURES,"diff_features":diff,
         "imputer_median":dict(zip(diff,map(float,imp.statistics_))),
         "scaler_mean":dict(zip(diff,map(float,sc.mean_))),
         "scaler_scale":dict(zip(diff,map(float,sc.scale_))),
         "coefficient":dict(zip(diff,map(float,lr.coef_[0]))),
         "intercept":float(lr.intercept_[0])}
    (MODEL/"spats_v1.json").write_text(json.dumps(obj,indent=2)+"\n")
    tg=pd.read_parquet(R/"data/curated/team_game_derived.parquet")
    tg=tg[tg.season<=2025].copy().sort_values(["team","game_date","game_id"])
    seed=tg.groupby("team",group_keys=False).tail(64)
    seed.to_csv(CACHE/"team_history_seed.csv",index=False)
    print(f"Frozen SPATS_V1 and {len(seed):,} compact seed rows.")

if __name__=="__main__": main()
