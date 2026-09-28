from pathlib import Path
import io,json,math,requests
import numpy as np
import pandas as pd
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.curate import build_games,build_team_game,pregame_roll
from src.releases import choose_assets

R=Path(__file__).resolve().parents[1]; LIVE=R/"live"; CACHE=LIVE/"cache"; MODEL=LIVE/"model"
SEASON=2026

def read_asset(a):
    r=requests.get(a["browser_download_url"],timeout=180); r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content)) if a["name"].lower().endswith(".parquet") else pd.read_csv(io.BytesIO(r.content),low_memory=False)

def current_sources():
    pa=choose_assets("pbp","parquet",SEASON,SEASON,True)
    if not pa: raise RuntimeError("No current-season PBP asset found")
    pbp=pd.concat([read_asset(a) for a in pa],ignore_index=True,sort=False)
    sa=choose_assets("schedules","parquet",SEASON,SEASON,False)
    if not sa: raise RuntimeError("No schedules asset found")
    sched=pd.concat([read_asset(a) for a in sa],ignore_index=True,sort=False)
    if "season" in sched: sched=sched[sched.season==SEASON].copy()
    return pbp,sched

def sigmoid(x): return 1/(1+math.exp(-max(min(x,35),-35)))

def predict_row(row,m):
    z=m["intercept"]
    for c in m["diff_features"]:
        v=row.get(c,np.nan)
        if pd.isna(v): v=m["imputer_median"][c]
        v=(float(v)-m["scaler_mean"][c])/m["scaler_scale"][c]
        z+=m["coefficient"][c]*v
    return sigmoid(z)

def main():
    LIVE.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    model=json.loads((MODEL/"spats_v1.json").read_text())
    pbp,sched=current_sources()
    games=build_games(pbp); cur=build_team_game(pbp)
    seed=pd.read_csv(CACHE/"team_history_seed.csv",low_memory=False)
    for d in (seed,cur):
        d["game_date"]=pd.to_datetime(d["game_date"],errors="coerce")
    hist=pd.concat([seed,cur],ignore_index=True,sort=False).drop_duplicates(["game_id","team"],keep="last")
    hist=hist.sort_values(["team","game_date","game_id"])
    # One synthetic next-game row per team exposes the exact pregame rolling state.
    last_date=pd.to_datetime(hist.game_date).max()+pd.Timedelta(days=1)
    maxweek=int(pd.to_numeric(cur.week,errors="coerce").max()) if len(cur) else 0
    dummy=[]
    for team in sorted(cur.team.dropna().unique()):
        row={c:np.nan for c in hist.columns}
        row.update({"game_id":f"LIVE_{team}","season":SEASON,"week":maxweek+1,
                    "game_date":last_date,"team":team,"opponent":"LIVE"})
        dummy.append(row)
    aug=pd.concat([hist,pd.DataFrame(dummy)],ignore_index=True,sort=False)
    pg=pregame_roll(aug); profiles=pg[pg.game_id.astype(str).str.startswith("LIVE_")].copy()
    keep=["team"]+model["features"]
    profiles[keep].sort_values("team").to_csv(LIVE/"current_team_profiles.csv",index=False)

    # Find the next schedule week with at least one game not present in completed PBP.
    sched["game_id"]=sched.game_id.astype(str); completed=set(games.game_id.astype(str))
    future=sched[~sched.game_id.isin(completed)].copy()
    if future.empty:
        upcoming=pd.DataFrame()
    else:
        wk=int(pd.to_numeric(future.week,errors="coerce").min()); upcoming=future[pd.to_numeric(future.week,errors="coerce")==wk].copy()
        p=profiles.set_index("team")
        rows=[]
        for _,g in upcoming.drop_duplicates("game_id").iterrows():
            h,a=str(g.home_team),str(g.away_team)
            if h not in p.index or a not in p.index: continue
            rec={"game_id":g.game_id,"season":SEASON,"week":wk,"away_team":a,"home_team":h}
            for f in model["features"]: rec[f+"_diff"]=p.loc[h,f]-p.loc[a,f]
            hp=predict_row(rec,model); rec.update({"home_win_prob":hp,"away_win_prob":1-hp,
                "pick":h if hp>=.5 else a,"confidence":max(hp,1-hp),"model_version":model["model_version"]})
            rows.append(rec)
        upcoming=pd.DataFrame(rows)
    upcoming.to_csv(LIVE/"upcoming_predictions.csv",index=False)

    # Append only new game IDs to immutable prediction archive.
    archive_path=LIVE/"season_predictions.csv"
    old=pd.read_csv(archive_path) if archive_path.exists() and archive_path.stat().st_size else pd.DataFrame()
    compact_cols=["game_id","season","week","away_team","home_team","home_win_prob","away_win_prob","pick","confidence","model_version"]
    new=upcoming[compact_cols].copy() if len(upcoming) else pd.DataFrame(columns=compact_cols)
    archive=pd.concat([old,new[~new.game_id.isin(old.game_id) if len(old) else [True]*len(new)]],ignore_index=True)
    # Grade archived games when results become available, without changing original probabilities.
    gm=games.set_index("game_id")
    if len(archive):
        archive["actual_winner"]=archive.game_id.map(lambda x: (gm.loc[x,"home_team"] if gm.loc[x,"home_margin"]>0 else gm.loc[x,"away_team"]) if x in gm.index and gm.loc[x,"home_margin"]!=0 else np.nan)
        archive["correct"]=np.where(archive.actual_winner.notna(),archive.pick.eq(archive.actual_winner),np.nan)
    archive.to_csv(archive_path,index=False)
    status={"model_version":model["model_version"],"season":SEASON,"completed_games":int(len(games)),
            "team_profiles":int(len(profiles)),"upcoming_games":int(len(upcoming)),
            "latest_completed_week":int(pd.to_numeric(games.week,errors="coerce").max()) if len(games) else None,
            "next_week":int(upcoming.week.iloc[0]) if len(upcoming) else None}
    (LIVE/"status.json").write_text(json.dumps(status,indent=2)+"\n")
    print(json.dumps(status,indent=2))

if __name__=="__main__": main()
