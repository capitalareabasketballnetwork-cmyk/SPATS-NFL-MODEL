from __future__ import annotations
import argparse,json,re,sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.releases import choose_assets,download_asset
from src.curate import build_games,build_team_game,pregame_roll,matchup_matrix

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data/raw"; CUR=ROOT/"data/curated"; FEAT=ROOT/"data/features"; REP=ROOT/"reports"

def read_any(p):
    return pd.read_parquet(p) if p.suffix==".parquet" else pd.read_csv(p,low_memory=False)

def ingest(name,cfg,start,end):
    tag=cfg["tag"]; seasonal=cfg.get("seasonal",False); minimum=cfg.get("min_season",start)
    assets=choose_assets(tag,cfg.get("preferred","parquet"),max(start,minimum),end,seasonal)
    paths=[]
    for a in assets:
        p=RAW/name/a["name"]
        if not p.exists(): download_asset(a,p)
        paths.append(p)
    return paths

def main(start,end):
    for p in [RAW,CUR,FEAT,REP]: p.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/"config/sources.json").read_text())
    manifest=[]; tables={}
    for name,s in cfg.items():
        try:
            paths=ingest(name,s,start,end)
            manifest.append({"dataset":name,"status":"ok","files":len(paths),"paths":";".join(str(p.relative_to(ROOT)) for p in paths)})
            if paths:
                frames=[read_any(p) for p in paths]
                df=pd.concat(frames,ignore_index=True,sort=False) if len(frames)>1 else frames[0]
                if "season" in df: df=df[(df.season>=start)&(df.season<=end)]
                out=CUR/f"{name}.parquet"; df.to_parquet(out,index=False); tables[name]=df
        except Exception as e:
            manifest.append({"dataset":name,"status":"unavailable","files":0,"paths":"","error":str(e)[:500]})
            print(f"WARN {name}: {e}")
    if "pbp" not in tables: raise RuntimeError("PBP is required; build cannot continue")
    pbp=tables["pbp"]
    games=build_games(pbp); tg=build_team_game(pbp); pg=pregame_roll(tg); mm=matchup_matrix(games,pg)
    games.to_parquet(CUR/"games.parquet",index=False)
    tg.to_parquet(CUR/"team_game_derived.parquet",index=False)
    pg.to_parquet(FEAT/"team_pregame.parquet",index=False)
    mm.to_parquet(FEAT/"matchups.parquet",index=False)
    pd.DataFrame(manifest).to_csv(REP/"source_manifest.csv",index=False)
    coverage=[]
    for name,df in {**tables,"games":games,"team_game_derived":tg}.items():
        coverage.append({"dataset":name,"rows":len(df),"columns":len(df.columns),
                         "min_season":df.season.min() if "season" in df and len(df) else None,
                         "max_season":df.season.max() if "season" in df and len(df) else None})
    pd.DataFrame(coverage).to_csv(REP/"coverage.csv",index=False)
    print(pd.DataFrame(coverage).to_string(index=False))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--start",type=int,default=1999); ap.add_argument("--end",type=int,default=2026)
    a=ap.parse_args(); main(a.start,a.end)
