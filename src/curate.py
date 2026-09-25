from __future__ import annotations
import numpy as np
import pandas as pd

IDCOLS={"game_id","season","week","season_type","game_date","gameday","team","opponent","home_team","away_team"}

def _s(df,n,default=0):
    return df[n] if n in df else pd.Series(default,index=df.index)

def build_games(p):
    p=p.sort_values(["game_id","play_id"])
    g=p.groupby("game_id",as_index=False).agg(
      season=("season","first"),season_type=("season_type","first"),week=("week","first"),
      game_date=("game_date","first"),home_team=("home_team","first"),away_team=("away_team","first"),
      home_score=("total_home_score","max"),away_score=("total_away_score","max"))
    g["game_date"]=pd.to_datetime(g["game_date"],errors="coerce")
    g["home_margin"]=g.home_score-g.away_score
    g["total_points"]=g.home_score+g.away_score
    g["home_win"]=(g.home_margin>0).astype("Int64")
    return g

def build_team_game(p):
    x=p[p.posteam.notna() & p.defteam.notna()].copy()
    x["pass_i"]=_s(x,"pass_attempt").fillna(0).eq(1)
    x["rush_i"]=_s(x,"rush_attempt").fillna(0).eq(1)
    x["success_i"]=_s(x,"success",np.nan)
    x["expl_pass_i"]=x.pass_i & (_s(x,"yards_gained")>=20)
    x["expl_rush_i"]=x.rush_i & (_s(x,"yards_gained")>=10)
    x["turnover_i"]=(_s(x,"interception")==1)|(_s(x,"fumble_lost")==1)
    x["third_i"]=_s(x,"down").eq(3)
    x["third_conv_i"]=x.third_i & _s(x,"first_down").eq(1)
    x["fourth_i"]=_s(x,"down").eq(4)
    x["fourth_conv_i"]=x.fourth_i & _s(x,"first_down").eq(1)
    x["early_i"]=_s(x,"down").isin([1,2])
    x["rz_i"]=_s(x,"yardline_100",999).le(20)
    x["td_i"]=_s(x,"touchdown").eq(1)
    x["penalty_i"]=_s(x,"penalty").fillna(0).eq(1)
    rows=[]
    keys=["game_id","season","week","game_date","posteam","defteam"]
    for k,d in x.groupby(keys,sort=False):
        pa=max(int(d.pass_i.sum()),1); ra=max(int(d.rush_i.sum()),1)
        drives=max(d["drive"].nunique() if "drive" in d else 0,1)
        comp=max(int(_s(d,"complete_pass").fillna(0).sum()),1)
        rz=d[d.rz_i]; third=d[d.third_i]; fourth=d[d.fourth_i]
        row=dict(zip(keys,k))
        row.update({
          "plays":len(d),"drives":drives,"epa_per_play":d.epa.mean(),"success_rate":d.success_i.mean(),
          "yards_per_play":_s(d,"yards_gained",np.nan).mean(),
          "pass_epa":d.loc[d.pass_i,"epa"].mean(),"rush_epa":d.loc[d.rush_i,"epa"].mean(),
          "pass_success":d.loc[d.pass_i,"success_i"].mean(),"rush_success":d.loc[d.rush_i,"success_i"].mean(),
          "early_down_epa":d.loc[d.early_i,"epa"].mean(),
          "explosive_pass_rate":d.expl_pass_i.sum()/pa,"explosive_rush_rate":d.expl_rush_i.sum()/ra,
          "third_down_rate":d.third_conv_i.sum()/max(len(third),1),"fourth_down_rate":d.fourth_conv_i.sum()/max(len(fourth),1),
          "red_zone_td_rate":rz.td_i.sum()/max(rz["drive"].nunique() if "drive" in rz else len(rz),1),
          "turnovers_per_drive":d.turnover_i.sum()/drives,
          "sack_rate":_s(d,"sack").fillna(0).sum()/pa,"qb_hit_rate":_s(d,"qb_hit").fillna(0).sum()/pa,
          "completion_rate":_s(d,"complete_pass").fillna(0).sum()/pa,
          "cpoe":pd.to_numeric(_s(d,"cpoe",np.nan),errors="coerce").mean(),
          "air_yards_per_attempt":pd.to_numeric(_s(d,"air_yards",np.nan),errors="coerce").sum()/pa,
          "yac_per_completion":pd.to_numeric(_s(d,"yards_after_catch",np.nan),errors="coerce").sum()/comp,
          "first_down_rate":_s(d,"first_down").fillna(0).sum()/max(len(d),1),
          "penalty_rate":d.penalty_i.sum()/max(len(d),1),
          "penalty_yards_per_play":pd.to_numeric(_s(d,"penalty_yards"),errors="coerce").fillna(0).sum()/max(len(d),1),
          "plays_per_drive":len(d)/drives,
          "punt_rate":_s(d,"punt_attempt").fillna(0).sum()/drives,
          "fg_attempt_rate":_s(d,"field_goal_attempt").fillna(0).sum()/drives,
          "wp_added":pd.to_numeric(_s(d,"wpa",np.nan),errors="coerce").sum(),
          "total_epa":pd.to_numeric(_s(d,"epa",np.nan),errors="coerce").sum()
        })
        rows.append(row)
    off=pd.DataFrame(rows).rename(columns={"posteam":"team","defteam":"opponent"})
    ids=["game_id","season","week","game_date","team","opponent"]
    metrics=[c for c in off if c not in ids]
    de=off.rename(columns={"team":"opponent","opponent":"team",**{c:"def_"+c for c in metrics}})
    return off.merge(de,on=ids,how="left")

def pregame_roll(t, windows=(3,5,8), alpha=.35):
    t=t.copy(); t["game_date"]=pd.to_datetime(t.game_date)
    t=t.sort_values(["team","game_date","game_id"]).reset_index(drop=True)
    ids=["game_id","season","week","game_date","team","opponent"]
    nums=[c for c in t.select_dtypes(include=np.number) if c not in {"season","week"}]
    o=t[ids].copy()
    for c in nums:
        s=t.groupby("team",sort=False)[c].shift(1)
        for w in windows:
            o[f"{c}_l{w}"]=s.groupby(t.team).transform(lambda q:q.rolling(w,min_periods=1).mean())
        o[f"{c}_std"]=s.groupby([t.team,t.season]).transform(lambda q:q.expanding(min_periods=1).mean())
        o[f"{c}_ewm"]=s.groupby(t.team).transform(lambda q:q.ewm(alpha=alpha,adjust=False,min_periods=1).mean())
    return o

def matchup_matrix(g,p):
    # Normalize merge keys explicitly: PBP-derived games can carry dates as strings
    # while rolling features use pandas datetimes.
    g=g.copy(); p=p.copy()
    g["game_date"]=pd.to_datetime(g["game_date"],errors="coerce")
    p["game_date"]=pd.to_datetime(p["game_date"],errors="coerce")
    h=p.rename(columns={"team":"home_team","opponent":"home_prev_opp"})
    a=p.rename(columns={"team":"away_team","opponent":"away_prev_opp"})
    z=g.merge(h,on=["game_id","season","week","game_date","home_team"],how="left")
    z=z.merge(a,on=["game_id","season","week","game_date","away_team"],how="left",suffixes=("_home","_away"))
    for hc in [c for c in z if c.endswith("_home")]:
        ac=hc[:-5]+"_away"
        if ac in z and pd.api.types.is_numeric_dtype(z[hc]):
            z[hc[:-5]+"_diff"]=z[hc]-z[ac]
    return z
