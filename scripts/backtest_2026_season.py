from pathlib import Path
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.curate import pregame_roll
R=Path(__file__).resolve().parents[1]
FEATURES=["total_epa_l8","def_pass_epa_l8","success_rate_ewm","sack_rate_l5","cpoe_l8","success_rate_l8","wp_added_l5","def_turnovers_per_drive_ewm","def_turnovers_per_drive_l8","explosive_pass_rate_l8","success_rate_l5"]
def mdl(): return Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("lr",LogisticRegression(C=.05,max_iter=5000))])
def profit(stake,ml,won):
 if not won:return -stake
 return stake*(100/ml if ml>0 else 100/abs(ml))
def main():
 tg=pd.read_parquet(R/"data/curated/team_game_derived.parquet"); games=pd.read_parquet(R/"data/curated/games.parquet")
 odds=pd.read_csv(R/"data/reference/2026_weeks1_3_moneylines.csv")
 for z in (tg,games): z["game_id"]=z.game_id.astype(str)
 pg=pregame_roll(tg); h=pg.add_prefix("home_"); a=pg.add_prefix("away_")
 d=games.merge(h,left_on=["game_id","home_team"],right_on=["home_game_id","home_team"],how="left")
 d=d.merge(a,left_on=["game_id","away_team"],right_on=["away_game_id","away_team"],how="left")
 diff=[]
 for f in FEATURES: c=f+"_diff";d[c]=d["home_"+f]-d["away_"+f];diff.append(c)
 # Fixed model architecture chosen from 2015-2025 only; no 2026 outcomes enter fitting.
 tr=d[(d.season>=2015)&(d.season<=2025)&d.home_margin.notna()&(d.home_margin!=0)].copy()
 m=mdl();m.fit(tr[diff],(tr.home_margin>0).astype(int))
 cur=d[(d.season==2026)&(d.week<=3)].copy()
 p=m.predict_proba(cur[diff])[:,1];cur["home_win_prob"]=p;cur["away_win_prob"]=1-p
 cur["pick"]=np.where(p>=.5,cur.home_team,cur.away_team);cur["confidence"]=np.maximum(p,1-p)
 out=odds.merge(cur[["game_id","home_margin","home_win_prob","away_win_prob","pick","confidence"]],on="game_id",how="left")
 out["winner"]=np.where(out.home_margin>0,out.home_team,np.where(out.home_margin<0,out.away_team,np.nan))
 out["correct"]=out.pick.eq(out.winner)
 out["pick_ml"]=np.where(out.pick.eq(out.home_team),out.home_ml,out.away_ml)
 out["stake"]=5.0
 out["profit"]=np.where(out.winner.notna(),[profit(5,float(ml),bool(w)) for ml,w in zip(out.pick_ml,out.correct)],np.nan)
 out.to_csv(R/"reports/2026_season_to_date_betting.csv",index=False)
 done=out[out.winner.notna()]
 summ=done.groupby("week").agg(games=("game_id","size"),wins=("correct","sum"),accuracy=("correct","mean"),wagered=("stake","sum"),net=("profit","sum")).reset_index()
 summ["roi"]=summ.net/summ.wagered
 total=pd.DataFrame([{"week":"TOTAL","games":len(done),"wins":int(done.correct.sum()),"accuracy":done.correct.mean(),"wagered":done.stake.sum(),"net":done.profit.sum(),"roi":done.profit.sum()/done.stake.sum()}])
 pd.concat([summ,total],ignore_index=True).to_csv(R/"reports/2026_season_to_date_summary.csv",index=False)
 print(pd.concat([summ,total],ignore_index=True).to_string(index=False))
if __name__=="__main__":main()
