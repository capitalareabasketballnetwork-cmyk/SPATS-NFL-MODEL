from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parents[1]

def main():
 d=pd.read_parquet(R/"data/features/matchups.parquet").copy()
 d=d[(d.home_margin.notna())&(d.home_margin!=0)&(d.season<=2025)].copy()
 y=(d.home_margin>0).astype(int)
 feats=[c for c in d.select_dtypes(include=np.number).columns if c.endswith("_diff")]
 rows=[]; bins=[]
 for c in feats:
  x=d[c]; ok=x.notna()
  if ok.sum()<500 or x[ok].nunique()<10: continue
  xx=x[ok]; yy=y[ok]
  try: auc=roc_auc_score(yy,xx)
  except: continue
  direction=1 if auc>=.5 else -1
  auc=max(auc,1-auc); adv=xx*direction
  # quintiles: win rate when home has increasing oriented advantage
  try:
   q=pd.qcut(adv,5,duplicates="drop")
   tmp=pd.DataFrame({"q":q,"win":yy.values}).groupby("q",observed=True).agg(games=("win","size"),win_rate=("win","mean")).reset_index()
   low=float(tmp.iloc[0].win_rate); high=float(tmp.iloc[-1].win_rate)
  except: low=high=np.nan
  rows.append({"feature":c,"games":int(ok.sum()),"auc":auc,"direction":direction,
               "bottom_quintile_win_rate":low,"top_quintile_win_rate":high,"spread":high-low})
  # percentile thresholds in favorable direction
  for pct in [0.60,0.70,0.80,0.90]:
   th=adv.quantile(pct); sel=adv>=th
   bins.append({"feature":c,"advantage_percentile":int(pct*100),"games":int(sel.sum()),
                "favored_side_win_rate":float(yy[sel].mean() if direction==1 else (1-yy[sel]).mean()),
                "oriented_threshold":float(th)})
 rank=pd.DataFrame(rows).sort_values(["auc","spread"],ascending=False)
 rank.to_csv(R/"reports/pregame_pattern_univariate.csv",index=False)
 pd.DataFrame(bins).to_csv(R/"reports/pregame_pattern_thresholds.csv",index=False)

 # Pairwise pattern screen among strongest 25 individual signals.
 top=rank.head(25).feature.tolist(); pairs=[]
 for i,a in enumerate(top):
  for b in top[i+1:]:
   ra=rank.set_index("feature").loc[a]; rb=rank.set_index("feature").loc[b]
   da=int(ra.direction); db=int(rb.direction)
   xa=d[a]*da; xb=d[b]*db; ok=xa.notna()&xb.notna()
   if ok.sum()<500: continue
   # each feature must place home in its favorable top 40% region
   sa=xa[ok]>=xa[ok].quantile(.60); sb=xb[ok]>=xb[ok].quantile(.60)
   sel=ok.copy(); sel.loc[ok]=sa&sb
   n=int(sel.sum())
   if n>=100:
    pairs.append({"feature_1":a,"feature_2":b,"games":n,"home_win_rate":float(y[sel].mean())})
 pd.DataFrame(pairs).sort_values(["home_win_rate","games"],ascending=[False,False]).to_csv(R/"reports/pregame_pattern_pairs.csv",index=False)
 print("\nTOP INDIVIDUAL PRE-GAME SIGNALS\n",rank.head(25).to_string(index=False))
 print("\nTOP TWO-SIGNAL PATTERNS\n",pd.DataFrame(pairs).sort_values(["home_win_rate","games"],ascending=[False,False]).head(25).to_string(index=False))

if __name__=="__main__": main()
