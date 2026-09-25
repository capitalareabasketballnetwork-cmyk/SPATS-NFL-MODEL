from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import accuracy_score,log_loss,mean_absolute_error,mean_squared_error
R=Path(__file__).resolve().parents[1]

def cols(df):
    ban={"season","week","home_score","away_score","home_margin","total_points","home_win"}
    return [c for c in df.select_dtypes(include=np.number) if c not in ban]

def pipe(kind):
    model=LogisticRegression(max_iter=4000,C=.5) if kind=="cls" else Ridge(alpha=10)
    return Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",model)])

def main():
    d=pd.read_parquet(R/"data/features/matchups.parquet").dropna(subset=["home_margin","total_points"])
    f=cols(d); seasons=sorted(d.season.unique()); rows=[]
    for yr in seasons:
        tr=d[d.season<yr]; te=d[d.season==yr]
        if len(tr)<500 or len(te)==0: continue
        c,ma,to=pipe("cls"),pipe("reg"),pipe("reg")
        c.fit(tr[f],tr.home_win); ma.fit(tr[f],tr.home_margin); to.fit(tr[f],tr.total_points)
        pr=c.predict_proba(te[f])[:,1]; pm=ma.predict(te[f]); pt=to.predict(te[f])
        rows.append({"season":int(yr),"games":len(te),"winner_accuracy":accuracy_score(te.home_win,pr>=.5),
          "log_loss":log_loss(te.home_win,pr,labels=[0,1]),"margin_mae":mean_absolute_error(te.home_margin,pm),
          "margin_rmse":mean_squared_error(te.home_margin,pm)**.5,"total_mae":mean_absolute_error(te.total_points,pt),
          "total_rmse":mean_squared_error(te.total_points,pt)**.5})
    pd.DataFrame(rows).to_csv(R/"reports/baseline_walk_forward.csv",index=False)
    c,ma,to=pipe("cls"),pipe("reg"),pipe("reg")
    c.fit(d[f],d.home_win); ma.fit(d[f],d.home_margin); to.fit(d[f],d.total_points)
    (R/"models").mkdir(exist_ok=True)
    joblib.dump({"features":f,"winner":c,"margin":ma,"total":to},R/"models/baselines.joblib")
if __name__=="__main__": main()
