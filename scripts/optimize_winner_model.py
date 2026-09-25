from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

R=Path(__file__).resolve().parents[1]
OUT=R/"reports"
OUT.mkdir(exist_ok=True)

def model(C=0.3):
    return Pipeline([
        ("impute",SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("lr",LogisticRegression(C=C,penalty="l2",max_iter=5000))
    ])

def score_seasons(d, feats, seasons, C=.3):
    vals=[]
    for yr in seasons:
        tr=d[d.season<yr]; te=d[d.season==yr]
        if len(tr)<1000 or len(te)<50: continue
        ytr=(tr.home_margin>0).astype(int); yte=(te.home_margin>0).astype(int)
        # Exclude ties from winner optimization.
        tr=tr[tr.home_margin!=0]; te=te[te.home_margin!=0]
        if len(te)<40: continue
        ytr=(tr.home_margin>0).astype(int); yte=(te.home_margin>0).astype(int)
        # A feature may exist overall but be entirely missing in an early
        # historical training fold (for example CPOE before it was tracked).
        # Skip that season for this candidate instead of allowing the imputer
        # to drop every column and hand StandardScaler a 0-column matrix.
        usable=[c for c in feats if tr[c].notna().any()]
        if not usable:
            continue
        m=model(C); m.fit(tr[usable],ytr)
        p=m.predict_proba(te[usable])[:,1]
        vals.append((len(te),accuracy_score(yte,p>=.5),log_loss(yte,p,labels=[0,1]),brier_score_loss(yte,p)))
    if not vals: return None
    n=sum(x[0] for x in vals)
    return {"games":n,"accuracy":sum(x[0]*x[1] for x in vals)/n,
            "log_loss":sum(x[0]*x[2] for x in vals)/n,
            "brier":sum(x[0]*x[3] for x in vals)/n}

def main():
    d=pd.read_parquet(R/"data/features/matchups.parquet")
    d=d[d.home_margin.notna()].copy()
    # Only matchup differences made from shifted/rolling pregame values.
    candidates=[c for c in d.columns if c.endswith("_diff") and pd.api.types.is_numeric_dtype(d[c])]
    candidates=[c for c in candidates if d[c].notna().mean()>=.45 and d[c].nunique(dropna=True)>5]
    seasons=sorted(int(x) for x in d.season.dropna().unique())
    # Tune on completed historical seasons only; 2026 remains a current/front-test season.
    completed=[s for s in seasons if s<=2025]
    eval_seasons=[s for s in completed if s>=2004]

    # Fast leakage-safe screening: each feature evaluated in walk-forward form.
    uni=[]
    for i,c in enumerate(candidates):
        r=score_seasons(d,[c],eval_seasons)
        if r: uni.append({"feature":c,**r})
    uni=pd.DataFrame(uni).sort_values(["log_loss","accuracy"],ascending=[True,False])
    uni.to_csv(OUT/"optimization_univariate.csv",index=False)
    pool=uni.head(40).feature.tolist()

    # Greedy forward selection minimizes walk-forward log loss.
    selected=[]; leaderboard=[]; remaining=pool.copy()
    for step in range(min(15,len(pool))):
        trials=[]
        for c in remaining:
            r=score_seasons(d,selected+[c],eval_seasons)
            if r: trials.append((r["log_loss"],-r["accuracy"],c,r))
        if not trials: break
        trials.sort()
        _,_,best,r=trials[0]
        if leaderboard and r["log_loss"] >= leaderboard[-1]["log_loss"]-0.00015:
            break
        selected.append(best); remaining.remove(best)
        leaderboard.append({"n_features":len(selected),"added":best,"features":";".join(selected),**r})
    lb=pd.DataFrame(leaderboard)
    lb.to_csv(OUT/"optimization_combinations.csv",index=False)
    if not selected: raise RuntimeError("No usable feature combination found")

    # Tune regularization for the selected combination.
    cs=[.01,.03,.1,.3,1,3,10]
    tune=[]
    for C in cs:
        r=score_seasons(d,selected,eval_seasons,C)
        tune.append({"C":C,**r})
    tune=pd.DataFrame(tune).sort_values(["log_loss","accuracy"],ascending=[True,False])
    tune.to_csv(OUT/"optimization_regularization.csv",index=False)
    bestC=float(tune.iloc[0].C)

    # Honest season-by-season walk-forward results.
    yearly=[]
    for yr in eval_seasons:
        r=score_seasons(d,selected,[yr],bestC)
        if r: yearly.append({"season":yr,**r})
    pd.DataFrame(yearly).to_csv(OUT/"optimization_yearly.csv",index=False)

    # Fit through 2025 and export standardized coefficient weights for current 2026 use.
    hist=d[(d.season<=2025)&(d.home_margin!=0)].copy()
    m=model(bestC); m.fit(hist[selected],(hist.home_margin>0).astype(int))
    coef=m.named_steps["lr"].coef_[0]
    abs_sum=np.abs(coef).sum()
    weights=pd.DataFrame({"feature":selected,"coefficient":coef,
                          "weight_pct":np.abs(coef)/abs_sum*100,
                          "direction":np.where(coef>=0,"home-positive","away-positive")})
    weights=weights.sort_values("weight_pct",ascending=False)
    weights.to_csv(OUT/"optimization_best_weights.csv",index=False)

    # Front-test on completed 2026 games, if any.
    cur=d[(d.season==2026)&(d.home_margin.notna())&(d.home_margin!=0)].copy()
    current={}
    if len(cur):
        p=m.predict_proba(cur[selected])[:,1]; y=(cur.home_margin>0).astype(int)
        current={"games":int(len(cur)),"accuracy":float(accuracy_score(y,p>=.5)),
                 "log_loss":float(log_loss(y,p,labels=[0,1])),"brier":float(brier_score_loss(y,p))}
        pd.DataFrame([current]).to_csv(OUT/"optimization_2026_fronttest.csv",index=False)

    summary={"selected_features":selected,"best_C":bestC,
             "historical_walk_forward":score_seasons(d,selected,eval_seasons,bestC),
             "current_2026_fronttest":current}
    (OUT/"optimization_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print("\nBEST STANDARDIZED WEIGHTS\n",weights.to_string(index=False))

if __name__=="__main__":
    warnings.filterwarnings("ignore",category=RuntimeWarning)
    main()
