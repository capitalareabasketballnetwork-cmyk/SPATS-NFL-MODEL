from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
CFG=ROOT/"config/model_lab.json"
WEB_TEAM=ROOT/"data/web/model_lab_team_games.csv.gz"
WEB_GAMES=ROOT/"data/web/model_lab_games.csv.gz"

LABELS={
"point_differential_per_game":"Point Differential",
"offensive_epa_per_play":"Offensive EPA / Play",
"defensive_success_rate":"Defensive Success Rate",
"third_down_conversion_rate":"3rd Down Conversion %",
"yards_per_play":"Yards / Play",
"turnover_margin_per_game":"Turnover Margin",
"yards_per_game":"Yards / Game",
"time_of_possession_seconds":"Time of Possession",
"fourth_down_conversion_rate":"4th Down Conversion %",
"penalty_yards_per_game":"Penalty Yards / Game",
"penalties_per_game":"Penalties / Game",
"strength_of_schedule":"Strength of Schedule",
"strength_of_record":"Strength of Record",
}
ALIASES={
"point_differential_per_game":["point_differential_per_game"],
"offensive_epa_per_play":["epa_per_play"],
"defensive_success_rate":["def_success_rate"],
"third_down_conversion_rate":["third_down_conversion_rate","third_down_rate"],
"yards_per_play":["yards_per_play"],
"turnover_margin_per_game":["turnover_margin_per_game"],
"yards_per_game":["yards_per_game"],
"time_of_possession_seconds":["time_of_possession_seconds"],
"fourth_down_conversion_rate":["fourth_down_conversion_rate","fourth_down_rate"],
"penalty_yards_per_game":["penalty_yards_per_game"],
"penalties_per_game":["penalties_per_game"],
}

st.set_page_config(page_title="SPATS Model Lab",page_icon="🏈",layout="wide")
st.title("🏈 SPATS Model Lab")
st.caption("Build, weight, and backtest NFL prediction models using SPATS historical data.")

cfg=json.loads(CFG.read_text())
if not (WEB_TEAM.exists() and WEB_GAMES.exists()):
    st.warning("The compact Model Lab dataset is still being prepared. The website is online; backtesting will unlock automatically when the dataset is available.")
    st.stop()

@st.cache_data
def load_data():
    return pd.read_csv(WEB_TEAM,compression="gzip"),pd.read_csv(WEB_GAMES,compression="gzip")

tg,games=load_data()
if "game_date" in tg: tg["game_date"]=pd.to_datetime(tg["game_date"],errors="coerce")

with st.sidebar:
    st.header("Backtest range")
    start=st.number_input("Start season",2015,2026,max(2015,int(cfg["seasons"]["start"])))
    end=st.number_input("End season",2015,2026,max(2015,int(cfg["seasons"]["end"])))
    if start>end: st.error("Start season must be before end season.")

st.subheader("Model Builder")
st.write("Turn statistics on or off and adjust their weights. Enabled weights are normalized automatically.")
edited={}
for key,spec in cfg["stats"].items():
    if key=="qbr": continue
    c1,c2,c3=st.columns([1,4,2])
    with c1: enabled=st.checkbox("Use",value=bool(spec.get("enabled")),key=f"use_{key}",label_visibility="collapsed")
    with c2: st.markdown(f"**{LABELS.get(key,key)}**")
    with c3: weight=st.number_input("Weight",0.0,100.0,float(spec.get("weight",0)),0.5,key=f"w_{key}",label_visibility="collapsed")
    edited[key]={**spec,"enabled":enabled,"weight":weight}

active={k:v for k,v in edited.items() if v["enabled"] and v["weight"]>0}
weight_total=sum(v["weight"] for v in active.values())
a,b,c=st.columns(3)
a.metric("Enabled statistics",len(active)); b.metric("Raw weight total",f"{weight_total:.1f}"); c.metric("Normalization","Automatic")

def find_col(df,key):
    for col in ALIASES.get(key,[key]):
        if col in df.columns: return col
    return None

def run_backtest(start,end,active):
    d=tg.copy()
    g=games.copy()
    if "point_differential_per_game" not in d.columns:
        hm=g[["game_id","home_team","home_margin"]].rename(columns={"home_team":"team","home_margin":"point_differential_per_game"})
        am=g[["game_id","away_team","home_margin"]].rename(columns={"away_team":"team"})
        am["point_differential_per_game"]=-am["home_margin"]
        d=d.merge(pd.concat([hm,am[["game_id","team","point_differential_per_game"]]],ignore_index=True),on=["game_id","team"],how="left")
    d=d.sort_values(["team","game_date","game_id"])
    missing=[]
    denom=sum(abs(float(v["weight"])) for v in active.values())
    d["rating"]=0.0
    for key,spec in active.items():
        col=find_col(d,key)
        if not col:
            missing.append(key); continue
        pre=d.groupby("team")[col].transform(lambda x:x.shift(1).expanding().mean())
        z=pd.DataFrame({"x":pre,"season":d["season"],"week":d["week"]}).groupby(["season","week"])["x"].transform(
            lambda x:(x-x.mean())/x.std(ddof=0) if x.notna().sum()>1 and x.std(ddof=0)>0 else 0.0)
        d["rating"]+=z*float(spec.get("direction",1))*float(spec["weight"])/denom
    if missing:
        raise ValueError("Enabled statistics missing from compact dataset: "+", ".join(missing))
    ratings=d[["game_id","team","rating"]]
    h=ratings.rename(columns={"team":"home_team","rating":"home_rating"})
    aw=ratings.rename(columns={"team":"away_team","rating":"away_rating"})
    out=g.merge(h,on=["game_id","home_team"],how="left").merge(aw,on=["game_id","away_team"],how="left")
    out=out[(out.season>=start)&(out.season<=end)&out.home_margin.notna()&(out.home_margin!=0)].copy()
    out=out[out.home_rating.notna()&out.away_rating.notna()].copy()
    out["model_edge"]=out.home_rating-out.away_rating
    out["pick"]=np.where(out.model_edge>=0,out.home_team,out.away_team)
    out["actual_winner"]=np.where(out.home_margin>0,out.home_team,out.away_team)
    out["correct"]=out.pick.eq(out.actual_winner)
    summary=out.groupby("season").agg(games=("correct","size"),wins=("correct","sum"),accuracy=("correct","mean")).reset_index()
    return out,summary

if weight_total<=0:
    st.warning("Enable at least one statistic with a positive weight.")
elif start>end:
    st.warning("Choose a valid season range.")
else:
    # Streamlit reruns automatically whenever a toggle, weight, or season changes,
    # so the displayed results always represent the controls currently on screen.
    try:
        results,summary=run_backtest(int(start),int(end),active)
        total_games=len(results)
        wins=int(results["correct"].sum())
        accuracy=wins/total_games if total_games else 0

        st.divider()
        st.subheader("Live Model Results")
        a,b,c=st.columns(3)
        a.metric("Overall Accuracy",f"{accuracy*100:.2f}%")
        b.metric("Record",f"{wins:,}–{total_games-wins:,}")
        c.metric("Games Tested",f"{total_games:,}")

        st.caption("Results update automatically whenever you change a statistic, weight, or season range.")

        st.subheader("Normalized Weights")
        norm=pd.DataFrame([
            {
                "Statistic":LABELS.get(k,k),
                "Raw Weight":float(v["weight"]),
                "Model Weight %":float(v["weight"])/weight_total*100,
            }
            for k,v in active.items()
        ])
        st.dataframe(norm,use_container_width=True,hide_index=True,column_config={
            "Raw Weight":st.column_config.NumberColumn(format="%.1f"),
            "Model Weight %":st.column_config.NumberColumn(format="%.2f%%"),
        })

        st.subheader("Season Results")
        chart=summary.copy()
        chart["accuracy"]=chart["accuracy"]*100
        st.bar_chart(chart.set_index("season")["accuracy"],y_label="Accuracy %")
        st.dataframe(
            chart.rename(columns={"season":"Season","games":"Games","wins":"Correct","accuracy":"Accuracy %"}),
            use_container_width=True,hide_index=True,
            column_config={"Accuracy %":st.column_config.NumberColumn(format="%.2f%%")}
        )

        with st.expander("View individual game results"):
            cols=[x for x in ["season","week","away_team","home_team","pick","actual_winner","correct","model_edge"] if x in results.columns]
            st.dataframe(results[cols].sort_values(["season","week"],ascending=[False,False]),use_container_width=True,hide_index=True)
    except Exception as e:
        st.error(f"Model could not be calculated: {e}")
