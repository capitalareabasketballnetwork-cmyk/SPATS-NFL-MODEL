from pathlib import Path
import json, subprocess, sys
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
CFG=ROOT/"config/model_lab.json"
SUMMARY=ROOT/"reports/model_lab_summary.csv"
RESULTS=ROOT/"reports/model_lab_results.csv"
TEAM_GAME=ROOT/"data/curated/team_game_derived.parquet"
GAMES=ROOT/"data/curated/games.parquet"
WEB_TEAM_GAME=ROOT/"data/web/model_lab_team_games.csv.gz"
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
"qbr":"QBR",
"strength_of_schedule":"Strength of Schedule",
"strength_of_record":"Strength of Record",
}

st.set_page_config(page_title="SPATS Model Lab",page_icon="🏈",layout="wide")
st.title("SPATS Model Lab")
st.caption("Build, weight, and backtest NFL prediction models using the SPATS database.")

cfg=json.loads(CFG.read_text())
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
c1,c2,c3=st.columns(3)
c1.metric("Enabled statistics",len(active))
c2.metric("Raw weight total",f"{weight_total:.1f}")
c3.metric("Normalization","Automatic")
if weight_total<=0: st.warning("Enable at least one statistic with a positive weight.")

def prepare_compact_data():
    """Materialize the small committed web dataset into the paths used by Model Lab."""
    if not (WEB_TEAM_GAME.exists() and WEB_GAMES.exists()):
        return False
    TEAM_GAME.parent.mkdir(parents=True,exist_ok=True)
    pd.read_csv(WEB_TEAM_GAME,compression="gzip").to_parquet(TEAM_GAME,index=False)
    pd.read_csv(WEB_GAMES,compression="gzip").to_parquet(GAMES,index=False)
    return True

run=st.button("Run Backtest",type="primary",disabled=(start>end or weight_total<=0),use_container_width=True)
if run:
    if not prepare_compact_data():
        st.error("The compact Model Lab dataset is still being prepared by GitHub Actions. Please try again in a few minutes.")
        st.stop()
    new_cfg={**cfg,"seasons":{"start":int(start),"end":int(end)}}
    new_cfg["stats"]={**cfg["stats"],**edited}
    new_cfg["stats"]["qbr"]={**cfg["stats"].get("qbr",{}),"enabled":False,"weight":0}
    original=CFG.read_text()
    try:
        CFG.write_text(json.dumps(new_cfg,indent=2)+"\n")
        with st.spinner("Running Model Lab backtest..."):
            p=subprocess.run([sys.executable,str(ROOT/"scripts/backtest_model_lab.py")],cwd=ROOT,text=True,capture_output=True)
        if p.returncode:
            st.error("Backtest stopped. No partial result was accepted.")
            st.code((p.stdout+"\n"+p.stderr)[-5000:])
        else:
            st.success("Backtest complete — every enabled statistic was found and used.")
    finally:
        CFG.write_text(original)

if SUMMARY.exists():
    s=pd.read_csv(SUMMARY)
    total=s[s["season"].astype(str)=="TOTAL"]
    if len(total):
        r=total.iloc[0]
        a,b,c=st.columns(3)
        a.metric("Overall Accuracy",f"{float(r.accuracy)*100:.2f}%")
        b.metric("Correct Picks",f"{int(r.wins):,}")
        c.metric("Games Tested",f"{int(r.games):,}")
    st.subheader("Season Results")
    view=s[s["season"].astype(str)!="TOTAL"].copy()
    view["accuracy"]=pd.to_numeric(view["accuracy"])*100
    st.bar_chart(view.set_index("season")["accuracy"],y_label="Accuracy %")
    st.dataframe(view.rename(columns={"season":"Season","games":"Games","wins":"Correct","accuracy":"Accuracy %"}),use_container_width=True,hide_index=True)

if RESULTS.exists():
    st.subheader("Game Results")
    d=pd.read_csv(RESULTS)
    st.dataframe(d.sort_values(["season","week"],ascending=[False,False]),use_container_width=True,hide_index=True)
