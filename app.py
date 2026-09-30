from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
CFG=ROOT/"config/model_lab.json"
WEB_TEAM=ROOT/"data/web/model_lab_team_games.csv.gz"
WEB_GAMES=ROOT/"data/web/model_lab_games.csv.gz"

KNOWN_LABELS={
"point_differential_per_game":"Point Differential","epa_per_play":"Offensive EPA / Play",
"offensive_epa_per_play":"Offensive EPA / Play","def_success_rate":"Defensive Success Rate",
"defensive_success_rate":"Defensive Success Rate","third_down_conversion_rate":"3rd Down Conversion %",
"third_down_rate":"3rd Down Conversion %","yards_per_play":"Yards / Play",
"turnover_margin_per_game":"Turnover Margin","yards_per_game":"Yards / Game",
"time_of_possession_seconds":"Time of Possession","fourth_down_conversion_rate":"4th Down Conversion %",
"fourth_down_rate":"4th Down Conversion %","penalty_yards_per_game":"Penalty Yards / Game",
"penalties_per_game":"Penalties / Game"}
KNOWN_DIR={"def_success_rate":-1,"defensive_success_rate":-1,"penalty_yards_per_game":-1,"penalties_per_game":-1}
ID_COLS={"game_id","season","week","game_date","team","home_team","away_team","home_margin","home_score","away_score","result","location"}

def label(k):
    suffix=""
    base=k
    for token,name in [("_l3"," · Last 3"),("_l5"," · Last 5"),("_l8"," · Last 8"),("_ewm"," · Recent EWM")]:
        if k.endswith(token): base=k[:-len(token)]; suffix=name; break
    return KNOWN_LABELS.get(base,base.replace("_"," ").replace(" pct"," %").title())+suffix
def default_dir(k):
    if k in KNOWN_DIR: return KNOWN_DIR[k]
    bad=("allowed","against","penalt","interception","sack_allowed","pressure_allowed","turnover_rate")
    return -1 if any(x in k.lower() for x in bad) else 1

st.set_page_config(page_title="SPATS Model Lab",page_icon="🏈",layout="wide")
st.title("🏈 SPATS Model Lab")
st.caption("Build, test, improve, and use NFL prediction models with the permanent SPATS historical dataset.")
if not (WEB_TEAM.exists() and WEB_GAMES.exists()):
    st.warning("Permanent Model Lab data is not available yet."); st.stop()

@st.cache_data
def load_data(): return pd.read_csv(WEB_TEAM,compression="gzip"),pd.read_csv(WEB_GAMES,compression="gzip")
tg,games=load_data()
tg["game_date"]=pd.to_datetime(tg.get("game_date"),errors="coerce")
games["game_date"]=pd.to_datetime(games.get("game_date"),errors="coerce")

numeric=[c for c in tg.columns if c not in ID_COLS and pd.api.types.is_numeric_dtype(tg[c]) and tg[c].notna().sum()>20 and tg[c].nunique(dropna=True)>1]
for dup,primary in [("third_down_rate","third_down_conversion_rate"),("fourth_down_rate","fourth_down_conversion_rate")]:
    if dup in numeric and primary in numeric: numeric.remove(dup)
COMMON_BASES=[
    "point_differential_per_game","epa_per_play","success_rate","def_epa_per_play",
    "def_success_rate","pass_epa","rush_epa","yards_per_play",
    "third_down_conversion_rate","fourth_down_conversion_rate",
    "turnover_margin_per_game","explosive_pass_rate","explosive_rush_rate",
    "red_zone_td_rate","sack_rate","qb_hit_rate","completion_rate","cpoe",
    "yards_per_game","penalty_yards_per_game","penalties_per_game"
]
def feature_base(k):
    for s in ("_l3","_l5","_l8","_ewm"):
        if k.endswith(s): return k[:-len(s)]
    return k
common_options=sorted([x for x in numeric if feature_base(x) in COMMON_BASES],key=lambda x:(COMMON_BASES.index(feature_base(x)),label(x)))
other_options=sorted([x for x in numeric if x not in common_options],key=lambda x:label(x))
options=common_options+other_options
dropdown_options=common_options+other_options
def dropdown_label(k):
    return ("🔵 " if k in common_options else "")+label(k)

cfg=json.loads(CFG.read_text()) if CFG.exists() else {"seasons":{"start":2015,"end":2026},"stats":{}}
def cfg_col(k): return {"offensive_epa_per_play":"epa_per_play","defensive_success_rate":"def_success_rate"}.get(k,k)
def initial_stats():
    out=[]
    for k,v in cfg.get("stats",{}).items():
        col=cfg_col(k)
        if v.get("enabled") and col in options and col not in out: out.append(col)
    return out or options[:min(6,len(options))]
if "selected_stats" not in st.session_state: st.session_state.selected_stats=[]
for k in options:
    st.session_state.setdefault(f"w_{k}",float(next((v.get("weight",5) for ck,v in cfg.get("stats",{}).items() if cfg_col(ck)==k),5)))
    st.session_state.setdefault(f"use_{k}",k in st.session_state.selected_stats)
    st.session_state.setdefault(f"dir_{k}","Higher is better" if default_dir(k)>0 else "Lower is better")

st.session_state.setdefault("random_count_all",5)
st.session_state.setdefault("random_count_common",5)

def _randomize_from(pool,count_key):
    if not pool: return
    n=max(1,min(int(st.session_state.get(count_key,5)),len(pool))); picks=random.sample(pool,n); st.session_state.selected_stats=picks
    vals=np.random.dirichlet(np.ones(n))
    rounded=[round(float(v),3) for v in vals]
    rounded[-1]=round(1.0-sum(rounded[:-1]),3)
    for k in options: st.session_state[f"use_{k}"]=k in picks
    for k,v in zip(picks,rounded): st.session_state[f"w_{k}"]=v

def randomize(): _randomize_from(options,"random_count_all")
def randomize_common(): _randomize_from(common_options,"random_count_common")

def equalize_weights():
    keys=[k for k in st.session_state.selected_stats if k in options and st.session_state.get(f"use_{k}",True)]
    if not keys: return
    vals=[round(1.0/len(keys),3) for _ in keys[:-1]]
    vals.append(round(1.0-sum(vals),3))
    for k,v in zip(keys,vals): st.session_state[f"w_{k}"]=v

def apply_weights(changes):
    for k,w in changes.items():
        if k not in st.session_state.selected_stats: st.session_state.selected_stats.append(k)
        st.session_state[f"use_{k}"]=True; st.session_state[f"w_{k}"]=round(float(w),1)

with st.sidebar:
    st.header("Backtest range")
    seasons=sorted(pd.to_numeric(games.season,errors="coerce").dropna().astype(int).unique())
    lo,hi=min(seasons),max(seasons)
    start=st.number_input("Start season",lo,hi,max(lo,int(cfg.get("seasons",{}).get("start",lo))))
    end=st.number_input("End season",lo,hi,hi)

builder,weekly,saved_page=st.tabs(["🧪 Model Lab","📅 Week Explorer","💾 Saved Models"])

st.session_state.setdefault("saved_models",{})
st.session_state.setdefault("computed_signature",None)
st.session_state.setdefault("computed_payload",None)

def load_saved_model(model):
    chosen=[k for k in model.get("stats",{}) if k in options]
    st.session_state.selected_stats=chosen
    for k in options:
        st.session_state[f"use_{k}"]=k in chosen
    for k,spec in model.get("stats",{}).items():
        if k in options:
            st.session_state[f"w_{k}"]=float(spec.get("weight",0))
    st.session_state["_loaded_model_name"]=model.get("name","Saved Model")

def active_from_state():
    a={}
    for k in st.session_state.selected_stats:
        if k in options and st.session_state.get(f"use_{k}",True) and st.session_state.get(f"w_{k}",0)>0:
            a[k]={"weight":float(st.session_state[f"w_{k}"]),"direction":default_dir(k)}
    return a

def model_signature(active,start,end):
    return (int(start),int(end),tuple(sorted((k,round(float(v["weight"]),6),int(v["direction"])) for k,v in active.items())))

def build_team_ratings(active):
    d=tg.copy().sort_values(["team","game_date","game_id"]); denom=sum(abs(v["weight"]) for v in active.values()); d["rating"]=0.0
    if denom<=0: return d
    for col,spec in active.items():
        if col not in d: raise ValueError(f"{label(col)} is missing from the permanent dataset.")
        # Window/EWM features above are already pregame values. Base columns use
        # season-to-date expanding averages here.
        if col.endswith(("_l3","_l5","_l8","_ewm")):
            pre=d[col]
        else:
            pre=d.groupby(["team","season"])[col].transform(lambda x:x.shift(1).expanding().mean())
        z=pd.DataFrame({"x":pre,"season":d.season,"week":d.week}).groupby(["season","week"])["x"].transform(
            lambda x:(x-x.mean())/x.std(ddof=0) if x.notna().sum()>1 and x.std(ddof=0)>0 else 0.0)
        d["rating"]+=z*spec["direction"]*spec["weight"]/denom
    return d

@st.cache_data(show_spinner=False)
def run_backtest(start,end,active_items):
    active={k:{"weight":w,"direction":d} for k,w,d in active_items}
    d=build_team_ratings(active); r=d[["game_id","team","rating"]]
    h=r.rename(columns={"team":"home_team","rating":"home_rating"}); a=r.rename(columns={"team":"away_team","rating":"away_rating"})
    out=games.merge(h,on=["game_id","home_team"],how="left").merge(a,on=["game_id","away_team"],how="left")
    out=out[(out.season>=start)&(out.season<=end)&out.home_margin.notna()&(out.home_margin!=0)&out.home_rating.notna()&out.away_rating.notna()].copy()
    out["model_edge"]=out.home_rating-out.away_rating
    out["pick"]=np.where(out.model_edge>=0,out.home_team,out.away_team)
    out["actual_winner"]=np.where(out.home_margin>0,out.home_team,out.away_team)
    out["correct"]=out.pick.eq(out.actual_winner)
    return out,out.groupby("season").agg(games=("correct","size"),wins=("correct","sum"),accuracy=("correct","mean")).reset_index()

def accuracy_of(a):
    items=tuple(sorted((k,float(v["weight"]),int(v["direction"])) for k,v in a.items()))
    r,_=run_backtest(int(start),int(end),items); return (float(r.correct.mean()) if len(r) else 0.0),len(r)

def score_color(x):
    palette=["#5b0a0a","#8b1111","#c44b12","#d98b00","#d9c92f","#b7c93b","#78c66a","#2f9e44","#126b2f"]
    return palette[min(8,max(0,int(np.ceil(x))-2))]
def accuracy_color(p):
    return ("#126b2f","#8DFFAE","🟢 Strong") if p>=60 else (("#5b4a00","#FFE66D","🟡 Moderate") if p>=55 else ("#5b1111","#FF9A90","🔴 Needs work"))

with builder:
    st.subheader("Model Builder")
    c1,c2,c3=st.columns([4.7,1.45,1.75])
    with c1:
        chosen=st.multiselect("Add statistics to your model",dropdown_options,default=[x for x in st.session_state.selected_stats if x in options],format_func=dropdown_label,placeholder="Choose a statistic…")
        st.caption("🔵 = common stat")
        if chosen!=st.session_state.selected_stats:
            st.session_state.selected_stats=chosen
            for k in options: st.session_state[f"use_{k}"]=k in chosen
            st.rerun()
    with c2:
        st.write("")
        st.number_input("Random stats",min_value=1,max_value=max(1,len(options)),step=1,key="random_count_all",help="Use −/+ to choose how many random statistics to add.")
        st.button("🎲 Randomize",use_container_width=True,on_click=randomize,help="Randomize this many statistics from all available stats.")
    with c3:
        st.write("")
        st.number_input("Common stats",min_value=1,max_value=max(1,len(common_options)),step=1,key="random_count_common",help="Use −/+ to choose how many common statistics to add.")
        st.button("🔵 Randomize Common",use_container_width=True,on_click=randomize_common,help="Randomize this many common statistics.")
    st.caption(f"{len(options)} usable statistics are currently available in the permanent web dataset. Add only the ones you want, then edit their weights.")
    if not st.session_state.selected_stats: st.info("Add at least one statistic from the dropdown above.")
    for k in list(st.session_state.selected_stats):
        a,b,c=st.columns([.7,5,2])
        with a: st.checkbox("Use",key=f"use_{k}",label_visibility="collapsed")
        with b: st.markdown(f"**{label(k)}**")
        with c: st.number_input("Weight",0.0,1.0,step=.001,format="%.3f",key=f"w_{k}",label_visibility="collapsed")
    if st.session_state.selected_stats:
        st.button("⚖️ Equal weights",on_click=equalize_weights,help="Distribute 1.000 equally across all selected active statistics.")
    active=active_from_state(); total=sum(v["weight"] for v in active.values())
    if not active or start>end:
        st.warning("Enable at least one statistic with a positive weight and choose a valid season range.")
    elif abs(total-1.0)>0.0005:
        st.error(f"⚠️ Make sure your active weights add up to **1.00**. Current total: **{total:.3f}**")
    else:
        sig=model_signature(active,start,end)
        compute_now=st.button("▶️ Compute Model",type="primary",use_container_width=True,help="Run the historical backtest using the current statistics and weights.")
        if compute_now:
            with st.spinner("Computing model…"):
                items=tuple(sorted((k,float(v["weight"]),int(v["direction"])) for k,v in active.items()))
                results,summary=run_backtest(int(start),int(end),items)
                st.session_state.computed_signature=sig
                st.session_state.computed_payload=(results,summary)
        if st.session_state.computed_signature!=sig or st.session_state.computed_payload is None:
            st.info("Model settings changed. Press **Compute Model** when you're ready to calculate the backtest.")
            st.stop()
        try:
            results,summary=st.session_state.computed_payload; n=len(results); wins=int(results.correct.sum()); acc=wins/n if n else 0; pct=acc*100
            bg,txt,status=accuracy_color(pct); st.divider(); st.subheader("Live Model Results")
            st.markdown(f'<div style="background:{bg};border-radius:18px;padding:22px;text-align:center"><div style="font-size:15px;font-weight:800">OVERALL BACKTEST ACCURACY</div><div style="font-size:68px;font-weight:900;color:{txt};line-height:1.05">{pct:.2f}%</div><div style="font-size:18px;font-weight:800">{status}</div></div>',unsafe_allow_html=True)
            x,y,z=st.columns(3); x.metric("Record",f"{wins:,}–{n-wins:,}"); y.metric("Games Tested",f"{n:,}"); z.metric("Active Stats",len(active))

            st.subheader("Save Model")
            save_name=st.text_input("Model name",placeholder="e.g. Week 4 Efficiency Model",key="save_model_name")
            if st.button("💾 Save Model",type="primary",disabled=not bool(save_name.strip())):
                name=save_name.strip()
                st.session_state.saved_models[name]={
                    "name":name,
                    "stats":{k:{"weight":float(v["weight"]),"direction":int(v["direction"])} for k,v in active.items()},
                    "accuracy":float(pct),"record":f"{wins}–{n-wins}","games":int(n),
                    "start_season":int(start),"end_season":int(end)
                }
                st.success(f'Saved **{name}**. You can reopen it from the Saved Models page.')

            consistency=max(0,1-float(summary.accuracy.std(ddof=0) if len(summary)>1 else 0))
            rating=float(np.clip(1+((pct-48)/17)*8 + min(n,2000)/2000*.5 + (consistency-.9)*1.5,1,10)); col=score_color(rating)
            st.subheader("Model Scorecard"); q1,q2=st.columns([1,3])
            with q1: st.markdown(f'<div style="background:{col};border-radius:16px;padding:18px;text-align:center"><div style="font-size:14px;font-weight:800">MODEL RATING</div><div style="font-size:52px;font-weight:900">{rating:.1f}<span style="font-size:22px">/10</span></div></div>',unsafe_allow_html=True)
            top=sorted(active.items(),key=lambda kv:kv[1]["weight"],reverse=True); topnames=", ".join(label(k) for k,_ in top[:3]); low=", ".join(label(k) for k,_ in top[-2:])
            with q2: st.info(f'**AI Snapshot:** This model is {"above" if pct>=55 else "near or below"} a useful historical baseline at **{pct:.2f}%** across **{n:,} games**. Its identity is driven most by **{topnames}**. The next improvement target is better out-of-sample validation and reducing weight on metrics that add little incremental accuracy.')
            st.markdown(f"**Model explanation:** Your model emphasizes **{topnames}** most heavily. **{low}** currently have the smallest influence. Weights are normalized automatically, so the model cares about each statistic in proportion to the other active weights.")

            st.subheader("🧠 Model Coach")
            st.caption("Tests nearby weight changes against the selected historical range. Improvements are in-sample clues, so a higher backtest number is not guaranteed to improve future picks.")
            def rebalance_candidate(base_weights,target,new_weight):
                # Change one stat while keeping the complete model at exactly 1.00.
                # The offset is spread proportionally across every other active stat,
                # preserving their relative importance.
                out={k:max(0.0,float(v["weight"])) for k,v in base_weights.items()}
                if target not in out or len(out)==1:
                    out[target]=1.0
                    return out
                new_weight=float(np.clip(new_weight,0.0,1.0))
                others=[k for k in out if k!=target]
                other_total=sum(out[k] for k in others)
                remainder=1.0-new_weight
                if other_total>0:
                    for k in others: out[k]=out[k]/other_total*remainder
                else:
                    for k in others: out[k]=remainder/len(others)
                out[target]=new_weight
                return out

            base=acc; trials=[]
            for k,s in active.items():
                ow=s["weight"]
                # Absolute weight-point moves are easier to understand and every
                # candidate is rebalanced before it is backtested.
                for delta in (.100,.050,.025,.010,-.010,-.025,-.050,-.100):
                    nw=float(np.clip(ow+delta,0.0,1.0))
                    if abs(nw-ow)<1e-9: continue
                    weights=rebalance_candidate(active,k,nw)
                    cand={a:{**dict(v),"weight":weights[a]} for a,v in active.items()}
                    ca,_=accuracy_of(cand)
                    trials.append((ca-base,k,nw,delta,ca,weights))
            trials.sort(reverse=True,key=lambda x:x[0]); good=[t for t in trials if t[0]>0][:5]
            suggested={k:v["weight"] for k,v in active.items()}
            if good:
                # Apply the single best complete rebalanced model, rather than
                # stacking independent suggestions that could conflict.
                best=good[0]; suggested=best[5]
                for imp,k,w,delta,ca,weights in good:
                    verb="Increase" if delta>0 else "Decrease"
                    st.success(f'**{label(k)}:** {verb} by **{abs(delta):.3f}** → **{w:.3f}** | complete rebalanced model: {ca*100:.2f}% (**+{imp*100:.2f} pts**)')
                st.caption("Each suggestion is tested as a complete 1.000-weight model. When one weight rises, the required amount is removed proportionally from the other active statistics; when it falls, that amount is redistributed proportionally.")
                st.button("✨ Apply best suggested model",type="primary",on_click=apply_weights,args=(suggested,))
            else:
                st.info("No nearby single-weight adjustment improved this backtest. Try Randomize, a different stat mix, or validate a different season range.")
            st.caption("Why models often stall near 59%: NFL outcomes contain injuries, turnovers, weather, matchup effects and randomness that season averages cannot fully capture. Better gains are more likely from stronger features and recency/matchup modeling than endlessly tuning the same weights.")

            st.subheader("Normalized Weights")
            norm=pd.DataFrame([{"Statistic":label(k),"Weight":v["weight"],"Model Weight %":v["weight"]/total*100,"Direction":"Higher" if v["direction"]>0 else "Lower"} for k,v in active.items()])
            st.dataframe(norm,use_container_width=True,hide_index=True,column_config={"Model Weight %":st.column_config.NumberColumn(format="%.2f%%")})
            st.subheader("Season Results")
            ch=summary.copy(); ch["accuracy"]*=100
            ch["Result"]=ch.apply(lambda r:f'{int(r["season"])}: {r["accuracy"]:.1f}%  ({int(r["wins"])}–{int(r["games"]-r["wins"])})',axis=1)
            def season_style(v):
                try:
                    p=float(v.split(": ")[1].split("%")[0])
                    color="#126b2f" if p>=60 else ("#2f9e44" if p>=57 else ("#d98b00" if p>=54 else "#8b1111"))
                    return f"color: {color}; font-weight: 800"
                except: return ""
            st.dataframe(ch[["Result"]].style.map(season_style),use_container_width=True,hide_index=True)
        except Exception as e:
            st.error(f"Model could not be calculated: {e}")

with weekly:
    st.subheader("Week Explorer")
    latest=int(pd.to_numeric(games.season,errors="coerce").dropna().max()); sg=games[games.season==latest].copy()
    weeks=sorted(pd.to_numeric(sg.week,errors="coerce").dropna().astype(int).unique())
    completed=pd.to_numeric(sg.home_margin,errors="coerce").notna()
    current=int(sg.loc[completed,"week"].max()) if completed.any() else (weeks[0] if weeks else 1)
    future_weeks=[w for w in weeks if w>=current and sg.loc[sg.week==w,"home_margin"].isna().any()]
    if future_weeks: current=future_weeks[0]
    season_pick=st.selectbox("Season",seasons,index=len(seasons)-1)
    wg=games[games.season==season_pick]; wks=sorted(pd.to_numeric(wg.week,errors="coerce").dropna().astype(int).unique())
    default=wks.index(current) if season_pick==latest and current in wks else len(wks)-1
    week_pick=st.selectbox("Week",wks,index=max(0,default))
    active=active_from_state()
    if not active:
        st.info("Build a model in the Model Lab tab first.")
    else:
        hist=tg[(tg.season==season_pick)&(pd.to_numeric(tg.week,errors="coerce")<week_pick)].copy(); team_rating={}
        if not hist.empty:
            teams=sorted(set(hist.team.dropna())); raw=pd.DataFrame(index=teams); raw["rating"]=0.0; denom=sum(abs(v["weight"]) for v in active.values())
            for col,spec in active.items():
                vals=hist.groupby("team")[col].mean() if col in hist else pd.Series(dtype=float)
                x=vals.reindex(teams); sd=x.std(ddof=0); zz=(x-x.mean())/sd if pd.notna(sd) and sd>0 else x*0
                raw["rating"]+=zz.fillna(0)*spec["direction"]*spec["weight"]/denom
            team_rating=raw["rating"].to_dict()
        view=wg[wg.week==week_pick].copy(); view["home_rating"]=view.home_team.map(team_rating); view["away_rating"]=view.away_team.map(team_rating)
        for _,g in view.sort_values("game_date").iterrows():
            finished=pd.notna(g.get("home_margin")); edge=g.get("home_rating",np.nan)-g.get("away_rating",np.nan)
            pick=(g.home_team if edge>=0 else g.away_team) if pd.notna(edge) else "—"
            hs=g.get("home_score",np.nan); aws=g.get("away_score",np.nan)
            if finished:
                hw=g.home_margin>0; away_col="#39d353" if not hw else "#ff6b6b"; home_col="#39d353" if hw else "#ff6b6b"
                score=(f"{int(aws)} – {int(hs)}" if pd.notna(aws) and pd.notna(hs) else f"Margin: {abs(float(g.home_margin)):.0f}")
                st.markdown(f'<div style="border:1px solid #555;border-radius:14px;padding:14px;margin:8px 0"><b>Final</b> &nbsp; <span style="color:{away_col};font-size:20px;font-weight:800">{g.away_team}</span> <b>{score}</b> <span style="color:{home_col};font-size:20px;font-weight:800">{g.home_team}</span><br><span style="opacity:.8">Model pick: <b>{pick}</b></span></div>',unsafe_allow_html=True)
            else:
                detail=f" · Edge {abs(edge):.2f}" if pd.notna(edge) else " · Waiting for enough prior data"
                st.markdown(f'<div style="border:1px solid #555;border-radius:14px;padding:14px;margin:8px 0"><b>{g.away_team} @ {g.home_team}</b><br>Model pick: <b>{pick}</b>{detail}</div>',unsafe_allow_html=True)


with saved_page:
    st.subheader("💾 Saved Models")
    st.caption("Choose a saved model to open its dashboard, adjust the model, or use it to look at upcoming games.")
    if not st.session_state.saved_models:
        st.info("You haven't saved any models yet. Build a model in Model Lab, give it a name, and press Save Model.")
    else:
        names=list(st.session_state.saved_models.keys())
        st.session_state.setdefault("open_saved_model",names[0])
        if st.session_state.open_saved_model not in names: st.session_state.open_saved_model=names[0]
        picked=st.selectbox("Saved models",names,index=names.index(st.session_state.open_saved_model))
        st.session_state.open_saved_model=picked
        model=st.session_state.saved_models[picked]
        with st.container(border=True):
            st.markdown(f"## {picked}")
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Accuracy",f'{model.get("accuracy",0):.2f}%')
            m2.metric("Record",model.get("record","—"))
            m3.metric("Stats",len(model.get("stats",{})))
            m4.metric("Games Tested",f'{model.get("games",0):,}')
            st.caption(f'Backtest: {model.get("start_season","—")}–{model.get("end_season","—")}')
            st.markdown("### Model weights")
            saved_table=pd.DataFrame([{"Statistic":label(k),"Weight":float(v.get("weight",0))} for k,v in model.get("stats",{}).items()])
            st.dataframe(saved_table,use_container_width=True,hide_index=True,column_config={"Weight":st.column_config.NumberColumn(format="%.3f")})
            a1,a2,a3=st.columns([1.3,1.6,4])
            if a1.button("✏️ Open & Adjust",type="primary",use_container_width=True):
                load_saved_model(model)
                st.success(f'**{picked}** is loaded into Model Lab. Open the Model Lab tab to adjust its stats or weights.')
            if a2.button("🏈 Use for Future Games",use_container_width=True):
                load_saved_model(model)
                st.session_state["saved_future_model"]=picked
                st.success(f'**{picked}** is loaded. Open Week Explorer to view its picks for upcoming games.')
            if a3.button("🗑️ Delete Model"):
                del st.session_state.saved_models[picked]
                st.session_state.pop("open_saved_model",None)
                st.rerun()

        st.markdown("### Saved model list")
        for name in names:
            sm=st.session_state.saved_models[name]
            left,right=st.columns([5,1])
            left.markdown(f'**{name}**  ·  {sm.get("accuracy",0):.2f}%  ·  {len(sm.get("stats",{}))} stats  ·  {sm.get("record","—")}')
            if right.button("Open",key=f"open_card_{name}",use_container_width=True):
                st.session_state.open_saved_model=name
                st.rerun()
