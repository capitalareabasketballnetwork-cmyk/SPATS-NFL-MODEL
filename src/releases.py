from __future__ import annotations
import json,re
from pathlib import Path
import requests

API="https://api.github.com/repos/nflverse/nflverse-data/releases/tags/{tag}"
UA={"User-Agent":"SPATS-NFL-MODEL"}

def release_assets(tag:str):
    r=requests.get(API.format(tag=tag),headers=UA,timeout=60)
    r.raise_for_status()
    return r.json().get("assets",[])

def choose_assets(tag, preferred="parquet", start=1999, end=2100, seasonal=False):
    assets=release_assets(tag)
    ext=("."+preferred)
    candidates=[a for a in assets if a["name"].lower().endswith(ext)]
    if not candidates:
        candidates=[a for a in assets if a["name"].lower().endswith(".csv")]
    if seasonal:
        out=[]
        for a in candidates:
            years=[int(y) for y in re.findall(r"(?:19|20)\d{2}",a["name"])]
            if years and start<=years[-1]<=end: out.append(a)
        return out
    return candidates

def download_asset(asset:dict,dest:Path):
    dest.parent.mkdir(parents=True,exist_ok=True)
    with requests.get(asset["browser_download_url"],headers=UA,stream=True,timeout=180) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(1024*1024):
                if chunk: f.write(chunk)
    return dest
