# SPATS NFL Database & Prediction Lab

A reproducible NFL analytics warehouse covering **1999-present**, with team, player, game, drive, and play-level data.

## Design
SPATS keeps source data separate from derived model features. Coverage is source-dependent: play-by-play is available from 1999; some player/roster/advanced sources begin later. Missing historical coverage is recorded rather than backfilled with invented values.

### Source families
- nflverse play-by-play (1999-present; ~372 play fields)
- schedules/game context and betting lines
- team weekly/season stats
- player weekly/season stats
- players, seasonal/weekly rosters
- snap counts
- PFR advanced passing/rushing/receiving/defense
- NFL Next Gen Stats
- ESPN QBR where available
- depth charts
- injury reports where available
- participation/personnel/coverage data where available
- FTN charting (2022-present)
- draft/combine/trades/contracts where available

## Warehouse layers
- `data/raw/`: immutable source snapshots (not committed)
- `data/curated/`: normalized parquet tables (not committed)
- `data/features/`: leakage-safe pregame features (not committed)
- `reports/`: compact manifests, coverage, validation, and backtests (committed)

## Build
```bash
pip install -r requirements.txt
python scripts/build_database.py --start 1999
python scripts/validate_database.py
python scripts/train_baselines.py
```

## Leakage rule
A prediction row for game G can use only information known before G. Rolling team/player features are shifted before rolling/expanding calculations. The project never uses end-of-season summaries to predict games from earlier in that season.

## Historical reality
"All possible modern data back to 1999" does **not** mean every metric exists in 1999. SPATS stores the maximum real coverage each source provides. For example, weekly rosters begin later than PBP and FTN charting begins in 2022. `reports/coverage.csv` documents actual observed coverage after each build.

## Refresh
The workflow can be run manually and on an in-season schedule. Large data files are rebuilt from source instead of bloating Git history.
