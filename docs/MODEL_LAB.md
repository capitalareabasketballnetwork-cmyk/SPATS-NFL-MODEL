# SPATS Model Lab

Model Lab lets you change a predictive model without editing Python.

## Configure
Open `config/model_lab.json`. For each statistic:
- `enabled`: `true` or `false`
- `weight`: any numeric weight
- `direction`: `1` when higher is better, `-1` when lower is better

Weights are automatically normalized across the enabled stats, so they do not have to add to 100.

## Run
```bash
python scripts/backtest_model_lab.py
```

Outputs:
- `reports/model_lab_results.csv`: every historical game, pick, winner, correctness and model edge.
- `reports/model_lab_summary.csv`: games, wins and accuracy by season plus total.

The backtest uses expanding pregame averages shifted by one game, preventing the current game's result from entering its own prediction.

If an enabled statistic is not currently present in `team_game_derived.parquet`, Model Lab prints it as unavailable and skips it rather than silently fabricating data. QBR, SOS and SOR are disabled in the starter config until their historical columns are wired into the curated dataset.
