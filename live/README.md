# SPATS Live

Fast operational layer for weekly SPATS queries.

## Contract
- SPATS V1 is frozen. Live never optimizes or retrains feature selection.
- Historical research is isolated in the manual SPATS Research Rebuild workflow.
- Live refresh downloads only the current-season PBP plus schedules.
- A compact historical seed preserves rolling/EWM continuity without rebuilding 1999-present.
- Predictions are snapshotted to `live/season_predictions.csv`; completed rows are never silently rewritten.

## Query files
- `live/current_team_profiles.csv` — current 32-team values for the 11 SPATS inputs.
- `live/upcoming_predictions.csv` — next scheduled week and probabilities.
- `live/season_predictions.csv` — archived SPATS predictions/results.
- `live/status.json` — freshness and model version.
- `live/model/spats_v1.json` — frozen inference parameters.
- `live/cache/team_history_seed.csv` — compact pre-2026 rolling-history seed.

The slow research workflow should only be run intentionally when developing a future model version.
