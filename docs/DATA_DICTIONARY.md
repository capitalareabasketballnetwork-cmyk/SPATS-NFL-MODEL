# SPATS data dictionary

## Core keys
- `game_id`: nflverse game identifier
- `season`, `week`, `season_type`
- `team`, `opponent`
- player tables retain source player identifiers (GSIS/PFR/etc.) whenever supplied

## Team derived metrics
SPATS derives offense and corresponding defense-allowed versions of:
EPA/play, total EPA, success rate, yards/play, pass EPA, rush EPA, pass/rush success, early-down EPA, explosive pass rate (20+), explosive rush rate (10+), third/fourth-down conversion, red-zone TD rate, turnovers/drive, sack rate, QB-hit rate, completion rate, CPOE, air yards/attempt, YAC/completion, first-down rate, penalty rate/yards, plays/drive, punt rate, field-goal-attempt rate, WPA.

Each numeric team-game metric is converted into pregame last-3, last-5, last-8, season-to-date, and exponentially weighted form. Defensive metrics are generated from the opponent's offensive game record.

## Player data
The raw/curated player tables preserve all source columns rather than selecting a small subset. This allows SPATS to retain passing, rushing, receiving, kicking, fantasy, EPA/CPOE, air-yard/YAC and other fields supplied by nflverse. Advanced source tables are stored separately to avoid losing source-specific definitions.

## Advanced coverage
Coverage varies. Do not interpret missing advanced fields in early seasons as zero. The build writes observed min/max season and row/column counts to `reports/coverage.csv`.
