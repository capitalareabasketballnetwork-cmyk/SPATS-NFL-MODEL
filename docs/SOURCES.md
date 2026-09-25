# Sources and coverage policy

Primary source: nflverse/nflverse-data release assets.

SPATS uses release discovery rather than hard-coding every asset filename. This makes refreshes resilient to year-specific files and allows the source manifest to show which optional datasets were actually available.

Known source-dependent boundaries include:
- Play-by-play: 1999-present.
- Weekly rosters: 2002-present.
- FTN charting: 2022-present.
- Participation: source/method changes over time; pre-2023 data includes NGS-derived fields, later availability differs.
- Injury history has source gaps; missing seasons are not fabricated.
- NGS/PFR/snap-count/depth-chart coverage begins later than 1999 and differs by table.

Every build records failures for optional sources and continues. PBP is required.
