# SPATS Model Lab Web UI

Run locally:

```bash
pip install -r requirements-web.txt
python scripts/build_database.py --start 2015 --end 2026
streamlit run app.py
```

The dashboard uses the same `scripts/backtest_model_lab.py` engine as the GitHub Action.

Features:
- Enable/disable model statistics
- Edit weights
- Select backtest season range
- Automatic weight normalization
- Hard failure if any enabled statistic is unavailable
- Overall and season-by-season accuracy
- Game-level results

The UI never enables QBR. Model configuration changes made in the web UI are temporary for that run and do not overwrite the repository's default configuration.
