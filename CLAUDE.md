# Clinical Trial Success Prediction — Oracle Bot

DONT PUT ANY COMMENTS UNLESS THE CODE IS REALLY DIFFICULT, NEVER PUT """ COMMENTS
## Project Goal

Build an **Oracle Bot** that:
1. Estimates the true biological probability of a clinical trial succeeding
2. Compares that estimate against market prices on **Polymarket**
3. Identifies mispriced positions (edge) and flags them for betting

Secondary use case: risk assessment tools for biopharma labs.

---

## Architecture (evolving)

```
Data Layer          →  Scoring Layer       →  Market Layer
────────────────       ─────────────────      ──────────────────
ClinicalTrials.gov     ML success model       Polymarket API
PubChem                AI signal filter       Edge calculator
(future: SEC, WHO)     Calibration            Alert / report
```

### Modules

| Path | Purpose | Status |
|------|---------|--------|
| `src/dataharvest/fdaparser.py` | Fetch & flatten trials from ClinicalTrials.gov v2 API + PubChem drug descriptions | Done |
| `src/dataharvest/aifilter.py` | AI-powered signal filtering / enrichment | In progress |
| `src/models/` | Success prediction models | Not started |
| `src/market/` | Polymarket integration, edge calculation | Not started |
| `src/oracle/` | Orchestration: pipeline end-to-end | Not started |

---

## Data Sources

- **ClinicalTrials.gov API v2** — `https://clinicaltrials.gov/api/v2/studies`
  - Training data: `COMPLETED | TERMINATED | WITHDRAWN` trials
  - Oracle leads: `RECRUITING | ACTIVE_NOT_RECRUITING` trials
- **PubChem** — drug descriptions via compound name lookup
- **Polymarket** — prediction market prices (integration TBD)

### Key fields used from ClinicalTrials.gov
`NCTId, OfficialTitle, OverallStatus, LeadSponsorName, LeadSponsorClass, Phase, InterventionName, PrimaryCompletionDate, ResultsFirstPostDate`

---

## Dev Commands

```bash
# Activate environment
source venv/bin/activate

# Run tests
python -m pytest tests/

# Lint
ruff check src/

# Quick pipeline smoke-test
python -c "from src.dataharvest.fdaparser import FDA_PCh_Parser; p = FDA_PCh_Parser(); print(p.fetch_training_data_df(limit=5))"
```

---

## Key Decisions & Conventions

- **Python 3.13**, venv at `venv/`
- **pandas** DataFrames as the standard data container between pipeline stages
- `FDA_PCh_Parser` is the entry point for all raw data fetching — do not bypass it
- Enriched/intermediate data lives in `src/dataharvest/` as CSV for now; will move to a DB later
- Phase classification: treat `PHASE1`, `PHASE2`, `PHASE3` as ordinal — higher phase = higher prior probability of success
- Sponsor class (`NIH`, `INDUSTRY`, `OTHER`) is a meaningful feature — keep it in all DataFrames
- `aifilter.py` will use an LLM to extract signal from free-text fields (title, drug description) — keep this separate from the deterministic parser

---

## What NOT to Do

- Don't fetch all fields from the API — the current field list is intentional to keep payloads small
- Don't remove `sponsor_class` from flattened data — it's a key predictor
- Don't hardcode queries — `query` param should always be configurable

---

## Open Questions / Next Steps

- [ ] Define success label: `COMPLETED` with results = success? Or need outcome data?
- [ ] Decide on model type: gradient boosting (tabular) vs. LLM-based scoring
- [ ] Polymarket API: REST or websocket? Which markets map to which NCT IDs?
- [ ] How to match Polymarket questions → ClinicalTrials.gov NCT IDs (likely needs NLP)
- [ ] Calibration strategy: how to convert model score → probability estimate