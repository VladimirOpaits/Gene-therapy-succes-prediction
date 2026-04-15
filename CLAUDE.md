# Clinical Trial Success Prediction — Oracle Bot

DONT PUT ANY COMMENTS UNLESS THE CODE IS REALLY DIFFICULT, NEVER PUT """ COMMENTS

## Project Goal

Build an **Oracle Bot** that predicts gene therapy trial success and finds mispriced positions on prediction markets.

**Core thesis:** Combine technical + business/research intelligence to generate alpha against market prices on Polymarket.

---

## Scope: Gene Therapy Only

**Focus:** Phase 2 INDUSTRY-funded gene therapy trials exclusively.
- Why: Narrow niche = less efficient market = easier to find alpha
- No other therapy types (immunotherapy, CAR-T, small molecules)

---

## Strategy

**Phase 1: Build ML Model (current)**
1. Collect large training dataset from ClinicalTrials.gov
2. Enrich with chemical + genetic + business intelligence
3. Train XGBoost on success patterns
4. Identify what predicts success vs failure

**Phase 2: Find Market Mispricing**
1. Match active trials to Polymarket markets (e.g., "Will FDA approve retatrutide by 2026?")
2. Compare model prediction vs market price
3. Identify edge: market underbets winners, overbets losers

**Phase 3: Production**
1. Real-time feed of new trials via API
2. Continuous prediction + market monitoring
3. Automated trade signals

---

## Data Architecture

```
ClinicalTrials.gov        →  Feature Engineering  →  XGBoost Model
━━━━━━━━━━━━━━━━━━━━━━━━      ━━━━━━━━━━━━━━━━      ━━━━━━━━━━━
- 17 completed Phase 2       - Chemical (MW, SMILES)  - Success
- 13 active Phase 2          - Genetic (target gene)  - Prediction
- Reasons for termination    - Business (sponsor,
                               risk factors, history)
```

### Current Datasets

| File | Trials | Use |
|------|--------|-----|
| `data/gene_therapy_phase2_completed.csv` | 17 | Training (success label: COMPLETED=1, TERMINATED=0) |
| `data/gene_therapy_phase2_private.csv` | 13 | Active leads (prediction targets) |

Success rate: **52.9%** (9 success, 8 failures) — good balance for classification.

### Key Fields

**From ClinicalTrials.gov:**
- `nct_id`, `title`, `status`, `drugs`, `sponsor`, `sponsor_class`, `phases`, `results_date`

**To extract (Phase 2):**
- `target_gene` — what gene is being targeted?
- `mechanism` — what's the delivery mechanism? (AAV, lipid nanoparticle, etc.)
- `termination_reason` — if failed, was it safety/efficacy/recruitment/funding?
- `risk_category` — classify failures into types
- `sponsor_reputation` — repeat player or first-timer?

---

## Modules

| Path | Purpose | Status |
|------|---------|--------|
| `src/dataharvest/fdaparser.py` | Fetch & flatten from ClinicalTrials.gov | Done |
| `src/dataharvest/aifilter.py` | LLM extraction: mechanism, target gene, risk factors | Next |
| `src/market/polymarket.py` | Polymarket API client (API key for future trading) | Ready |
| `src/models/xgboost.py` | Success prediction XGBoost | Next |
| `src/oracle/` | Pipeline orchestration | TBD |

---

## Dev Commands

```bash
source venv/bin/activate

python -m pytest tests/

ruff check src/

python -c "from src.dataharvest.fdaparser import FDA_PCh_Parser; p = FDA_PCh_Parser(); print(p.fetch_phase2_private_df(limit=5))"
```

---

## Key Decisions

- **Success label:** `COMPLETED` = success, `TERMINATED` = failure (binary classification)
- **Model:** XGBoost (fast, interpretable, tabular data)
- **Feature sources:** ClinicalTrials fields + LLM extraction + domain knowledge
- **Alpha generation:** NOT from technical features (everyone sees those) but from:
  - Why trials fail (safety vs efficacy vs recruitment)
  - Sponsor track record + patterns
  - Market sentiment vs biological reality
- **Polymarket:** Found markets like "Will FDA approve retatrutide by 2026?" (32% price)
  - Will match active trials to these markets
  - API key stored for future trading layer

---

## Next Steps

- [ ] Expand training data: broader gene therapy queries (CRISPR, mRNA, etc.) → 50-100+ trials
- [ ] LLM extraction: parse trial titles + descriptions → target_gene, mechanism, risk_factors
- [ ] Feature engineering: chemical (SMILES), genetic (pathway), business (sponsor history)
- [ ] XGBoost training: predict success on completed trials
- [ ] Backtesting: would model have beaten Polymarket prices?
- [ ] Find more Polymarket markets for gene therapy approvals
- [ ] Production pipeline: real-time trial feed + predictions