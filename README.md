# Credit for Thin-File Customers — Team 2

**Purple Rockets Hackathon · ADA Final Project**

Alexa Pereda · Diego Mellado · Emilio Macías · Emmanuel Santos · Jorge Siegrist · Luis Mendoza · Santiago Donoso

---

## Objective

Design an approach based on **behavioural signals** to extend credit approval to **thin-file** customers (no bureau history) who **are already Nu customers**, reducing operational friction (analyst hours, manual review) **without exceeding the ~2.6% delinquency guardrail** on the approved book.

The Machine Learning model (Random Forest) acts as a **laboratory**: it prioritises which variables to request in a **Phase 2** (payroll, secured card, deposits), not as a production-ready engine for deployment.

**North Star:** raise the thin-file approval rate (today ~**12.1%** vs ~**81.3%** for bureau-hit) while holding expected loss constant.

---

## Challenge

### Business problem

- **34.3%** of applications (~178k of 520k) are thin-file.
- **60.4%** of those thin-file applications (**107,768**) are already Nu customers — we have deposits, balances and payroll patterns; bureau silence does not mean we know nothing.
- Thin-file faces **54%** manual review vs **19%** for bureau-hit, and **32h** vs **3h** median latency.
- Only **12.1%** of thin-file is approved vs **81.3%** for bureau-hit.

### Analytical challenge

1. Train on the **bureau + policy** population (stable label).
2. Validate on the **pilot holdout** (riskier profiles that policy would have rejected).
3. Apply to **thin-file without bureau** (no label) to estimate operational volume.
4. Define a **band policy** (auto-approve / manual / auto-decline), not just a 0.5 threshold.
5. Communicate that the model's probabilities are a **ranking**, not a literal delinquency % (threshold calibration required).

### Data sources

| Resource | Location |
|---------|-----------|
| Base view | `usr.diego_mellado.v_application_base` |
| Clean table | `usr.diego_mellado.application_base_clean` |
| Summary tables (QuickSight) | `usr.santiagodonoso.ada_*` |
| Notebook | Databricks → `ada_hackathon` |
| Dashboard | [QuickSight — Operations & Risk Bands](https://us-east1.quicksight.aws.amazon.com/sn/account/nu-qs-prod/dashboards/252b5fdd-d2aa-4cfb-ac70-7af40ef8ad89) |
| Presentation | `Credit for Thin-File Customers — Team 2.pdf` |

---

## Key results

| Metric | Value | Notes |
|---------|-------|-------|
| AUC (pilot holdout) | **0.68** | Moderate separation; same set as eval |
| Top signals | payroll **26.5%**, secured card **23%**, deposits **14.4%** | Feature importance |
| Auto-approve band (score < **0.40**) | **27.9%** of apply (~**30k** of 107,768) | vs **12.1%** current approval (~**13k**) |
| Auto-approve band delinquency (pilot) | **1.9%** | < **2.6%** guardrail |
| Manual band (0.40 – 0.70) | ~**66.5%** | Phase 2 / analyst |
| Auto-decline band (score > **0.70**) | ~**5.5%** | Highest risk |
| Analyst hours saved (estimated) | **~2,047 h** | Automated decisions × average review minutes |
| FN in auto-approve | **7.7%** of pilot delinquents | % of **all** delinquents falling into the safe band |
| FP in auto-decline | **5.1%** of pilot goods | % of **all** good accounts flagged as high risk |

**Volume upside:** ~**+17k** applications eligible for auto-approval (+**~131%** vs the 12.1% status quo).

---

## Instructions

### Prerequisites

- Databricks access (`nubank-e2-general`)
- `SELECT` permission on `usr.diego_mellado.v_application_base` and write access to the summary-table schema
- Cluster with `%run` to toolbox UC and utils UC (notebook cells 0–1)

### Notebook execution order (`ada_hackathon`)

```text
1. Cells 0–1    → %run toolbox + utils
2. Cells 3–13   → EDA + Scala cleaning → saves application_base_clean
3. Cells 15–16  → feature_cols (14) and feature_cols_v2 (12)
4. Cell 18      → Train RF + prob_eval + AUC  ⚠️ use prob_eval, not pred_eval
5. Cell 19      → Correlations (after training)
6. Cell 20      → Feature importance
7. Cells 25–26  → Threshold sweep + matrix with THRESHOLD=0.70
8. Apply cell   → Thin-file segmentation + hours saved
9. Final cell   → Save ada_* tables to schema for QuickSight
```

### Important model rules

```python
# AUC and bands — always probabilities
prob_eval = model.predict_proba(eval_df[feature_cols_v2])[:, 1]

# Classification with a business threshold (do NOT use model.predict() = fixed 0.5)
pred_hard = (prob_eval >= THRESHOLD).astype(int)

# Three bands
LOW  = 0.40   # auto-approve (LOW score = less risk)
HIGH = 0.70   # auto-decline (HIGH score = more risk)
```

### Model populations

| Flag | Filter | n | Use |
|------|--------|---|-----|
| `is_train = 1` | Nu + bureau + policy + label | **181,151** | Train RF |
| `is_eval = 1` | pilot_holdout + label | **9,681** | AUC, KS, bands, 1.9% delinquency |
| `is_apply = 1` | thin_file = true | **107,768** | Operational projection |

### Saving tables for QuickSight

Run the export cell to the configured schema (default: `usr.santiagodonoso`):

| Table | Contents |
|-------|-----------|
| `ada_population_summary` | train / eval / apply sizes / delinquencies |
| `ada_threshold_config` | LOW, HIGH, guardrail, AUC |
| `ada_pilot_band_summary` | Pilot bands + delinquency + FP/FN % |
| `ada_low_threshold_sensitivity` | Delinquency sensitivity by LOW threshold |
| `ada_apply_band_summary` | Bands on thin-file apply |
| `ada_ops_savings` | Hours saved |
| `ada_feature_importance` | Feature ranking |

### Permissions (collaborators / QuickSight)

```sql
GRANT USE SCHEMA ON SCHEMA usr.santiagodonoso TO `user@nubank.com.mx`;
GRANT SELECT ON SCHEMA usr.santiagodonoso TO `user@nubank.com.mx`;
```

---

## Contents

### Presentation (8 slides + annex)

| # | Slide | Contents |
|---|-------|-----------|
| 01 | Context | Thin-file friction, 34%, 12.1% vs 81.3%, 107k Nu |
| 02 | Diagnostic | 520k, north star, Nu 63%, approval gap |
| 03 | Data prep | Joins, dedup, leakage, nulls |
| 04 | Other KPIs | Pilot 5.7% vs policy 2.5%, funnel |
| 05 | Test design | Hypothesis, 181k train, 9.6k eval, 12 features |
| 06 | Early model read | Importance → AUC → 40/70 bands → 28% auto / 1.9% delinquency / 2,047h |
| 07 | Definition of success | Defensible segment, less manual review |
| 08 | Outcomes & risks | Upside, correlational, thin-file sample |
| 09 | Next steps | Prospects, Phase 2, gradual rollout |
| — | Annex | QuickSight dashboards |

### Notebook `ada_hackathon` (Databricks)

| Section | Cells | Language | Description |
|---------|--------|----------|-------------|
| Setup | 0–1 | — | Toolbox UC |
| Data exploration | 3–13 | Scala | EDA, nulls, pilot vs policy, cleaning + flags |
| Model improvement | 15–16 | Python | Features v1 (14) and v2 (12) post-correlation |
| Random Forest | 18–20 | Python | Train, AUC, correlations, importance |
| Evaluation | 21–26 | Python | Matrix, threshold sweep, 0.40 / 0.70 bands |
| Apply + export | — | Python | Thin-file scoring, hours, QuickSight tables |

### Model variables (`feature_cols_v2`, 12)

1. `payroll_deposit_flag` (~26.5%)
2. `secured_card_utilization_pct` (~23%)
3. `deposits_90d_count` (~14.4%)
4. `balance_avg_90d_mxn`
5. `sessions_30d`
6. `avg_session_minutes`
7. `p2p_inbound_90d_count`
8. `bill_payments_90d_count`
9. `days_since_last_login`
10. `declared_income_mxn`
11. `age`
12. `months_with_nu`

*Excluded for correlation > 0.7:* `deposits_90d_amount_mxn`, `has_secured_card`

### Team artefacts

| Artefact | Path / link |
|-----------|-------------|
| Notebook | `/Users/santiago.donoso@nubank.com.br/ada_hackathon` |
| Clean table | `usr.diego_mellado.application_base_clean` |
| Dashboard tables | `usr.santiagodonoso.ada_*` |
| PDF presentation | `Credit for Thin-File Customers — Team 2` |
| Brief / data dictionary | `Downloads/README.md` (Challenge 2) |

---

## Limitations (read before presenting)

1. **Synthetic data** — recommendations concern methodology and test design, not direct deployment.
2. **Train with bureau, apply without bureau** — transfer is assumed; delinquency validated on the pilot, not on apply.
3. **Uncalibrated probabilities** — `class_weight=balanced` inflates scores; the 0.40 / 0.70 thresholds are policy, not sklearn defaults.
4. **28% auto-approve ≠ 28% total approval** — the ~66% middle band requires Phase 2 (payroll, income) or an analyst.

---

## References

- PRD / data spec: `Downloads/README.md` — Challenge 2 · The test before the model
- Challenge PDF: `Equipo 2 · Crédito para quienes no tienen historial.pdf`
