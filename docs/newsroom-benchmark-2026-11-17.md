# Multi-Beat Newsroom Simulation, Low-Polish Fairness & Interleaving Benchmark (`2026-11-17`)

- **Date:** 2026-11-17
- **Repository:** `https://github.com/Swivver/swivver_algorithm` (`Apache-2.0`, zero external dependencies)
- **Evaluated model:** `swivver-ranker/1.0.0` (`pkg/ranker`, `pkg/interleave`, `eval/evaluate_newsroom_simulation.py`)
- **Deterministic seeds:** `corpus_seed = 20261117`, `interleave_outcome_seed = 20261118`

## 1. Benchmark Objective & Clean-Room Scope

This benchmark evaluates whether Swivver's clean-room recommendation engine (`pkg/ranker` and `pkg/interleave`) upholds four core discovery commitments across a realistic multi-beat, multilingual (`en`, `pt`, `es`), multi-region newsroom simulation using **only** versioned `CandidateFeatureVector` and `ScopedSignalFeatures` wire contracts:

1. **Plain-language `why_this` auditability:** Surface at most two concise, human-readable reasons per item (followed writer/publication, single or multi-beat topic match, regional relevance, corroboration density across independent source clusters, reader article/source depth, reading language match, or timely developing update) without leaking internal identifiers or raw scores.
2. **Low-production-polish journalism fairness (*"Production polish is not the same as human value"*):** Ensure corroborated handheld field reporting with noisy street/weather capture (`technical_legibility_tier = "degraded"`), urgent camera movement, and ordinary unsigned phone capture (`provenance_status = "missing"`) is never crushed behind studio-polished packages with equivalent corroboration.
3. **Uncertainty-calibrated suppression of unsourced attention-bait:** Demote slickly edited high-arousal or bait-and-switch clips **only** when zero-source status (`independent_source_clusters = 0` without `supported` evidence alignment) or active claim/consistency dispute is explicitly `available`, while maintaining an exact `0.0` `signal_delta` for `nil`, `unknown`, `unsupported`, `not_applicable`, and `stale` observations.
4. **Multi-beat session diversity & Team-Draft Interleaving sensitivity:** Prevent consecutive back-to-back author or micro-incident streaks when diverse reporting is available, and verify statistically significant (`p < 0.01`) deliberative usefulness gains over a naive engagement-only ranker via deterministic Team-Draft Interleaving.

## 2. Simulated Newsroom Corpus & Reader Cohort

The harness (`eval/evaluate_newsroom_simulation.py`) generates `1,200` candidate packages (`200` matched time-slots across `6` creator archetypes and `36` creators in `en`, `pt`, and `es`, with `4%` policy-ineligible items injected to verify fail-closed `0.00%` policy-sandwich exposure) and evaluates `2,000` reader sessions (`48`-candidate retrieval pools, `Top-10` feeds):

| Archetype | Beats & Micro-Incidents | Scoped Signal & Capture Profile | Reader Behavior Profile |
| --- | --- | --- | --- |
| **1. Investigative Infrastructure** | `infrastructure`, `energy`, `transit`, `civic` | Studio/desk polish (`clear`, `verified`/`declared`), `supported`, `2–4` independent source clusters | Moderate watch (`0.47–0.76`), high article (`0.27–0.49`) & source (`0.17–0.36`) depth |
| **2. Maritime/Climate Science** | `climate`, `maritime`, `science`, `economy` | Studio/data polish (`clear`, `verified`/`declared`), `supported`, `2–4` independent source clusters | Moderate watch (`0.47–0.76`), high article & source inspection depth |
| **3. Public Health Desk** | `health`, `science`, `civic`, `policy` | `86%` full `supported` studio polish + `14%` `unsupported` partial coverage (uncertainty neutrality check) | Moderate watch (`0.47–0.76`), high article & source inspection depth |
| **4. Regional Breaking Low-Polish Field Reporter** | `civic`, `transit`, `infrastructure`, `climate` | Handheld field capture (`degraded` legibility, `missing` provenance, `0.54–0.81` urgent camera pacing, low bait risk `<= 0.13`), `supported` (`2–4` clusters) + `14%` cold-start `nil`/`unknown` dispatches | Matched deliberative engagement (`0.27–0.49` article open, `0.17–0.36` source inspect) |
| **5. Tech/AI Policy Analyst** | `technology`, `policy`, `economy`, `science` | Studio/desk polish (`clear`, `verified`/`declared`), `supported`, `2–4` independent source clusters | Moderate watch (`0.47–0.76`), high article & source inspection depth |
| **6. High-Arousal Unsourced Bait** | Broad popular tags (`economy`, `technology`, `health`, `climate`, `civic`, `infrastructure`) | Slick studio editing (`clear` legibility), `0` independent sources, high-arousal jolts (`0.78–0.99`), bait-switch risk (`0.58–0.96`), disputed/mixed alignment + `20%` `anomaly_flagged` velocity spikes | Inflated passive watch completion (`0.89–0.995`), near-zero article/source depth, high post-click rapid skips |

## 3. Measured Ranking, Diversity & Fairness Results (`2,000` Sessions)

We compare three deterministic rankers over identical `2,000` reader sessions:
1. **`swivver-ranker/1.0.0` (Signal-Informed Multi-Beat Ranker):** Full clean-room ranker with corroborated low-polish protection, uncertainty-calibrated zero-source sensationalism demotion, and 5-tier non-consecutive author/incident diversity.
2. **`naive-engagement-ranker/1.0` (Naive Engagement-Only Baseline):** Watch-ratio-dominated engagement ranking without Signal observations or streak caps.
3. **`studio-polish-ablation/1.0` (Single-Factor Studio-Polish Bias Ablation):** Identical diversity engine, but penalizes `degraded` field legibility (`-0.09`) and `missing` provenance (`-0.03`) while rewarding studio `clear` + `verified` capture.

| Metric | Signal-Informed Ranker (`swivver-ranker/1.0.0`) | Naive Engagement-Only (`naive-engagement-ranker/1.0`) | Studio-Polish-Biased Ablation |
| --- | ---: | ---: | ---: |
| `NDCG@5` | **0.9182** | `0.0058` | `0.8786` |
| `NDCG@10` | **0.9607** | `0.1196` | `0.9278` |
| `MRR` (first Grade-3 deliberative match) | **0.9934** | `0.0787` | `0.9905` |
| `Topic Diversity@10` (distinct beats / tags) | **`6.75` beats / `16.01` tags** | `6.46` beats / `12.88` tags | `6.98` beats / `16.11` tags |
| Max consecutive same-author streak | **`1`** | `2` | `1` |
| **Low-Polish Journalism Fairness Gap** (`\|mean_rank(low_polish) - mean_rank(studio_polish)\|`) | **`0.3634`** (`18.81` vs `19.17`) | `1.0029` (`28.11` vs `27.11`) | `10.9671` (`28.08` vs `17.11`) |
| Top-5 unsourced attention-bait exposure rate | **`0.00%`** (`0 / 10,000`) | `98.95%` (`9,895 / 10,000`) | `0.00%` (`0 / 10,000`) |
| **Top-5 Unsourced Attention-Bait Suppression Rate** | **`100.00%`** | baseline | — |
| Policy-sandwich ineligible leaks (`!eligible` or `policy_blocked`) | **`0`** (`0.00%`) | `0` (`0.00%`) | `0` (`0.00%`) |

### Per-Archetype Exposure & Pool Rank Breakdown (`swivver-ranker/1.0.0`)

| Archetype | Capture & Evidence Group | Top-5 Exposure Share | Top-10 Exposure Share | Mean Pool Rank (`/ 46`) | Mean `signal_delta` |
| --- | --- | ---: | ---: | ---: | ---: |
| Investigative Infrastructure | `studio_polish` | `22.29%` | `22.20%` | `18.89` | `+0.1800` |
| Maritime/Climate Science | `studio_polish` | `18.60%` | `20.20%` | `19.30` | `+0.1800` |
| Public Health Desk | `studio_polish` (+ `14%` `unsupported`) | `19.20%` | `18.34%` | `20.85` | `+0.1537` |
| **Regional Breaking Low-Polish Field Reporter** | **`low_polish`** (+ `14%` cold-start) | **`18.48%`** | **`18.52%`** | **`20.46`** | **`+0.1537`** |
| Tech/AI Policy Analyst | `studio_polish` | `21.43%` | `20.74%` | `19.19` | `+0.1800` |
| High-Arousal Unsourced Bait | `attention_bait` (`clear` polish, `0` sources) | **`0.00%`** | **`0.00%`** | `42.57` | `-0.3312` |

Key observations:
- **96.7% fairness-gap reduction for field reporters:** Waiving degraded-legibility and urgent-camera penalties when reporting is corroborated reduces `|mean_rank(corroborated_low_polish) - mean_rank(corroborated_studio_polish)|` from **`10.9671` ranks** (under studio-polish bias) down to **`0.3634` ranks** (`18.8103` vs `19.1737` across 46-item pools), giving Regional Breaking Field Reporters parity in Top-10 feed share (`18.52%` vs `18.34%` Public Health Desk).
- **100.00% Top-5 unsourced bait suppression despite slick studio polish:** Even though every `High-Arousal Unsourced Bait` item carries `technical_legibility_tier = "clear"` and `0.89–0.995` passive watch ratios, deliberative engagement weighting plus uncertainty-calibrated zero-source sensationalism demotion (`mean signal_delta = -0.3312`) eliminates `9,895 / 9,895` (`100.00%`) Top-5 bait exposures.

## 4. Team-Draft Interleaving Study (`2,000` Sessions)

Using deterministic Team-Draft Interleaving (Radlinski, Kurup & Joachims, 2008) matching `pkg/interleave`, Arm A (`naive-engagement-ranker/1.0`) and Arm B (`swivver-ranker/1.0.0`) draft 10-item interleaved feeds across all `2,000` sessions with position-decayed reader examination (`1 / log2(rank + 1)`) and deliberative outcome attribution (`+1.0` completed read/watch, `+1.0` source inspection, `+1.5` positive usefulness, `-1.0` rapid skip, `-2.0` correction impression):

| Study Metric | Measured Value |
| --- | ---: |
| Sessions evaluated (`PageSize = 10`) | `2,000` |
| `swivver-ranker/1.0.0` session wins / Naive wins / Ties | **`1,983`** / `2` / `15` |
| Decisive session win-rate (excluding ties) | **`99.90%`** (`1,983 / 1,985`) |
| Overall session win-rate (all sessions) | **`99.15%`** (`1,983 / 2,000`) |
| Mean deliberative usefulness gain per session | **`+6.4023`** |
| 2,000-resample percentile bootstrap `95%` CI | **`[+6.2683, +6.5382]`** |
| Post-exposure correction impressions (`Signal` vs `Naive`) | **`0`** vs `878` |
| Exact two-sided binomial sign-test `p`-value | **`< 1e-100`** (`10^-590.9`, `p < 0.01` **PASS**) |

## 5. Reproducible JSON Summary & CLI Commands

Reproduce the full JSON + Markdown benchmark report and Go interleaving verification from the repository root:

```sh
go test -race -cover ./...
python3.11 eval/evaluate_newsroom_simulation.py --markdown
```

Excerpt of deterministic JSON output (`seed = 20261117`):

```json
{
  "passed": true,
  "benchmark_date": "2026-11-17",
  "seed": 20261117,
  "sessions_simulated": 2000,
  "corpus_size": 1200,
  "archetypes_simulated": 6,
  "policy_sandwich_leaks": 0,
  "low_polish_journalism_fairness": {
    "signal_ranker_gap": 0.3634,
    "signal_mean_rank_low_polish": 18.8103,
    "signal_mean_rank_studio_polish": 19.1737,
    "polish_biased_ablation_gap": 10.9671,
    "gap_reduction_ratio": 0.966865
  },
  "unsourced_bait_suppression_top5": {
    "naive_top5_exposures": 9895,
    "naive_top5_rate": 0.9895,
    "signal_top5_exposures": 0,
    "signal_top5_rate": 0.0,
    "suppression_rate": 1.0
  },
  "team_draft_interleaving": {
    "sessions_evaluated": 2000,
    "session_wins_signal": 1983,
    "session_wins_naive": 2,
    "session_ties": 15,
    "signal_win_rate_decisive": 0.998992,
    "signal_win_rate_all_sessions": 0.9915,
    "mean_usefulness_gain_per_session": 6.40225,
    "bootstrap_95_lcb": 6.26825,
    "bootstrap_95_ucb": 6.53825,
    "sign_test_p_value": 0.0,
    "sign_test_log10_p_value": -590.9488,
    "passed_significance_gate": true
  }
}
```
