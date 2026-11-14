# Swivver Algorithm (`v1.0.0`)

Open-source recommendation, candidate ranking, diversity constraints, online Team-Draft Interleaving, and validation tooling for Swivver. Swivver builds [source-linked short-form news](https://swivver.ai/product): a short video update connected to a written article, inspectable sources, and a visible change history.

## Package & Contract Surface (`v1.0.0`)

| Surface | Path | Implemented scope (`v1.0.0`) |
| --- | --- | --- |
| In-memory Go ranker | [`pkg/ranker/`](pkg/ranker) | Deterministic scoring (`swivver-ranker/1.0.0`), fail-closed policy-gate defense-in-depth, uncertainty-neutral Signal handling, sliding-window + MMR diversity, `D01`/`D03` session pacing advisories, plain-language `WhyThis`, cursor pagination (`98.4%` `-race` statement coverage) |
| Online interleaving & attribution | [`pkg/interleave/`](pkg/interleave) | Deterministic, seed-reproducible Team-Draft Interleaving (`Interleave`), bilateral policy-sandwich enforcement (`0.00%` policy leak), opaque HMAC-SHA256 signed `AttributionMap` tokens, idempotent `M01` deliberative usefulness credit attribution (`ScoreOutcomes`, `AggregateStudy`), exact two-sided sign test, and 2,000-resample percentile bootstrap 95% confidence bounds (`100.0%` `-race` statement coverage) |
| Wire contracts | [`schemas/`](schemas) | Draft 2020-12 JSON Schemas: `scoped-signal-features-v1`, `candidate-feature-vector-v1`, `ranked-feed-result-v1` (`additionalProperties: false`) |
| Evaluation & 5k validation suite | [`eval/`](eval) | Zero-dependency Python 3.11+ offline metrics (`NDCG@k`, author Gini, language parity, counterfactual `unknown` rank-delta), 5,000-package stratified `VG-3`/`VG-5` benchmark (`validate_5k_corpus.py`), Draft 2020-12 schema validator, and AST/source clean-room boundary scanner |
| Architecture decision | [`docs/adr/0001-clean-room-scoped-signal-contract.md`](docs/adr/0001-clean-room-scoped-signal-contract.md) | Accepted (`A2` pure-Go library + JSON Schema contracts) |
| Platform backend integration | External | **Separate repository boundary**: the platform backend enforces pre- and post-ranking policy checks and calls this module in-process; this repository alone does not establish a deployed production feed |

## Design Invariants

- **Policy precedes personalization ("policy sandwich"):** Publication state, legal/regional rules, rights, moderation enforcements, and safety blocks are enforced before and after ranking by the caller. As defense-in-depth, both `pkg/ranker` and `pkg/interleave` hard-drop any candidate with `Eligible: false`, `PolicyBlocked: true`, or asymmetric arm eligibility (`0.00%` exposure).
- **Uncertainty neutrality:** Optional `ScopedSignalFeatures` carry explicit availability (`available`, `unknown`, `unsupported`, `not_applicable`, `stale`). Missing or non-available observations produce an exact `0.0` Signal delta and `0` rank demotions relative to cold-start priors.
- **Anti-engagement-trap & bot immunity:** Deliberative reader depth (`article_open_rate`, `source_inspect_rate`, `save_rate`) outweighs passive video completion (`valid_watch_ratio`). When `anomaly_flagged: true`, engagement and cold-start exploration boosts are discounted to `0.0`.
- **Opaque online experimentation:** `pkg/interleave` encodes per-slot `TeamA`/`TeamB` assignments inside an HMAC-SHA256 authenticated token so client surfaces receive standard user-safe `RankedItem` records with zero leaked experiment internals.
- **Inspectable reasons (`WhyThis`):** Every ranked item returns at most 2 plain-language reasons without leaking internal identifiers, dimension codes, or raw floating-point scores.
- **Clean-room open-source scope:** Zero external Go or Python dependencies, no editorial merit or author-credibility scoring, and no universal quality oracle score.

For reader product context see [Swivver for readers](https://swivver.ai/for-readers) and contributor guidelines in [AGENTS.md](AGENTS.md).

## Measured Validation Results (`VG-3` / `VG-5` Suite)

Reproducible across both Go (`pkg/ranker/corpus_validation_test.go`) and Python (`eval/validate_5k_corpus.py`) over the deterministic 5,000-package stratified multilingual corpus (`3,500` news packages + `1,500` standalone/partner media packages across `en`, `pt`, and `es`):

| Validation gate | Target threshold | Measured result (Go & Python) |
| --- | --- | --- |
| **1. Policy-sandwich fuzzing (`10,000` trials)** | `0.00%` (`0 / 10,000`) ineligible/blocked leak | **`0.00%`** (`0 / 10,000` trials; `0 / 68,159` adversarial maximal-score ineligible items exposed) |
| **2. Counterfactual uncertainty neutrality (`5,000` sweep)** | `SignalDelta == 0.0` (`< 1e-12`), `0` rank demotions | **`0.0` max `|SignalDelta|`** across `25,000` state checks (`nil`, `unknown`, `unsupported`, `not_applicable`, `stale`); **`0 / 5,000` rank demotions** |
| **3. Contradiction & ragebait suppression** | `>= 85%` top-10 drop vs. pure watch-time sort | **`100.00%` reduction** (`2,669 / 5,000` [`53.38%`] watch-sort top-10 ragebait slots reduced to `0 / 5,000` [`0.00%`]) |
| **4. Bot / brigading immunity** | `100%` elimination of `anomaly_flagged` boosts | **`100.0%`** (`350 / 350` brigaded items stripped of engagement and cold-start boosts) |
| **5. Author fairness & minority-language parity** | Within `+- 3%` (`<= 0.03`) of eligible supply | Author Gini delta **`-0.0027`** (`0.2291` vs. `0.2318` supply); `pt + es` share delta **`-0.0050`** (`0.3950` vs. `0.4000` supply; `pt` `-0.0012`, `es` `-0.0038`) |
| **6. 200-candidate ranking latency** | `p95 <= 25.0ms` | Go **`p50 = 0.09ms`**, **`p95 = 0.20ms`** (`65 µs/op` benchmark; `2.91ms` `p95` under `-race`); Python reference **`p95 = 0.82ms`** |

## Running Tests and Validation

Requirements: Go `1.25+` and Python `3.11+` (standard library only).

```sh
gofmt -s -w .
go vet ./...
go test -v -race -cover ./...
python3.11 -m unittest discover -s eval -v
python3.11 eval/evaluate_offline.py
python3.11 eval/validate_5k_corpus.py
```

## License

Apache-2.0. See [LICENSE](LICENSE).
