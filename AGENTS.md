# Working in `swivver_algorithm`

`swivver_algorithm` is Swivver's **public, Apache-2.0** repository for explainable recommendation, candidate ranking, diversity constraints, and offline evaluation (`https://github.com/Swivver/swivver_algorithm`).

Every change in this repository must preserve the clean-room, policy, and neutrality invariants below.

## Non-negotiable invariants

1. **Public Apache-2.0 clean-room boundary**
   - Keep zero imports, prompts, editorial rubrics, model checkpoints, calibration weights, SQL queries, or private source code from Swivver's private repositories.
   - Do not implement editorial/informational merit scoring, truth arbitration, or author-credibility ranking.
   - Never accept or compute a single "universal quality oracle" score.
   - Keep machine-specific paths, credentials, internal hostnames, and private dataset samples out of all repository files.

2. **Policy sandwich ownership (defense-in-depth)**
   - Platform eligibility (publication state, legal/regional availability, rights, moderation enforcements, and strict safety blocks) is owned and enforced by the caller **before** and **after** ranking.
   - As defense-in-depth, `pkg/ranker` **must hard-drop** (`0%` exposure) any candidate with `Eligible: false`, `PolicyBlocked: true`, or an empty `CandidateID` regardless of preference match, engagement, or Signal attributes.

3. **Uncertainty neutrality**
   - Optional `ScopedSignalFeatures` (and each individual dimension) carry an explicit `availability` status: `available`, `unknown`, `unsupported`, `not_applicable`, or `stale`.
   - Whenever `ScopedSignalFeatures` is absent or a dimension's status is `unknown`, `unsupported`, `not_applicable`, or `stale`, the ranker **must** apply an exact `0.0` Signal delta relative to the unassessed baseline and **must never** penalize or demote the candidate.
   - Fresh unassessed candidates may receive a bounded, configurable cold-start exploration bonus; adding `unknown`, `unsupported`, `not_applicable`, or `stale` observations must never reduce a candidate's score below its unassessed baseline.

4. **Anti-engagement-trap & bot immunity**
   - Prioritize deliberative reader actions (`article_open_rate`, `source_inspect_rate`, `save_rate`) over passive watch ratio (`valid_watch_ratio`).
   - When `anomaly_flagged` is `true`, engagement rollup terms must be discounted to `0.0` and never rewarded.

5. **Inspectable user-facing reasons**
   - `WhyThis` explanations must return at most 2 concise, plain-language reasons and must never leak internal IDs, raw floating-point scores, or internal dimension codes into user-facing copy.

## Architecture & repository layout

| Path | Responsibility |
| --- | --- |
| `go.mod` | Pure-Go module (`github.com/Swivver/swivver_algorithm`, standard library only) |
| `pkg/ranker/` | Deterministic, thread-safe in-memory candidate ranker, diversity re-ranker, pacing advisories, cursor pagination, and plain-language explanations |
| `schemas/` | Draft 2020-12 JSON Schemas (`scoped-signal-features-v1`, `candidate-feature-vector-v1`, `ranked-feed-result-v1`) |
| `eval/` | Standard-library Python offline evaluation metrics (`NDCG@k`, author Gini, language parity, counterfactual `unknown` rank-delta), schema checker, and clean-room scanner |
| `docs/adr/` | Architecture Decision Records |

**Implementation vs. integration status:** This repository provides the local `v0.1` Go `pkg/ranker` library, versioned JSON Schemas, and offline evaluation suite. Cross-repository adapter wiring into the platform backend feed (`M3`) is separate and not established by files in this repository alone.

## Validation commands

Run all checks from the root of `swivver_algorithm`:

```sh
gofmt -s -w .
go vet ./...
go test -race -cover ./...
python3 -m unittest discover -s eval -v
python3 eval/evaluate_offline.py
```

### Quality gates

- Statement coverage in `pkg/ranker` must remain **`>= 95%`**.
- Ranking 200 eligible candidates must remain deterministic, allocation-bounded, thread-safe under `-race`, and complete in **`< 5ms`**.
- `python3 eval/evaluate_offline.py` must verify schema validity, counterfactual `unknown` non-demotion, and zero clean-room boundary violations.
