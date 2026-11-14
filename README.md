# Swivver Algorithm

Open-source recommendation, candidate ranking, diversity constraints, and offline evaluation for Swivver. Swivver builds [source-linked short-form news](https://swivver.ai/product): a short video update connected to a written article, inspectable sources, and a visible change history.

## Status (`v0.1` clean-room library & contracts)

| Surface | Path | Current state |
| --- | --- | --- |
| In-memory Go ranker | [`pkg/ranker/`](pkg/ranker) | Implemented (`v0.1`): deterministic scoring, policy-gate defense-in-depth, uncertainty-neutral Signal handling, sliding-window/MMR diversity, `D01`/`D03` pacing advisories, plain-language `WhyThis`, cursor pagination |
| Wire contracts | [`schemas/`](schemas) | Implemented (Draft 2020-12 JSON Schemas): `scoped-signal-features-v1`, `candidate-feature-vector-v1`, `ranked-feed-result-v1` |
| Offline evaluation & boundary suite | [`eval/`](eval) | Implemented (Python 3 stdlib): `NDCG@k`, author Gini diversity, minority-language parity, counterfactual `unknown` rank-delta, schema validation, clean-room AST/source scanner |
| Architecture decision | [`docs/adr/0001-clean-room-scoped-signal-contract.md`](docs/adr/0001-clean-room-scoped-signal-contract.md) | Accepted (`A2` pure-Go library + JSON Schema contracts) |
| Platform backend feed integration | External (`M3`) | **Not yet wired**: the platform backend currently runs its own feed logic; this repository alone does not establish a deployed production feed |

## Design invariants

- **Policy precedes personalization ("policy sandwich"):** Publication state, legal/regional rules, rights, moderation enforcements, and safety blocks are enforced before and after ranking by the caller. As defense-in-depth, `pkg/ranker` drops any candidate with `Eligible: false` or `PolicyBlocked: true` (`0%` exposure).
- **Uncertainty neutrality:** Optional `ScopedSignalFeatures` carry explicit availability (`available`, `unknown`, `unsupported`, `not_applicable`, `stale`). Non-available or missing observations produce an exact `0.0` Signal delta and never demote a candidate relative to an unassessed baseline; fresh unassessed stories may receive a bounded cold-start discovery envelope.
- **Anti-engagement-trap weighting:** Deliberative depth (`article_open_rate`, `source_inspect_rate`, `save_rate`) outweighs passive video completion (`valid_watch_ratio`). Engagement flagged as anomalous (`anomaly_flagged: true`) is discounted to `0.0`.
- **Inspectable reasons (`WhyThis`):** Every ranked item returns at most 2 plain-language reasons explaining why it was surfaced without leaking internal identifiers or raw scores.
- **Clean-room open-source scope:** Zero external Go or Python dependencies, no editorial merit or author-credibility scoring, and no universal quality oracle score.

For reader product context see [Swivver for readers](https://swivver.ai/for-readers) and contributor guidelines in [AGENTS.md](AGENTS.md).

## Running tests and evaluation

Requirements: Go `1.25+` and Python `3.11+` (standard library only).

```sh
gofmt -s -w .
go vet ./...
go test -race -cover ./...
python3 -m unittest discover -s eval -v
python3 eval/evaluate_offline.py
```

## License

Apache-2.0. See [LICENSE](LICENSE).
