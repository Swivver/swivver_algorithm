# ADR 0001: Clean-Room Go Library and Scoped Signal Contract (`A2`)

- **Date:** 2026-11-14
- **Status:** Accepted (M0 library, schemas, and offline evaluation; platform feed adapter integration deferred to M3)
- **License boundary:** Apache-2.0 (`https://github.com/Swivver/swivver_algorithm`)

## Context

Swivver's recommendation and discovery layer must rank eligible Story candidates transparently while upholding four strict architectural constraints:

1. **Clean-room open-source scope:** `swivver_algorithm` is a public Apache-2.0 repository. It must not import code, prompts, model weights, or private assessment schemas from private platform or media-analysis repositories, and it must never compute editorial merit or author-credibility scores.
2. **Policy precedence ("policy sandwich"):** Publication state, legal/regional rules, rights, moderation enforcements, and safety blocks are hard constraints owned by the platform backend before and after ranking. Ranking weights must never rescue an ineligible item.
3. **Uncertainty neutrality:** Media and source-alignment observations are optional, asynchronous, and partial. Newly published stories, unsupported languages/modalities, or stale snapshots frequently carry `unknown`, `unsupported`, `not_applicable`, or `stale` statuses. Penalizing missing coverage would suppress breaking reporting and minority-language creators.
4. **Anti-engagement-trap alignment:** Optimizing for raw short-video completion rewards sensory jolts (`D01`), bait-and-switch hooks (`D02`), and compulsive scrolling (`D03`) over reading the connected article and inspecting sources.

### Options considered

| Option | Description | Trade-off |
| --- | --- | --- |
| `A0`: SQL-only feed ordering | Keep all scoring inside backend SQL queries | Hard to test counterfactual invariants, MMR diversity windows, explainable reason selection, and offline parity metrics cleanly in open source |
| `A1`: Networked ranking service | Deploy a separate HTTP/gRPC Python or Go ranking service | Adds serialization overhead, tail-latency failure modes, auth/operational surface, and deployment coupling before learned models are warranted |
| `A2`: Pure-Go library + JSON Schema contracts (**Chosen**) | Provide zero-dependency `pkg/ranker` Go package alongside Draft 2020-12 JSON Schemas (`schemas/`) and a standard-library Python offline evaluator (`eval/`) | Sub-millisecond in-process execution (`< 5ms` for 200 candidates), zero network failure modes, strict clean-room decoupling, and language-neutral JSON contracts for audit/replay |

## Decision

We adopt **Option `A2`**:

1. **Pure-Go ranking engine (`pkg/ranker`):** Standard-library-only Go package (`github.com/Swivver/swivver_algorithm/pkg/ranker`) with deterministic inputs, explicit reference clock `Now`, zero mutable global state, and thread-safe execution.
2. **Versioned Draft 2020-12 JSON Schemas (`schemas/`):**
   - `scoped-signal-features-v1.schema.json` (`scoped-signal-features/1.0`): normalized, opaque Signal feature projection with explicit `availability` (`available`, `unknown`, `unsupported`, `not_applicable`, `stale`), `coverage_ratio`, `freshness_sec`, and bounded observation tiers/risks (`additionalProperties: false`, excluding any universal quality oracle score).
   - `candidate-feature-vector-v1.schema.json` (`candidate-feature-vector/1.0`): candidate metadata, explicit `eligible` and `policy_blocked` flags, viewer preference matches, anti-abuse-filtered engagement rollups, and optional `signal`.
   - `ranked-feed-result-v1.schema.json` (`ranked-feed-result/1.0`): ordered items, plain-language `why_this` (`<= 2` reasons), `session_pacing_advisory` (`D01`/`D03`), opaque deterministic `next_cursor`, and `diversity_audit`.
3. **Offline evaluation and clean-room scanner (`eval/`):** Zero-dependency Python 3 suite verifying `NDCG@k`, author Gini diversity, minority-language exposure parity, counterfactual `unknown` non-demotion, JSON Schema validity, and AST/source clean-room compliance.

## Mathematical Formulation

### 1. Defense-in-depth policy gate

Every candidate $c$ is checked before scoring:

$$\text{Admit}(c) \iff c.\text{Eligible} \land \neg c.\text{PolicyBlocked} \land (c.\text{CandidateID} \neq \epsilon)$$

Ineligible candidates receive `0%` exposure and are counted in `diversity_audit.dropped_ineligible`.

### 2. Base relevance and anti-engagement-trap scoring

For admitted candidate $c$ at reference timestamp $T_{\text{now}}$ with age $\Delta t_h = \max(0, (T_{\text{now}} - T_{\text{pub}}(c)) / 1\text{h})$:

$$S_{\text{base}}(c) = S_{\text{pref}}(c) + S_{\text{fresh}}(c) + S_{\text{eng}}(c)$$

- **Preference alignment** $S_{\text{pref}}(c)$: additive contributions from explicit `followed_author` ($0.30$), bounded diminishing-returns `followed_topic_matches` ($0.22 \cdot \min(1, \text{matches}/2)$ for $1+$ matches with $0.65$ at $1$ match), and `language_match` ($0.15$). Non-matching languages receive `0.0` preference bonus rather than exclusion.
- **Freshness decay** $S_{\text{fresh}}(c) = W_{\text{fresh}} \cdot 2^{-\Delta t_h / H}$ with default $W_{\text{fresh}} = 0.20$ and half-life $H = 18\text{h}$.
- **Quality-adjusted engagement** $S_{\text{eng}}(c)$:
  - If `anomaly_flagged == true`, $S_{\text{eng}}(c) = 0.0$ (bot/velocity spikes are completely ignored).
  - Otherwise, deliberative depth dominates passive watch ratio:
    $$S_{\text{eng}}(c) = W_{\text{eng}} \cdot \bigl(0.40 \cdot r_{\text{article}} + 0.35 \cdot r_{\text{source}} + 0.15 \cdot r_{\text{save}} + 0.10 \cdot r_{\text{watch}}\bigr)$$
    where $W_{\text{eng}} = 0.25$ and each rate is clamped to $[0, 1]$. A sensational video with high completion ($r_{\text{watch}} = 1.0, r_{\text{article}} = 0, r_{\text{source}} = 0$) earns at most $0.025$, whereas a story whose readers open the article and inspect sources earns up to $0.25$.

### 3. Uncertainty neutrality and cold-start envelope

Let $A(s, d)$ denote whether Signal profile $s$ and dimension $d$ are both `available` and fresh ($0 \le s.\text{freshness\_sec} \le F_{\max}$).

- Whenever $s = \text{nil}$, $s.\text{availability} \in \{\text{unknown}, \text{unsupported}, \text{not\_applicable}, \text{stale}\}$, or $s.\text{freshness\_sec} > F_{\max}$, every dimension has $A(s, d) = \text{false}$ and:
  $$\Delta S_{\text{signal}}(c) = 0.0$$
- For each individual dimension $d$, if $A(s, d) = \text{false}$ or its tier is `"unknown"`, its contribution $\delta_d(c) = 0.0$ identically. Note also that `provenance_status == "missing"` and `"mixed"` alignment tiers yield `0.0` so ordinary un-signed cameras and developing stories are never penalized.
- Available negative risk observations apply bounded rank demotions rather than hard bans:
  - `evidence_support_tier == "contradicted"` ($-0.22$), `consistency_tier == "inconsistent"` ($-0.10$), `technical_legibility_tier == "unusable"` ($-0.12$), `provenance_status == "invalid"` ($-0.10$), elevated `bait_switch_risk > 0.35` (`D02`, up to $-0.20$), and extreme `attention_pacing_score > 0.70` (`D01`, up to $-0.08$), clamped to $[-0.35, +0.18]$.
- **Cold-start exploration envelope:** Let $b_0(c) \ge 0$ be the configured cold-start bonus for fresh ($\Delta t_h \le H_{\text{cold}}$), low-impression, non-anomalous candidates. We define:
  $$B_{\text{explore}}(c) = \begin{cases} 0.0 & \text{if } \Delta S_{\text{signal}}(c) < 0 \\ \max\bigl(0.0,\, b_0(c) - \Delta S_{\text{signal}}(c)\bigr) & \text{if } \Delta S_{\text{signal}}(c) \ge 0 \end{cases}$$
  This guarantees two simultaneous properties:
  1. Unassessed (`nil`), `unknown`, `unsupported`, `not_applicable`, and `stale` profiles receive $\Delta S_{\text{signal}}(c) = 0.0$ and $B_{\text{explore}}(c) = b_0(c)$, producing **bit-for-bit identical** scores.
  2. Attaching a neutral or positive partial observation ($\Delta S_{\text{signal}}(c) \ge 0$) yields $\Delta S_{\text{signal}}(c) + B_{\text{explore}}(c) = \max(b_0(c), \Delta S_{\text{signal}}(c)) \ge b_0(c)$, so completing an assessment without negative findings can **never** lower a fresh candidate below its unassessed cold-start baseline.

### 4. Diversity constraints and session pacing (`D01` / `D03`)

- **Sliding-window + MMR diversity:** Greedy selection evaluates remaining candidates with a Maximal Marginal Relevance penalty against already-selected authors and Jaccard topic overlap while enforcing sliding-window caps (`MaxSameAuthorPerWindow`, `MaxSameTopicPerWindow`). Candidates exceeding window caps are deferred behind diverse alternatives (and tracked in `diversity_audit`), preventing single-author or single-topic feed flooding.
- **Session pacing cues:** When consecutive high-intensity (`D01` `attention_pacing_score >= 0.75`) items reach `MaxHighIntensityStreak` or cumulative session items reach `SessionDepthAdvisoryThreshold` (`D03`), the result emits a non-coercive `session_pacing_advisory` suggesting either opening a full article for slower context or pausing at a natural stopping point.
- **Deterministic cursor pagination:** `next_cursor` base64url-encodes the versioned set of already-delivered candidate IDs plus trailing diversity/pacing window state, ensuring deferred candidates appear on subsequent pages without duplication or cross-page author clustering.

## Consequences

- **Positive:** Zero external Go or Python dependencies; deterministic `< 5ms` ranking; provable `0%` policy-blocked leakage and `0.0` `unknown`-status penalty; full offline inspectability.
- **Limitation / Next steps:** Core currently executes its own feed logic. Milestone `M3` will introduce an adapter in Core's feed application layer translating domain projections into `CandidateFeatureVector` and enforcing post-ranking delivery checks.
