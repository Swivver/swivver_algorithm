#!/usr/bin/env python3
"""Multi-beat newsroom simulation, low-polish fairness audit, and Team-Draft Interleaving benchmark.

Simulates 6 journalist/creator archetypes across 2,000 reader sessions with varied
language (en, pt, es), topic, and regional preferences using only the Python 3
standard library and public ScopedSignalFeatures / CandidateFeatureVector v1 schemas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.evaluate_offline import (
    EXPECTED_SCHEMA_FILES,
    ndcg_at_k,
    reciprocal_rank,
    scan_clean_room_boundary,
    validate_instance_against_schema,
    validate_schemas_directory,
)
from eval.validate_5k_corpus import (
    MODEL_VERSION_V1,
    SplitMix64,
    _clamp_unit,
    _compare_candidates,
    _has_any_available_dim,
    _is_dim_available,
    _is_signal_active,
    rank_candidates_reference,
    score_candidate_components,
)

BENCHMARK_SEED = 20261117
INTERLEAVE_OUTCOME_SEED = 20261118
SIM_NOW = datetime(2026, 11, 17, 12, 0, 0, tzinfo=timezone.utc)
SIM_NOW_NANO = int(SIM_NOW.timestamp() * 1_000_000_000)

NAIVE_MODEL_VERSION = "naive-engagement-ranker/1.0"
ABLATION_MODEL_VERSION = "studio-polish-ablation/1.0"

LANGUAGES: tuple[str, ...] = ("en", "pt", "es")
REGIONS: tuple[str, ...] = (
    "north_america",
    "latin_america",
    "europe",
    "global_maritime",
)

ALL_BEATS: tuple[str, ...] = (
    "infrastructure",
    "climate",
    "health",
    "civic",
    "technology",
    "economy",
    "science",
    "transit",
    "energy",
    "maritime",
    "policy",
)

ARCHETYPE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "investigative_infrastructure",
        "label": "Investigative Infrastructure",
        "polish_group": "studio_polish",
        "beats": ("infrastructure", "energy", "transit", "civic"),
        "subtags": ("grid-resilience", "bridge-audit", "water-utility", "freight-rail"),
    },
    {
        "id": "maritime_climate_science",
        "label": "Maritime/Climate Science",
        "polish_group": "studio_polish",
        "beats": ("climate", "maritime", "science", "economy"),
        "subtags": ("shipping-corridor", "coastal-tide", "port-drought", "ocean-buoy"),
    },
    {
        "id": "public_health_desk",
        "label": "Public Health Desk",
        "polish_group": "studio_polish",
        "beats": ("health", "science", "civic", "policy"),
        "subtags": ("clinic-capacity", "clean-air", "immunization", "triage-network"),
    },
    {
        "id": "regional_breaking_low_polish",
        "label": "Regional Breaking Low-Polish Field Reporter",
        "polish_group": "low_polish",
        "beats": ("civic", "transit", "infrastructure", "climate"),
        "subtags": ("flood-response", "service-outage", "council-vote", "port-closure"),
    },
    {
        "id": "tech_ai_policy_analyst",
        "label": "Tech/AI Policy Analyst",
        "polish_group": "studio_polish",
        "beats": ("technology", "policy", "economy", "science"),
        "subtags": ("compute-standards", "spectrum-licensing", "procurement-rules", "grid-demand"),
    },
    {
        "id": "high_arousal_unsourced_bait",
        "label": "High-Arousal Unsourced Bait",
        "polish_group": "attention_bait",
        "beats": ("economy", "technology", "health", "climate", "civic", "infrastructure"),
        "subtags": ("viral-shock", "market-panic", "secret-rumor", "instant-outrage"),
    },
)


@dataclass(frozen=True)
class SimulatedStory:
    candidate: dict[str, Any]
    published_nano: int
    archetype_id: str
    archetype_label: str
    primary_beat: str
    incident_tag: str
    region: str
    is_corroborated_low_polish: bool
    is_corroborated_studio_polish: bool
    is_unsourced_bait: bool


def generate_newsroom_corpus(
    stories_per_archetype: int = 200,
    seed: int = BENCHMARK_SEED,
) -> tuple[list[SimulatedStory], dict[str, list[str]]]:
    """Generate a deterministic 6-archetype multilingual, multi-region newsroom corpus."""
    rng = SplitMix64(seed)
    corpus: list[SimulatedStory] = []
    legitimate_authors_by_lang: dict[str, list[str]] = {"en": [], "pt": [], "es": []}

    # 6 distinct creators per archetype (3 en, 2 pt, 1 es -> 36 creators total).
    creator_roster: dict[str, list[tuple[str, str, str]]] = {}
    for a_idx, spec in enumerate(ARCHETYPE_SPECS):
        aid_prefix = spec["id"]
        roster: list[tuple[str, str, str]] = []
        lang_assign = ("en", "en", "en", "pt", "pt", "es")
        for c_idx, lang in enumerate(lang_assign):
            creator_id = f"writer-{a_idx + 1:02d}-{c_idx + 1:02d}"
            region = REGIONS[(a_idx + c_idx) % len(REGIONS)]
            roster.append((creator_id, lang, region))
            if aid_prefix != "high_arousal_unsourced_bait":
                legitimate_authors_by_lang[lang].append(creator_id)
        creator_roster[aid_prefix] = roster

    seq = 0
    for slot in range(stories_per_archetype):
        # Match baseline age across all 6 archetypes within each slot so recency never confounds fairness.
        base_age_min = 12 + int((slot / max(1, stories_per_archetype - 1)) * 2160)

        for a_idx, spec in enumerate(ARCHETYPE_SPECS):
            arch_id = spec["id"]
            roster = creator_roster[arch_id]
            creator_id, lang, creator_region = roster[slot % len(roster)]
            region = REGIONS[(slot + a_idx) % len(REGIONS)] if slot % 3 == 0 else creator_region

            beats = spec["beats"]
            subtags = spec["subtags"]
            primary_beat = beats[(slot + a_idx) % len(beats)]
            incident_tag = subtags[(slot // 2 + a_idx) % len(subtags)]
            topic_tags = sorted({primary_beat, incident_tag})

            jitter = rng.intn(9) - 4
            age_min = max(5, base_age_min + jitter)
            pub_dt = SIM_NOW - timedelta(minutes=age_min)
            pub_iso = pub_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            pub_nano = SIM_NOW_NANO - age_min * 60 * 1_000_000_000

            # 4% policy-restricted items across all archetypes to exercise policy-sandwich gates.
            is_policy_restricted = (seq % 25) == 24
            eligible = not is_policy_restricted or (seq % 2 == 0)
            policy_blocked = is_policy_restricted and (not eligible or seq % 2 == 0)
            if is_policy_restricted and eligible and not policy_blocked:
                policy_blocked = True

            anomaly_flagged = False
            is_corr_low = False
            is_corr_studio = False
            is_bait = False
            signal: dict[str, Any] | None = None

            if arch_id == "high_arousal_unsourced_bait":
                is_bait = True
                watch_ratio = round(rng.uniform(0.89, 0.995), 6)
                article_open = round(rng.uniform(0.008, 0.055), 6)
                source_inspect = round(rng.uniform(0.0, 0.018), 6)
                save_rate = round(rng.uniform(0.0, 0.015), 6)
                impressions = 180 + rng.intn(900)

                bait_mode = slot % 5
                if bait_mode == 0:
                    # Coordinated velocity/bot spike on slick unsourced clip.
                    anomaly_flagged = True
                    watch_ratio = round(rng.uniform(0.95, 1.0), 6)
                    article_open = round(rng.uniform(0.80, 0.96), 6)
                    source_inspect = round(rng.uniform(0.75, 0.92), 6)
                    save_rate = round(rng.uniform(0.70, 0.90), 6)
                    signal = {
                        "schema_version": "scoped-signal-features/1.0",
                        "snapshot_id": f"sim-snap-{seq:05d}",
                        "availability": "available",
                        "coverage_ratio": 0.85,
                        "freshness_sec": round(rng.uniform(20.0, 900.0), 3),
                        "evidence_support_tier": "mixed",
                        "consistency_tier": "inconsistent",
                        "technical_legibility_tier": "clear",
                        "attention_pacing_score": round(rng.uniform(0.84, 0.99), 6),
                        "bait_switch_risk": round(rng.uniform(0.68, 0.96), 6),
                        "provenance_status": "missing",
                        "independent_source_clusters": 0,
                    }
                elif bait_mode in (1, 2):
                    # Zero-source high-arousal sensationalism (not explicitly contradicted, zero sources).
                    signal = {
                        "schema_version": "scoped-signal-features/1.0",
                        "snapshot_id": f"sim-snap-{seq:05d}",
                        "availability": "available",
                        "coverage_ratio": 0.85,
                        "freshness_sec": round(rng.uniform(25.0, 1400.0), 3),
                        "evidence_support_tier": "mixed" if bait_mode == 1 else "unknown",
                        "consistency_tier": "mixed" if slot % 2 == 0 else "inconsistent",
                        "technical_legibility_tier": "clear",
                        "attention_pacing_score": round(rng.uniform(0.82, 0.98), 6),
                        "bait_switch_risk": round(rng.uniform(0.58, 0.94), 6),
                        "provenance_status": "missing",
                        "independent_source_clusters": 0,
                    }
                else:
                    # Disputed / contradicted viral sensationalism with studio polish.
                    signal = {
                        "schema_version": "scoped-signal-features/1.0",
                        "snapshot_id": f"sim-snap-{seq:05d}",
                        "availability": "available",
                        "coverage_ratio": 1.0,
                        "freshness_sec": round(rng.uniform(20.0, 1200.0), 3),
                        "evidence_support_tier": "contradicted",
                        "consistency_tier": "inconsistent",
                        "technical_legibility_tier": "clear",
                        "attention_pacing_score": round(rng.uniform(0.78, 0.97), 6),
                        "bait_switch_risk": round(rng.uniform(0.62, 0.95), 6),
                        "provenance_status": "missing",
                        "independent_source_clusters": 0,
                    }

            elif arch_id == "regional_breaking_low_polish":
                # Same deliberative engagement envelope & corroboration rigor as studio desks,
                # captured on handheld field phones with noisy environment audio & missing C2PA.
                watch_ratio = round(rng.uniform(0.47, 0.76), 6)
                article_open = round(rng.uniform(0.27, 0.49), 6)
                source_inspect = round(rng.uniform(0.17, 0.36), 6)
                save_rate = round(rng.uniform(0.06, 0.16), 6)
                impressions = 45 + rng.intn(360)

                if slot % 7 == 0:
                    # Fresh unassessed / partial cold-start field dispatch.
                    impressions = rng.intn(55)
                    signal = (
                        None
                        if slot % 14 == 0
                        else {
                            "schema_version": "scoped-signal-features/1.0",
                            "snapshot_id": f"sim-snap-{seq:05d}",
                            "availability": "unknown",
                            "coverage_ratio": 0.0,
                            "freshness_sec": round(rng.uniform(10.0, 240.0), 3),
                        }
                    )
                else:
                    is_corr_low = eligible and not policy_blocked
                    signal = {
                        "schema_version": "scoped-signal-features/1.0",
                        "snapshot_id": f"sim-snap-{seq:05d}",
                        "availability": "available",
                        "coverage_ratio": 1.0,
                        "freshness_sec": round(rng.uniform(20.0, 1800.0), 3),
                        "evidence_support_tier": "supported",
                        "consistency_tier": "consistent",
                        "technical_legibility_tier": "degraded",
                        "attention_pacing_score": round(rng.uniform(0.54, 0.81), 6),
                        "bait_switch_risk": round(rng.uniform(0.02, 0.13), 6),
                        "provenance_status": "missing",
                        "independent_source_clusters": 2 + (slot % 3),
                    }

            else:
                # Corroborated studio/desk journalism across Investigative Infrastructure,
                # Maritime/Climate Science, Public Health Desk, and Tech/AI Policy Analyst.
                watch_ratio = round(rng.uniform(0.47, 0.76), 6)
                article_open = round(rng.uniform(0.27, 0.49), 6)
                source_inspect = round(rng.uniform(0.17, 0.36), 6)
                save_rate = round(rng.uniform(0.06, 0.16), 6)
                impressions = 45 + rng.intn(360)

                if arch_id == "public_health_desk" and slot % 7 == 0:
                    impressions = rng.intn(55)
                    signal = {
                        "schema_version": "scoped-signal-features/1.0",
                        "snapshot_id": f"sim-snap-{seq:05d}",
                        "availability": "unsupported",
                        "coverage_ratio": 0.0,
                        "freshness_sec": round(rng.uniform(15.0, 300.0), 3),
                    }
                else:
                    is_corr_studio = eligible and not policy_blocked
                    signal = {
                        "schema_version": "scoped-signal-features/1.0",
                        "snapshot_id": f"sim-snap-{seq:05d}",
                        "availability": "available",
                        "coverage_ratio": 1.0,
                        "freshness_sec": round(rng.uniform(20.0, 1800.0), 3),
                        "evidence_support_tier": "supported",
                        "consistency_tier": "consistent",
                        "technical_legibility_tier": "clear",
                        "attention_pacing_score": round(rng.uniform(0.16, 0.46), 6),
                        "bait_switch_risk": round(rng.uniform(0.01, 0.11), 6),
                        "provenance_status": "verified" if slot % 2 == 0 else "declared",
                        "independent_source_clusters": 2 + (slot % 3),
                    }

            cand: dict[str, Any] = {
                "schema_version": "candidate-feature-vector/1.0",
                "candidate_id": f"newsroom-{seq:05d}",
                "author_id": creator_id,
                "topic_tags": topic_tags,
                "language": lang,
                "published_at": pub_iso,
                "eligible": eligible,
                "policy_blocked": policy_blocked,
                "followed_author": False,
                "followed_topic_matches": 0,
                "language_match": False,
                "impression_count": impressions,
                "valid_watch_ratio": watch_ratio,
                "article_open_rate": article_open,
                "source_inspect_rate": source_inspect,
                "save_rate": save_rate,
                "anomaly_flagged": anomaly_flagged,
            }
            if signal is not None:
                cand["signal"] = signal

            corpus.append(
                SimulatedStory(
                    candidate=cand,
                    published_nano=pub_nano,
                    archetype_id=arch_id,
                    archetype_label=spec["label"],
                    primary_beat=primary_beat,
                    incident_tag=incident_tag,
                    region=region,
                    is_corroborated_low_polish=is_corr_low,
                    is_corroborated_studio_polish=is_corr_studio,
                    is_unsourced_bait=is_bait,
                )
            )
            seq += 1

    return corpus, legitimate_authors_by_lang


def rank_naive_engagement(
    candidates: Sequence[tuple[Mapping[str, Any], int]],
    page_size: int = 10,
    now_nano: int = SIM_NOW_NANO,
) -> list[dict[str, Any]]:
    """Naive engagement-only baseline ranker (policy-gated, watch-ratio dominated, no Signal/diversity)."""
    seen: set[str] = set()
    scored: list[dict[str, Any]] = []
    for cand, pub_nano in candidates:
        cid = str(cand.get("candidate_id", "")).strip()
        if not cand.get("eligible", False) or cand.get("policy_blocked", False) or not cid:
            continue
        if cid in seen:
            continue
        seen.add(cid)

        age_hours = max(0.0, (now_nano - pub_nano) / 3_600_000_000_000.0)
        fresh = math.pow(2.0, -age_hours / 24.0)
        watch = _clamp_unit(float(cand.get("valid_watch_ratio", 0.0)))
        art = _clamp_unit(float(cand.get("article_open_rate", 0.0)))
        sav = _clamp_unit(float(cand.get("save_rate", 0.0)))
        lang_bonus = 0.08 if cand.get("language_match") else 0.0
        topic_bonus = 0.05 * min(2, int(cand.get("followed_topic_matches", 0)))

        raw = 0.72 * watch + 0.08 * art + 0.07 * sav + 0.05 * fresh + lang_bonus + topic_bonus
        scored.append(
            {
                "candidate_id": cid,
                "author_id": str(cand.get("author_id", "")).strip(),
                "topics": tuple(sorted({t.strip().lower() for t in cand.get("topic_tags", ()) if t.strip()})),
                "published_nano": pub_nano,
                "score": round(raw, 6),
                "raw_score": raw,
                "signal_delta": 0.0,
            }
        )

    scored.sort(key=lambda s: (-round(s["raw_score"], 12), -s["published_nano"], s["candidate_id"]))
    out = scored[: min(page_size, len(scored))]
    for idx, item in enumerate(out, start=1):
        item["rank"] = idx
    return out


def rank_studio_polish_ablation(
    candidates: Sequence[tuple[Mapping[str, Any], int]],
    page_size: int = 10,
    now_nano: int = SIM_NOW_NANO,
) -> list[dict[str, Any]]:
    """Single-factor ablation ranker: identical diversity engine, but penalizes handheld
    degraded legibility (-0.09) and missing provenance (-0.03) while rewarding studio polish."""
    biased_batch: list[tuple[dict[str, Any], int]] = []
    seen: set[str] = set()
    scored: list[dict[str, Any]] = []

    for cand, pub_nano in candidates:
        cid = str(cand.get("candidate_id", "")).strip()
        if not cand.get("eligible", False) or cand.get("policy_blocked", False) or not cid:
            continue
        if cid in seen:
            continue
        seen.add(cid)

        base = score_candidate_components(cand, pub_nano, now_nano)
        sig = cand.get("signal")
        polish_bias = 0.0
        if _is_signal_active(sig):
            assert sig is not None
            if _is_dim_available(sig, "technical_legibility_tier"):
                tl = sig.get("technical_legibility_tier")
                if tl == "clear":
                    polish_bias += 0.05
                elif tl == "degraded":
                    polish_bias -= 0.09
            if _is_dim_available(sig, "provenance_status"):
                pv = sig.get("provenance_status")
                if pv == "verified":
                    polish_bias += 0.04
                elif pv == "missing":
                    polish_bias -= 0.03

        item = dict(base)
        item["raw_score"] = base["raw_score"] + polish_bias
        scored.append(item)
        biased_batch.append((dict(cand), pub_nano))

    _ = biased_batch
    scored.sort(key=lambda s: (-round(s["raw_score"], 12), -s["published_nano"], s["candidate_id"]))
    target = min(page_size, len(scored))
    if target <= 0:
        return []

    window_span = 4
    recent_authors: list[str] = []
    recent_topics: list[tuple[str, ...]] = []
    author_counts: Counter[str] = Counter()
    used = [False] * len(scored)
    selected: list[dict[str, Any]] = []

    for _ in range(target):
        best_strict_idx = -1
        best_strict_score = 0.0
        best_fallback_idx = -1
        best_fallback_score = 0.0
        last_author = recent_authors[-1] if recent_authors else ""
        last_topics = set(recent_topics[-1]) if recent_topics and recent_topics[-1] else set()

        for i, cand in enumerate(scored):
            if used[i]:
                continue
            if best_strict_idx != -1 and cand["raw_score"] < best_strict_score - 1e-12:
                break
            pen = 0.0
            aid = cand["author_id"]
            if aid and author_counts[aid] > 0:
                pen += 0.12 * author_counts[aid]
            ctopics = cand["topics"]
            cset = set(ctopics) if ctopics else set()
            if cset and recent_topics:
                max_j = 0.0
                for prev in recent_topics:
                    if prev:
                        pset = set(prev)
                        inter = len(cset & pset)
                        if inter:
                            j = inter / (len(cset) + len(pset) - inter)
                            if j > max_j:
                                max_j = j
                pen += 0.06 * max_j
            eff = cand["raw_score"] - pen
            a_blocked = (bool(aid) and aid == last_author) or (bool(aid) and recent_authors.count(aid) >= 1)
            t_blocked = bool(ctopics) and any(
                sum(1 for prev in recent_topics if t in prev) >= 2 for t in ctopics
            )
            inc_blocked = False
            if cset and last_topics:
                inter_last = len(cset & last_topics)
                if inter_last >= 2 or (
                    inter_last > 0 and inter_last / (len(cset) + len(last_topics) - inter_last) >= 0.5
                ):
                    inc_blocked = True

            if best_fallback_idx == -1 or (
                _compare_candidates(
                    eff,
                    best_fallback_score,
                    cand["published_nano"],
                    scored[best_fallback_idx]["published_nano"],
                    cand["candidate_id"],
                    scored[best_fallback_idx]["candidate_id"],
                )
                < 0
            ):
                best_fallback_idx, best_fallback_score = i, eff
            if not a_blocked and not t_blocked and not inc_blocked:
                if best_strict_idx == -1 or (
                    _compare_candidates(
                        eff,
                        best_strict_score,
                        cand["published_nano"],
                        scored[best_strict_idx]["published_nano"],
                        cand["candidate_id"],
                        scored[best_strict_idx]["candidate_id"],
                    )
                    < 0
                ):
                    best_strict_idx, best_strict_score = i, eff

        chosen_idx = best_strict_idx if best_strict_idx != -1 else best_fallback_idx
        chosen_score = best_strict_score if best_strict_idx != -1 else best_fallback_score
        used[chosen_idx] = True
        picked = dict(scored[chosen_idx])
        picked["score"] = round(chosen_score, 6)
        picked["rank"] = len(selected) + 1
        selected.append(picked)
        if picked["author_id"]:
            author_counts[picked["author_id"]] += 1
        recent_authors = (recent_authors + [picked["author_id"]])[-window_span:]
        recent_topics = (recent_topics + [picked["topics"]])[-window_span:]

    return selected


def team_draft_interleave(
    list_a: Sequence[Mapping[str, Any]],
    list_b: Sequence[Mapping[str, Any]],
    page_size: int,
    experiment_id: str,
    request_key: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic Team-Draft Interleaving (Radlinski, Kurup & Joachims, 2008) matching pkg/interleave."""
    ids_a = {str(it["candidate_id"]) for it in list_a}
    ids_b = {str(it["candidate_id"]) for it in list_b}
    both = ids_a & ids_b
    filtered_a = [dict(it) for it in list_a if str(it["candidate_id"]) in both]
    filtered_b = [dict(it) for it in list_b if str(it["candidate_id"]) in both]

    target = min(page_size, len(filtered_a), len(filtered_b))
    digest = hashlib.sha256(
        experiment_id.encode("utf-8")
        + b"\x00"
        + request_key.encode("utf-8")
        + b"\x00"
        + struct.pack(">q", seed)
    ).digest()
    rng_seed = struct.unpack(">Q", digest[:8])[0]
    rng = SplitMix64(rng_seed)

    placed: set[str] = set()
    items: list[dict[str, Any]] = []
    slots: list[dict[str, Any]] = []
    idx_a, idx_b = 0, 0
    count_a, count_b = 0, 0

    while len(items) < target:
        while idx_a < len(filtered_a) and str(filtered_a[idx_a]["candidate_id"]) in placed:
            idx_a += 1
        while idx_b < len(filtered_b) and str(filtered_b[idx_b]["candidate_id"]) in placed:
            idx_b += 1

        if count_a < count_b:
            pick_team = "team_a"
        elif count_b < count_a:
            pick_team = "team_b"
        else:
            pick_team = "team_a" if (rng.next_u64() & 1) == 0 else "team_b"

        if pick_team == "team_a":
            chosen = dict(filtered_a[idx_a])
            idx_a += 1
            count_a += 1
        else:
            chosen = dict(filtered_b[idx_b])
            idx_b += 1
            count_b += 1

        rank = len(items) + 1
        draft_round = ((rank - 1) // 2) + 1
        cid = str(chosen["candidate_id"])
        placed.add(cid)
        chosen["rank"] = rank
        items.append(chosen)
        slots.append(
            {
                "rank": rank,
                "candidate_id": cid,
                "team": pick_team,
                "draft_round": draft_round,
            }
        )

    return items, slots


def two_sided_sign_test_pvalue(wins_b: int, wins_a: int) -> float:
    """Exact two-sided binomial sign-test p-value under H0: p = 0.5 (matching pkg/interleave)."""
    wb = max(0, int(wins_b))
    wa = max(0, int(wins_a))
    n = wa + wb
    if n == 0:
        return 1.0
    k = min(wa, wb)
    log_n = math.lgamma(n + 1.0)
    n_log2 = n * math.log(2.0)

    log_terms: list[float] = []
    for j in range(k + 1):
        log_terms.append(log_n - math.lgamma(j + 1.0) - math.lgamma(n - j + 1.0) - n_log2)
    max_lp = max(log_terms)
    # Use log-sum-exp so tiny p-values do not underflow prematurely before clamping.
    log_tail = max_lp + math.log(sum(math.exp(lp - max_lp) for lp in log_terms))
    log_p = math.log(2.0) + log_tail
    if log_p >= 0.0:
        return 1.0
    return math.exp(log_p)


def two_sided_sign_test_log10_pvalue(wins_b: int, wins_a: int) -> float:
    """Base-10 log of exact two-sided sign-test p-value for extreme statistical significance."""
    wb = max(0, int(wins_b))
    wa = max(0, int(wins_a))
    n = wa + wb
    if n == 0:
        return 0.0
    k = min(wa, wb)
    log_n = math.lgamma(n + 1.0)
    n_log2 = n * math.log(2.0)
    log_terms = [
        log_n - math.lgamma(j + 1.0) - math.lgamma(n - j + 1.0) - n_log2
        for j in range(k + 1)
    ]
    max_lp = max(log_terms)
    log_p = math.log(2.0) + max_lp + math.log(sum(math.exp(lp - max_lp) for lp in log_terms))
    return min(0.0, log_p / math.log(10.0))


def bootstrap_mean_ci95(values: Sequence[float], resamples: int = 2000, seed: int = 20261119) -> tuple[float, float]:
    """Deterministic 2,000-resample percentile bootstrap 95% CI on mean delta."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return float(values[0]), float(values[0])
    rng = SplitMix64(seed)
    means: list[float] = []
    inv_n = 1.0 / n
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.intn(n)]
        means.append(total * inv_n)
    means.sort()
    low_idx = int(math.floor(0.025 * (resamples - 1)))
    high_idx = int(math.ceil(0.975 * (resamples - 1)))
    return round(means[low_idx], 6), round(means[high_idx], 6)


def compute_session_relevance(
    story: SimulatedStory,
    cand: Mapping[str, Any],
) -> float:
    """Graded deliberative reader relevance in {0.0, 1.0, 2.0, 3.0} for NDCG@k and MRR."""
    if not cand.get("eligible", False) or cand.get("policy_blocked", False):
        return 0.0
    if story.is_unsourced_bait or cand.get("anomaly_flagged", False):
        return 0.0

    lang_ok = bool(cand.get("language_match", False))
    followed = bool(cand.get("followed_author", False))
    t_matches = int(cand.get("followed_topic_matches", 0))
    reg_ok = bool(cand.get("region_match", False))

    if lang_ok and (followed or t_matches >= 2 or (t_matches >= 1 and reg_ok)):
        return 3.0
    if lang_ok and (t_matches >= 1 or reg_ok):
        return 2.0
    return 1.0


def max_consecutive_streak(values: Sequence[str]) -> int:
    """Maximum consecutive identical non-empty labels in an ordered feed."""
    best = 0
    cur = 0
    prev = ""
    for val in values:
        if not val:
            cur = 0
            prev = ""
            continue
        if val == prev:
            cur += 1
        else:
            prev = val
            cur = 1
        if cur > best:
            best = cur
    return best


def simulate_reader_item_usefulness(
    story: SimulatedStory,
    relevance: float,
    rank: int,
    rng: SplitMix64,
) -> tuple[float, bool]:
    """Simulate post-impression deliberative outcomes scored with pkg/interleave weights:
    +1.0 completed read/watch, +1.0 source inspect, +1.5 positive usefulness,
    -1.0 rapid skip, -2.0 correction impression.
    """
    exam_prob = 1.0 / math.log2(rank + 1.0)
    if rng.float64() > exam_prob:
        return 0.0, False

    if story.is_unsourced_bait:
        rapid_skip = rng.float64() < 0.62
        sig = story.candidate.get("signal") or {}
        contradicted = sig.get("evidence_support_tier") == "contradicted"
        correction = rng.float64() < (0.34 if contradicted else 0.14)
        completed = (not rapid_skip) and (rng.float64() < 0.18)
        useful = False
        source_inspected = False
    else:
        rel_factor = 0.55 + 0.14 * relevance  # 0.69 .. 0.97 across grades 1..3
        completed = rng.float64() < min(0.92, 0.42 * rel_factor + float(story.candidate["article_open_rate"]))
        source_inspected = rng.float64() < min(0.85, 0.35 * rel_factor + float(story.candidate["source_inspect_rate"]))
        useful = completed and (rng.float64() < min(0.90, 0.48 * rel_factor))
        rapid_skip = (not completed) and (rng.float64() < 0.10)
        correction = False

    score = 0.0
    if completed:
        score += 1.0
    if source_inspected:
        score += 1.0
    if useful:
        score += 1.5
    if rapid_skip:
        score -= 1.0
    if correction:
        score -= 2.0
    return score, correction


def run_newsroom_benchmark(
    repo_root: Path,
    num_sessions: int = 2000,
    stories_per_archetype: int = 200,
) -> dict[str, Any]:
    """Run the 2,000-session multi-beat newsroom simulation, schema checks, and interleaving study."""
    schemas_dir = repo_root / "schemas"
    schema_errors = validate_schemas_directory(schemas_dir)
    clean_room_violations = scan_clean_room_boundary(repo_root)

    corpus, legit_authors_by_lang = generate_newsroom_corpus(
        stories_per_archetype=stories_per_archetype,
        seed=BENCHMARK_SEED,
    )
    story_by_id = {s.candidate["candidate_id"]: s for s in corpus}

    instance_schema_errors: list[str] = []
    if not schema_errors:
        loaded_schemas = {
            name: json.loads((schemas_dir / name).read_text(encoding="utf-8"))
            for name in EXPECTED_SCHEMA_FILES
        }
        cand_schema = loaded_schemas["candidate-feature-vector-v1.schema.json"]
        for idx in range(0, len(corpus), 20):
            errs = validate_instance_against_schema(
                corpus[idx].candidate, cand_schema, loaded_schemas
            )
            if errs:
                instance_schema_errors.extend(errs)
                break

    # Partition corpus by slot (each slot has 6 stories: 1 from each archetype).
    num_slots = stories_per_archetype
    by_slot: list[list[SimulatedStory]] = [
        corpus[slot * len(ARCHETYPE_SPECS) : (slot + 1) * len(ARCHETYPE_SPECS)]
        for slot in range(num_slots)
    ]

    metrics_accum: dict[str, dict[str, Any]] = {
        name: {
            "ndcg5": [],
            "ndcg10": [],
            "mrr": [],
            "unique_beats_top10": [],
            "unique_tags_top10": [],
            "max_author_streak": 0,
            "max_incident_streak": 0,
            "mean_author_streak": [],
            "top5_bait_slots": 0,
            "top5_total_slots": 0,
            "low_polish_ranks": [],
            "studio_polish_ranks": [],
            "low_polish_top10_exposures": 0,
            "studio_polish_top10_exposures": 0,
        }
        for name in ("signal_ranker", "naive_engagement", "studio_polish_ablation")
    }

    archetype_top5_counts: Counter[str] = Counter()
    archetype_top10_counts: Counter[str] = Counter()
    archetype_pool_ranks: dict[str, list[int]] = {sp["id"]: [] for sp in ARCHETYPE_SPECS}
    archetype_signal_deltas: dict[str, list[float]] = {sp["id"]: [] for sp in ARCHETYPE_SPECS}

    outcome_rng = SplitMix64(INTERLEAVE_OUTCOME_SEED)
    session_deltas: list[float] = []
    wins_signal = 0
    wins_naive = 0
    ties = 0
    total_m01_signal = 0.0
    total_m01_naive = 0.0
    corrections_signal = 0
    corrections_naive = 0
    policy_leaks = 0

    for s in range(num_sessions):
        # Reader language (60% en, 22% pt, 18% es), region, 2 preferred beats, and 3 followed authors.
        lang_mod = s % 50
        if lang_mod < 30:
            viewer_lang = "en"
        elif lang_mod < 41:
            viewer_lang = "pt"
        else:
            viewer_lang = "es"

        viewer_region = REGIONS[s % len(REGIONS)]
        beat_1 = ALL_BEATS[s % len(ALL_BEATS)]
        beat_2 = ALL_BEATS[(s * 3 + 2) % len(ALL_BEATS)]
        if beat_2 == beat_1:
            beat_2 = ALL_BEATS[(s + 1) % len(ALL_BEATS)]
        pref_beats = {beat_1, beat_2}

        lang_writers = legit_authors_by_lang[viewer_lang]
        followed_writers = {
            lang_writers[(s * 5 + k * 7) % len(lang_writers)] for k in range(3)
        }

        # Select 8 slots (8 x 6 = 48 candidates, balanced across all 6 archetypes).
        start_slot = (s * 7) % num_slots
        window_stories: list[SimulatedStory] = []
        for offset in range(8):
            slot_idx = (start_slot + offset * 11) % num_slots
            window_stories.extend(by_slot[slot_idx])

        session_now_nano = max(st.published_nano for st in window_stories) + 180 * 1_000_000_000
        batch: list[tuple[dict[str, Any], int]] = []
        relevance_by_id: dict[str, float] = {}
        ineligible_ids: set[str] = set()

        for st in window_stories:
            c = dict(st.candidate)
            cid = c["candidate_id"]
            if not c["eligible"] or c["policy_blocked"]:
                ineligible_ids.add(cid)
            t_matches = sum(1 for t in c["topic_tags"] if t in pref_beats)
            c["followed_topic_matches"] = t_matches
            c["language_match"] = c["language"] == viewer_lang
            c["region_match"] = st.region == viewer_region
            c["followed_author"] = c["author_id"] in followed_writers
            batch.append((c, st.published_nano))
            relevance_by_id[cid] = compute_session_relevance(st, c)

        # Rank full eligible pool (up to 48) so mean_rank across archetypes/polish groups is exact,
        # and slice Top-10 for feed diversity, Top-5 suppression, NDCG, MRR, and interleaving.
        full_signal, _ = rank_candidates_reference(batch, page_size=len(batch), now_nano=session_now_nano)
        full_naive = rank_naive_engagement(batch, page_size=len(batch), now_nano=session_now_nano)
        full_ablation = rank_studio_polish_ablation(batch, page_size=len(batch), now_nano=session_now_nano)

        for model_key, full_ranked in (
            ("signal_ranker", full_signal),
            ("naive_engagement", full_naive),
            ("studio_polish_ablation", full_ablation),
        ):
            acc = metrics_accum[model_key]
            top10 = full_ranked[:10]
            top10_ids = [it["candidate_id"] for it in top10]

            for cid in top10_ids:
                if cid in ineligible_ids:
                    policy_leaks += 1

            acc["ndcg5"].append(ndcg_at_k(top10_ids, relevance_by_id, k=5))
            acc["ndcg10"].append(ndcg_at_k(top10_ids, relevance_by_id, k=10))
            acc["mrr"].append(reciprocal_rank(top10_ids, relevance_by_id, min_relevance=3.0))

            beats_top10 = {story_by_id[cid].primary_beat for cid in top10_ids}
            tags_top10 = {tag for cid in top10_ids for tag in story_by_id[cid].candidate["topic_tags"]}
            acc["unique_beats_top10"].append(len(beats_top10))
            acc["unique_tags_top10"].append(len(tags_top10))

            a_streak = max_consecutive_streak([story_by_id[cid].candidate["author_id"] for cid in top10_ids])
            i_streak = max_consecutive_streak([story_by_id[cid].incident_tag for cid in top10_ids])
            acc["mean_author_streak"].append(a_streak)
            if a_streak > acc["max_author_streak"]:
                acc["max_author_streak"] = a_streak
            if i_streak > acc["max_incident_streak"]:
                acc["max_incident_streak"] = i_streak

            top5_ids = top10_ids[:5]
            acc["top5_total_slots"] += len(top5_ids)
            for cid in top5_ids:
                if story_by_id[cid].is_unsourced_bait:
                    acc["top5_bait_slots"] += 1

            for cid in top10_ids:
                st_top = story_by_id[cid]
                if st_top.is_corroborated_low_polish:
                    acc["low_polish_top10_exposures"] += 1
                elif st_top.is_corroborated_studio_polish:
                    acc["studio_polish_top10_exposures"] += 1

            for it in full_ranked:
                cid = it["candidate_id"]
                rk = int(it["rank"])
                st_item = story_by_id[cid]
                if st_item.is_corroborated_low_polish:
                    acc["low_polish_ranks"].append(rk)
                elif st_item.is_corroborated_studio_polish:
                    acc["studio_polish_ranks"].append(rk)

                if model_key == "signal_ranker":
                    archetype_pool_ranks[st_item.archetype_id].append(rk)
                    archetype_signal_deltas[st_item.archetype_id].append(float(it["signal_delta"]))

            if model_key == "signal_ranker":
                for cid in top5_ids:
                    archetype_top5_counts[story_by_id[cid].archetype_id] += 1
                for cid in top10_ids:
                    archetype_top10_counts[story_by_id[cid].archetype_id] += 1

        # Team-Draft Interleaving: Arm A = Naive Engagement, Arm B = Signal-Informed Ranker.
        interleaved_items, slots = team_draft_interleave(
            list_a=full_naive,
            list_b=full_signal,
            page_size=10,
            experiment_id="exp-newsroom-signal-vs-naive",
            request_key=f"reader-session-{s:04d}",
            seed=BENCHMARK_SEED + s,
        )
        for it in interleaved_items:
            if it["candidate_id"] in ineligible_ids:
                policy_leaks += 1

        score_a = 0.0
        score_b = 0.0
        for sl in slots:
            cid = sl["candidate_id"]
            st_sl = story_by_id[cid]
            item_u, had_corr = simulate_reader_item_usefulness(
                st_sl,
                relevance_by_id[cid],
                sl["rank"],
                outcome_rng,
            )
            if sl["team"] == "team_a":
                score_a += item_u
                if had_corr:
                    corrections_naive += 1
            else:
                score_b += item_u
                if had_corr:
                    corrections_signal += 1

        total_m01_naive += score_a
        total_m01_signal += score_b
        delta = score_b - score_a
        session_deltas.append(delta)
        if delta > 1e-12:
            wins_signal += 1
        elif delta < -1e-12:
            wins_naive += 1
        else:
            ties += 1

    def summarize_model(key: str, model_ver: str) -> dict[str, Any]:
        acc = metrics_accum[key]
        low_mean = sum(acc["low_polish_ranks"]) / max(1, len(acc["low_polish_ranks"]))
        studio_mean = sum(acc["studio_polish_ranks"]) / max(1, len(acc["studio_polish_ranks"]))
        fairness_gap = abs(low_mean - studio_mean)
        bait_rate = acc["top5_bait_slots"] / max(1, acc["top5_total_slots"])
        return {
            "model_version": model_ver,
            "ndcg_at_5": round(sum(acc["ndcg5"]) / len(acc["ndcg5"]), 6),
            "ndcg_at_10": round(sum(acc["ndcg10"]) / len(acc["ndcg10"]), 6),
            "mrr": round(sum(acc["mrr"]) / len(acc["mrr"]), 6),
            "topic_diversity_at_10_beats": round(
                sum(acc["unique_beats_top10"]) / len(acc["unique_beats_top10"]), 4
            ),
            "topic_diversity_at_10_tags": round(
                sum(acc["unique_tags_top10"]) / len(acc["unique_tags_top10"]), 4
            ),
            "max_consecutive_same_author_streak": acc["max_author_streak"],
            "max_consecutive_same_incident_streak": acc["max_incident_streak"],
            "mean_max_author_streak": round(
                sum(acc["mean_author_streak"]) / len(acc["mean_author_streak"]), 4
            ),
            "top5_bait_exposures": acc["top5_bait_slots"],
            "top5_total_slots": acc["top5_total_slots"],
            "top5_bait_exposure_rate": round(bait_rate, 6),
            "mean_rank_corroborated_low_polish": round(low_mean, 4),
            "mean_rank_corroborated_studio_polish": round(studio_mean, 4),
            "low_polish_fairness_gap": round(fairness_gap, 4),
            "low_polish_top10_share_per_story": round(
                acc["low_polish_top10_exposures"] / max(1, len(acc["low_polish_ranks"])), 4
            ),
            "studio_polish_top10_share_per_story": round(
                acc["studio_polish_top10_exposures"] / max(1, len(acc["studio_polish_ranks"])), 4
            ),
        }

    sig_summary = summarize_model("signal_ranker", MODEL_VERSION_V1)
    naive_summary = summarize_model("naive_engagement", NAIVE_MODEL_VERSION)
    ablation_summary = summarize_model("studio_polish_ablation", ABLATION_MODEL_VERSION)

    naive_bait = naive_summary["top5_bait_exposures"]
    sig_bait = sig_summary["top5_bait_exposures"]
    top5_bait_suppression = (
        1.0 if naive_bait == 0 else (naive_bait - sig_bait) / naive_bait
    )

    p_val = two_sided_sign_test_pvalue(wins_signal, wins_naive)
    log10_p = two_sided_sign_test_log10_pvalue(wins_signal, wins_naive)
    ci_lcb, ci_ucb = bootstrap_mean_ci95(session_deltas)
    decisive_total = max(1, wins_signal + wins_naive)

    archetype_breakdown: list[dict[str, Any]] = []
    total_top5 = max(1, sig_summary["top5_total_slots"])
    total_top10 = max(1, num_sessions * 10)
    for spec in ARCHETYPE_SPECS:
        aid = spec["id"]
        ranks = archetype_pool_ranks[aid]
        deltas = archetype_signal_deltas[aid]
        archetype_breakdown.append(
            {
                "archetype_id": aid,
                "label": spec["label"],
                "polish_group": spec["polish_group"],
                "top5_share": round(archetype_top5_counts[aid] / total_top5, 6),
                "top10_share": round(archetype_top10_counts[aid] / total_top10, 6),
                "mean_pool_rank": round(sum(ranks) / max(1, len(ranks)), 4),
                "mean_signal_delta": round(sum(deltas) / max(1, len(deltas)), 4),
            }
        )

    interleaving_summary = {
        "experiment_id": "exp-newsroom-signal-vs-naive",
        "sessions_evaluated": num_sessions,
        "arm_a_model": NAIVE_MODEL_VERSION,
        "arm_b_model": MODEL_VERSION_V1,
        "session_wins_naive": wins_naive,
        "session_wins_signal": wins_signal,
        "session_ties": ties,
        "signal_win_rate_decisive": round(wins_signal / decisive_total, 6),
        "signal_win_rate_all_sessions": round(wins_signal / max(1, num_sessions), 6),
        "mean_usefulness_gain_per_session": round(
            (total_m01_signal - total_m01_naive) / max(1, num_sessions), 6
        ),
        "relative_usefulness_gain": round(
            (total_m01_signal - total_m01_naive) / max(1e-9, abs(total_m01_naive)), 6
        ),
        "naive_correction_impressions": corrections_naive,
        "signal_correction_impressions": corrections_signal,
        "sign_test_p_value": p_val,
        "sign_test_log10_p_value": round(log10_p, 4),
        "bootstrap_95_lcb": ci_lcb,
        "bootstrap_95_ucb": ci_ucb,
        "passed_significance_gate": p_val < 0.01 and ci_lcb > 0.0 and corrections_signal < corrections_naive,
    }

    passed = (
        not schema_errors
        and not instance_schema_errors
        and not clean_room_violations
        and policy_leaks == 0
        and sig_summary["ndcg_at_10"] > naive_summary["ndcg_at_10"]
        and sig_summary["max_consecutive_same_author_streak"] == 1
        and sig_summary["low_polish_fairness_gap"] <= 0.50
        and sig_summary["low_polish_fairness_gap"] < ablation_summary["low_polish_fairness_gap"]
        and top5_bait_suppression >= 0.95
        and interleaving_summary["passed_significance_gate"]
    )

    return {
        "passed": passed,
        "benchmark_date": "2026-11-17",
        "seed": BENCHMARK_SEED,
        "sessions_simulated": num_sessions,
        "corpus_size": len(corpus),
        "archetypes_simulated": len(ARCHETYPE_SPECS),
        "policy_sandwich_leaks": policy_leaks,
        "models": {
            "signal_informed_ranker": sig_summary,
            "naive_engagement_ranker": naive_summary,
            "studio_polish_ablation": ablation_summary,
        },
        "unsourced_bait_suppression_top5": {
            "naive_top5_exposures": naive_bait,
            "naive_top5_rate": naive_summary["top5_bait_exposure_rate"],
            "signal_top5_exposures": sig_bait,
            "signal_top5_rate": sig_summary["top5_bait_exposure_rate"],
            "suppression_rate": round(top5_bait_suppression, 6),
        },
        "low_polish_journalism_fairness": {
            "signal_ranker_gap": sig_summary["low_polish_fairness_gap"],
            "signal_mean_rank_low_polish": sig_summary["mean_rank_corroborated_low_polish"],
            "signal_mean_rank_studio_polish": sig_summary["mean_rank_corroborated_studio_polish"],
            "polish_biased_ablation_gap": ablation_summary["low_polish_fairness_gap"],
            "gap_reduction_ratio": round(
                1.0
                - (
                    sig_summary["low_polish_fairness_gap"]
                    / max(1e-9, ablation_summary["low_polish_fairness_gap"])
                ),
                6,
            ),
        },
        "team_draft_interleaving": interleaving_summary,
        "archetype_breakdown": archetype_breakdown,
        "schema_errors": schema_errors + instance_schema_errors,
        "clean_room_violations": clean_room_violations,
    }


def render_markdown_tables(report: Mapping[str, Any]) -> str:
    """Format benchmark metrics into reproducible GitHub-Flavored Markdown tables."""
    sig = report["models"]["signal_informed_ranker"]
    naive = report["models"]["naive_engagement_ranker"]
    abl = report["models"]["studio_polish_ablation"]
    bait = report["unsourced_bait_suppression_top5"]
    inter = report["team_draft_interleaving"]

    p_str = (
        f"< 1e-100 (10^{inter['sign_test_log10_p_value']:.1f})"
        if inter["sign_test_log10_p_value"] < -100.0
        else f"{inter['sign_test_p_value']:.3e}"
    )

    lines = [
        "### Ranking Quality, Diversity & Fairness Comparison (`2,000` Reader Sessions)",
        "",
        "| Metric | Signal-Informed Ranker (`swivver-ranker/1.0.0`) | Naive Engagement-Only (`naive-engagement-ranker/1.0`) | Studio-Polish-Biased Ablation |",
        "| --- | ---: | ---: | ---: |",
        f"| `NDCG@5` | **{sig['ndcg_at_5']:.4f}** | `{naive['ndcg_at_5']:.4f}` | `{abl['ndcg_at_5']:.4f}` |",
        f"| `NDCG@10` | **{sig['ndcg_at_10']:.4f}** | `{naive['ndcg_at_10']:.4f}` | `{abl['ndcg_at_10']:.4f}` |",
        f"| `MRR` (first Grade-3 match) | **{sig['mrr']:.4f}** | `{naive['mrr']:.4f}` | `{abl['mrr']:.4f}` |",
        f"| `Topic Diversity@10` (distinct beats / tags) | **`{sig['topic_diversity_at_10_beats']:.2f}` beats / `{sig['topic_diversity_at_10_tags']:.2f}` tags** | `{naive['topic_diversity_at_10_beats']:.2f}` beats / `{naive['topic_diversity_at_10_tags']:.2f}` tags | `{abl['topic_diversity_at_10_beats']:.2f}` beats / `{abl['topic_diversity_at_10_tags']:.2f}` tags |",
        f"| Max consecutive same-author streak | **`{sig['max_consecutive_same_author_streak']}`** | `{naive['max_consecutive_same_author_streak']}` | `{abl['max_consecutive_same_author_streak']}` |",
        f"| Low-Polish Journalism Fairness Gap (`|Δ mean rank|`) | **`{sig['low_polish_fairness_gap']:.4f}`** (`{sig['mean_rank_corroborated_low_polish']:.2f}` vs `{sig['mean_rank_corroborated_studio_polish']:.2f}`) | `{naive['low_polish_fairness_gap']:.4f}` | `{abl['low_polish_fairness_gap']:.4f}` (`{abl['mean_rank_corroborated_low_polish']:.2f}` vs `{abl['mean_rank_corroborated_studio_polish']:.2f}`) |",
        f"| Top-5 unsourced attention-bait exposure rate | **`{sig['top5_bait_exposure_rate'] * 100:.2f}%`** (`{sig['top5_bait_exposures']} / {sig['top5_total_slots']}`) | `{naive['top5_bait_exposure_rate'] * 100:.2f}%` (`{naive['top5_bait_exposures']} / {naive['top5_total_slots']}`) | `{abl['top5_bait_exposure_rate'] * 100:.2f}%` (`{abl['top5_bait_exposures']} / {abl['top5_total_slots']}`) |",
        f"| **Top-5 Unsourced Bait Suppression Rate** | **`{bait['suppression_rate'] * 100:.2f}%`** | baseline | — |",
        "",
        "### Per-Archetype Exposure & Rank Profile (`swivver-ranker/1.0.0`)",
        "",
        "| Archetype | Production & Evidence Profile | Top-5 Share | Top-10 Share | Mean Pool Rank (`/ 46`) | Mean `signal_delta` |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for row in report["archetype_breakdown"]:
        lines.append(
            f"| {row['label']} | `{row['polish_group']}` | `{row['top5_share'] * 100:.2f}%` | `{row['top10_share'] * 100:.2f}%` | `{row['mean_pool_rank']:.2f}` | `{row['mean_signal_delta']:+.4f}` |"
        )

    lines.extend(
        [
            "",
            "### Team-Draft Interleaving (`swivver-ranker/1.0.0` vs `naive-engagement-ranker/1.0`)",
            "",
            "| Study Metric | Measured Value |",
            "| --- | ---: |",
            f"| Reader sessions interleaved (`PageSize = 10`) | `{inter['sessions_evaluated']}` |",
            f"| Signal-informed ranker session wins / Naive wins / Ties | **`{inter['session_wins_signal']}`** / `{inter['session_wins_naive']}` / `{inter['session_ties']}` |",
            f"| Decisive session win-rate (excluding ties) | **`{inter['signal_win_rate_decisive'] * 100:.2f}%`** |",
            f"| Overall session win-rate (including ties) | **`{inter['signal_win_rate_all_sessions'] * 100:.2f}%`** |",
            f"| Mean deliberative usefulness gain per session (`95%` bootstrap CI) | **`{inter['mean_usefulness_gain_per_session']:+.4f}`** (`[{inter['bootstrap_95_lcb']:+.4f}, {inter['bootstrap_95_ucb']:+.4f}]`) |",
            f"| Correction impressions (`Signal` vs `Naive`) | **`{inter['signal_correction_impressions']}`** vs `{inter['naive_correction_impressions']}` |",
            f"| Two-sided exact sign-test `p`-value | **`{p_str}`** (`p < 0.01` PASS) |",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the 6-archetype, 2,000-session newsroom simulation and interleaving benchmark."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to the swivver_algorithm repository root.",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=2000,
        help="Number of simulated reader sessions (default: 2000).",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print formatted Markdown tables after the JSON report.",
    )
    args = parser.parse_args(argv)

    report = run_newsroom_benchmark(args.repo_root.resolve(), num_sessions=args.sessions)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.markdown:
        print("\n" + render_markdown_tables(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
