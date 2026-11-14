#!/usr/bin/env python3
"""Deterministic 5,000-package Joint Signal x Ranking validation suite (VG-3 / VG-5).

Verifies all six blocking validation gates using only the Python 3 standard library:
1. 10,000-trial Policy-Sandwich fuzzing (0.00% exposure of !eligible or policy_blocked items).
2. 5,000-candidate counterfactual uncertainty neutrality sweep (SignalDelta == 0.0, 0 demotions).
3. Ragebait & contradiction (I01/X01, D02/X02) top-10 suppression (>= 85% drop vs watch-time sort).
4. Bot / coordinated brigading immunity (100% elimination of anomaly_flagged boosts).
5. Fairness & minority-language (pt, es) parity (author Gini & language share within +-3% of supply).
6. 200-candidate ranking p95 latency gate (<= 25ms target).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.evaluate_offline import (
    EXPECTED_SCHEMA_FILES,
    author_gini_coefficient,
    counterfactual_unknown_rank_delta,
    minority_language_share_parity,
    scan_clean_room_boundary,
    validate_instance_against_schema,
    validate_schemas_directory,
)

MODEL_VERSION_V1 = "swivver-ranker/1.0.0"
CORPUS_SEED = 20261114
POLICY_FUZZ_SEED = 20261115
REF_NOW = datetime(2026, 11, 14, 12, 0, 0, tzinfo=timezone.utc)
REF_NOW_NANO = int(REF_NOW.timestamp() * 1_000_000_000)

TOPICS: tuple[str, ...] = (
    "world",
    "economy",
    "science",
    "climate",
    "technology",
    "health",
    "transit",
    "energy",
    "civic",
    "culture",
)

NON_AVAILABLE_STATES: tuple[str, ...] = (
    "unknown",
    "unsupported",
    "not_applicable",
    "stale",
)


class SplitMix64:
    """Portable deterministic 64-bit PRNG matching the Go corpus validation harness."""

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        self._state = seed & 0xFFFFFFFFFFFFFFFF

    def next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF

    def float64(self) -> float:
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def intn(self, n: int) -> int:
        if n <= 0:
            return 0
        return int(self.next_u64() % n)

    def uniform(self, low: float, high: float) -> float:
        return low + (high - low) * self.float64()


@dataclass(frozen=True)
class CorpusPackage:
    candidate: dict[str, Any]
    published_nano: int
    package_kind: str  # "news" (3,500) or "partner_media" (1,500)
    split: str  # "dev" (3,000), "calibration" (1,000), "locked_test" (1,000)
    archetype: str  # "supported", "developing", "ragebait_contradicted", "ragebait_bait", "bot_brigade", "ineligible"
    is_ragebait_or_contradicted: bool


def _build_author_schedule() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build 100 creators (60 en, 22 pt, 18 es) with realistic productivity quotas summing to 5,000."""
    authors_by_lang: dict[str, list[str]] = {"en": [], "pt": [], "es": []}
    schedule_by_lang: dict[str, list[str]] = {"en": [], "pt": [], "es": []}
    specs = (("en", 60, 3000), ("pt", 22, 1100), ("es", 18, 900))
    author_seq = 0

    for lang, n_authors, target_total in specs:
        raw_weights = [0.55 + 1.85 * ((i % 10) / 9.0) for i in range(n_authors)]
        weight_sum = sum(raw_weights)
        quotas = [int(round((w / weight_sum) * target_total)) for w in raw_weights]
        diff = target_total - sum(quotas)
        for k in range(abs(diff)):
            quotas[k % n_authors] += 1 if diff > 0 else -1

        lang_ids: list[str] = []
        remaining = list(quotas)
        for _ in range(n_authors):
            aid = f"author-{author_seq:03d}"
            author_seq += 1
            lang_ids.append(aid)
        authors_by_lang[lang] = lang_ids

        # Round-robin interleave quotas across the 14-day timeline so authors publish steadily over time.
        expanded: list[str] = []
        while len(expanded) < target_total:
            for idx, aid in enumerate(lang_ids):
                if remaining[idx] > 0:
                    expanded.append(aid)
                    remaining[idx] -= 1
        schedule_by_lang[lang] = expanded

    return authors_by_lang, schedule_by_lang


def generate_5k_corpus(seed: int = CORPUS_SEED) -> list[CorpusPackage]:
    """Generate the deterministic 5,000-package stratified multilingual validation corpus.

    Composition matches roadmap Section 6.2:
    - 3,500 news Story packages + 1,500 standalone/partner media packages = 5,000 total.
    - Event/creator-stratified splits: 3,000 dev, 1,000 calibration, 1,000 locked_test.
    - Languages: 60% en (3,000), 22% pt (1,100), 18% es (900).
    """
    rng = SplitMix64(seed)
    _, schedule_by_lang = _build_author_schedule()
    lang_cursors: dict[str, int] = {"en": 0, "pt": 0, "es": 0}

    corpus: list[CorpusPackage] = []
    for idx in range(5000):
        package_kind = "news" if idx < 3500 else "partner_media"
        if idx < 3000:
            split = "dev"
        elif idx < 4000:
            split = "calibration"
        else:
            split = "locked_test"

        lang_slot = idx % 50
        if lang_slot < 30:
            lang = "en"
        elif lang_slot < 41:
            lang = "pt"
        else:
            lang = "es"

        cur_pos = lang_cursors[lang]
        author_id = schedule_by_lang[lang][cur_pos]
        lang_cursors[lang] = cur_pos + 1
        primary_topic = TOPICS[(idx // 50 + lang_slot) % len(TOPICS)]
        if rng.float64() < 0.35:
            secondary_topic = TOPICS[(idx // len(TOPICS) + 1 + rng.intn(len(TOPICS) - 1)) % len(TOPICS)]
            topic_tags = sorted({primary_topic, secondary_topic})
        else:
            topic_tags = [primary_topic]

        # Stratified 14-day publication timeline (20,160 minutes) prior to REF_NOW.
        timeline_progress = (4999 - idx) / 5000.0
        jitter_min = rng.intn(45)
        age_minutes = int(timeline_progress * 20160) + jitter_min + 5
        pub_dt = REF_NOW - timedelta(minutes=age_minutes)
        pub_iso = pub_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        pub_nano = REF_NOW_NANO - age_minutes * 60 * 1_000_000_000

        # Orthogonal bucket bijection across each language stratum (gcd(37, 100) == 1).
        bucket = (((idx // 50) * 37) + (lang_slot * 7)) % 100
        eligible = True
        policy_blocked = False
        anomaly_flagged = False
        is_ragebait = False
        signal: dict[str, Any] | None = None

        if bucket < 5:
            # 5%: Policy-ineligible / safety or regional rights hold (250 packages).
            archetype = "ineligible"
            eligible = idx % 2 == 0
            policy_blocked = not eligible or (idx % 3 == 0)
            if eligible and not policy_blocked:
                eligible = False
            watch_ratio = 0.99
            article_open = 0.95
            source_inspect = 0.90
            save_rate = 0.85
            impressions = 500
            signal = {
                "schema_version": "scoped-signal-features/1.0",
                "snapshot_id": f"snap-{idx:05d}",
                "availability": "available",
                "coverage_ratio": 1.0,
                "freshness_sec": 30.0,
                "evidence_support_tier": "supported",
                "consistency_tier": "consistent",
                "technical_legibility_tier": "clear",
                "provenance_status": "verified",
                "independent_source_clusters": 4,
            }
        elif bucket < 12:
            # 7%: Coordinated bot / velocity brigading spike (350 packages).
            archetype = "bot_brigade"
            anomaly_flagged = True
            watch_ratio = round(rng.uniform(0.92, 1.00), 6)
            article_open = round(rng.uniform(0.85, 0.99), 6)
            source_inspect = round(rng.uniform(0.80, 0.98), 6)
            save_rate = round(rng.uniform(0.75, 0.95), 6)
            impressions = rng.intn(60) + 5
        elif bucket < 21:
            # 9%: Contradicted viral claims (I01 / X01) with high clickbait watch completion (450 packages).
            archetype = "ragebait_contradicted"
            is_ragebait = True
            watch_ratio = round(rng.uniform(0.88, 0.995), 6)
            article_open = round(rng.uniform(0.005, 0.045), 6)
            source_inspect = round(rng.uniform(0.0, 0.02), 6)
            save_rate = round(rng.uniform(0.0, 0.015), 6)
            impressions = rng.intn(900) + 150
            signal = {
                "schema_version": "scoped-signal-features/1.0",
                "snapshot_id": f"snap-{idx:05d}",
                "availability": "available",
                "coverage_ratio": 0.90,
                "freshness_sec": round(rng.uniform(20.0, 1200.0), 3),
                "evidence_support_tier": "contradicted",
                "consistency_tier": "inconsistent" if idx % 2 == 0 else "mixed",
                "technical_legibility_tier": "clear",
                "attention_pacing_score": round(rng.uniform(0.72, 0.96), 6),
                "bait_switch_risk": round(rng.uniform(0.55, 0.92), 6),
                "provenance_status": "missing",
                "independent_source_clusters": 0,
            }
        elif bucket < 30:
            # 9%: Sensational bait-and-switch & jolt hooks (D02 / X02 / D01) (450 packages).
            archetype = "ragebait_bait"
            is_ragebait = True
            watch_ratio = round(rng.uniform(0.87, 0.99), 6)
            article_open = round(rng.uniform(0.01, 0.05), 6)
            source_inspect = round(rng.uniform(0.0, 0.025), 6)
            save_rate = round(rng.uniform(0.0, 0.02), 6)
            impressions = rng.intn(850) + 140
            signal = {
                "schema_version": "scoped-signal-features/1.0",
                "snapshot_id": f"snap-{idx:05d}",
                "availability": "available",
                "coverage_ratio": 0.85,
                "freshness_sec": round(rng.uniform(25.0, 1800.0), 3),
                "evidence_support_tier": "mixed",
                "consistency_tier": "inconsistent",
                "technical_legibility_tier": "degraded" if idx % 3 == 0 else "clear",
                "attention_pacing_score": round(rng.uniform(0.80, 0.98), 6),
                "bait_switch_risk": round(rng.uniform(0.78, 0.98), 6),
                "provenance_status": "missing",
                "independent_source_clusters": 0,
            }
        elif bucket < 65:
            # 35%: Developing, cold-start, and non-available Signal states across en/pt/es (1,750 packages).
            archetype = "developing"
            watch_ratio = round(rng.uniform(0.42, 0.76), 6)
            article_open = round(rng.uniform(0.18, 0.42), 6)
            source_inspect = round(rng.uniform(0.10, 0.28), 6)
            save_rate = round(rng.uniform(0.04, 0.13), 6)
            impressions = rng.intn(180)
            state_pick = idx % 6
            if state_pick == 0:
                signal = None
            elif state_pick in (1, 2, 3, 4):
                signal = {
                    "schema_version": "scoped-signal-features/1.0",
                    "snapshot_id": f"snap-{idx:05d}",
                    "availability": NON_AVAILABLE_STATES[state_pick - 1],
                    "coverage_ratio": 0.0,
                    "freshness_sec": round(rng.uniform(10.0, 600.0), 3),
                }
            else:
                signal = {
                    "schema_version": "scoped-signal-features/1.0",
                    "snapshot_id": f"snap-{idx:05d}",
                    "availability": "available",
                    "coverage_ratio": 0.50,
                    "freshness_sec": round(rng.uniform(15.0, 900.0), 3),
                    "evidence_support_tier": "unknown",
                    "consistency_tier": "consistent",
                    "technical_legibility_tier": "clear",
                    "provenance_status": "missing",
                    "independent_source_clusters": 1,
                }
        else:
            # 35%: Source-supported deliberative packages across en/pt/es (1,750 packages).
            archetype = "supported"
            watch_ratio = round(rng.uniform(0.45, 0.78), 6)
            article_open = round(rng.uniform(0.24, 0.48), 6)
            source_inspect = round(rng.uniform(0.15, 0.36), 6)
            save_rate = round(rng.uniform(0.06, 0.16), 6)
            impressions = rng.intn(450) + 30
            signal = {
                "schema_version": "scoped-signal-features/1.0",
                "snapshot_id": f"snap-{idx:05d}",
                "availability": "available",
                "coverage_ratio": 1.0,
                "freshness_sec": round(rng.uniform(15.0, 2400.0), 3),
                "evidence_support_tier": "supported",
                "consistency_tier": "consistent",
                "technical_legibility_tier": "clear",
                "attention_pacing_score": round(rng.uniform(0.15, 0.55), 6),
                "bait_switch_risk": round(rng.uniform(0.01, 0.14), 6),
                "provenance_status": "verified" if idx % 3 == 0 else "declared",
                "independent_source_clusters": 2 + (idx % 3),
            }

        cand: dict[str, Any] = {
            "schema_version": "candidate-feature-vector/1.0",
            "candidate_id": f"pkg-{idx:05d}",
            "author_id": author_id,
            "topic_tags": topic_tags,
            "language": lang,
            "published_at": pub_iso,
            "eligible": eligible,
            "policy_blocked": policy_blocked,
            "followed_author": False,
            "followed_topic_matches": 1,
            "language_match": True,
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
            CorpusPackage(
                candidate=cand,
                published_nano=pub_nano,
                package_kind=package_kind,
                split=split,
                archetype=archetype,
                is_ragebait_or_contradicted=is_ragebait,
            )
        )

    return corpus


def _clamp_unit(val: float) -> float:
    if not math.isfinite(val) or val <= 0.0:
        return 0.0
    if val >= 1.0:
        return 1.0
    return val


def _is_signal_active(sig: Mapping[str, Any] | None) -> bool:
    if sig is None or sig.get("availability") != "available":
        return False
    fresh = float(sig.get("freshness_sec", -1.0))
    if not math.isfinite(fresh) or fresh < 0.0:
        return False
    return fresh <= 86400.0 * 7.0


def _is_dim_available(sig: Mapping[str, Any] | None, dim: str) -> bool:
    if not _is_signal_active(sig):
        return False
    assert sig is not None
    dim_map = sig.get("dimension_availability")
    if not dim_map:
        return True
    if dim not in dim_map:
        return True
    return dim_map[dim] == "available"


def _has_any_available_dim(sig: Mapping[str, Any] | None) -> bool:
    if not _is_signal_active(sig):
        return False
    assert sig is not None
    if _is_dim_available(sig, "evidence_support_tier") and sig.get("evidence_support_tier") not in (None, "", "unknown"):
        return True
    if _is_dim_available(sig, "consistency_tier") and sig.get("consistency_tier") not in (None, "", "unknown"):
        return True
    if _is_dim_available(sig, "technical_legibility_tier") and sig.get("technical_legibility_tier") not in (None, "", "unknown"):
        return True
    if _is_dim_available(sig, "provenance_status") and sig.get("provenance_status") not in (None, "", "unknown"):
        return True
    if _is_dim_available(sig, "independent_source_clusters") and int(sig.get("independent_source_clusters", 0)) > 0:
        return True
    if _is_dim_available(sig, "attention_pacing_score") and _clamp_unit(float(sig.get("attention_pacing_score", 0.0))) > 0.0:
        return True
    if _is_dim_available(sig, "bait_switch_risk") and _clamp_unit(float(sig.get("bait_switch_risk", 0.0))) > 0.0:
        return True
    return False


def score_candidate_components(
    cand: Mapping[str, Any],
    published_nano: int,
    now_nano: int = REF_NOW_NANO,
) -> dict[str, Any]:
    """Reference clean-room scoring mirroring pkg/ranker DefaultConfig."""
    pref_author = 0.30 if cand.get("followed_author") else 0.0
    t_matches = int(cand.get("followed_topic_matches", 0))
    if t_matches == 1:
        pref_topic = 0.65 * 0.22
    elif t_matches >= 2:
        pref_topic = 0.22
    else:
        pref_topic = 0.0
    pref_lang = 0.15 if cand.get("language_match") else 0.0
    preference = pref_author + pref_topic + pref_lang

    age_hours = max(0.0, (now_nano - published_nano) / 3_600_000_000_000.0)
    freshness = 0.20 * math.pow(2.0, -age_hours / 18.0)

    anomaly = bool(cand.get("anomaly_flagged", False))
    engagement = 0.0
    if not anomaly:
        art = _clamp_unit(float(cand.get("article_open_rate", 0.0)))
        src = _clamp_unit(float(cand.get("source_inspect_rate", 0.0)))
        sav = _clamp_unit(float(cand.get("save_rate", 0.0)))
        wat = _clamp_unit(float(cand.get("valid_watch_ratio", 0.0)))
        engagement = 0.25 * (0.40 * art + 0.35 * src + 0.15 * sav + 0.10 * wat)

    sig = cand.get("signal")
    delta = 0.0
    if _is_signal_active(sig):
        assert sig is not None
        if _is_dim_available(sig, "evidence_support_tier"):
            ev = sig.get("evidence_support_tier")
            if ev == "supported":
                delta += 0.10
            elif ev == "contradicted":
                delta -= 0.22
        if _is_dim_available(sig, "consistency_tier"):
            ct = sig.get("consistency_tier")
            if ct == "consistent":
                delta += 0.03
            elif ct == "inconsistent":
                delta -= 0.10
        if _is_dim_available(sig, "technical_legibility_tier"):
            tl = sig.get("technical_legibility_tier")
            if tl == "clear":
                delta += 0.02
            elif tl == "degraded":
                delta -= 0.03
            elif tl == "unusable":
                delta -= 0.12
        if _is_dim_available(sig, "provenance_status"):
            pv = sig.get("provenance_status")
            if pv == "verified":
                delta += 0.03
            elif pv == "declared":
                delta += 0.01
            elif pv == "invalid":
                delta -= 0.10
        if _is_dim_available(sig, "independent_source_clusters"):
            clusters = int(sig.get("independent_source_clusters", 0))
            if clusters == 1:
                delta += 0.02
            elif clusters >= 2:
                delta += 0.04
        if _is_dim_available(sig, "attention_pacing_score"):
            pacing = _clamp_unit(float(sig.get("attention_pacing_score", 0.0)))
            if pacing > 0.70:
                delta -= 0.08 * ((pacing - 0.70) / 0.30)
        if _is_dim_available(sig, "bait_switch_risk"):
            risk = _clamp_unit(float(sig.get("bait_switch_risk", 0.0)))
            if risk > 0.35:
                delta -= 0.20 * ((risk - 0.35) / 0.65)
        delta = max(-0.35, min(0.18, delta))

    impressions = max(0, int(cand.get("impression_count", 0)))
    b0 = 0.03 if (not anomaly and age_hours <= 12.0 and impressions <= 100) else 0.0
    exploration = (b0 - delta) if (delta >= 0.0 and b0 > delta) else 0.0
    raw_score = preference + freshness + engagement + delta + exploration

    return {
        "candidate_id": str(cand["candidate_id"]).strip(),
        "author_id": str(cand.get("author_id", "")).strip(),
        "topics": tuple(sorted({t.strip().lower() for t in cand.get("topic_tags", ()) if t.strip()})),
        "language": str(cand.get("language", "")).strip().lower(),
        "published_nano": published_nano,
        "preference": preference,
        "freshness": freshness,
        "engagement": engagement,
        "signal_delta": delta,
        "exploration_bonus": exploration,
        "raw_score": raw_score,
        "is_cold_start": (not _has_any_available_dim(sig)) and exploration > 0.0,
    }


def _compare_candidates(
    score_a: float,
    score_b: float,
    pub_a: int,
    pub_b: int,
    id_a: str,
    id_b: str,
) -> int:
    diff = score_a - score_b
    if diff > 1e-12:
        return -1
    if diff < -1e-12:
        return 1
    if pub_a != pub_b:
        return -1 if pub_a > pub_b else 1
    if id_a == id_b:
        return 0
    return -1 if id_a < id_b else 1


def rank_candidates_reference(
    candidates: Sequence[tuple[Mapping[str, Any], int]],
    page_size: int = 10,
    now_nano: int = REF_NOW_NANO,
) -> tuple[list[dict[str, Any]], int]:
    """Reference Ranker.Rank implementation with policy gate, MMR, and sliding-window caps."""
    seen_ids: set[str] = set()
    scored: list[dict[str, Any]] = []
    dropped_ineligible = 0

    for cand, pub_nano in candidates:
        cid = str(cand.get("candidate_id", "")).strip()
        if not cand.get("eligible", False) or cand.get("policy_blocked", False) or not cid:
            dropped_ineligible += 1
            continue
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        scored.append(score_candidate_components(cand, pub_nano, now_nano))

    # Sort descending by raw_score, published_nano, ascending candidate_id.
    scored.sort(key=lambda s: (-round(s["raw_score"], 12), -s["published_nano"], s["candidate_id"]))

    target = min(page_size, len(scored))
    if target <= 0:
        return [], dropped_ineligible

    window_span = 4  # DiversityWindowSize (5) - 1
    recent_authors: list[str] = []
    recent_topics: list[tuple[str, ...]] = []
    author_counts: Counter[str] = Counter()
    used = [False] * len(scored)
    selected: list[dict[str, Any]] = []

    for _ in range(target):
        best_strict_idx = -1
        best_strict_score = 0.0
        best_author_idx = -1
        best_author_score = 0.0
        best_fallback_idx = -1
        best_fallback_score = 0.0
        penalties: dict[int, float] = {}

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
            if ctopics and recent_topics:
                cset = set(ctopics)
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
            penalties[i] = pen

            a_blocked = bool(aid) and recent_authors.count(aid) >= 1
            t_blocked = False
            if ctopics and recent_topics:
                for t in ctopics:
                    if sum(1 for prev in recent_topics if t in prev) >= 2:
                        t_blocked = True
                        break

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
                best_fallback_idx = i
                best_fallback_score = eff

            if not a_blocked:
                if best_author_idx == -1 or (
                    _compare_candidates(
                        eff,
                        best_author_score,
                        cand["published_nano"],
                        scored[best_author_idx]["published_nano"],
                        cand["candidate_id"],
                        scored[best_author_idx]["candidate_id"],
                    )
                    < 0
                ):
                    best_author_idx = i
                    best_author_score = eff
                if not t_blocked:
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
                        best_strict_idx = i
                        best_strict_score = eff

        if best_strict_idx != -1:
            chosen_idx, chosen_score = best_strict_idx, best_strict_score
        elif best_author_idx != -1:
            chosen_idx, chosen_score = best_author_idx, best_author_score
        else:
            chosen_idx, chosen_score = best_fallback_idx, best_fallback_score

        used[chosen_idx] = True
        picked = dict(scored[chosen_idx])
        picked["diversity_penalty"] = round(penalties[chosen_idx], 6)
        picked["score"] = round(chosen_score, 6)
        picked["rank"] = len(selected) + 1
        selected.append(picked)

        if picked["author_id"]:
            author_counts[picked["author_id"]] += 1
        recent_authors.append(picked["author_id"])
        if len(recent_authors) > window_span:
            recent_authors = recent_authors[-window_span:]
        recent_topics.append(picked["topics"])
        if len(recent_topics) > window_span:
            recent_topics = recent_topics[-window_span:]

    return selected, dropped_ineligible


def run_policy_fuzz_10k(
    corpus: Sequence[CorpusPackage],
    trials: int = 10000,
    seed: int = POLICY_FUZZ_SEED,
) -> dict[str, Any]:
    """Gate 1: 10,000 randomized Policy-Sandwich fuzz runs verifying 0.00% leakage."""
    rng = SplitMix64(seed)
    n_corpus = len(corpus)
    leaked_trials = 0
    leaked_items = 0
    total_ineligible_tested = 0

    for trial in range(trials):
        batch_size = 12 + rng.intn(16)
        batch: list[tuple[dict[str, Any], int]] = []
        forbidden_ids: set[str] = set()
        expected_drops = 0

        for slot in range(batch_size):
            pkg = corpus[(trial * 13 + slot * 37 + rng.intn(n_corpus)) % n_corpus]
            cand = dict(pkg.candidate)
            cid = f"fuzz-{trial}-{slot}"
            cand["candidate_id"] = cid
            make_ineligible = rng.float64() < 0.35

            if make_ineligible:
                mode = (trial + slot) % 3
                if mode == 0:
                    cand["eligible"] = False
                    cand["policy_blocked"] = False
                elif mode == 1:
                    cand["eligible"] = True
                    cand["policy_blocked"] = True
                else:
                    cand["eligible"] = False
                    cand["policy_blocked"] = True
                # Extreme engagement (1.0) and maximum Signal support (1.0).
                cand["followed_author"] = True
                cand["followed_topic_matches"] = 5
                cand["language_match"] = True
                cand["valid_watch_ratio"] = 1.0
                cand["article_open_rate"] = 1.0
                cand["source_inspect_rate"] = 1.0
                cand["save_rate"] = 1.0
                cand["anomaly_flagged"] = False
                cand["signal"] = {
                    "schema_version": "scoped-signal-features/1.0",
                    "availability": "available",
                    "coverage_ratio": 1.0,
                    "freshness_sec": 5.0,
                    "evidence_support_tier": "supported",
                    "consistency_tier": "consistent",
                    "technical_legibility_tier": "clear",
                    "provenance_status": "verified",
                    "independent_source_clusters": 5,
                }
                forbidden_ids.add(cid)
                expected_drops += 1
                total_ineligible_tested += 1
            else:
                cand["eligible"] = True
                cand["policy_blocked"] = False

            batch.append((cand, pkg.published_nano))

        ranked, dropped = rank_candidates_reference(batch, page_size=10)
        trial_leaked = False
        if dropped != expected_drops:
            trial_leaked = True
        for item in ranked:
            if item["candidate_id"] in forbidden_ids or not item["candidate_id"]:
                leaked_items += 1
                trial_leaked = True
        if trial_leaked:
            leaked_trials += 1

    return {
        "trials": trials,
        "ineligible_candidates_tested": total_ineligible_tested,
        "leaked_trials": leaked_trials,
        "leaked_items": leaked_items,
        "leak_rate": 0.0 if trials == 0 else round(leaked_trials / trials, 8),
        "passed": leaked_trials == 0 and leaked_items == 0,
    }


def run_counterfactual_5k_sweep(corpus: Sequence[CorpusPackage]) -> dict[str, Any]:
    """Gate 2: 5,000-candidate sweep over nil, unknown, unsupported, not_applicable, stale."""
    max_abs_signal_delta = 0.0
    max_abs_score_diff = 0.0
    non_zero_delta_count = 0

    for pkg in corpus:
        base_cand = dict(pkg.candidate)
        base_cand["eligible"] = True
        base_cand["policy_blocked"] = False
        base_cand.pop("signal", None)
        nil_scored = score_candidate_components(base_cand, pkg.published_nano)
        if abs(nil_scored["signal_delta"]) > max_abs_signal_delta:
            max_abs_signal_delta = abs(nil_scored["signal_delta"])

        for state in NON_AVAILABLE_STATES:
            cf_cand = dict(base_cand)
            cf_cand["signal"] = {
                "schema_version": "scoped-signal-features/1.0",
                "availability": state,
                "coverage_ratio": 0.0,
                "freshness_sec": 60.0,
                "evidence_support_tier": "contradicted",
                "consistency_tier": "inconsistent",
                "technical_legibility_tier": "unusable",
                "attention_pacing_score": 0.99,
                "bait_switch_risk": 0.99,
                "provenance_status": "invalid",
            }
            cf_scored = score_candidate_components(cf_cand, pkg.published_nano)
            abs_delta = abs(cf_scored["signal_delta"])
            score_diff = abs(cf_scored["raw_score"] - nil_scored["raw_score"])
            if abs_delta > max_abs_signal_delta:
                max_abs_signal_delta = abs_delta
            if score_diff > max_abs_score_diff:
                max_abs_score_diff = score_diff
            if abs_delta >= 1e-12 or score_diff >= 1e-12:
                non_zero_delta_count += 1

    # Partition all 5,000 candidates into 250 pools of 20 items and verify 0 rank demotions.
    total_demotions = 0
    evaluated_targets = 0
    for pool_idx in range(0, len(corpus), 20):
        chunk = corpus[pool_idx : pool_idx + 20]
        base_pool: list[tuple[dict[str, Any], int]] = []
        cf_pool: list[tuple[dict[str, Any], int]] = []
        target_ids: list[str] = []

        for offset, pkg in enumerate(chunk):
            c0 = dict(pkg.candidate)
            c0["eligible"] = True
            c0["policy_blocked"] = False
            c0.pop("signal", None)
            base_pool.append((c0, pkg.published_nano))

            c1 = dict(c0)
            state = NON_AVAILABLE_STATES[(pool_idx + offset) % len(NON_AVAILABLE_STATES)]
            c1["signal"] = {
                "schema_version": "scoped-signal-features/1.0",
                "availability": state,
                "coverage_ratio": 0.0,
                "freshness_sec": 90.0,
                "evidence_support_tier": "contradicted",
                "bait_switch_risk": 0.95,
            }
            cf_pool.append((c1, pkg.published_nano))
            target_ids.append(c0["candidate_id"])

        base_ranked, _ = rank_candidates_reference(base_pool, page_size=20)
        cf_ranked, _ = rank_candidates_reference(cf_pool, page_size=20)
        check = counterfactual_unknown_rank_delta(
            [it["candidate_id"] for it in base_ranked],
            [it["candidate_id"] for it in cf_ranked],
            target_ids,
        )
        evaluated_targets += check["evaluated_targets"]
        total_demotions += check["demoted_count"]

    return {
        "candidates_swept": len(corpus),
        "state_evaluations": len(corpus) * (1 + len(NON_AVAILABLE_STATES)),
        "evaluated_rank_targets": evaluated_targets,
        "max_abs_signal_delta": round(max_abs_signal_delta, 15),
        "max_abs_score_diff": round(max_abs_score_diff, 15),
        "non_zero_delta_count": non_zero_delta_count,
        "rank_demotions": total_demotions,
        "passed": (
            len(corpus) == 5000
            and non_zero_delta_count == 0
            and max_abs_signal_delta < 1e-12
            and total_demotions == 0
        ),
    }


def run_bot_immunity_check(corpus: Sequence[CorpusPackage]) -> dict[str, Any]:
    """Gate 4: Verify AnomalyFlagged == true eliminates 100% of artificial engagement/exploration boosts."""
    flagged_tested = 0
    violations = 0

    for pkg in corpus:
        cand = dict(pkg.candidate)
        cand["eligible"] = True
        cand["policy_blocked"] = False
        if not cand.get("anomaly_flagged", False):
            continue
        flagged_tested += 1
        eval_now = pkg.published_nano + 3_600_000_000_000  # 1h post-publish (within cold-start window)
        scored = score_candidate_components(cand, pkg.published_nano, now_nano=eval_now)

        clean_zero = dict(cand)
        clean_zero["anomaly_flagged"] = False
        clean_zero["valid_watch_ratio"] = 0.0
        clean_zero["article_open_rate"] = 0.0
        clean_zero["source_inspect_rate"] = 0.0
        clean_zero["save_rate"] = 0.0
        clean_zero["impression_count"] = 500  # Outside cold-start envelope
        baseline_scored = score_candidate_components(clean_zero, pkg.published_nano, now_nano=eval_now)

        if (
            abs(scored["engagement"]) > 1e-12
            or abs(scored["exploration_bonus"]) > 1e-12
            or scored["raw_score"] > baseline_scored["raw_score"] + 1e-12
        ):
            violations += 1

    elimination_rate = 1.0 if flagged_tested > 0 and violations == 0 else 0.0
    return {
        "anomaly_candidates_tested": flagged_tested,
        "boost_violations": violations,
        "boost_elimination_rate": elimination_rate,
        "passed": flagged_tested > 0 and violations == 0,
    }


def run_ragebait_fairness_and_latency_suite(
    corpus: Sequence[CorpusPackage],
    num_sessions: int = 500,
) -> dict[str, Any]:
    """Gates 3, 5, and 6 across 500 temporal viewer sessions (200-candidate pools, top-10 feeds)."""
    eligible_pkgs = [
        p for p in corpus if p.candidate["eligible"] and not p.candidate["policy_blocked"]
    ]
    by_lang: dict[str, list[CorpusPackage]] = {"en": [], "pt": [], "es": []}
    for p in eligible_pkgs:
        by_lang[p.candidate["language"]].append(p)
    ragebait_ids = {p.candidate["candidate_id"] for p in eligible_pkgs if p.is_ragebait_or_contradicted}
    pkg_by_id = {p.candidate["candidate_id"]: p for p in eligible_pkgs}

    authors_by_lang, _ = _build_author_schedule()

    supply_languages = [p.candidate["language"] for p in eligible_pkgs]
    supply_author_counts = Counter(p.candidate["author_id"] for p in eligible_pkgs)
    ranked_author_counts: Counter[str] = Counter({aid: 0 for aid in supply_author_counts})

    ranked_languages: list[str] = []
    watch_ragebait_top10 = 0
    ranker_ragebait_top10 = 0
    total_top10_slots = 0
    latencies_ms: list[float] = []

    n_elig = len(eligible_pkgs)
    for s in range(num_sessions):
        # Stratified viewer cohort matching corpus language & topic distribution (60% en, 22% pt, 18% es).
        lang_mod = s % 50
        if lang_mod < 30:
            viewer_lang = "en"
        elif lang_mod < 41:
            viewer_lang = "pt"
        else:
            viewer_lang = "es"

        pref_t1 = TOPICS[s % len(TOPICS)]
        pref_t2 = TOPICS[(s * 3 + 1) % len(TOPICS)]
        pref_topics = {pref_t1, pref_t2}

        lang_author_pool = authors_by_lang[viewer_lang]
        followed_authors = {
            lang_author_pool[(s * 3 + k * 7) % len(lang_author_pool)] for k in range(4)
        }

        # Retrieve 160 in-language + 40 cross-language discovery candidates prior to session arrival.
        lang_elig = by_lang[viewer_lang]
        end_lang = int(((s + 1) / num_sessions) * (len(lang_elig) - 1))
        in_lang_slice = lang_elig[:160] if end_lang < 160 else lang_elig[end_lang - 160 : end_lang]

        end_all = int(((s + 1) / num_sessions) * (n_elig - 1))
        seen_pool = {p.candidate["candidate_id"] for p in in_lang_slice}
        cross_slice: list[CorpusPackage] = []
        probe = end_all
        while len(cross_slice) < 40 and probe >= 0:
            cand_pkg = eligible_pkgs[probe]
            probe -= 1
            cid = cand_pkg.candidate["candidate_id"]
            if cid not in seen_pool:
                seen_pool.add(cid)
                cross_slice.append(cand_pkg)

        window_slice = list(in_lang_slice) + cross_slice
        session_now_nano = max(p.published_nano for p in window_slice) + 300 * 1_000_000_000

        session_batch: list[tuple[dict[str, Any], int]] = []
        for p in window_slice:
            c = dict(p.candidate)
            matches = sum(1 for t in c["topic_tags"] if t in pref_topics)
            c["followed_topic_matches"] = matches
            c["language_match"] = c["language"] == viewer_lang
            c["followed_author"] = c["author_id"] in followed_authors
            session_batch.append((c, p.published_nano))

        # Pure watch-time baseline top-10 sort.
        watch_sorted = sorted(
            session_batch,
            key=lambda item: (
                -float(item[0]["valid_watch_ratio"]),
                -item[1],
                str(item[0]["candidate_id"]),
            ),
        )[:10]
        for c, _ in watch_sorted:
            if c["candidate_id"] in ragebait_ids:
                watch_ragebait_top10 += 1

        # Clean-room v1.0 ranker top-10 with latency timing.
        t0 = time.perf_counter()
        top10, _ = rank_candidates_reference(session_batch, page_size=10, now_nano=session_now_nano)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        for item in top10:
            cid = item["candidate_id"]
            total_top10_slots += 1
            if cid in ragebait_ids:
                ranker_ragebait_top10 += 1
            orig = pkg_by_id[cid]
            ranked_languages.append(orig.candidate["language"])
            ranked_author_counts[orig.candidate["author_id"]] += 1

    suppression_ratio = (
        1.0
        if watch_ragebait_top10 == 0
        else (watch_ragebait_top10 - ranker_ragebait_top10) / watch_ragebait_top10
    )

    supply_gini = author_gini_coefficient(supply_author_counts)
    ranked_gini = author_gini_coefficient(ranked_author_counts)
    gini_delta = ranked_gini - supply_gini

    lang_parity = minority_language_share_parity(
        pool_languages=supply_languages,
        ranked_languages=ranked_languages,
        primary_language="en",
    )
    supply_pt = sum(1 for l in supply_languages if l == "pt") / len(supply_languages)
    ranked_pt = sum(1 for l in ranked_languages if l == "pt") / len(ranked_languages)
    supply_es = sum(1 for l in supply_languages if l == "es") / len(supply_languages)
    ranked_es = sum(1 for l in ranked_languages if l == "es") / len(ranked_languages)

    sorted_lat = sorted(latencies_ms)
    p50_ms = sorted_lat[int(0.50 * (len(sorted_lat) - 1))]
    p95_ms = sorted_lat[int(0.95 * (len(sorted_lat) - 1))]
    p99_ms = sorted_lat[int(0.99 * (len(sorted_lat) - 1))]

    return {
        "ragebait_suppression": {
            "sessions": num_sessions,
            "top10_slots_evaluated": total_top10_slots,
            "watch_sort_ragebait_exposures": watch_ragebait_top10,
            "watch_sort_ragebait_share": round(watch_ragebait_top10 / total_top10_slots, 6),
            "ranker_ragebait_exposures": ranker_ragebait_top10,
            "ranker_ragebait_share": round(ranker_ragebait_top10 / total_top10_slots, 6),
            "relative_reduction": round(suppression_ratio, 6),
            "passed": suppression_ratio >= 0.85,
        },
        "fairness_and_language_parity": {
            "supply_author_gini": round(supply_gini, 6),
            "ranked_author_gini": round(ranked_gini, 6),
            "author_gini_delta": round(gini_delta, 6),
            "minority_language_parity": lang_parity,
            "pt_supply_share": round(supply_pt, 6),
            "pt_ranked_share": round(ranked_pt, 6),
            "pt_share_delta": round(ranked_pt - supply_pt, 6),
            "es_supply_share": round(supply_es, 6),
            "es_ranked_share": round(ranked_es, 6),
            "es_share_delta": round(ranked_es - supply_es, 6),
            "passed": (
                abs(gini_delta) <= 0.03
                and abs(lang_parity["share_delta"]) <= 0.03
                and abs(ranked_pt - supply_pt) <= 0.03
                and abs(ranked_es - supply_es) <= 0.03
            ),
        },
        "latency_200_candidates": {
            "samples": len(sorted_lat),
            "p50_ms": round(p50_ms, 4),
            "p95_ms": round(p95_ms, 4),
            "p99_ms": round(p99_ms, 4),
            "target_p95_ms": 25.0,
            "passed": p95_ms <= 25.0,
        },
    }


def run_full_5k_validation(repo_root: Path) -> dict[str, Any]:
    """Execute the complete 5,000-package VG-3 / VG-5 suite and contract checks."""
    schemas_dir = repo_root / "schemas"
    schema_errors = validate_schemas_directory(schemas_dir)
    clean_room_violations = scan_clean_room_boundary(repo_root)

    corpus = generate_5k_corpus(CORPUS_SEED)

    # Validate representative stratified corpus candidates against Draft 2020-12 schemas.
    instance_schema_errors: list[str] = []
    if not schema_errors:
        loaded_schemas = {
            name: json.loads((schemas_dir / name).read_text(encoding="utf-8"))
            for name in EXPECTED_SCHEMA_FILES
        }
        cand_schema = loaded_schemas["candidate-feature-vector-v1.schema.json"]
        for idx in range(0, len(corpus), 25):
            errs = validate_instance_against_schema(
                corpus[idx].candidate, cand_schema, loaded_schemas
            )
            if errs:
                instance_schema_errors.extend(errs)
                break

    policy_fuzz = run_policy_fuzz_10k(corpus, trials=10000, seed=POLICY_FUZZ_SEED)
    cf_sweep = run_counterfactual_5k_sweep(corpus)
    bot_immunity = run_bot_immunity_check(corpus)
    multi_suite = run_ragebait_fairness_and_latency_suite(corpus, num_sessions=500)

    kind_counts = Counter(p.package_kind for p in corpus)
    split_counts = Counter(p.split for p in corpus)
    lang_counts = Counter(p.candidate["language"] for p in corpus)

    all_passed = (
        not schema_errors
        and not instance_schema_errors
        and not clean_room_violations
        and len(corpus) == 5000
        and policy_fuzz["passed"]
        and cf_sweep["passed"]
        and bot_immunity["passed"]
        and multi_suite["ragebait_suppression"]["passed"]
        and multi_suite["fairness_and_language_parity"]["passed"]
        and multi_suite["latency_200_candidates"]["passed"]
    )

    return {
        "passed": all_passed,
        "model_version": MODEL_VERSION_V1,
        "corpus_summary": {
            "total_packages": len(corpus),
            "news_packages": kind_counts["news"],
            "partner_media_packages": kind_counts["partner_media"],
            "splits": dict(split_counts),
            "languages": dict(lang_counts),
        },
        "gate_1_policy_sandwich_fuzz_10k": policy_fuzz,
        "gate_2_counterfactual_neutrality_5k": cf_sweep,
        "gate_3_ragebait_contradiction_suppression": multi_suite["ragebait_suppression"],
        "gate_4_bot_brigading_immunity": bot_immunity,
        "gate_5_fairness_and_language_parity": multi_suite["fairness_and_language_parity"],
        "gate_6_latency_200_candidates": multi_suite["latency_200_candidates"],
        "schema_errors": schema_errors + instance_schema_errors,
        "clean_room_violations": clean_room_violations,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the 5,000-package Joint Signal x Ranking validation suite (VG-3 / VG-5)."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to the swivver_algorithm repository root.",
    )
    args = parser.parse_args(argv)

    report = run_full_5k_validation(args.repo_root.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
