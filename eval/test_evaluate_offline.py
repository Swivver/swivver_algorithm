"""Unit tests for offline evaluation metrics, JSON Schemas, and clean-room scanner."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from eval.evaluate_offline import (
    EXPECTED_SCHEMA_FILES,
    author_gini_coefficient,
    counterfactual_unknown_rank_delta,
    main,
    minority_language_share_parity,
    ndcg_at_k,
    run_synthetic_evaluation,
    scan_clean_room_boundary,
    validate_instance_against_schema,
    validate_schemas_directory,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class EvaluateOfflineTests(unittest.TestCase):
    def test_ndcg_at_k_ordering_and_edge_cases(self) -> None:
        rels = {"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.0}
        self.assertAlmostEqual(ndcg_at_k(["a", "b", "c", "d"], rels, k=3), 1.0, places=6)
        suboptimal = ndcg_at_k(["d", "c", "b", "a"], rels, k=3)
        self.assertGreater(suboptimal, 0.0)
        self.assertLess(suboptimal, 0.6)
        self.assertEqual(ndcg_at_k([], rels, k=3), 0.0)
        self.assertEqual(ndcg_at_k(["a"], rels, k=0), 0.0)
        self.assertEqual(ndcg_at_k(["x"], {"x": 0.0}, k=3), 0.0)

    def test_author_gini_coefficient(self) -> None:
        # Equal exposure across 5 authors -> Gini = 0.0
        self.assertAlmostEqual(author_gini_coefficient(["a1", "a2", "a3", "a4", "a5"]), 0.0, places=6)
        # Heavy concentration -> high Gini
        skewed = author_gini_coefficient({"a1": 90, "a2": 5, "a3": 3, "a4": 2})
        self.assertGreater(skewed, 0.60)
        self.assertLess(skewed, 1.0)
        self.assertEqual(author_gini_coefficient([]), 0.0)
        self.assertEqual(author_gini_coefficient({"solo": 10}), 0.0)

    def test_minority_language_share_parity(self) -> None:
        pool = ["en", "en", "en", "es", "fr"]
        ranked_equal = ["en", "es", "en", "fr", "en"]
        parity = minority_language_share_parity(pool, ranked_equal)
        self.assertAlmostEqual(parity["pool_minority_share"], 0.4, places=6)
        self.assertAlmostEqual(parity["ranked_minority_share"], 0.4, places=6)
        self.assertAlmostEqual(parity["parity_ratio"], 1.0, places=6)

        suppressed = minority_language_share_parity(pool, ["en", "en", "en", "en", "en"], primary_language="en")
        self.assertEqual(suppressed["ranked_minority_share"], 0.0)
        self.assertEqual(suppressed["parity_ratio"], 0.0)

        empty = minority_language_share_parity([], [])
        self.assertEqual(empty["parity_ratio"], 1.0)

    def test_counterfactual_unknown_rank_delta(self) -> None:
        base = ["c1", "c2", "c3", "c4"]
        non_demoted = ["c2", "c1", "c3", "c4"]
        res_ok = counterfactual_unknown_rank_delta(base, non_demoted, target_ids={"c2", "c3"})
        self.assertTrue(res_ok["passed"])
        self.assertEqual(res_ok["demoted_count"], 0)
        self.assertEqual(res_ok["max_demotion_positions"], 0)

        demoted = ["c1", "c3", "c4", "c2"]
        res_bad = counterfactual_unknown_rank_delta(base, demoted, target_ids={"c2"})
        self.assertFalse(res_bad["passed"])
        self.assertEqual(res_bad["demoted_count"], 1)
        self.assertEqual(res_bad["max_demotion_positions"], 2)

        dropped = ["c1", "c3", "c4"]
        res_missing = counterfactual_unknown_rank_delta(base, dropped, target_ids={"c2"})
        self.assertFalse(res_missing["passed"])
        self.assertEqual(res_missing["missing_ids"], ["c2"])

    def test_json_schemas_valid_and_instance_validation(self) -> None:
        schemas_dir = REPO_ROOT / "schemas"
        errors = validate_schemas_directory(schemas_dir)
        self.assertEqual(errors, [], f"Unexpected schema validation errors: {errors}")

        loaded = {
            name: json.loads((schemas_dir / name).read_text(encoding="utf-8"))
            for name in EXPECTED_SCHEMA_FILES
        }

        valid_signal = {
            "schema_version": "scoped-signal-features/1.0",
            "availability": "available",
            "coverage_ratio": 0.75,
            "freshness_sec": 45.0,
            "evidence_support_tier": "supported",
            "consistency_tier": "consistent",
            "technical_legibility_tier": "clear",
            "attention_pacing_score": 0.25,
            "bait_switch_risk": 0.05,
            "provenance_status": "verified",
            "independent_source_clusters": 2,
            "dimension_availability": {
                "evidence_support_tier": "available",
                "provenance_status": "available",
            },
        }
        self.assertEqual(
            validate_instance_against_schema(
                valid_signal, loaded["scoped-signal-features-v1.schema.json"], loaded
            ),
            [],
        )

        # Reject undeclared universal quality oracle score.
        oracle_violation = dict(valid_signal)
        oracle_violation["_".join(("universal", "quality", "score"))] = 0.99
        self.assertNotEqual(
            validate_instance_against_schema(
                oracle_violation, loaded["scoped-signal-features-v1.schema.json"], loaded
            ),
            [],
        )

        valid_candidate = {
            "schema_version": "candidate-feature-vector/1.0",
            "candidate_id": "story-101",
            "author_id": "author-7",
            "topic_tags": ["science", "energy"],
            "language": "en",
            "published_at": "2026-11-14T10:00:00Z",
            "eligible": True,
            "policy_blocked": False,
            "followed_author": True,
            "followed_topic_matches": 1,
            "language_match": True,
            "valid_watch_ratio": 0.55,
            "article_open_rate": 0.28,
            "source_inspect_rate": 0.16,
            "save_rate": 0.06,
            "anomaly_flagged": False,
            "signal": valid_signal,
        }
        self.assertEqual(
            validate_instance_against_schema(
                valid_candidate, loaded["candidate-feature-vector-v1.schema.json"], loaded
            ),
            [],
        )

        valid_result = {
            "schema_version": "ranked-feed-result/1.0",
            "model_version": "swivver-ranker/0.1.0",
            "items": [
                {
                    "candidate_id": "story-101",
                    "rank": 1,
                    "score": 0.812,
                    "score_components": {
                        "preference": 0.443,
                        "freshness": 0.185,
                        "engagement": 0.054,
                        "signal_delta": 0.13,
                        "exploration_bonus": 0.0,
                        "diversity_penalty": 0.0,
                    },
                    "why_this": {
                        "reason_codes": ["FOLLOWED_AUTHOR", "TOPIC_MATCH"],
                        "reasons": [
                            "From a writer or publication you follow",
                            "Matches topics in your preferences",
                        ],
                        "summary": "From a writer or publication you follow · Matches topics in your preferences.",
                    },
                }
            ],
            "session_pacing_advisory": {
                "triggered": False,
                "trigger_codes": [],
                "advisory_kind": "none",
                "insert_after_rank": 0,
                "message": "",
                "high_intensity_streak": 0,
                "session_depth_items": 1,
            },
            "next_cursor": "",
            "diversity_audit": {
                "eligible_candidates": 1,
                "dropped_ineligible": 0,
                "returned_count": 1,
                "unique_authors": 1,
                "unique_topics": 2,
                "max_author_share": 1.0,
                "author_cap_deferrals": 0,
                "topic_cap_deferrals": 0,
                "cold_start_items": 0,
                "anomaly_discounted_items": 0,
            },
        }
        self.assertEqual(
            validate_instance_against_schema(
                valid_result, loaded["ranked-feed-result-v1.schema.json"], loaded
            ),
            [],
        )

    def test_clean_room_scanner_passes_repo_and_catches_violations(self) -> None:
        self.assertEqual(scan_clean_room_boundary(REPO_ROOT), [])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_py = root / "leak.py"
            bad_pkg = "_".join(("swivver", "signal"))
            bad_py.write_text(f"import {bad_pkg}.rules\n", encoding="utf-8")

            bad_go = root / "leak.go"
            bad_go.write_text('package x\nimport "example.com/external/pkg"\n', encoding="utf-8")

            bad_md = root / "notes.md"
            bad_md.write_text("Path /" + "Users/alice/secret\n", encoding="utf-8")

            violations = scan_clean_room_boundary(root)
            self.assertGreaterEqual(len(violations), 3)

    def test_cli_entrypoint_passes(self) -> None:
        summary = run_synthetic_evaluation(REPO_ROOT)
        self.assertTrue(summary["passed"], f"Summary failed: {summary}")
        rc = main(["--repo-root", str(REPO_ROOT)])
        self.assertEqual(rc, 0)

        proc = subprocess.run(
            ["python3", str(REPO_ROOT / "eval" / "evaluate_offline.py")],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
