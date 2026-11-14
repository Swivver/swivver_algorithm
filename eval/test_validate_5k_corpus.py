"""Unit tests for the 5,000-package Joint Signal x Ranking validation suite (VG-3 / VG-5)."""

from __future__ import annotations

import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

from eval.validate_5k_corpus import (
    CORPUS_SEED,
    MODEL_VERSION_V1,
    SplitMix64,
    generate_5k_corpus,
    main,
    rank_candidates_reference,
    run_bot_immunity_check,
    run_counterfactual_5k_sweep,
    run_full_5k_validation,
    run_policy_fuzz_10k,
    run_ragebait_fairness_and_latency_suite,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class Validate5KCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = generate_5k_corpus(CORPUS_SEED)

    def test_splitmix64_determinism_and_corpus_stratification(self) -> None:
        r1 = SplitMix64(42)
        r2 = SplitMix64(42)
        seq1 = [r1.next_u64() for _ in range(10)]
        seq2 = [r2.next_u64() for _ in range(10)]
        self.assertEqual(seq1, seq2)
        self.assertEqual(r1.intn(0), 0)

        self.assertEqual(len(self.corpus), 5000)
        kinds = Counter(p.package_kind for p in self.corpus)
        splits = Counter(p.split for p in self.corpus)
        langs = Counter(p.candidate["language"] for p in self.corpus)
        archetypes = Counter(p.archetype for p in self.corpus)

        self.assertEqual(kinds["news"], 3500)
        self.assertEqual(kinds["partner_media"], 1500)
        self.assertEqual(splits, {"dev": 3000, "calibration": 1000, "locked_test": 1000})
        self.assertEqual(langs, {"en": 3000, "pt": 1100, "es": 900})
        self.assertEqual(archetypes["ineligible"], 250)
        self.assertEqual(archetypes["bot_brigade"], 350)
        self.assertEqual(archetypes["ragebait_contradicted"] + archetypes["ragebait_bait"], 900)

    def test_gate_1_policy_sandwich_fuzz_10k(self) -> None:
        res = run_policy_fuzz_10k(self.corpus, trials=10000)
        self.assertTrue(res["passed"])
        self.assertEqual(res["leaked_trials"], 0)
        self.assertEqual(res["leaked_items"], 0)
        self.assertEqual(res["leak_rate"], 0.0)
        self.assertGreater(res["ineligible_candidates_tested"], 50000)

    def test_gate_2_counterfactual_uncertainty_neutrality_5k(self) -> None:
        res = run_counterfactual_5k_sweep(self.corpus)
        self.assertTrue(res["passed"])
        self.assertEqual(res["candidates_swept"], 5000)
        self.assertEqual(res["state_evaluations"], 25000)
        self.assertLess(res["max_abs_signal_delta"], 1e-12)
        self.assertEqual(res["rank_demotions"], 0)

    def test_gate_3_5_6_ragebait_fairness_and_latency(self) -> None:
        res = run_ragebait_fairness_and_latency_suite(self.corpus, num_sessions=500)
        rb = res["ragebait_suppression"]
        self.assertTrue(rb["passed"])
        self.assertGreaterEqual(rb["relative_reduction"], 0.85)

        fp = res["fairness_and_language_parity"]
        self.assertTrue(fp["passed"])
        self.assertLessEqual(abs(fp["author_gini_delta"]), 0.03)
        self.assertLessEqual(abs(fp["minority_language_parity"]["share_delta"]), 0.03)
        self.assertLessEqual(abs(fp["pt_share_delta"]), 0.03)
        self.assertLessEqual(abs(fp["es_share_delta"]), 0.03)

        lat = res["latency_200_candidates"]
        self.assertTrue(lat["passed"])
        self.assertLessEqual(lat["p95_ms"], 25.0)

    def test_gate_4_bot_brigading_immunity(self) -> None:
        res = run_bot_immunity_check(self.corpus)
        self.assertTrue(res["passed"])
        self.assertEqual(res["boost_violations"], 0)
        self.assertEqual(res["boost_elimination_rate"], 1.0)

    def test_reference_ranker_edge_cases_and_full_cli(self) -> None:
        empty_ranked, dropped = rank_candidates_reference([], page_size=10)
        self.assertEqual(empty_ranked, [])
        self.assertEqual(dropped, 0)

        summary = run_full_5k_validation(REPO_ROOT)
        self.assertTrue(summary["passed"], f"Full 5K validation failed: {summary}")
        self.assertEqual(summary["model_version"], MODEL_VERSION_V1)
        self.assertEqual(summary["schema_errors"], [])
        self.assertEqual(summary["clean_room_violations"], [])

        rc = main(["--repo-root", str(REPO_ROOT)])
        self.assertEqual(rc, 0)

        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "eval" / "validate_5k_corpus.py")],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
