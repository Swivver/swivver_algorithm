"""Unit tests for the 6-archetype, 2,000-session newsroom simulation and interleaving benchmark."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from eval.evaluate_newsroom_simulation import (
    ARCHETYPE_SPECS,
    BENCHMARK_SEED,
    bootstrap_mean_ci95,
    generate_newsroom_corpus,
    main,
    max_consecutive_streak,
    render_markdown_tables,
    run_newsroom_benchmark,
    two_sided_sign_test_pvalue,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class EvaluateNewsroomSimulationTests(unittest.TestCase):
    def test_corpus_generation_and_helper_edge_cases(self) -> None:
        corpus, authors_by_lang = generate_newsroom_corpus(stories_per_archetype=30, seed=BENCHMARK_SEED)
        self.assertEqual(len(corpus), 30 * len(ARCHETYPE_SPECS))
        self.assertEqual(set(authors_by_lang.keys()), {"en", "pt", "es"})
        for lang_list in authors_by_lang.values():
            self.assertGreater(len(lang_list), 0)

        self.assertEqual(max_consecutive_streak([]), 0)
        self.assertEqual(max_consecutive_streak(["a", "b", "b", "", "c"]), 2)
        self.assertEqual(two_sided_sign_test_pvalue(0, 0), 1.0)
        self.assertEqual(two_sided_sign_test_pvalue(5, 5), 1.0)
        self.assertEqual(bootstrap_mean_ci95([]), (0.0, 0.0))
        self.assertEqual(bootstrap_mean_ci95([2.5]), (2.5, 2.5))

    def test_2000_session_benchmark_gates_and_markdown_output(self) -> None:
        report = run_newsroom_benchmark(REPO_ROOT, num_sessions=2000, stories_per_archetype=200)
        self.assertTrue(report["passed"], f"Benchmark failed: {report}")
        self.assertEqual(report["policy_sandwich_leaks"], 0)
        self.assertEqual(report["schema_errors"], [])
        self.assertEqual(report["clean_room_violations"], [])

        sig = report["models"]["signal_informed_ranker"]
        naive = report["models"]["naive_engagement_ranker"]
        abl = report["models"]["studio_polish_ablation"]

        self.assertGreater(sig["ndcg_at_5"], 0.90)
        self.assertGreater(sig["ndcg_at_10"], 0.95)
        self.assertGreater(sig["mrr"], 0.98)
        self.assertGreater(sig["ndcg_at_10"], naive["ndcg_at_10"])
        self.assertEqual(sig["max_consecutive_same_author_streak"], 1)

        # Low-polish journalism fairness gap must stay near 0 and far below studio-biased ablation.
        self.assertLessEqual(sig["low_polish_fairness_gap"], 0.50)
        self.assertLess(sig["low_polish_fairness_gap"], abl["low_polish_fairness_gap"] * 0.10)

        # Top-5 unsourced attention-bait suppression rate >= 95%.
        self.assertGreaterEqual(report["unsourced_bait_suppression_top5"]["suppression_rate"], 0.95)

        # Team-Draft Interleaving win-rate & two-sided sign-test p < 0.01.
        inter = report["team_draft_interleaving"]
        self.assertTrue(inter["passed_significance_gate"])
        self.assertGreater(inter["signal_win_rate_decisive"], 0.90)
        self.assertLess(inter["sign_test_p_value"], 0.01)
        self.assertGreater(inter["bootstrap_95_lcb"], 0.0)

        md = render_markdown_tables(report)
        self.assertIn("NDCG@10", md)
        self.assertIn("Low-Polish Journalism Fairness Gap", md)
        self.assertIn("Team-Draft Interleaving", md)

    def test_cli_entrypoint_with_markdown_flag(self) -> None:
        rc = main(["--repo-root", str(REPO_ROOT), "--sessions", "200", "--markdown"])
        self.assertEqual(rc, 0)

        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "eval" / "evaluate_newsroom_simulation.py"), "--sessions", "200"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
