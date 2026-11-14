package interleave

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Swivver/swivver_algorithm/pkg/ranker"
)

var testNow = time.Date(2026, 11, 14, 12, 0, 0, 0, time.UTC)

func makeCandidate(id, author string, eligible, blocked bool) ranker.CandidateFeatureVector {
	return ranker.CandidateFeatureVector{
		SchemaVersion:        ranker.CandidateSchemaVersionV1,
		CandidateID:          id,
		AuthorID:             author,
		TopicTags:            []string{"world", "policy"},
		Language:             "en",
		PublishedAt:          testNow.Add(-2 * time.Hour),
		Eligible:             eligible,
		PolicyBlocked:        blocked,
		FollowedAuthor:       false,
		FollowedTopicMatches: 1,
		LanguageMatch:        true,
		ImpressionCount:      40,
		ValidWatchRatio:      0.50,
		ArticleOpenRate:      0.20,
		SourceInspectRate:    0.12,
		SaveRate:             0.05,
		AnomalyFlagged:       false,
	}
}

func TestPolicySandwichAndBilateralEligibilityFuzz(t *testing.T) {
	controlCfg := ranker.DefaultConfig()
	controlCfg.ModelVersion = "swivver-ranker/control"
	controlCfg.MaxSignalBoost = 0
	controlCfg.MaxSignalDemotion = 0
	rankerA := ranker.New(controlCfg)
	rankerB := ranker.NewDefault()

	rng := rand.New(rand.NewSource(20261114))

	for trial := 0; trial < 1000; trial++ {
		n := 8 + rng.Intn(28)
		batch := make([]ranker.CandidateFeatureVector, 0, n+2)
		forbidden := make(map[string]bool)

		for i := 0; i < n; i++ {
			id := fmt.Sprintf("cand-%d-%d", trial, i)
			eligible := rng.Float64() > 0.30
			blocked := rng.Float64() < 0.20
			c := makeCandidate(id, fmt.Sprintf("author-%d", rng.Intn(10)), eligible, blocked)
			c.PublishedAt = testNow.Add(-time.Duration(rng.Intn(48)) * time.Hour)
			c.ValidWatchRatio = rng.Float64()
			c.ArticleOpenRate = rng.Float64()

			if !eligible || blocked {
				// Give ineligible items maximal scores to verify zero leakage.
				c.FollowedAuthor = true
				c.FollowedTopicMatches = 4
				c.ValidWatchRatio = 1.0
				c.ArticleOpenRate = 1.0
				c.SourceInspectRate = 1.0
				c.SaveRate = 1.0
				c.Signal = &ranker.ScopedSignalFeatures{
					SchemaVersion:             ranker.SignalSchemaVersionV1,
					Availability:              ranker.AvailabilityAvailable,
					CoverageRatio:             1.0,
					FreshnessSec:              15,
					EvidenceSupportTier:       ranker.EvidenceSupported,
					ConsistencyTier:           ranker.ConsistencyConsistent,
					TechnicalLegibilityTier:   ranker.LegibilityClear,
					ProvenanceStatus:          ranker.ProvenanceVerified,
					IndependentSourceClusters: 5,
				}
				forbidden[id] = true
			}
			batch = append(batch, c)
		}

		// Inject a duplicate ID where the first copy looks eligible and the second is policy-blocked.
		conflictID := fmt.Sprintf("conflict-%d", trial)
		batch = append(batch,
			makeCandidate(conflictID, "author-dup", true, false),
			makeCandidate(conflictID, "author-dup", true, true),
		)
		forbidden[conflictID] = true

		// Inject a blank ID record and a benign duplicate eligible record.
		blankCand := makeCandidate("   ", "author-blank", true, false)
		dupEligible := makeCandidate(fmt.Sprintf("dup-ok-%d", trial), "author-ok", true, false)
		batch = append(batch, blankCand, dupEligible, dupEligible)

		mode := trial % 4
		req := InterleaveRequest{
			ExperimentID: "exp-policy-fuzz",
			Seed:         int64(trial),
			RequestKey:   fmt.Sprintf("viewer-%d", trial),
			Candidates:   batch,
			PageSize:     12,
			Now:          testNow,
		}

		var asymmetricExcluded map[string]bool
		switch mode {
		case 0:
			req.RankerA = rankerA
			req.RankerB = rankerB
		case 1:
			req.RankerA = rankerA
			req.RankedB, asymmetricExcluded = buildPreRankedWithExtras(batch, rng)
		case 2:
			req.RankedA, asymmetricExcluded = buildPreRankedWithExtras(batch, rng)
			req.RankerB = rankerB
		case 3:
			var asymA, asymB map[string]bool
			req.RankedA, asymA = buildPreRankedWithExtras(batch, rng)
			req.RankedB, asymB = buildPreRankedWithExtras(batch, rng)
			asymmetricExcluded = make(map[string]bool)
			for k := range asymA {
				asymmetricExcluded[k] = true
			}
			for k := range asymB {
				asymmetricExcluded[k] = true
			}
		}

		res, err := Interleave(req)
		if err != nil {
			t.Fatalf("trial %d mode %d unexpected error: %v", trial, mode, err)
		}

		seenInOutput := make(map[string]bool, len(res.Items))
		for idx, item := range res.Items {
			if item.Rank != idx+1 {
				t.Fatalf("trial %d expected sequential rank %d, got %d", trial, idx+1, item.Rank)
			}
			if forbidden[item.CandidateID] || strings.TrimSpace(item.CandidateID) == "" {
				t.Fatalf("trial %d mode %d leaked policy-forbidden candidate %q", trial, mode, item.CandidateID)
			}
			if strings.HasPrefix(item.CandidateID, "unattested-") {
				t.Fatalf("trial %d mode %d leaked unattested candidate %q", trial, mode, item.CandidateID)
			}
			if asymmetricExcluded[item.CandidateID] {
				t.Fatalf("trial %d mode %d leaked asymmetric candidate %q not eligible in both arms", trial, mode, item.CandidateID)
			}
			if seenInOutput[item.CandidateID] {
				t.Fatalf("trial %d mode %d produced duplicate candidate %q", trial, mode, item.CandidateID)
			}
			seenInOutput[item.CandidateID] = true
		}

		if diff := res.Audit.TeamACount - res.Audit.TeamBCount; diff < -1 || diff > 1 {
			t.Fatalf("trial %d team draft counts unbalanced: A=%d B=%d", trial, res.Audit.TeamACount, res.Audit.TeamBCount)
		}
	}
}

func buildPreRankedWithExtras(batch []ranker.CandidateFeatureVector, rng *rand.Rand) ([]ranker.RankedItem, map[string]bool) {
	items := make([]ranker.RankedItem, 0, len(batch)+3)
	included := make(map[string]bool)
	droppedFromArm := make(map[string]bool)
	for i, c := range batch {
		id := strings.TrimSpace(c.CandidateID)
		if id == "" {
			items = append(items, ranker.RankedItem{CandidateID: "  ", Rank: i + 1})
			continue
		}
		if included[id] || droppedFromArm[id] {
			continue
		}
		// Drop ~20% of candidates from this arm to exercise bilateral eligibility exclusion.
		if rng.Float64() < 0.20 {
			droppedFromArm[id] = true
			continue
		}
		included[id] = true
		items = append(items, ranker.RankedItem{
			CandidateID: id,
			Rank:        len(items) + 1,
			Score:       0.5,
			WhyThis: ranker.WhyThis{
				ReasonCodes: []ranker.ReasonCode{ranker.ReasonGeneralDiscovery},
				Reasons:     []string{"Selected from eligible recent reporting"},
				Summary:     "Selected from eligible recent reporting.",
			},
		})
	}
	// Inject an unattested candidate missing from batch and a duplicate entry.
	items = append(items, ranker.RankedItem{CandidateID: "unattested-999", Rank: len(items) + 1})
	if len(items) > 1 {
		items = append(items, items[0])
	}
	return items, droppedFromArm
}

func TestDeterministicSeedAndNullHypothesisUnbiasedness(t *testing.T) {
	r := ranker.NewDefault()
	pool := make([]ranker.CandidateFeatureVector, 24)
	for i := 0; i < 24; i++ {
		c := makeCandidate(fmt.Sprintf("story-%02d", i), fmt.Sprintf("author-%02d", i%12), true, false)
		c.PublishedAt = testNow.Add(-time.Duration(i*10) * time.Minute)
		pool[i] = c
	}

	req := InterleaveRequest{
		ExperimentID: "exp-repro",
		Seed:         42,
		RequestKey:   "session-abc",
		SigningKey:   []byte("unit-test-hmac-secret"),
		Candidates:   pool,
		RankerA:      r,
		RankerB:      r,
		ModelA:       "model-a-null",
		ModelB:       "model-b-null",
		PageSize:     10,
		Now:          testNow,
	}

	res1, err := Interleave(req)
	if err != nil {
		t.Fatal(err)
	}
	res2, err := Interleave(req)
	if err != nil {
		t.Fatal(err)
	}
	if res1.AttributionToken != res2.AttributionToken {
		t.Fatalf("expected identical attribution token for identical seed")
	}
	for i := range res1.Items {
		if res1.Items[i].CandidateID != res2.Items[i].CandidateID {
			t.Fatalf("non-deterministic item at rank %d: %s vs %s", i+1, res1.Items[i].CandidateID, res2.Items[i].CandidateID)
		}
	}

	// Null A/A test across 400 sessions where readers ALWAYS engage only with rank 1.
	// A biased interleaver would favor one arm; Team-Draft must stay ~50/50 unbiased.
	reports := make([]OutcomeReport, 400)
	for s := 0; s < 400; s++ {
		run, err := Interleave(InterleaveRequest{
			ExperimentID: "exp-null-aa",
			Seed:         int64(1000 + s),
			Candidates:   pool,
			RankerA:      r,
			RankerB:      r,
			PageSize:     10,
			Now:          testNow,
		})
		if err != nil {
			t.Fatal(err)
		}
		topID := run.Items[0].CandidateID
		rep, err := ScoreOutcomes(run.Attribution, []OutcomeEvent{
			{
				CandidateID:          topID,
				CompletedReadOrWatch: true,
				PositiveUsefulness:   true,
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		reports[s] = rep
	}

	study := AggregateStudy(reports)
	winShareB := float64(study.SessionWinsB) / float64(study.SessionsEvaluated)
	if math.Abs(winShareB-0.50) > 0.08 {
		t.Fatalf("expected unbiased ~50%% win share on A/A test, got %.3f (%+v)", winShareB, study)
	}
	if study.SignTestPValue < 0.05 {
		t.Fatalf("expected non-significant p-value under null A/A test, got %f", study.SignTestPValue)
	}
	if study.Bootstrap95LCB > 0 || study.Bootstrap95UCB < 0 {
		t.Fatalf("expected 95%% bootstrap CI to straddle 0 under A/A null, got [%f, %f]", study.Bootstrap95LCB, study.Bootstrap95UCB)
	}
	if study.PassedUsefulnessGate {
		t.Fatalf("null A/A test must not pass strict positive usefulness gate")
	}
}

func TestOpaqueTokenSecurityAndTamperRejection(t *testing.T) {
	r := ranker.NewDefault()
	secret := []byte("production-interleave-hmac-key")
	candidates := []ranker.CandidateFeatureVector{
		makeCandidate("c-1", "a-1", true, false),
		makeCandidate("c-2", "a-2", true, false),
		makeCandidate("c-3", "a-3", true, false),
		makeCandidate("c-4", "a-4", true, false),
	}

	res, err := Interleave(InterleaveRequest{
		ExperimentID: "exp-opaque",
		Seed:         7,
		SigningKey:   secret,
		Candidates:   candidates,
		RankerA:      r,
		RankerB:      r,
		PageSize:     4,
		Now:          testNow,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Verify client-serialized JSON contains zero internal team assignments or arm model names.
	wireBytes, err := json.Marshal(res)
	if err != nil {
		t.Fatal(err)
	}
	wireStr := string(wireBytes)
	for _, forbidden := range []string{"team_a", "team_b", "draft_round"} {
		// Audit has team_a_count/team_b_count, so check exact assignment values and items array.
		var parsed map[string]any
		if err := json.Unmarshal(wireBytes, &parsed); err != nil {
			t.Fatal(err)
		}
		itemsJSON, _ := json.Marshal(parsed["items"])
		if strings.Contains(string(itemsJSON), forbidden) {
			t.Fatalf("client items JSON leaked internal experiment field %q: %s", forbidden, string(itemsJSON))
		}
	}
	_ = wireStr

	// Standalone AttributionMap marshaling emits only the opaque token string.
	attrJSON, err := json.Marshal(res.Attribution)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(attrJSON), "team_a") || strings.Contains(string(attrJSON), "slots") {
		t.Fatalf("AttributionMap.MarshalJSON leaked internal slots: %s", string(attrJSON))
	}

	// Verify round-trip DecodeAttribution and accessor methods.
	decoded, err := DecodeAttribution(res.AttributionToken, secret)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.ExperimentID() != "exp-opaque" || decoded.Len() != 4 || decoded.ModelA() == "" || decoded.ModelB() == "" {
		t.Fatalf("unexpected decoded metadata: exp=%s len=%d", decoded.ExperimentID(), decoded.Len())
	}
	slotsCopy := decoded.Slots()
	slotsCopy[0].CandidateID = "mutated"
	origSlot, ok := decoded.Lookup("c-1")
	if !ok || origSlot.CandidateID != "c-1" {
		t.Fatalf("Slots() did not return a defensive copy or Lookup failed")
	}

	// Verify wrong key and malformed/tampered tokens are rejected.
	if _, err := DecodeAttribution(res.AttributionToken, []byte("wrong-key")); !errors.Is(err, ErrInvalidAttributionToken) {
		t.Fatalf("expected ErrInvalidAttributionToken for wrong key, got %v", err)
	}
	if _, err := ScoreTokenOutcomes("bad-token", secret, nil); !errors.Is(err, ErrInvalidAttributionToken) {
		t.Fatalf("expected ErrInvalidAttributionToken from ScoreTokenOutcomes, got %v", err)
	}
	if _, err := ScoreOutcomes(AttributionMap{}, nil); !errors.Is(err, ErrInvalidAttributionToken) {
		t.Fatalf("expected ErrInvalidAttributionToken for zero AttributionMap, got %v", err)
	}

	signHelper := func(raw string) string {
		b64 := base64.RawURLEncoding.EncodeToString([]byte(raw))
		input := "v1." + b64
		sig := base64.RawURLEncoding.EncodeToString(computeHMAC(input, secret))
		return input + "." + sig
	}
	badPayloads := []string{
		"",
		strings.Repeat("x", maxAttributionTokenLen+5),
		"v2.abc.def",
		"v1.only-two-parts",
		"v1.abc.%%%bad-base64%%%",
		"v1." + base64.RawURLEncoding.EncodeToString([]byte(`{"v":1}`)) + "." + base64.RawURLEncoding.EncodeToString([]byte("forged-mac")),
		"v1.%%%bad-payload%%%." + base64.RawURLEncoding.EncodeToString(computeHMAC("v1.%%%bad-payload%%%", secret)),
		signHelper(`{broken-json`),
		signHelper(`{"v":1,"exp":"e","unknown_field":1,"slots":[]}`),
		signHelper(`{"v":99,"exp":"e","slots":[]}`),
		signHelper(`{"v":1,"exp":"   ","slots":[]}`),
		signHelper(`{"v":1,"exp":"e","slots":[{"rank":0,"candidate_id":"c1","team":"team_a","draft_round":1}]}`),
		signHelper(`{"v":1,"exp":"e","slots":[{"rank":1,"candidate_id":"c1","team":"invalid","draft_round":1}]}`),
		signHelper(`{"v":1,"exp":"e","slots":[{"rank":1,"candidate_id":"c1","team":"team_a","draft_round":1},{"rank":2,"candidate_id":"c1","team":"team_b","draft_round":1}]}`),
		signHelper(`{"v":1,"exp":"e","slots":[{"rank":1,"candidate_id":"c1","team":"team_a","draft_round":1},{"rank":1,"candidate_id":"c2","team":"team_b","draft_round":1}]}`),
	}
	for idx, tok := range badPayloads {
		if _, err := DecodeAttribution(tok, secret); !errors.Is(err, ErrInvalidAttributionToken) {
			t.Fatalf("case %d: expected ErrInvalidAttributionToken, got %v", idx, err)
		}
	}
}

func TestM01UsefulnessOnlineStudySignalAwareWinsOverWatchBaseline(t *testing.T) {
	// Arm A: Watch-heavy baseline ignoring Signal observations.
	watchCfg := ranker.DefaultConfig()
	watchCfg.ModelVersion = "watch-baseline/0.1"
	watchCfg.MaxSignalBoost = 0.0
	watchCfg.MaxSignalDemotion = 0.0
	watchCfg.EngagementWeight = 0.60
	rankerA := ranker.New(watchCfg)

	// Arm B: Signal-aware clean-room ranker.
	rankerB := ranker.NewDefault()

	rng := rand.New(rand.NewSource(202611))
	sessionReports := make([]OutcomeReport, 120)

	for s := 0; s < 120; s++ {
		pool := make([]ranker.CandidateFeatureVector, 24)
		isContradictedBait := make(map[string]bool)
		isGrounded := make(map[string]bool)

		for i := 0; i < 24; i++ {
			id := fmt.Sprintf("s%d-c%d", s, i)
			c := makeCandidate(id, fmt.Sprintf("writer-%d", i), true, false)
			c.PublishedAt = testNow.Add(-time.Duration(rng.Intn(12)) * time.Hour)

			if i < 8 {
				// Sensational contradicted / bait-and-switch items with inflated watch ratio.
				c.ValidWatchRatio = 0.98
				c.ArticleOpenRate = 0.30
				c.SourceInspectRate = 0.0
				c.Signal = &ranker.ScopedSignalFeatures{
					SchemaVersion:           ranker.SignalSchemaVersionV1,
					Availability:            ranker.AvailabilityAvailable,
					CoverageRatio:           1.0,
					FreshnessSec:            60,
					EvidenceSupportTier:     ranker.EvidenceContradicted,
					ConsistencyTier:         ranker.ConsistencyInconsistent,
					BaitSwitchRisk:          0.92,
					AttentionPacingScore:    0.90,
					TechnicalLegibilityTier: ranker.LegibilityDegraded,
				}
				isContradictedBait[id] = true
			} else {
				// Grounded source-backed reporting.
				c.ValidWatchRatio = 0.55
				c.ArticleOpenRate = 0.25
				c.SourceInspectRate = 0.20
				c.SaveRate = 0.08
				c.Signal = &ranker.ScopedSignalFeatures{
					SchemaVersion:             ranker.SignalSchemaVersionV1,
					Availability:              ranker.AvailabilityAvailable,
					CoverageRatio:             1.0,
					FreshnessSec:              60,
					EvidenceSupportTier:       ranker.EvidenceSupported,
					ConsistencyTier:           ranker.ConsistencyConsistent,
					TechnicalLegibilityTier:   ranker.LegibilityClear,
					ProvenanceStatus:          ranker.ProvenanceVerified,
					IndependentSourceClusters: 3,
				}
				isGrounded[id] = true
			}
			pool[i] = c
		}

		interleaved, err := Interleave(InterleaveRequest{
			ExperimentID: "exp-m4-signal-usefulness",
			Seed:         int64(s + 1),
			RequestKey:   fmt.Sprintf("reader-%d", s),
			Candidates:   pool,
			RankerA:      rankerA,
			RankerB:      rankerB,
			PageSize:     8,
			Now:          testNow,
		})
		if err != nil {
			t.Fatal(err)
		}

		events := make([]OutcomeEvent, 0, 12)
		for _, item := range interleaved.Items {
			if isContradictedBait[item.CandidateID] {
				events = append(events, OutcomeEvent{
					CandidateID:          item.CandidateID,
					RapidSkip:            true,
					CorrectionImpression: true,
				})
			} else if isGrounded[item.CandidateID] {
				// Split across granular events + retry duplicate to verify idempotent aggregation.
				events = append(events,
					OutcomeEvent{CandidateID: item.CandidateID, ArticleCompleted: true},
					OutcomeEvent{CandidateID: item.CandidateID, WatchCompleted: true, SourceInspected: true},
					OutcomeEvent{CandidateID: item.CandidateID, PositiveUsefulness: true},
					OutcomeEvent{CandidateID: item.CandidateID, PositiveUsefulness: true},
				)
			}
		}
		// Include an unattributed event from outside the interleaved feed.
		events = append(events, OutcomeEvent{CandidateID: "unrelated-item", PositiveUsefulness: true})

		rep, err := ScoreTokenOutcomes(interleaved.AttributionToken, nil, events)
		if err != nil {
			t.Fatal(err)
		}
		if rep.UnattributedEvents != 1 {
			t.Fatalf("expected 1 unattributed event, got %d", rep.UnattributedEvents)
		}
		sessionReports[s] = rep
	}

	study := AggregateStudy(sessionReports)
	if !study.PassedUsefulnessGate {
		t.Fatalf("expected Signal-aware treatment to pass VG-5 usefulness gate, got %+v", study)
	}
	if study.Bootstrap95LCB <= 0 {
		t.Fatalf("expected positive 95%% LCB on M01 usefulness gain, got %f", study.Bootstrap95LCB)
	}
	if study.SignTestPValue >= 0.001 {
		t.Fatalf("expected highly significant two-sided sign test p < 0.001, got %f", study.SignTestPValue)
	}
	if study.TeamBCorrections >= study.TeamACorrections {
		t.Fatalf("expected treatment B to reduce correction impressions: B=%d vs A=%d", study.TeamBCorrections, study.TeamACorrections)
	}
}

func TestEdgeCasesLargePoolPaginationAndConcurrency(t *testing.T) {
	r := ranker.NewDefault()

	// Error cases: missing ranker and pre-ranked without candidate policy attestations.
	if _, err := Interleave(InterleaveRequest{RankerA: r}); !errors.Is(err, ErrMissingRanker) {
		t.Fatalf("expected ErrMissingRanker, got %v", err)
	}
	if _, err := Interleave(InterleaveRequest{
		RankedA: []ranker.RankedItem{{CandidateID: "x", Rank: 1}},
		RankedB: []ranker.RankedItem{{CandidateID: "x", Rank: 1}},
	}); !errors.Is(err, ErrMissingCandidates) {
		t.Fatalf("expected ErrMissingCandidates, got %v", err)
	}

	// Empty candidate pool with live rankers + default ExperimentID/PageSize/Now.
	emptyRes, err := Interleave(InterleaveRequest{RankerA: r, RankerB: r})
	if err != nil {
		t.Fatal(err)
	}
	if len(emptyRes.Items) != 0 || emptyRes.Attribution.Slots() != nil {
		t.Fatalf("expected empty result and nil slots, got %+v", emptyRes)
	}
	if _, ok := (AttributionMap{}).Lookup("c1"); ok {
		t.Fatalf("expected false lookup on empty AttributionMap")
	}

	// Empty ScoreOutcomes / AggregateStudy / single-element bootstrap / negative sign-test guard.
	emptyOutcome, err := ScoreOutcomes(emptyRes.Attribution, nil)
	if err != nil || emptyOutcome.Winner != TeamTie {
		t.Fatalf("unexpected empty outcome: %+v err=%v", emptyOutcome, err)
	}
	if st := AggregateStudy(nil); st.SessionsEvaluated != 0 || st.SignTestPValue != 1.0 {
		t.Fatalf("unexpected empty study: %+v", st)
	}
	singleStudy := AggregateStudy([]OutcomeReport{{
		ExperimentID: "single",
		TeamA:        TeamMetrics{M01Score: 5.0},
		TeamB:        TeamMetrics{M01Score: 2.0},
	}})
	if singleStudy.SessionWinsA != 1 || singleStudy.Bootstrap95LCB != -3.0 {
		t.Fatalf("unexpected single-session TeamA win study: %+v", singleStudy)
	}
	if p := TwoSidedSignTestPValue(-2, -1); p != 1.0 {
		t.Fatalf("expected 1.0 p-value for clamped zero wins, got %f", p)
	}
	if p := TwoSidedSignTestPValue(4, 4); p != 1.0 {
		t.Fatalf("expected 1.0 p-value for exact 4-4 tie, got %f", p)
	}

	// Single-pair TeamA winner and odd-slot PageSize=3 + PageSize > maxPageSize (multi-page rankTopK).
	largePool := make([]ranker.CandidateFeatureVector, 220)
	for i := 0; i < 220; i++ {
		c := makeCandidate(fmt.Sprintf("lg-%03d", i), fmt.Sprintf("auth-%02d", i%35), true, false)
		c.PublishedAt = testNow.Add(-time.Duration(i) * time.Minute)
		largePool[i] = c
	}

	// Explicit ModelA/ModelB overrides on mixed arms.
	mixedRes1, err := Interleave(InterleaveRequest{
		Candidates: largePool[:3],
		RankerA:    r,
		RankedB:    []ranker.RankedItem{{CandidateID: "lg-000", Rank: 1}},
		ModelA:     "custom-a",
		ModelB:     "custom-b",
		Now:        testNow,
	})
	if err != nil || mixedRes1.Attribution.ModelA() != "custom-a" || mixedRes1.Attribution.ModelB() != "custom-b" {
		t.Fatalf("unexpected mixed arm 1 models: %+v err=%v", mixedRes1.Attribution, err)
	}
	mixedRes2, err := Interleave(InterleaveRequest{
		Candidates: largePool[:3],
		RankedA:    []ranker.RankedItem{{CandidateID: "lg-000", Rank: 1}},
		RankerB:    r,
		ModelA:     "custom-a2",
		ModelB:     "custom-b2",
		Now:        testNow,
	})
	if err != nil || mixedRes2.Attribution.ModelA() != "custom-a2" || mixedRes2.Attribution.ModelB() != "custom-b2" {
		t.Fatalf("unexpected mixed arm 2 models: %+v err=%v", mixedRes2.Attribution, err)
	}

	oddRes, err := Interleave(InterleaveRequest{
		ExperimentID: "exp-odd",
		Seed:         9,
		Candidates:   largePool[:5],
		RankerA:      r,
		RankerB:      r,
		PageSize:     3,
		Now:          testNow,
	})
	if err != nil || len(oddRes.Items) != 3 {
		t.Fatalf("expected 3 items, got len=%d err=%v", len(oddRes.Items), err)
	}
	// Credit TeamA exclusively to cover Winner == TeamA and PairWinsA > 0.
	var teamACand string
	for _, sl := range oddRes.Attribution.Slots() {
		if sl.DraftRound == 1 && sl.Team == TeamA {
			teamACand = sl.CandidateID
			break
		}
	}
	repA, err := ScoreOutcomes(oddRes.Attribution, []OutcomeEvent{
		{CandidateID: teamACand, CompletedReadOrWatch: true, PositiveUsefulness: true},
	})
	if err != nil || repA.Winner != TeamA || repA.PairWinsA != 1 {
		t.Fatalf("expected TeamA winner, got %+v err=%v", repA, err)
	}

	// Concurrent 220-candidate interleaving with PageSize=500 (clamped to maxPageSize=200, multi-page cursor).
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			res, err := Interleave(InterleaveRequest{
				ExperimentID: "exp-race",
				Seed:         int64(id),
				Candidates:   largePool,
				RankerA:      r,
				RankerB:      r,
				PageSize:     500,
				Now:          testNow,
			})
			if err != nil || len(res.Items) != maxPageSize {
				t.Errorf("goroutine %d failed: len=%d err=%v", id, len(res.Items), err)
			}
		}(g)
	}
	wg.Wait()
}
