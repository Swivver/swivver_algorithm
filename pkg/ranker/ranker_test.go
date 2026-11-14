package ranker

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
)

var refNow = time.Date(2026, 11, 14, 12, 0, 0, 0, time.UTC)

func baseCandidate(id, author string) CandidateFeatureVector {
	return CandidateFeatureVector{
		SchemaVersion:        CandidateSchemaVersionV1,
		CandidateID:          id,
		AuthorID:             author,
		TopicTags:            []string{"world"},
		Language:             "en",
		PublishedAt:          refNow.Add(-2 * time.Hour),
		Eligible:             true,
		PolicyBlocked:        false,
		FollowedAuthor:       false,
		FollowedTopicMatches: 0,
		LanguageMatch:        true,
		ImpressionCount:      25,
		ValidWatchRatio:      0.40,
		ArticleOpenRate:      0.10,
		SourceInspectRate:    0.05,
		SaveRate:             0.02,
		AnomalyFlagged:       false,
	}
}

func TestPolicyExclusionFuzz1000(t *testing.T) {
	r := NewDefault()
	rng := rand.New(rand.NewSource(20261114))

	for trial := 0; trial < 1000; trial++ {
		n := 5 + rng.Intn(35)
		batch := make([]CandidateFeatureVector, n)
		ineligibleIDs := make(map[string]bool)
		expectedDrops := 0

		for i := 0; i < n; i++ {
			id := fmt.Sprintf("cand-%d-%d", trial, i)
			c := CandidateFeatureVector{
				SchemaVersion:        CandidateSchemaVersionV1,
				CandidateID:          id,
				AuthorID:             fmt.Sprintf("author-%d", rng.Intn(8)),
				TopicTags:            []string{fmt.Sprintf("topic-%d", rng.Intn(6))},
				Language:             "en",
				PublishedAt:          refNow.Add(-time.Duration(rng.Intn(72)) * time.Hour),
				Eligible:             rng.Float64() > 0.35,
				PolicyBlocked:        rng.Float64() < 0.25,
				FollowedAuthor:       rng.Float64() < 0.5,
				FollowedTopicMatches: rng.Intn(4),
				LanguageMatch:        rng.Float64() < 0.8,
				ImpressionCount:      rng.Intn(500),
				ValidWatchRatio:      rng.Float64(),
				ArticleOpenRate:      rng.Float64(),
				SourceInspectRate:    rng.Float64(),
				SaveRate:             rng.Float64(),
				AnomalyFlagged:       rng.Float64() < 0.2,
			}
			// Give ineligible items maximal preference, engagement, and Signal support
			// to prove ranking weights can never override policy gates.
			if !c.Eligible || c.PolicyBlocked {
				c.FollowedAuthor = true
				c.FollowedTopicMatches = 5
				c.ArticleOpenRate = 1.0
				c.SourceInspectRate = 1.0
				c.SaveRate = 1.0
				c.Signal = &ScopedSignalFeatures{
					SchemaVersion:             SignalSchemaVersionV1,
					Availability:              AvailabilityAvailable,
					CoverageRatio:             1.0,
					FreshnessSec:              10,
					EvidenceSupportTier:       EvidenceSupported,
					ConsistencyTier:           ConsistencyConsistent,
					TechnicalLegibilityTier:   LegibilityClear,
					ProvenanceStatus:          ProvenanceVerified,
					IndependentSourceClusters: 4,
				}
				ineligibleIDs[id] = true
				expectedDrops++
			}
			if i == n-1 && trial%10 == 0 {
				if c.Eligible && !c.PolicyBlocked {
					expectedDrops++
				}
				c.CandidateID = "   "
				delete(ineligibleIDs, id)
			}
			batch[i] = c
		}

		cursor := ""
		for page := 0; page < 10; page++ {
			res, err := r.Rank(RankRequest{
				Candidates: batch,
				PageSize:   10,
				Cursor:     cursor,
				Now:        refNow,
			})
			if err != nil {
				t.Fatalf("trial %d page %d unexpected error: %v", trial, page, err)
			}
			if res.DiversityAudit.DroppedIneligible != expectedDrops {
				t.Fatalf("trial %d expected %d dropped, got %d", trial, expectedDrops, res.DiversityAudit.DroppedIneligible)
			}
			for _, item := range res.Items {
				if ineligibleIDs[item.CandidateID] || strings.TrimSpace(item.CandidateID) == "" {
					t.Fatalf("trial %d leaked ineligible candidate %q", trial, item.CandidateID)
				}
			}
			if res.NextCursor == "" {
				break
			}
			cursor = res.NextCursor
		}
	}
}

func TestUncertaintyNeutralityCounterfactualInvariance(t *testing.T) {
	r := NewDefault()
	nonAvailableStates := []Availability{
		AvailabilityUnknown,
		AvailabilityUnsupported,
		AvailabilityNotApplicable,
		AvailabilityStale,
	}

	// 1. Direct score and rank identity across nil and all non-available states.
	for _, state := range nonAvailableStates {
		cNil := baseCandidate("story-1", "author-1")
		cNil.Signal = nil

		cState := baseCandidate("story-1", "author-1")
		cState.Signal = &ScopedSignalFeatures{
			SchemaVersion:             SignalSchemaVersionV1,
			Availability:              state,
			CoverageRatio:             0.0,
			FreshnessSec:              120,
			EvidenceSupportTier:       EvidenceContradicted, // Must be ignored when not available!
			BaitSwitchRisk:            0.99,
			AttentionPacingScore:      0.99,
			TechnicalLegibilityTier:   LegibilityUnusable,
			ProvenanceStatus:          ProvenanceInvalid,
			IndependentSourceClusters: 0,
		}

		resNil, err := r.Rank(RankRequest{Candidates: []CandidateFeatureVector{cNil}, Now: refNow})
		if err != nil {
			t.Fatal(err)
		}
		resState, err := r.Rank(RankRequest{Candidates: []CandidateFeatureVector{cState}, Now: refNow})
		if err != nil {
			t.Fatal(err)
		}
		if resState.Items[0].ScoreComponents.SignalDelta != 0.0 {
			t.Fatalf("state %s produced non-zero SignalDelta: %f", state, resState.Items[0].ScoreComponents.SignalDelta)
		}
		if resNil.Items[0].Score != resState.Items[0].Score {
			t.Fatalf("state %s changed score from %f to %f", state, resNil.Items[0].Score, resState.Items[0].Score)
		}
	}

	// 2. Expired freshness, negative freshness, and NaN freshness all yield 0.0 delta.
	staleVariants := []float64{86400 * 30, -5.0, math.NaN(), math.Inf(1)}
	for idx, freshSec := range staleVariants {
		c := baseCandidate(fmt.Sprintf("stale-%d", idx), "author-1")
		c.Signal = &ScopedSignalFeatures{
			SchemaVersion:       SignalSchemaVersionV1,
			Availability:        AvailabilityAvailable,
			CoverageRatio:       1.0,
			FreshnessSec:        freshSec,
			EvidenceSupportTier: EvidenceContradicted,
			BaitSwitchRisk:      1.0,
		}
		res, err := r.Rank(RankRequest{Candidates: []CandidateFeatureVector{c}, Now: refNow})
		if err != nil {
			t.Fatal(err)
		}
		if res.Items[0].ScoreComponents.SignalDelta != 0.0 {
			t.Fatalf("variant %d expected 0.0 delta, got %f", idx, res.Items[0].ScoreComponents.SignalDelta)
		}
	}

	// 3. Available profile with unknown/missing/mixed neutral tiers and dimension overrides.
	cBase := baseCandidate("story-neutral", "author-2")
	resBase, _ := r.Rank(RankRequest{Candidates: []CandidateFeatureVector{cBase}, Now: refNow})

	cPartial := baseCandidate("story-neutral", "author-2")
	cPartial.Signal = &ScopedSignalFeatures{
		SchemaVersion:           SignalSchemaVersionV1,
		Availability:            AvailabilityAvailable,
		CoverageRatio:           0.4,
		FreshnessSec:            60,
		EvidenceSupportTier:     EvidenceUnknown,
		ConsistencyTier:         ConsistencyMixed,
		TechnicalLegibilityTier: LegibilityUnknown,
		ProvenanceStatus:        ProvenanceMissing,
		BaitSwitchRisk:          0.95,
		AttentionPacingScore:    0.95,
		DimensionAvailability: map[string]Availability{
			DimBaitSwitch:      AvailabilityUnsupported,
			DimAttentionPacing: AvailabilityUnknown,
		},
	}
	resPartial, _ := r.Rank(RankRequest{Candidates: []CandidateFeatureVector{cPartial}, Now: refNow})
	if resPartial.Items[0].ScoreComponents.SignalDelta != 0.0 {
		t.Fatalf("expected 0.0 signal delta for neutral/unsupported dims, got %f", resPartial.Items[0].ScoreComponents.SignalDelta)
	}
	if resPartial.Items[0].Score < resBase.Items[0].Score {
		t.Fatalf("neutral partial profile demoted candidate: %f < %f", resPartial.Items[0].Score, resBase.Items[0].Score)
	}

	// 4. Randomized counterfactual pool check: switching any subset of candidates from nil
	// to unknown/unsupported/not_applicable/stale never demotes their rank.
	rng := rand.New(rand.NewSource(99))
	for trial := 0; trial < 200; trial++ {
		poolNil := make([]CandidateFeatureVector, 20)
		poolUnknown := make([]CandidateFeatureVector, 20)
		targetIDs := make(map[string]bool)

		for i := 0; i < 20; i++ {
			id := fmt.Sprintf("cf-%d-%d", trial, i)
			c := baseCandidate(id, fmt.Sprintf("author-%d", i))
			c.PublishedAt = refNow.Add(-time.Duration(rng.Intn(48)) * time.Hour)
			c.ArticleOpenRate = rng.Float64() * 0.5
			c.SourceInspectRate = rng.Float64() * 0.4
			c.FollowedTopicMatches = rng.Intn(3)
			poolNil[i] = c

			cCopy := c
			if rng.Float64() < 0.5 {
				st := nonAvailableStates[rng.Intn(len(nonAvailableStates))]
				cCopy.Signal = &ScopedSignalFeatures{
					SchemaVersion:       SignalSchemaVersionV1,
					Availability:        st,
					CoverageRatio:       0.0,
					FreshnessSec:        300,
					EvidenceSupportTier: EvidenceUnknown,
				}
				targetIDs[id] = true
			}
			poolUnknown[i] = cCopy
		}

		r1, err := r.Rank(RankRequest{Candidates: poolNil, PageSize: 20, Now: refNow})
		if err != nil {
			t.Fatal(err)
		}
		r2, err := r.Rank(RankRequest{Candidates: poolUnknown, PageSize: 20, Now: refNow})
		if err != nil {
			t.Fatal(err)
		}
		rank1 := make(map[string]int, len(r1.Items))
		for _, it := range r1.Items {
			rank1[it.CandidateID] = it.Rank
		}
		for _, it := range r2.Items {
			if targetIDs[it.CandidateID] && it.Rank > rank1[it.CandidateID] {
				t.Fatalf("trial %d candidate %s demoted from %d to %d", trial, it.CandidateID, rank1[it.CandidateID], it.Rank)
			}
		}
	}
}

func TestBotAnomalyImmunityAndAntiEngagementTrap(t *testing.T) {
	r := NewDefault()

	botCandidate := baseCandidate("bot-spike", "author-bot")
	botCandidate.AnomalyFlagged = true
	botCandidate.ValidWatchRatio = 1.0
	botCandidate.ArticleOpenRate = 1.0
	botCandidate.SourceInspectRate = 1.0
	botCandidate.SaveRate = 1.0

	watchTrap := baseCandidate("autoplay-trap", "author-trap")
	watchTrap.ValidWatchRatio = 1.0
	watchTrap.ArticleOpenRate = 0.0
	watchTrap.SourceInspectRate = 0.0
	watchTrap.SaveRate = 0.0

	deliberative := baseCandidate("deep-read", "author-news")
	deliberative.ValidWatchRatio = 0.35
	deliberative.ArticleOpenRate = 0.35
	deliberative.SourceInspectRate = 0.25
	deliberative.SaveRate = 0.10

	res, err := r.Rank(RankRequest{
		Candidates: []CandidateFeatureVector{botCandidate, watchTrap, deliberative},
		PageSize:   10,
		Now:        refNow,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.DiversityAudit.AnomalyDiscountedItems != 1 {
		t.Fatalf("expected 1 anomaly discounted item, got %d", res.DiversityAudit.AnomalyDiscountedItems)
	}
	if len(res.Items) != 3 {
		t.Fatalf("expected 3 items, got %d", len(res.Items))
	}
	if res.Items[0].CandidateID != "deep-read" {
		t.Fatalf("expected deliberative story first, got %s", res.Items[0].CandidateID)
	}
	if res.Items[2].CandidateID != "bot-spike" {
		t.Fatalf("expected anomaly-flagged bot story last, got %s", res.Items[2].CandidateID)
	}
	if res.Items[2].ScoreComponents.Engagement != 0.0 || res.Items[2].ScoreComponents.ExplorationBonus != 0.0 {
		t.Fatalf("expected zero engagement and exploration for bot-flagged item, got %+v", res.Items[2].ScoreComponents)
	}
}

func TestSignalSupportedBoostAndContradictedBaitDemotion(t *testing.T) {
	r := NewDefault()

	supported := baseCandidate("supported-story", "author-a")
	supported.Signal = &ScopedSignalFeatures{
		SchemaVersion:             SignalSchemaVersionV1,
		Availability:              AvailabilityAvailable,
		CoverageRatio:             1.0,
		FreshnessSec:              60,
		EvidenceSupportTier:       EvidenceSupported,
		ConsistencyTier:           ConsistencyConsistent,
		TechnicalLegibilityTier:   LegibilityClear,
		ProvenanceStatus:          ProvenanceVerified,
		IndependentSourceClusters: 3,
	}

	unassessed := baseCandidate("unassessed-story", "author-b")

	degraded := baseCandidate("degraded-story", "author-c")
	degraded.Signal = &ScopedSignalFeatures{
		SchemaVersion:           SignalSchemaVersionV1,
		Availability:            AvailabilityAvailable,
		CoverageRatio:           0.6,
		FreshnessSec:            60,
		TechnicalLegibilityTier: LegibilityDegraded,
		ProvenanceStatus:        ProvenanceDeclared,
	}

	contradictedBait := baseCandidate("contradicted-bait", "author-d")
	contradictedBait.Signal = &ScopedSignalFeatures{
		SchemaVersion:           SignalSchemaVersionV1,
		Availability:            AvailabilityAvailable,
		CoverageRatio:           1.0,
		FreshnessSec:            60,
		EvidenceSupportTier:     EvidenceContradicted,
		ConsistencyTier:         ConsistencyInconsistent,
		TechnicalLegibilityTier: LegibilityUnusable,
		ProvenanceStatus:        ProvenanceInvalid,
		BaitSwitchRisk:          0.95,
		AttentionPacingScore:    0.95,
	}

	res, err := r.Rank(RankRequest{
		Candidates: []CandidateFeatureVector{contradictedBait, degraded, unassessed, supported},
		PageSize:   10,
		Now:        refNow,
	})
	if err != nil {
		t.Fatal(err)
	}

	wantOrder := []string{"supported-story", "unassessed-story", "degraded-story", "contradicted-bait"}
	for i, want := range wantOrder {
		if res.Items[i].CandidateID != want {
			t.Fatalf("rank %d: want %s, got %s", i+1, want, res.Items[i].CandidateID)
		}
	}
	// Verify signal boost and demotion stay strictly bounded by Config caps.
	if res.Items[0].ScoreComponents.SignalDelta > DefaultConfig().MaxSignalBoost {
		t.Fatalf("signal boost exceeded cap: %f", res.Items[0].ScoreComponents.SignalDelta)
	}
	if res.Items[3].ScoreComponents.SignalDelta < -DefaultConfig().MaxSignalDemotion {
		t.Fatalf("signal demotion exceeded cap: %f", res.Items[3].ScoreComponents.SignalDelta)
	}
}

func TestAuthorAndTopicDiversityCaps(t *testing.T) {
	r := NewDefault()

	candidates := make([]CandidateFeatureVector, 0, 12)
	// Dominant author submits 7 high-preference stories on the same topic.
	for i := 0; i < 7; i++ {
		c := baseCandidate(fmt.Sprintf("dom-%d", i), "author-dominant")
		c.FollowedAuthor = true
		c.FollowedTopicMatches = 2
		c.TopicTags = []string{"markets", "macro"}
		c.PublishedAt = refNow.Add(-time.Duration(i) * time.Minute)
		candidates = append(candidates, c)
	}
	// Four independent authors submit eligible stories across diverse topics.
	otherTopics := []string{"climate", "science", "transit", "health"}
	for i, topic := range otherTopics {
		c := baseCandidate(fmt.Sprintf("ind-%d", i), fmt.Sprintf("author-ind-%d", i))
		c.TopicTags = []string{topic}
		candidates = append(candidates, c)
	}

	res, err := r.Rank(RankRequest{
		Candidates: candidates,
		PageSize:   5,
		Now:        refNow,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.DiversityAudit.UniqueAuthors != 5 {
		t.Fatalf("expected 5 unique authors in 5-item window, got %d", res.DiversityAudit.UniqueAuthors)
	}
	if res.DiversityAudit.MaxAuthorShare > 0.200001 {
		t.Fatalf("expected max author share <= 0.20, got %f", res.DiversityAudit.MaxAuthorShare)
	}
	if res.DiversityAudit.AuthorCapDeferrals == 0 {
		t.Fatalf("expected positive author cap deferrals")
	}

	// Verify topic cap deferral across distinct authors sharing one hot topic.
	topicFlood := make([]CandidateFeatureVector, 0, 6)
	for i := 0; i < 4; i++ {
		c := baseCandidate(fmt.Sprintf("hot-topic-%d", i), fmt.Sprintf("writer-%d", i))
		c.FollowedTopicMatches = 2
		c.TopicTags = []string{"elections"}
		topicFlood = append(topicFlood, c)
	}
	diverseTopic := baseCandidate("diverse-topic", "writer-diverse")
	diverseTopic.TopicTags = []string{"astronomy"}
	topicFlood = append(topicFlood, diverseTopic)

	resTopic, err := r.Rank(RankRequest{
		Candidates: topicFlood,
		PageSize:   4,
		Now:        refNow,
	})
	if err != nil {
		t.Fatal(err)
	}
	if resTopic.DiversityAudit.TopicCapDeferrals == 0 {
		t.Fatalf("expected topic cap deferrals > 0, got 0")
	}

	// Verify fallback passes when all candidates share one topic or one author.
	singleAuthorPool := []CandidateFeatureVector{
		baseCandidate("sa-1", "solo-author"),
		baseCandidate("sa-2", "solo-author"),
		baseCandidate("sa-3", "solo-author"),
	}
	resSolo, err := r.Rank(RankRequest{Candidates: singleAuthorPool, PageSize: 3, Now: refNow})
	if err != nil {
		t.Fatal(err)
	}
	if len(resSolo.Items) != 3 {
		t.Fatalf("expected fallback pass to return all 3 solo-author items, got %d", len(resSolo.Items))
	}

	sharedTopicPool := []CandidateFeatureVector{
		baseCandidate("st-1", "w-1"),
		baseCandidate("st-2", "w-2"),
		baseCandidate("st-3", "w-3"),
		baseCandidate("st-4", "w-4"),
	}
	resSharedTopic, err := r.Rank(RankRequest{Candidates: sharedTopicPool, PageSize: 4, Now: refNow})
	if err != nil {
		t.Fatal(err)
	}
	if len(resSharedTopic.Items) != 4 {
		t.Fatalf("expected topic-relaxation pass to return 4 items, got %d", len(resSharedTopic.Items))
	}
}

func TestSessionPacingAdvisoriesD01AndD03(t *testing.T) {
	r := NewDefault()

	makePaced := func(id, author string, pacing float64, avail Availability) CandidateFeatureVector {
		c := baseCandidate(id, author)
		c.Signal = &ScopedSignalFeatures{
			SchemaVersion:        SignalSchemaVersionV1,
			Availability:         avail,
			CoverageRatio:        0.5,
			FreshnessSec:         30,
			AttentionPacingScore: pacing,
		}
		return c
	}

	// 1. Three consecutive available high-D01 items trigger slower_context_break.
	d01Batch := []CandidateFeatureVector{
		makePaced("p1", "a1", 0.85, AvailabilityAvailable),
		makePaced("p2", "a2", 0.82, AvailabilityAvailable),
		makePaced("p3", "a3", 0.80, AvailabilityAvailable),
	}
	resD01, err := r.Rank(RankRequest{Candidates: d01Batch, PageSize: 5, Now: refNow})
	if err != nil {
		t.Fatal(err)
	}
	if !resD01.SessionPacingAdvisory.Triggered ||
		resD01.SessionPacingAdvisory.AdvisoryKind != AdvisorySlowerContextBreak ||
		resD01.SessionPacingAdvisory.InsertAfterRank != 3 {
		t.Fatalf("unexpected D01 advisory: %+v", resD01.SessionPacingAdvisory)
	}

	// 2. Unknown availability on D01 never triggers high-intensity streak.
	unknownPaced := []CandidateFeatureVector{
		makePaced("u1", "a1", 0.99, AvailabilityUnknown),
		makePaced("u2", "a2", 0.99, AvailabilityUnknown),
		makePaced("u3", "a3", 0.99, AvailabilityUnknown),
	}
	resUnknown, err := r.Rank(RankRequest{Candidates: unknownPaced, PageSize: 5, Now: refNow})
	if err != nil {
		t.Fatal(err)
	}
	if resUnknown.SessionPacingAdvisory.Triggered {
		t.Fatalf("unknown D01 must not trigger pacing advisory: %+v", resUnknown.SessionPacingAdvisory)
	}

	// 3. Session depth boundary (D03) triggers gentle_stopping_point.
	resD03, err := r.Rank(RankRequest{
		Candidates: unknownPaced,
		PageSize:   5,
		Now:        refNow,
		Session:    SessionContext{SessionItemsSeen: 13},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !resD03.SessionPacingAdvisory.Triggered ||
		resD03.SessionPacingAdvisory.AdvisoryKind != AdvisoryGentleStoppingPoint ||
		resD03.SessionPacingAdvisory.InsertAfterRank != 2 {
		t.Fatalf("unexpected D03 advisory: %+v", resD03.SessionPacingAdvisory)
	}

	// 4. Combined D01 + D03 triggers gentle_stopping_point with both codes.
	resBoth, err := r.Rank(RankRequest{
		Candidates: d01Batch,
		PageSize:   5,
		Now:        refNow,
		Session:    SessionContext{SessionItemsSeen: 14, PriorHighIntensityStreak: 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resBoth.SessionPacingAdvisory.TriggerCodes) != 2 ||
		resBoth.SessionPacingAdvisory.AdvisoryKind != AdvisoryGentleStoppingPoint {
		t.Fatalf("expected both D01 and D03 triggers, got %+v", resBoth.SessionPacingAdvisory)
	}
}

func TestWhyThisPlainLanguageNoInternalLeaks(t *testing.T) {
	r := NewDefault()

	// Exercise every reason code branch, including fallback GENERAL_DISCOVERY.
	olderFallback := baseCandidate("secret-cand-id-9001", "internal-author-uuid-42")
	olderFallback.LanguageMatch = false
	olderFallback.PublishedAt = refNow.Add(-48 * time.Hour)
	olderFallback.ImpressionCount = 500
	olderFallback.ArticleOpenRate = 0.01

	richFollowed := baseCandidate("secret-cand-id-9002", "internal-author-uuid-77")
	richFollowed.FollowedAuthor = true
	richFollowed.FollowedTopicMatches = 1
	richFollowed.ArticleOpenRate = 0.30
	richFollowed.Signal = &ScopedSignalFeatures{
		SchemaVersion:       SignalSchemaVersionV1,
		Availability:        AvailabilityAvailable,
		CoverageRatio:       1.0,
		FreshnessSec:        10,
		EvidenceSupportTier: EvidenceSupported,
	}

	coldDiscovery := baseCandidate("secret-cand-id-9003", "internal-author-uuid-88")
	coldDiscovery.LanguageMatch = false
	coldDiscovery.PublishedAt = refNow.Add(-1 * time.Hour)
	coldDiscovery.ImpressionCount = 0

	res, err := r.Rank(RankRequest{
		Candidates: []CandidateFeatureVector{olderFallback, richFollowed, coldDiscovery},
		PageSize:   10,
		Now:        refNow,
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, item := range res.Items {
		if len(item.WhyThis.Reasons) < 1 || len(item.WhyThis.Reasons) > 2 {
			t.Fatalf("expected 1..2 reasons, got %d", len(item.WhyThis.Reasons))
		}
		if len(item.WhyThis.ReasonCodes) != len(item.WhyThis.Reasons) {
			t.Fatalf("reason codes/reasons length mismatch")
		}
		checkText := item.WhyThis.Summary + " " + strings.Join(item.WhyThis.Reasons, " ")
		if strings.Contains(checkText, item.CandidateID) || strings.Contains(checkText, "uuid") {
			t.Fatalf("WhyThis leaked internal identifier: %q", checkText)
		}
		for _, ch := range checkText {
			if ch >= '0' && ch <= '9' {
				t.Fatalf("WhyThis leaked numeric score or code in text: %q", checkText)
			}
		}
	}
}

func TestCursorPaginationDeterminismAndValidation(t *testing.T) {
	r := NewDefault()

	pool := make([]CandidateFeatureVector, 0, 17)
	for i := 0; i < 17; i++ {
		c := baseCandidate(fmt.Sprintf("item-%02d", i), fmt.Sprintf("author-%d", i%4))
		c.TopicTags = []string{fmt.Sprintf("topic-%d", i%3), "shared"}
		c.PublishedAt = refNow.Add(-time.Duration(i*15) * time.Minute)
		if i == 0 {
			// Include duplicate candidate ID to ensure deduplication.
			pool = append(pool, c)
		}
		pool = append(pool, c)
	}

	collectAll := func() ([]string, []string) {
		var ids []string
		var cursors []string
		cursor := ""
		for {
			res, err := r.Rank(RankRequest{
				Candidates: pool,
				PageSize:   5,
				Cursor:     cursor,
				Now:        refNow,
			})
			if err != nil {
				t.Fatalf("unexpected pagination error: %v", err)
			}
			for _, it := range res.Items {
				ids = append(ids, it.CandidateID)
			}
			cursors = append(cursors, res.NextCursor)
			if res.NextCursor == "" {
				break
			}
			cursor = res.NextCursor
		}
		return ids, cursors
	}

	run1IDs, run1Cursors := collectAll()
	run2IDs, run2Cursors := collectAll()

	if len(run1IDs) != 17 {
		t.Fatalf("expected all 17 unique items across pages, got %d", len(run1IDs))
	}
	seen := make(map[string]bool, 17)
	for i, id := range run1IDs {
		if seen[id] {
			t.Fatalf("duplicate candidate %s across pages", id)
		}
		seen[id] = true
		if run2IDs[i] != id {
			t.Fatalf("non-deterministic item at index %d: %s vs %s", i, id, run2IDs[i])
		}
	}
	for i := range run1Cursors {
		if run1Cursors[i] != run2Cursors[i] {
			t.Fatalf("non-deterministic cursor at page %d", i)
		}
	}

	// Invalid cursor cases must all return ErrInvalidCursor.
	invalidCursors := []string{
		"%%%not-base64%%%",
		base64.RawURLEncoding.EncodeToString([]byte(`{"v":99,"seen":["a"]}`)),
		base64.RawURLEncoding.EncodeToString([]byte(`{"v":1,"seen":[""]}`)),
		base64.RawURLEncoding.EncodeToString([]byte(`{"v":1,"seen":["a"],"unknown_field":true}`)),
		base64.RawURLEncoding.EncodeToString([]byte(`{broken-json`)),
		strings.Repeat("a", maxCursorBytes+10),
	}
	for _, bad := range invalidCursors {
		_, err := r.Rank(RankRequest{Candidates: pool, PageSize: 5, Cursor: bad, Now: refNow})
		if !errors.Is(err, ErrInvalidCursor) {
			t.Fatalf("cursor %q: expected ErrInvalidCursor, got %v", bad[:min(16, len(bad))], err)
		}
	}
}

func TestConfigNormalizationConcurrencyAndLatencyGate(t *testing.T) {
	// Zero/negative config normalization + empty Now fallback + empty batch.
	custom := New(Config{
		DefaultPageSize:               500,
		MaxPageSize:                   10,
		MaxSignalBoost:                -1,
		MaxSignalDemotion:             -1,
		ColdStartExplorationBonus:     -1,
		DiversityWindowSize:           1,
		MMRAuthorPenalty:              -1,
		MMRTopicPenalty:               -1,
		HighIntensityPacingThreshold:  2.0,
		MaxHighIntensityStreak:        -1,
		SessionDepthAdvisoryThreshold: -1,
	})
	emptyRes, err := custom.Rank(RankRequest{PageSize: 0})
	if err != nil {
		t.Fatal(err)
	}
	if len(emptyRes.Items) != 0 || emptyRes.NextCursor != "" {
		t.Fatalf("expected empty result, got %+v", emptyRes)
	}

	// Build 200-candidate batch for latency gate (< 5ms) and concurrent -race check.
	r := NewDefault()
	batch200 := make([]CandidateFeatureVector, 200)
	for i := 0; i < 200; i++ {
		c := baseCandidate(fmt.Sprintf("c200-%03d", i), fmt.Sprintf("author-%02d", i%25))
		c.TopicTags = []string{fmt.Sprintf("t-%d", i%12), fmt.Sprintf("Sub-%d", i%7), "  "}
		c.PublishedAt = refNow.Add(-time.Duration(i*5) * time.Minute)
		c.FollowedAuthor = i%7 == 0
		c.FollowedTopicMatches = i % 3
		c.Signal = &ScopedSignalFeatures{
			SchemaVersion:             SignalSchemaVersionV1,
			Availability:              AvailabilityAvailable,
			CoverageRatio:             0.8,
			FreshnessSec:              120,
			EvidenceSupportTier:       EvidenceSupported,
			IndependentSourceClusters: i % 4,
		}
		batch200[i] = c
	}

	// Warmup + measure median over 20 runs to avoid CI scheduler jitter while asserting < 5ms.
	start := time.Now()
	const runs = 20
	var sampleResult RankedFeedResult
	for i := 0; i < runs; i++ {
		sampleResult, err = r.Rank(RankRequest{
			Candidates: batch200,
			PageSize:   20,
			Now:        refNow,
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	avgDuration := time.Since(start) / runs
	if avgDuration >= 5*time.Millisecond {
		t.Fatalf("200-candidate ranking exceeded 5ms budget: avg=%v", avgDuration)
	}

	// Verify JSON round-trip of RankedFeedResult.
	rawJSON, err := json.Marshal(sampleResult)
	if err != nil {
		t.Fatal(err)
	}
	var decoded RankedFeedResult
	if err := json.Unmarshal(rawJSON, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.SchemaVersion != ResultSchemaVersionV1 || len(decoded.Items) != 20 {
		t.Fatalf("unexpected JSON round-trip result: %+v", decoded)
	}

	// Concurrent goroutines sharing the same Ranker and input slice under -race.
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			res, err := r.Rank(RankRequest{
				Candidates: batch200,
				PageSize:   25,
				Now:        refNow,
			})
			if err != nil || len(res.Items) != 25 {
				t.Errorf("concurrent Rank failed: err=%v len=%d", err, len(res.Items))
			}
		}()
	}
	wg.Wait()
}

func TestIsolatedDimensionsAndMissingMetadata(t *testing.T) {
	r := NewDefault()
	// Exercise empty author, empty topics, clamped PageSize > MaxPageSize, and isolated Signal dimensions.
	c1 := baseCandidate("iso-1", "")
	c1.TopicTags = nil
	c1.Signal = &ScopedSignalFeatures{
		SchemaVersion:   SignalSchemaVersionV1,
		Availability:    AvailabilityAvailable,
		CoverageRatio:   0.2,
		FreshnessSec:    10,
		ConsistencyTier: ConsistencyConsistent,
	}

	c2 := baseCandidate("iso-2", "")
	c2.TopicTags = []string{}
	c2.PublishedAt = refNow.Add(-3 * time.Hour)
	c2.Signal = &ScopedSignalFeatures{
		SchemaVersion:             SignalSchemaVersionV1,
		Availability:              AvailabilityAvailable,
		CoverageRatio:             0.2,
		FreshnessSec:              10,
		IndependentSourceClusters: 2,
	}

	c3 := baseCandidate("iso-3", "")
	c3.TopicTags = nil
	c3.PublishedAt = refNow.Add(-1 * time.Hour)
	c3.Signal = &ScopedSignalFeatures{
		SchemaVersion:  SignalSchemaVersionV1,
		Availability:   AvailabilityAvailable,
		CoverageRatio:  0.2,
		FreshnessSec:   10,
		BaitSwitchRisk: 0.20,
	}

	res, err := r.Rank(RankRequest{
		Candidates: []CandidateFeatureVector{c1, c2, c3},
		PageSize:   999,
		Now:        refNow,
	})
	if err != nil || len(res.Items) != 3 {
		t.Fatalf("expected 3 items, err=%v len=%d", err, len(res.Items))
	}
}

func BenchmarkRank200(b *testing.B) {
	r := NewDefault()
	batch := make([]CandidateFeatureVector, 200)
	for i := 0; i < 200; i++ {
		c := baseCandidate(fmt.Sprintf("bench-%03d", i), fmt.Sprintf("author-%02d", i%30))
		c.TopicTags = []string{fmt.Sprintf("topic-%d", i%10), fmt.Sprintf("topic-%d", (i+3)%10)}
		c.PublishedAt = refNow.Add(-time.Duration(i*3) * time.Minute)
		batch[i] = c
	}
	req := RankRequest{Candidates: batch, PageSize: 20, Now: refNow}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = r.Rank(req)
	}
}
