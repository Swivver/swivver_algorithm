package ranker

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"
	"time"
)

const (
	corpusSeed5K     uint64 = 20261114
	policyFuzzSeed5K uint64 = 20261115
)

var corpusTopics = []string{
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
}

var nonAvailableSignalStates = []Availability{
	AvailabilityUnknown,
	AvailabilityUnsupported,
	AvailabilityNotApplicable,
	AvailabilityStale,
}

type splitMix64 struct {
	state uint64
}

func newSplitMix64(seed uint64) *splitMix64 {
	return &splitMix64{state: seed}
}

func (s *splitMix64) nextUint64() uint64 {
	s.state += 0x9e3779b97f4a7c15
	z := s.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func (s *splitMix64) float64() float64 {
	return float64(s.nextUint64()>>11) * (1.0 / float64(uint64(1)<<53))
}

func (s *splitMix64) intn(n int) int {
	if n <= 0 {
		return 0
	}
	return int(s.nextUint64() % uint64(n))
}

func (s *splitMix64) uniform(low, high float64) float64 {
	return low + (high-low)*s.float64()
}

type corpusPackage struct {
	candidate                CandidateFeatureVector
	packageKind              string
	split                    string
	archetype                string
	isRagebaitOrContradicted bool
}

func buildAuthorSchedule5K() (map[string][]string, map[string][]string) {
	authorsByLang := map[string][]string{"en": {}, "pt": {}, "es": {}}
	scheduleByLang := map[string][]string{"en": {}, "pt": {}, "es": {}}

	type langSpec struct {
		lang   string
		count  int
		target int
	}
	specs := []langSpec{
		{lang: "en", count: 60, target: 3000},
		{lang: "pt", count: 22, target: 1100},
		{lang: "es", count: 18, target: 900},
	}

	authorSeq := 0
	for _, sp := range specs {
		rawWeights := make([]float64, sp.count)
		weightSum := 0.0
		for i := 0; i < sp.count; i++ {
			w := 0.55 + 1.85*(float64(i%10)/9.0)
			rawWeights[i] = w
			weightSum += w
		}
		quotas := make([]int, sp.count)
		qSum := 0
		for i, w := range rawWeights {
			q := int(math.Round((w / weightSum) * float64(sp.target)))
			quotas[i] = q
			qSum += q
		}
		diff := sp.target - qSum
		steps := diff
		if steps < 0 {
			steps = -steps
		}
		for k := 0; k < steps; k++ {
			if diff > 0 {
				quotas[k%sp.count]++
			} else {
				quotas[k%sp.count]--
			}
		}

		langIDs := make([]string, sp.count)
		remaining := make([]int, sp.count)
		copy(remaining, quotas)
		for i := 0; i < sp.count; i++ {
			langIDs[i] = fmt.Sprintf("author-%03d", authorSeq)
			authorSeq++
		}
		authorsByLang[sp.lang] = langIDs

		expanded := make([]string, 0, sp.target)
		for len(expanded) < sp.target {
			for idx, aid := range langIDs {
				if remaining[idx] > 0 {
					expanded = append(expanded, aid)
					remaining[idx]--
				}
			}
		}
		scheduleByLang[sp.lang] = expanded
	}
	return authorsByLang, scheduleByLang
}

func generate5KCorpus(seed uint64) ([]corpusPackage, map[string][]string) {
	rng := newSplitMix64(seed)
	authorsByLang, scheduleByLang := buildAuthorSchedule5K()
	langCursors := map[string]int{"en": 0, "pt": 0, "es": 0}

	corpus := make([]corpusPackage, 5000)
	for idx := 0; idx < 5000; idx++ {
		pkgKind := "news"
		if idx >= 3500 {
			pkgKind = "partner_media"
		}
		split := "dev"
		if idx >= 4000 {
			split = "locked_test"
		} else if idx >= 3000 {
			split = "calibration"
		}

		langSlot := idx % 50
		lang := "en"
		if langSlot >= 41 {
			lang = "es"
		} else if langSlot >= 30 {
			lang = "pt"
		}

		curPos := langCursors[lang]
		authorID := scheduleByLang[lang][curPos]
		langCursors[lang] = curPos + 1

		primaryTopic := corpusTopics[(idx/50+langSlot)%len(corpusTopics)]
		var topicTags []string
		if rng.float64() < 0.35 {
			sec := corpusTopics[(idx/len(corpusTopics)+1+rng.intn(len(corpusTopics)-1))%len(corpusTopics)]
			if sec == primaryTopic {
				topicTags = []string{primaryTopic}
			} else if primaryTopic < sec {
				topicTags = []string{primaryTopic, sec}
			} else {
				topicTags = []string{sec, primaryTopic}
			}
		} else {
			topicTags = []string{primaryTopic}
		}

		timelineProgress := float64(4999-idx) / 5000.0
		jitterMin := rng.intn(45)
		ageMinutes := int(timelineProgress*20160.0) + jitterMin + 5
		pubAt := refNow.Add(-time.Duration(ageMinutes) * time.Minute)

		bucket := (((idx / 50) * 37) + (langSlot * 7)) % 100
		eligible := true
		policyBlocked := false
		anomalyFlagged := false
		isRagebait := false
		var archetype string
		var watchRatio, articleOpen, sourceInspect, saveRate float64
		var impressions int
		var sig *ScopedSignalFeatures

		switch {
		case bucket < 5:
			archetype = "ineligible"
			eligible = idx%2 == 0
			policyBlocked = !eligible || idx%3 == 0
			if eligible && !policyBlocked {
				eligible = false
			}
			watchRatio = 0.99
			articleOpen = 0.95
			sourceInspect = 0.90
			saveRate = 0.85
			impressions = 500
			sig = &ScopedSignalFeatures{
				SchemaVersion:             SignalSchemaVersionV1,
				SnapshotID:                fmt.Sprintf("snap-%05d", idx),
				Availability:              AvailabilityAvailable,
				CoverageRatio:             1.0,
				FreshnessSec:              30.0,
				EvidenceSupportTier:       EvidenceSupported,
				ConsistencyTier:           ConsistencyConsistent,
				TechnicalLegibilityTier:   LegibilityClear,
				ProvenanceStatus:          ProvenanceVerified,
				IndependentSourceClusters: 4,
			}

		case bucket < 12:
			archetype = "bot_brigade"
			anomalyFlagged = true
			watchRatio = roundScore(rng.uniform(0.92, 1.00))
			articleOpen = roundScore(rng.uniform(0.85, 0.99))
			sourceInspect = roundScore(rng.uniform(0.80, 0.98))
			saveRate = roundScore(rng.uniform(0.75, 0.95))
			impressions = rng.intn(60) + 5

		case bucket < 21:
			archetype = "ragebait_contradicted"
			isRagebait = true
			watchRatio = roundScore(rng.uniform(0.88, 0.995))
			articleOpen = roundScore(rng.uniform(0.005, 0.045))
			sourceInspect = roundScore(rng.uniform(0.0, 0.02))
			saveRate = roundScore(rng.uniform(0.0, 0.015))
			impressions = rng.intn(900) + 150
			cons := ConsistencyMixed
			if idx%2 == 0 {
				cons = ConsistencyInconsistent
			}
			sig = &ScopedSignalFeatures{
				SchemaVersion:             SignalSchemaVersionV1,
				SnapshotID:                fmt.Sprintf("snap-%05d", idx),
				Availability:              AvailabilityAvailable,
				CoverageRatio:             0.90,
				FreshnessSec:              math.Round(rng.uniform(20.0, 1200.0)*1e3) / 1e3,
				EvidenceSupportTier:       EvidenceContradicted,
				ConsistencyTier:           cons,
				TechnicalLegibilityTier:   LegibilityClear,
				AttentionPacingScore:      roundScore(rng.uniform(0.72, 0.96)),
				BaitSwitchRisk:            roundScore(rng.uniform(0.55, 0.92)),
				ProvenanceStatus:          ProvenanceMissing,
				IndependentSourceClusters: 0,
			}

		case bucket < 30:
			archetype = "ragebait_bait"
			isRagebait = true
			watchRatio = roundScore(rng.uniform(0.87, 0.99))
			articleOpen = roundScore(rng.uniform(0.01, 0.05))
			sourceInspect = roundScore(rng.uniform(0.0, 0.025))
			saveRate = roundScore(rng.uniform(0.0, 0.02))
			impressions = rng.intn(850) + 140
			leg := LegibilityClear
			if idx%3 == 0 {
				leg = LegibilityDegraded
			}
			sig = &ScopedSignalFeatures{
				SchemaVersion:             SignalSchemaVersionV1,
				SnapshotID:                fmt.Sprintf("snap-%05d", idx),
				Availability:              AvailabilityAvailable,
				CoverageRatio:             0.85,
				FreshnessSec:              math.Round(rng.uniform(25.0, 1800.0)*1e3) / 1e3,
				EvidenceSupportTier:       EvidenceMixed,
				ConsistencyTier:           ConsistencyInconsistent,
				TechnicalLegibilityTier:   leg,
				AttentionPacingScore:      roundScore(rng.uniform(0.80, 0.98)),
				BaitSwitchRisk:            roundScore(rng.uniform(0.78, 0.98)),
				ProvenanceStatus:          ProvenanceMissing,
				IndependentSourceClusters: 0,
			}

		case bucket < 65:
			archetype = "developing"
			watchRatio = roundScore(rng.uniform(0.42, 0.76))
			articleOpen = roundScore(rng.uniform(0.18, 0.42))
			sourceInspect = roundScore(rng.uniform(0.10, 0.28))
			saveRate = roundScore(rng.uniform(0.04, 0.13))
			impressions = rng.intn(180)
			switch statePick := idx % 6; statePick {
			case 0:
				sig = nil
			case 1, 2, 3, 4:
				sig = &ScopedSignalFeatures{
					SchemaVersion: SignalSchemaVersionV1,
					SnapshotID:    fmt.Sprintf("snap-%05d", idx),
					Availability:  nonAvailableSignalStates[statePick-1],
					CoverageRatio: 0.0,
					FreshnessSec:  math.Round(rng.uniform(10.0, 600.0)*1e3) / 1e3,
				}
			default:
				sig = &ScopedSignalFeatures{
					SchemaVersion:             SignalSchemaVersionV1,
					SnapshotID:                fmt.Sprintf("snap-%05d", idx),
					Availability:              AvailabilityAvailable,
					CoverageRatio:             0.50,
					FreshnessSec:              math.Round(rng.uniform(15.0, 900.0)*1e3) / 1e3,
					EvidenceSupportTier:       EvidenceUnknown,
					ConsistencyTier:           ConsistencyConsistent,
					TechnicalLegibilityTier:   LegibilityClear,
					ProvenanceStatus:          ProvenanceMissing,
					IndependentSourceClusters: 1,
				}
			}

		default:
			archetype = "supported"
			watchRatio = roundScore(rng.uniform(0.45, 0.78))
			articleOpen = roundScore(rng.uniform(0.24, 0.48))
			sourceInspect = roundScore(rng.uniform(0.15, 0.36))
			saveRate = roundScore(rng.uniform(0.06, 0.16))
			impressions = rng.intn(450) + 30
			prov := ProvenanceDeclared
			if idx%3 == 0 {
				prov = ProvenanceVerified
			}
			sig = &ScopedSignalFeatures{
				SchemaVersion:             SignalSchemaVersionV1,
				SnapshotID:                fmt.Sprintf("snap-%05d", idx),
				Availability:              AvailabilityAvailable,
				CoverageRatio:             1.0,
				FreshnessSec:              math.Round(rng.uniform(15.0, 2400.0)*1e3) / 1e3,
				EvidenceSupportTier:       EvidenceSupported,
				ConsistencyTier:           ConsistencyConsistent,
				TechnicalLegibilityTier:   LegibilityClear,
				AttentionPacingScore:      roundScore(rng.uniform(0.15, 0.55)),
				BaitSwitchRisk:            roundScore(rng.uniform(0.01, 0.14)),
				ProvenanceStatus:          prov,
				IndependentSourceClusters: 2 + (idx % 3),
			}
		}

		corpus[idx] = corpusPackage{
			candidate: CandidateFeatureVector{
				SchemaVersion:        CandidateSchemaVersionV1,
				CandidateID:          fmt.Sprintf("pkg-%05d", idx),
				AuthorID:             authorID,
				TopicTags:            topicTags,
				Language:             lang,
				PublishedAt:          pubAt,
				Eligible:             eligible,
				PolicyBlocked:        policyBlocked,
				FollowedAuthor:       false,
				FollowedTopicMatches: 1,
				LanguageMatch:        true,
				ImpressionCount:      impressions,
				ValidWatchRatio:      watchRatio,
				ArticleOpenRate:      articleOpen,
				SourceInspectRate:    sourceInspect,
				SaveRate:             saveRate,
				AnomalyFlagged:       anomalyFlagged,
				Signal:               sig,
			},
			packageKind:              pkgKind,
			split:                    split,
			archetype:                archetype,
			isRagebaitOrContradicted: isRagebait,
		}
	}

	return corpus, authorsByLang
}

func computeGiniFromCounts(counts map[string]int) float64 {
	if len(counts) <= 1 {
		return 0.0
	}
	vals := make([]int, 0, len(counts))
	total := 0
	for _, v := range counts {
		if v >= 0 {
			vals = append(vals, v)
			total += v
		}
	}
	n := len(vals)
	if n <= 1 || total <= 0 {
		return 0.0
	}
	slices.Sort(vals)
	var weightedSum float64
	for idx, v := range vals {
		pos := idx + 1
		weightedSum += float64((2*pos - n - 1) * v)
	}
	return math.Max(0.0, weightedSum/float64(n*total))
}

func TestVG3PolicySandwichFuzz10000(t *testing.T) {
	r := NewDefault()
	corpus, _ := generate5KCorpus(corpusSeed5K)
	rng := newSplitMix64(policyFuzzSeed5K)

	const trials = 10000
	totalIneligibleTested := 0

	for trial := 0; trial < trials; trial++ {
		batchSize := 12 + rng.intn(16)
		batch := make([]CandidateFeatureVector, batchSize)
		forbidden := make(map[string]bool, batchSize)
		expectedDrops := 0

		for slot := 0; slot < batchSize; slot++ {
			pkg := corpus[(trial*13+slot*37+rng.intn(len(corpus)))%len(corpus)]
			cand := pkg.candidate
			cid := fmt.Sprintf("fuzz-%d-%d", trial, slot)
			cand.CandidateID = cid

			if rng.float64() < 0.35 {
				switch (trial + slot) % 3 {
				case 0:
					cand.Eligible = false
					cand.PolicyBlocked = false
				case 1:
					cand.Eligible = true
					cand.PolicyBlocked = true
				default:
					cand.Eligible = false
					cand.PolicyBlocked = true
				}
				cand.FollowedAuthor = true
				cand.FollowedTopicMatches = 5
				cand.LanguageMatch = true
				cand.ValidWatchRatio = 1.0
				cand.ArticleOpenRate = 1.0
				cand.SourceInspectRate = 1.0
				cand.SaveRate = 1.0
				cand.AnomalyFlagged = false
				cand.Signal = &ScopedSignalFeatures{
					SchemaVersion:             SignalSchemaVersionV1,
					Availability:              AvailabilityAvailable,
					CoverageRatio:             1.0,
					FreshnessSec:              5.0,
					EvidenceSupportTier:       EvidenceSupported,
					ConsistencyTier:           ConsistencyConsistent,
					TechnicalLegibilityTier:   LegibilityClear,
					ProvenanceStatus:          ProvenanceVerified,
					IndependentSourceClusters: 5,
				}
				forbidden[cid] = true
				expectedDrops++
				totalIneligibleTested++
			} else {
				cand.Eligible = true
				cand.PolicyBlocked = false
			}
			batch[slot] = cand
		}

		res, err := r.Rank(RankRequest{
			Candidates: batch,
			PageSize:   10,
			Now:        refNow,
		})
		if err != nil {
			t.Fatalf("trial %d unexpected error: %v", trial, err)
		}
		if res.DiversityAudit.DroppedIneligible != expectedDrops {
			t.Fatalf("trial %d expected %d dropped, got %d", trial, expectedDrops, res.DiversityAudit.DroppedIneligible)
		}
		for _, item := range res.Items {
			if forbidden[item.CandidateID] || strings.TrimSpace(item.CandidateID) == "" {
				t.Fatalf("trial %d leaked ineligible candidate %q", trial, item.CandidateID)
			}
		}
	}

	t.Logf("VG-3 Policy-Sandwich Fuzzing: 0/%d leaked trials (0/%d adversarial ineligible candidates exposed, 0.00%% leak rate)",
		trials, totalIneligibleTested)
}

func TestVG3CounterfactualUncertaintyNeutrality5000(t *testing.T) {
	r := NewDefault()
	corpus, _ := generate5KCorpus(corpusSeed5K)
	if len(corpus) != 5000 {
		t.Fatalf("expected 5000 corpus packages, got %d", len(corpus))
	}

	// 1. Per-candidate sweep across all 5,000 packages x (nil + 4 non-available states = 25,000 evaluations).
	for idx := range corpus {
		base := corpus[idx].candidate
		base.Eligible = true
		base.PolicyBlocked = false
		base.Signal = nil

		baseScored := r.scoreCandidate(&base, base.CandidateID, refNow)
		if math.Abs(baseScored.signalDelta) >= 1e-12 {
			t.Fatalf("candidate %s nil signal produced non-zero delta: %.15f", base.CandidateID, baseScored.signalDelta)
		}

		for _, state := range nonAvailableSignalStates {
			cf := base
			cf.Signal = &ScopedSignalFeatures{
				SchemaVersion:           SignalSchemaVersionV1,
				Availability:            state,
				CoverageRatio:           0.0,
				FreshnessSec:            60.0,
				EvidenceSupportTier:     EvidenceContradicted,
				ConsistencyTier:         ConsistencyInconsistent,
				TechnicalLegibilityTier: LegibilityUnusable,
				AttentionPacingScore:    0.99,
				BaitSwitchRisk:          0.99,
				ProvenanceStatus:        ProvenanceInvalid,
			}
			cfScored := r.scoreCandidate(&cf, cf.CandidateID, refNow)
			if math.Abs(cfScored.signalDelta) >= 1e-12 {
				t.Fatalf("candidate %s state %s produced non-zero SignalDelta: %.15f", cf.CandidateID, state, cfScored.signalDelta)
			}
			if math.Abs(cfScored.rawScore-baseScored.rawScore) >= 1e-12 {
				t.Fatalf("candidate %s state %s altered score: %.12f vs %.12f", cf.CandidateID, state, cfScored.rawScore, baseScored.rawScore)
			}
		}
	}

	// 2. Multi-pool rank non-demotion sweep across all 5,000 candidates (250 pools x 20 candidates).
	demotions := 0
	for start := 0; start < len(corpus); start += 20 {
		basePool := make([]CandidateFeatureVector, 20)
		cfPool := make([]CandidateFeatureVector, 20)
		for offset := 0; offset < 20; offset++ {
			c0 := corpus[start+offset].candidate
			c0.Eligible = true
			c0.PolicyBlocked = false
			c0.Signal = nil
			basePool[offset] = c0

			c1 := c0
			state := nonAvailableSignalStates[(start+offset)%len(nonAvailableSignalStates)]
			c1.Signal = &ScopedSignalFeatures{
				SchemaVersion:       SignalSchemaVersionV1,
				Availability:        state,
				CoverageRatio:       0.0,
				FreshnessSec:        90.0,
				EvidenceSupportTier: EvidenceContradicted,
				BaitSwitchRisk:      0.95,
			}
			cfPool[offset] = c1
		}

		resBase, err := r.Rank(RankRequest{Candidates: basePool, PageSize: 20, Now: refNow})
		if err != nil {
			t.Fatal(err)
		}
		resCF, err := r.Rank(RankRequest{Candidates: cfPool, PageSize: 20, Now: refNow})
		if err != nil {
			t.Fatal(err)
		}

		baseRank := make(map[string]int, 20)
		for _, it := range resBase.Items {
			baseRank[it.CandidateID] = it.Rank
		}
		for _, it := range resCF.Items {
			if it.Rank > baseRank[it.CandidateID] {
				demotions++
			}
		}
	}
	if demotions != 0 {
		t.Fatalf("expected 0 counterfactual rank demotions across 5000 candidates, got %d", demotions)
	}
	t.Logf("VG-3 Counterfactual Uncertainty Neutrality: 5,000 candidates (25,000 evaluations), SignalDelta=0.0, rank demotions=0")
}

func TestVG5BotBrigadingImmunity5000(t *testing.T) {
	r := NewDefault()
	corpus, _ := generate5KCorpus(corpusSeed5K)

	tested := 0
	for idx := range corpus {
		cand := corpus[idx].candidate
		if !cand.AnomalyFlagged {
			continue
		}
		cand.Eligible = true
		cand.PolicyBlocked = false
		tested++

		evalNow := cand.PublishedAt.Add(1 * time.Hour)
		scored := r.scoreCandidate(&cand, cand.CandidateID, evalNow)
		if scored.engagement != 0.0 || scored.explorationBonus != 0.0 {
			t.Fatalf("anomaly-flagged candidate %s retained boost: engagement=%f exploration=%f",
				cand.CandidateID, scored.engagement, scored.explorationBonus)
		}

		cleanZero := cand
		cleanZero.AnomalyFlagged = false
		cleanZero.ValidWatchRatio = 0.0
		cleanZero.ArticleOpenRate = 0.0
		cleanZero.SourceInspectRate = 0.0
		cleanZero.SaveRate = 0.0
		cleanZero.ImpressionCount = 500
		baseScored := r.scoreCandidate(&cleanZero, cleanZero.CandidateID, evalNow)
		if scored.rawScore > baseScored.rawScore+1e-12 {
			t.Fatalf("anomaly-flagged candidate %s exceeded zero-engagement baseline: %f > %f",
				cand.CandidateID, scored.rawScore, baseScored.rawScore)
		}
	}
	if tested == 0 {
		t.Fatalf("expected non-zero anomaly-flagged candidates in 5000 corpus")
	}
	t.Logf("VG-5 Bot/Brigading Immunity: %d/%d (100.0%%) anomaly-flagged candidates stripped of artificial boosts", tested, tested)
}

func TestVG5RagebaitSuppressionFairnessParityAndLatency5000(t *testing.T) {
	r := NewDefault()
	corpus, authorsByLang := generate5KCorpus(corpusSeed5K)

	eligiblePkgs := make([]corpusPackage, 0, len(corpus))
	byLang := map[string][]corpusPackage{"en": {}, "pt": {}, "es": {}}
	ragebaitIDs := make(map[string]bool)
	pkgByID := make(map[string]corpusPackage, len(corpus))
	supplyAuthorCounts := make(map[string]int)
	rankedAuthorCounts := make(map[string]int)

	for _, p := range corpus {
		if !p.candidate.Eligible || p.candidate.PolicyBlocked {
			continue
		}
		eligiblePkgs = append(eligiblePkgs, p)
		byLang[p.candidate.Language] = append(byLang[p.candidate.Language], p)
		pkgByID[p.candidate.CandidateID] = p
		supplyAuthorCounts[p.candidate.AuthorID]++
		rankedAuthorCounts[p.candidate.AuthorID] = 0
		if p.isRagebaitOrContradicted {
			ragebaitIDs[p.candidate.CandidateID] = true
		}
	}

	const numSessions = 500
	nElig := len(eligiblePkgs)
	watchRagebaitTop10 := 0
	rankerRagebaitTop10 := 0
	totalTop10Slots := 0
	rankedLangCounts := map[string]int{"en": 0, "pt": 0, "es": 0}
	latenciesMs := make([]float64, numSessions)

	for s := 0; s < numSessions; s++ {
		langMod := s % 50
		viewerLang := "en"
		if langMod >= 41 {
			viewerLang = "es"
		} else if langMod >= 30 {
			viewerLang = "pt"
		}

		prefT1 := corpusTopics[s%len(corpusTopics)]
		prefT2 := corpusTopics[(s*3+1)%len(corpusTopics)]

		langAuthors := authorsByLang[viewerLang]
		followedAuthors := make(map[string]bool, 4)
		for k := 0; k < 4; k++ {
			followedAuthors[langAuthors[(s*3+k*7)%len(langAuthors)]] = true
		}

		langElig := byLang[viewerLang]
		endLang := int((float64(s+1) / float64(numSessions)) * float64(len(langElig)-1))
		var inLangSlice []corpusPackage
		if endLang < 160 {
			inLangSlice = langElig[:160]
		} else {
			inLangSlice = langElig[endLang-160 : endLang]
		}

		seenPool := make(map[string]bool, 200)
		windowSlice := make([]corpusPackage, 0, 200)
		var maxPub time.Time
		for _, p := range inLangSlice {
			seenPool[p.candidate.CandidateID] = true
			windowSlice = append(windowSlice, p)
			if p.candidate.PublishedAt.After(maxPub) {
				maxPub = p.candidate.PublishedAt
			}
		}

		endAll := int((float64(s+1) / float64(numSessions)) * float64(nElig-1))
		for probe := endAll; len(windowSlice) < 200 && probe >= 0; probe-- {
			candPkg := eligiblePkgs[probe]
			if !seenPool[candPkg.candidate.CandidateID] {
				seenPool[candPkg.candidate.CandidateID] = true
				windowSlice = append(windowSlice, candPkg)
				if candPkg.candidate.PublishedAt.After(maxPub) {
					maxPub = candPkg.candidate.PublishedAt
				}
			}
		}
		sessionNow := maxPub.Add(5 * time.Minute)

		batch := make([]CandidateFeatureVector, len(windowSlice))
		for i, p := range windowSlice {
			c := p.candidate
			matches := 0
			for _, tag := range c.TopicTags {
				if tag == prefT1 || tag == prefT2 {
					matches++
				}
			}
			c.FollowedTopicMatches = matches
			c.LanguageMatch = c.Language == viewerLang
			c.FollowedAuthor = followedAuthors[c.AuthorID]
			batch[i] = c
		}

		// Pure watch-time sort top-10 baseline.
		watchSorted := make([]CandidateFeatureVector, len(batch))
		copy(watchSorted, batch)
		slices.SortFunc(watchSorted, func(a, b CandidateFeatureVector) int {
			return compareCandidates(
				a.ValidWatchRatio,
				b.ValidWatchRatio,
				a.PublishedAt.UnixNano(),
				b.PublishedAt.UnixNano(),
				a.CandidateID,
				b.CandidateID,
			)
		})
		for i := 0; i < 10 && i < len(watchSorted); i++ {
			if ragebaitIDs[watchSorted[i].CandidateID] {
				watchRagebaitTop10++
			}
		}

		start := time.Now()
		res, err := r.Rank(RankRequest{
			Candidates: batch,
			PageSize:   10,
			Now:        sessionNow,
		})
		latenciesMs[s] = float64(time.Since(start).Nanoseconds()) / 1e6
		if err != nil {
			t.Fatalf("session %d rank error: %v", s, err)
		}

		for _, item := range res.Items {
			totalTop10Slots++
			if ragebaitIDs[item.CandidateID] {
				rankerRagebaitTop10++
			}
			orig := pkgByID[item.CandidateID]
			rankedLangCounts[orig.candidate.Language]++
			rankedAuthorCounts[orig.candidate.AuthorID]++
		}
	}

	suppression := float64(watchRagebaitTop10-rankerRagebaitTop10) / float64(watchRagebaitTop10)
	if suppression < 0.85 {
		t.Fatalf("VG-5 ragebait suppression %.4f below 0.85 threshold (watch=%d ranker=%d)",
			suppression, watchRagebaitTop10, rankerRagebaitTop10)
	}

	supplyGini := computeGiniFromCounts(supplyAuthorCounts)
	rankedGini := computeGiniFromCounts(rankedAuthorCounts)
	giniDelta := rankedGini - supplyGini
	if math.Abs(giniDelta) > 0.03 {
		t.Fatalf("VG-5 author Gini delta %.6f exceeded +-0.03 bound (supply=%.6f ranked=%.6f)",
			giniDelta, supplyGini, rankedGini)
	}

	supplyMinorityShare := float64(len(byLang["pt"])+len(byLang["es"])) / float64(nElig)
	rankedMinorityShare := float64(rankedLangCounts["pt"]+rankedLangCounts["es"]) / float64(totalTop10Slots)
	minorityDelta := rankedMinorityShare - supplyMinorityShare
	if math.Abs(minorityDelta) > 0.03 {
		t.Fatalf("VG-5 minority language share delta %.6f exceeded +-0.03 bound (supply=%.4f ranked=%.4f)",
			minorityDelta, supplyMinorityShare, rankedMinorityShare)
	}

	slices.Sort(latenciesMs)
	p50Ms := latenciesMs[int(0.50*float64(len(latenciesMs)-1))]
	p95Ms := latenciesMs[int(0.95*float64(len(latenciesMs)-1))]
	p99Ms := latenciesMs[int(0.99*float64(len(latenciesMs)-1))]
	if p95Ms > 25.0 {
		t.Fatalf("VG-5 200-candidate ranking p95 latency %.4fms exceeded 25ms target", p95Ms)
	}

	t.Logf("VG-5 Ragebait Suppression: %.2f%% reduction (watch_top10=%d/%d [%.2f%%] -> ranker_top10=%d/%d [%.2f%%])",
		suppression*100.0,
		watchRagebaitTop10, totalTop10Slots, 100.0*float64(watchRagebaitTop10)/float64(totalTop10Slots),
		rankerRagebaitTop10, totalTop10Slots, 100.0*float64(rankerRagebaitTop10)/float64(totalTop10Slots))
	t.Logf("VG-5 Fairness & Language Parity: supply_gini=%.4f ranked_gini=%.4f (delta=%+.4f); minority_lang_supply=%.4f ranked=%.4f (delta=%+.4f)",
		supplyGini, rankedGini, giniDelta, supplyMinorityShare, rankedMinorityShare, minorityDelta)
	t.Logf("VG-5 200-Candidate Latency: p50=%.4fms p95=%.4fms p99=%.4fms (target p95 <= 25.0ms)",
		p50Ms, p95Ms, p99Ms)
}
