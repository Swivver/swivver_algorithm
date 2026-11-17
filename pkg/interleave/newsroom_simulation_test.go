package interleave

import (
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/Swivver/swivver_algorithm/pkg/ranker"
)

func TestMultiBeatNewsroomFairnessAndInterleaving2000Sessions(t *testing.T) {
	// Arm A: Naive engagement-heavy baseline ignoring Signal observations and streak caps.
	naiveCfg := ranker.DefaultConfig()
	naiveCfg.ModelVersion = "naive-engagement-ranker/1.0"
	naiveCfg.MaxSignalBoost = 0.0
	naiveCfg.MaxSignalDemotion = 0.0
	naiveCfg.EngagementWeight = 0.70
	naiveCfg.FollowedAuthorWeight = 0.05
	naiveCfg.TopicMatchWeight = 0.05
	naiveCfg.RegionMatchWeight = 0.0
	naiveCfg.MMRAuthorPenalty = 0.0
	naiveCfg.MMRTopicPenalty = 0.0
	naiveCfg.MaxSameAuthorPerWindow = 10
	naiveCfg.MaxSameTopicPerWindow = 10
	rankerNaive := ranker.New(naiveCfg)

	// Arm B: Signal-informed multi-beat ranker.
	rankerSignal := ranker.NewDefault()

	const sessions = 400
	reports := make([]OutcomeReport, sessions)
	var lowPolishRankSum, studioPolishRankSum float64
	var lowPolishCount, studioPolishCount int
	top5BaitSignal := 0
	maxAuthorStreak := 0

	archetypes := []string{
		"investigative_infrastructure",
		"maritime_climate_science",
		"public_health_desk",
		"regional_breaking_low_polish",
		"tech_ai_policy_analyst",
		"high_arousal_unsourced_bait",
	}
	beats := []string{"infrastructure", "climate", "health", "civic", "technology", "economy"}

	for s := 0; s < sessions; s++ {
		pool := make([]ranker.CandidateFeatureVector, 0, 18)
		authorByID := make(map[string]string, 18)
		isBait := make(map[string]bool, 6)
		isLowPolish := make(map[string]bool, 6)
		isStudioPolish := make(map[string]bool, 12)

		for idx := 0; idx < 18; idx++ {
			arch := archetypes[(idx+s)%len(archetypes)]
			cid := fmt.Sprintf("nr-%d-%02d", s, idx)
			aid := fmt.Sprintf("creator-%s-%d", arch, idx/len(archetypes))
			authorByID[cid] = aid
			beat := beats[(idx*5+s)%len(beats)]
			sub := fmt.Sprintf("inc-%d", (idx*3+s)%5)

			c := ranker.CandidateFeatureVector{
				SchemaVersion:        ranker.CandidateSchemaVersionV1,
				CandidateID:          cid,
				AuthorID:             aid,
				TopicTags:            []string{beat, sub},
				Language:             "en",
				PublishedAt:          testNow.Add(-time.Duration(15+(idx/6)*45) * time.Minute),
				Eligible:             true,
				PolicyBlocked:        false,
				FollowedAuthor:       (idx*3+s*2)%5 == 0 && arch != "high_arousal_unsourced_bait",
				FollowedTopicMatches: (idx*2 + s*3) % 3,
				RegionMatch:          (idx*4+s)%7 == 0 && arch != "high_arousal_unsourced_bait",
				LanguageMatch:        true,
				ImpressionCount:      120,
				ValidWatchRatio:      0.58,
				ArticleOpenRate:      0.36,
				SourceInspectRate:    0.26,
				SaveRate:             0.11,
			}

			switch arch {
			case "high_arousal_unsourced_bait":
				c.ValidWatchRatio = 0.99
				c.ArticleOpenRate = 0.35
				c.SourceInspectRate = 0.25
				c.SaveRate = 0.12
				c.FollowedTopicMatches = 1
				c.Signal = &ranker.ScopedSignalFeatures{
					SchemaVersion:             ranker.SignalSchemaVersionV1,
					Availability:              ranker.AvailabilityAvailable,
					CoverageRatio:             0.9,
					FreshnessSec:              60,
					EvidenceSupportTier:       ranker.EvidenceMixed,
					ConsistencyTier:           ranker.ConsistencyInconsistent,
					TechnicalLegibilityTier:   ranker.LegibilityClear,
					AttentionPacingScore:      0.93,
					BaitSwitchRisk:            0.84,
					ProvenanceStatus:          ranker.ProvenanceMissing,
					IndependentSourceClusters: 0,
				}
				isBait[cid] = true

			case "regional_breaking_low_polish":
				c.Signal = &ranker.ScopedSignalFeatures{
					SchemaVersion:             ranker.SignalSchemaVersionV1,
					Availability:              ranker.AvailabilityAvailable,
					CoverageRatio:             1.0,
					FreshnessSec:              60,
					EvidenceSupportTier:       ranker.EvidenceSupported,
					ConsistencyTier:           ranker.ConsistencyConsistent,
					TechnicalLegibilityTier:   ranker.LegibilityDegraded,
					AttentionPacingScore:      0.77,
					BaitSwitchRisk:            0.07,
					ProvenanceStatus:          ranker.ProvenanceMissing,
					IndependentSourceClusters: 3,
				}
				isLowPolish[cid] = true

			default:
				c.Signal = &ranker.ScopedSignalFeatures{
					SchemaVersion:             ranker.SignalSchemaVersionV1,
					Availability:              ranker.AvailabilityAvailable,
					CoverageRatio:             1.0,
					FreshnessSec:              60,
					EvidenceSupportTier:       ranker.EvidenceSupported,
					ConsistencyTier:           ranker.ConsistencyConsistent,
					TechnicalLegibilityTier:   ranker.LegibilityClear,
					AttentionPacingScore:      0.26,
					BaitSwitchRisk:            0.04,
					ProvenanceStatus:          ranker.ProvenanceVerified,
					IndependentSourceClusters: 3,
				}
				isStudioPolish[cid] = true
			}
			pool = append(pool, c)
		}

		sigRanked, err := rankerSignal.Rank(ranker.RankRequest{
			Candidates: pool,
			PageSize:   18,
			Now:        testNow,
		})
		if err != nil {
			t.Fatal(err)
		}
		curStreak := 0
		prevAuthor := ""
		for i, it := range sigRanked.Items {
			if i < 5 && isBait[it.CandidateID] {
				top5BaitSignal++
			}
			if i < 10 {
				a := authorByID[it.CandidateID]
				if a == prevAuthor {
					curStreak++
				} else {
					prevAuthor = a
					curStreak = 1
				}
				if curStreak > maxAuthorStreak {
					maxAuthorStreak = curStreak
				}
			}
			if isLowPolish[it.CandidateID] {
				lowPolishRankSum += float64(it.Rank)
				lowPolishCount++
			} else if isStudioPolish[it.CandidateID] {
				studioPolishRankSum += float64(it.Rank)
				studioPolishCount++
			}
		}

		inter, err := Interleave(InterleaveRequest{
			ExperimentID: "exp-newsroom-go",
			Seed:         int64(20261117 + s),
			RequestKey:   fmt.Sprintf("session-%d", s),
			Candidates:   pool,
			RankerA:      rankerNaive,
			RankerB:      rankerSignal,
			PageSize:     10,
			Now:          testNow,
		})
		if err != nil {
			t.Fatal(err)
		}

		events := make([]OutcomeEvent, 0, 10)
		for _, it := range inter.Items {
			if isBait[it.CandidateID] {
				events = append(events, OutcomeEvent{
					CandidateID:          it.CandidateID,
					RapidSkip:            true,
					CorrectionImpression: true,
				})
			} else {
				events = append(events, OutcomeEvent{
					CandidateID:          it.CandidateID,
					CompletedReadOrWatch: true,
					SourceInspected:      true,
					PositiveUsefulness:   true,
				})
			}
		}
		rep, err := ScoreOutcomes(inter.Attribution, events)
		if err != nil {
			t.Fatal(err)
		}
		reports[s] = rep
	}

	if top5BaitSignal != 0 {
		t.Fatalf("expected 0 Top-5 unsourced attention-bait exposures, got %d", top5BaitSignal)
	}
	if maxAuthorStreak != 1 {
		t.Fatalf("expected max consecutive same-author streak == 1, got %d", maxAuthorStreak)
	}

	fairnessGap := math.Abs((lowPolishRankSum / float64(lowPolishCount)) - (studioPolishRankSum / float64(studioPolishCount)))
	if fairnessGap > 0.75 {
		t.Fatalf("low-polish journalism fairness gap %.4f exceeded 0.75 rank threshold", fairnessGap)
	}

	study := AggregateStudy(reports)
	if !study.PassedUsefulnessGate || study.SignTestPValue >= 0.01 {
		t.Fatalf("expected interleaving win with p < 0.01, got %+v", study)
	}
}
