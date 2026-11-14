package ranker

import (
	"math"
	"slices"
)

type weightedReason struct {
	code     ReasonCode
	text     string
	weight   float64
	priority int
}

func buildWhyThis(sc *scoredCandidate) WhyThis {
	candidates := make([]weightedReason, 0, 6)

	if sc.followedAuthor {
		candidates = append(candidates, weightedReason{
			code:     ReasonFollowedAuthor,
			text:     "From a writer or publication you follow",
			weight:   sc.prefAuthor,
			priority: 1,
		})
	}
	if sc.followedTopicMatches > 0 {
		candidates = append(candidates, weightedReason{
			code:     ReasonTopicMatch,
			text:     "Matches topics in your preferences",
			weight:   sc.prefTopic,
			priority: 2,
		})
	}
	if sc.sourceBacked {
		candidates = append(candidates, weightedReason{
			code:     ReasonSourceBacked,
			text:     "Includes linked sources you can inspect",
			weight:   sc.signalPositive,
			priority: 3,
		})
	}
	if sc.engagementEligible {
		candidates = append(candidates, weightedReason{
			code:     ReasonReaderEngagement,
			text:     "Readers are opening the full article and sources",
			weight:   sc.engagement,
			priority: 4,
		})
	}
	if sc.languageMatch {
		candidates = append(candidates, weightedReason{
			code:     ReasonLanguageMatch,
			text:     "In your preferred reading language",
			weight:   sc.prefLang,
			priority: 5,
		})
	}
	if sc.ageHours <= 12.0 {
		candidates = append(candidates, weightedReason{
			code:     ReasonRecentUpdate,
			text:     "Published recently",
			weight:   sc.freshness,
			priority: 6,
		})
	}
	if sc.isColdStart {
		candidates = append(candidates, weightedReason{
			code:     ReasonFreshDiscovery,
			text:     "New story surfaced for discovery",
			weight:   sc.explorationBonus,
			priority: 7,
		})
	}

	if len(candidates) == 0 {
		code := ReasonGeneralDiscovery
		text := "Selected from eligible recent reporting"
		return WhyThis{
			ReasonCodes: []ReasonCode{code},
			Reasons:     []string{text},
			Summary:     text + ".",
		}
	}

	slices.SortFunc(candidates, func(a, b weightedReason) int {
		const eps = 1e-12
		if diff := a.weight - b.weight; math.Abs(diff) > eps {
			if diff > 0 {
				return -1
			}
			return 1
		}
		return a.priority - b.priority
	})

	limit := min(2, len(candidates))
	codes := make([]ReasonCode, limit)
	reasons := make([]string, limit)
	for i := 0; i < limit; i++ {
		codes[i] = candidates[i].code
		reasons[i] = candidates[i].text
	}

	summary := reasons[0] + "."
	if limit == 2 {
		summary = reasons[0] + " · " + reasons[1] + "."
	}

	return WhyThis{
		ReasonCodes: codes,
		Reasons:     reasons,
		Summary:     summary,
	}
}
