package ranker

import "math"

type weightedReason struct {
	code     ReasonCode
	text     string
	weight   float64
	priority int
}

func betterReason(a, b weightedReason) bool {
	const eps = 1e-12
	if diff := a.weight - b.weight; math.Abs(diff) > eps {
		return diff > 0
	}
	return a.priority < b.priority
}

func buildWhyThis(sc *scoredCandidate) WhyThis {
	var buf [8]weightedReason
	n := 0

	if sc.followedAuthor {
		buf[n] = weightedReason{
			code:     ReasonFollowedAuthor,
			text:     "From a writer or publication you follow",
			weight:   sc.prefAuthor,
			priority: 1,
		}
		n++
	}
	if sc.followedTopicMatches > 0 {
		topicText := "Matches topics in your preferences"
		if sc.followedTopicMatches >= 2 {
			topicText = "Matches multiple topics in your preferences"
		}
		buf[n] = weightedReason{
			code:     ReasonTopicMatch,
			text:     topicText,
			weight:   sc.prefTopic,
			priority: 2,
		}
		n++
	}
	if sc.regionMatch {
		buf[n] = weightedReason{
			code:     ReasonRegionMatch,
			text:     "Relevant to your selected region",
			weight:   sc.prefRegion,
			priority: 3,
		}
		n++
	}
	if sc.sourceBacked {
		sourceText := "Includes linked sources you can inspect"
		if sc.evidenceSupported && sc.corroborationClusters >= 2 {
			sourceText = "Corroborated across multiple independent sources"
		} else if sc.corroborationClusters >= 2 {
			sourceText = "Backed by multiple independent source clusters"
		}
		buf[n] = weightedReason{
			code:     ReasonSourceBacked,
			text:     sourceText,
			weight:   sc.signalPositive,
			priority: 4,
		}
		n++
	}
	if sc.engagementEligible {
		buf[n] = weightedReason{
			code:     ReasonReaderEngagement,
			text:     "Readers are opening the full article and sources",
			weight:   sc.engagement,
			priority: 5,
		}
		n++
	}
	if sc.languageMatch {
		buf[n] = weightedReason{
			code:     ReasonLanguageMatch,
			text:     "In your preferred reading language",
			weight:   sc.prefLang,
			priority: 6,
		}
		n++
	}
	if sc.ageHours <= 12.0 {
		recencyText := "Published recently"
		if sc.ageHours <= 3.0 {
			recencyText = "Timely developing update"
		}
		buf[n] = weightedReason{
			code:     ReasonRecentUpdate,
			text:     recencyText,
			weight:   sc.freshness,
			priority: 7,
		}
		n++
	}
	if sc.isColdStart {
		buf[n] = weightedReason{
			code:     ReasonFreshDiscovery,
			text:     "New story surfaced for discovery",
			weight:   sc.explorationBonus,
			priority: 8,
		}
		n++
	}

	if n == 0 {
		code := ReasonGeneralDiscovery
		text := "Selected from eligible recent reporting"
		return WhyThis{
			ReasonCodes: []ReasonCode{code},
			Reasons:     []string{text},
			Summary:     text + ".",
		}
	}

	bestIdx := 0
	for i := 1; i < n; i++ {
		if betterReason(buf[i], buf[bestIdx]) {
			bestIdx = i
		}
	}

	if n == 1 {
		r0 := buf[bestIdx]
		return WhyThis{
			ReasonCodes: []ReasonCode{r0.code},
			Reasons:     []string{r0.text},
			Summary:     r0.text + ".",
		}
	}

	secondIdx := -1
	for i := 0; i < n; i++ {
		if i == bestIdx {
			continue
		}
		if secondIdx == -1 || betterReason(buf[i], buf[secondIdx]) {
			secondIdx = i
		}
	}

	r0, r1 := buf[bestIdx], buf[secondIdx]
	return WhyThis{
		ReasonCodes: []ReasonCode{r0.code, r1.code},
		Reasons:     []string{r0.text, r1.text},
		Summary:     r0.text + " · " + r1.text + ".",
	}
}
