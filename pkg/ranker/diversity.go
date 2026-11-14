package ranker

import (
	"slices"
	"strings"
)

type candidateEval struct {
	idx           int
	effScore      float64
	penalty       float64
	authorBlocked bool
	topicBlocked  bool
}

func (r *Ranker) selectDiversePage(
	pool []scoredCandidate,
	pageSize int,
	numAuthors int,
	seedAuthors []string,
	seedAuthorIDs []int,
	seedTopics [][]string,
	seedTopicIDs [][]int,
) (selected []scoredCandidate, authorDeferrals int, topicDeferrals int, tailAuthors []string, tailTopics [][]string) {
	target := min(pageSize, len(pool))
	if target == 0 {
		return []scoredCandidate{}, 0, 0, nil, nil
	}

	windowSpan := r.cfg.DiversityWindowSize - 1
	recentAuthors := trimTailStrings(seedAuthors, windowSpan)
	recentAuthorIDs := trimTailInts(seedAuthorIDs, windowSpan)
	recentTopics := trimTailTopicSlices(seedTopics, windowSpan)
	recentTopicIDs := trimTailIntSlices(seedTopicIDs, windowSpan)

	authorPageCounts := make([]int, numAuthors)
	for _, aIdx := range recentAuthorIDs {
		if aIdx >= 0 && aIdx < numAuthors {
			authorPageCounts[aIdx]++
		}
	}

	used := make([]bool, len(pool))
	selected = make([]scoredCandidate, 0, target)
	evals := make([]candidateEval, 0, len(pool))

	for len(selected) < target {
		bestStrictIdx := -1
		bestStrictScore := 0.0
		bestAuthorOnlyIdx := -1
		bestAuthorOnlyScore := 0.0
		bestFallbackIdx := -1
		bestFallbackScore := 0.0
		evals = evals[:0]

		for i := range pool {
			if used[i] {
				continue
			}
			cand := &pool[i]
			if bestStrictIdx != -1 && cand.rawScore < bestStrictScore-1e-12 {
				break
			}
			penalty := r.computeMMRPenalty(cand, authorPageCounts, recentTopicIDs)
			effScore := cand.rawScore - penalty
			aBlocked := r.isAuthorWindowBlocked(cand.authorIdx, recentAuthorIDs)
			tBlocked := r.isTopicWindowBlocked(cand.topicIDs, recentTopicIDs)

			evals = append(evals, candidateEval{
				idx:           i,
				effScore:      effScore,
				penalty:       penalty,
				authorBlocked: aBlocked,
				topicBlocked:  tBlocked,
			})

			if bestFallbackIdx == -1 ||
				compareCandidates(effScore, bestFallbackScore, cand.publishedUnixNano, pool[bestFallbackIdx].publishedUnixNano, cand.candidateID, pool[bestFallbackIdx].candidateID) < 0 {
				bestFallbackIdx = i
				bestFallbackScore = effScore
			}
			if !aBlocked {
				if bestAuthorOnlyIdx == -1 ||
					compareCandidates(effScore, bestAuthorOnlyScore, cand.publishedUnixNano, pool[bestAuthorOnlyIdx].publishedUnixNano, cand.candidateID, pool[bestAuthorOnlyIdx].candidateID) < 0 {
					bestAuthorOnlyIdx = i
					bestAuthorOnlyScore = effScore
				}
				if !tBlocked {
					if bestStrictIdx == -1 ||
						compareCandidates(effScore, bestStrictScore, cand.publishedUnixNano, pool[bestStrictIdx].publishedUnixNano, cand.candidateID, pool[bestStrictIdx].candidateID) < 0 {
						bestStrictIdx = i
						bestStrictScore = effScore
					}
				}
			}
		}

		chosenIdx := bestStrictIdx
		chosenScore := bestStrictScore
		if chosenIdx == -1 {
			if bestAuthorOnlyIdx != -1 {
				chosenIdx = bestAuthorOnlyIdx
				chosenScore = bestAuthorOnlyScore
			} else {
				chosenIdx = bestFallbackIdx
				chosenScore = bestFallbackScore
			}
		}

		chosen := &pool[chosenIdx]
		var chosenPenalty float64
		for i := range evals {
			ev := &evals[i]
			if ev.idx == chosenIdx {
				chosenPenalty = ev.penalty
				continue
			}
			other := &pool[ev.idx]
			if compareCandidates(ev.effScore, chosenScore, other.publishedUnixNano, chosen.publishedUnixNano, other.candidateID, chosen.candidateID) < 0 {
				if ev.authorBlocked {
					authorDeferrals++
				} else if ev.topicBlocked {
					topicDeferrals++
				}
			}
		}

		used[chosenIdx] = true
		picked := *chosen
		picked.diversityPenalty = chosenPenalty
		picked.finalScore = chosenScore
		selected = append(selected, picked)

		if picked.authorIdx >= 0 && picked.authorIdx < len(authorPageCounts) {
			authorPageCounts[picked.authorIdx]++
		}
		recentAuthors = appendBoundedString(recentAuthors, picked.authorID, windowSpan)
		recentAuthorIDs = appendBoundedInt(recentAuthorIDs, picked.authorIdx, windowSpan)
		recentTopics = appendBoundedTopicSlice(recentTopics, picked.topics, windowSpan)
		recentTopicIDs = appendBoundedIntSlice(recentTopicIDs, picked.topicIDs, windowSpan)
	}

	return selected, authorDeferrals, topicDeferrals, recentAuthors, recentTopics
}

func (r *Ranker) computeMMRPenalty(
	cand *scoredCandidate,
	authorCounts []int,
	recentTopicIDs [][]int,
) float64 {
	var penalty float64
	if cand.authorIdx >= 0 && cand.authorIdx < len(authorCounts) {
		if repeats := authorCounts[cand.authorIdx]; repeats > 0 {
			penalty += r.cfg.MMRAuthorPenalty * float64(repeats)
		}
	}
	if len(cand.topicIDs) > 0 && len(recentTopicIDs) > 0 {
		maxJaccard := 0.0
		for _, prev := range recentTopicIDs {
			if j := sortedIntsJaccard(cand.topicIDs, prev); j > maxJaccard {
				maxJaccard = j
			}
		}
		penalty += r.cfg.MMRTopicPenalty * maxJaccard
	}
	return penalty
}

func (r *Ranker) isAuthorWindowBlocked(authorIdx int, recentAuthorIDs []int) bool {
	if authorIdx < 0 {
		return false
	}
	count := 0
	for _, a := range recentAuthorIDs {
		if a == authorIdx {
			count++
		}
	}
	return count >= r.cfg.MaxSameAuthorPerWindow
}

func (r *Ranker) isTopicWindowBlocked(topicIDs []int, recentTopicIDs [][]int) bool {
	if len(topicIDs) == 0 || len(recentTopicIDs) == 0 {
		return false
	}
	for _, tid := range topicIDs {
		count := 0
		for _, prevSlice := range recentTopicIDs {
			if slices.Contains(prevSlice, tid) {
				count++
			}
		}
		if count >= r.cfg.MaxSameTopicPerWindow {
			return true
		}
	}
	return false
}

func (r *Ranker) evaluateSessionPacing(
	selected []scoredCandidate,
	priorItemsSeen int,
	priorHighIntensityStreak int,
) (advisory SessionPacingAdvisory, endingStreak int) {
	triggers := make([]PacingTriggerCode, 0, 2)
	streak := priorHighIntensityStreak
	maxStreak := streak
	d01Rank := 0
	d03Rank := 0

	for i := range selected {
		rank := i + 1
		if selected[i].highIntensityD01 {
			streak++
			if streak > maxStreak {
				maxStreak = streak
			}
			if d01Rank == 0 && streak >= r.cfg.MaxHighIntensityStreak {
				d01Rank = rank
			}
		} else {
			streak = 0
		}

		if d03Rank == 0 && priorItemsSeen+rank >= r.cfg.SessionDepthAdvisoryThreshold {
			d03Rank = rank
		}
	}

	totalSessionDepth := priorItemsSeen + len(selected)
	if len(selected) == 0 {
		return SessionPacingAdvisory{
			Triggered:           false,
			TriggerCodes:        triggers,
			AdvisoryKind:        AdvisoryNone,
			InsertAfterRank:     0,
			Message:             "",
			HighIntensityStreak: maxStreak,
			SessionDepthItems:   totalSessionDepth,
		}, streak
	}

	if d01Rank > 0 {
		triggers = append(triggers, TriggerHighIntensityStreak)
	}
	if d03Rank > 0 {
		triggers = append(triggers, TriggerSessionDepthBoundary)
	}

	if len(triggers) == 0 {
		return SessionPacingAdvisory{
			Triggered:           false,
			TriggerCodes:        triggers,
			AdvisoryKind:        AdvisoryNone,
			InsertAfterRank:     0,
			Message:             "",
			HighIntensityStreak: maxStreak,
			SessionDepthItems:   totalSessionDepth,
		}, streak
	}

	insertRank := d01Rank
	if insertRank == 0 || (d03Rank > 0 && d03Rank < insertRank) {
		insertRank = d03Rank
	}

	kind := AdvisorySlowerContextBreak
	msg := "Several rapid-fire updates in a row — consider opening a full story for slower, source-linked context."
	if d01Rank > 0 && d03Rank > 0 {
		kind = AdvisoryGentleStoppingPoint
		msg = "You have caught up on several fast-paced updates. Take a breather or open a full article for deeper context."
	} else if d03Rank > 0 {
		kind = AdvisoryGentleStoppingPoint
		msg = "You are caught up on recent updates — a good stopping point whenever you are ready."
	}

	return SessionPacingAdvisory{
		Triggered:           true,
		TriggerCodes:        triggers,
		AdvisoryKind:        kind,
		InsertAfterRank:     insertRank,
		Message:             msg,
		HighIntensityStreak: maxStreak,
		SessionDepthItems:   totalSessionDepth,
	}, streak
}

func buildDiversityAudit(
	selected []scoredCandidate,
	eligibleTotal int,
	droppedIneligible int,
	authorDeferrals int,
	topicDeferrals int,
	coldStartItems int,
	anomalyDiscounted int,
) DiversityAudit {
	authors := make(map[string]int, len(selected))
	topics := make(map[string]struct{}, len(selected)*2)
	maxAuthorCount := 0

	for i := range selected {
		sc := &selected[i]
		if sc.authorID != "" {
			authors[sc.authorID]++
			if authors[sc.authorID] > maxAuthorCount {
				maxAuthorCount = authors[sc.authorID]
			}
		}
		for _, t := range sc.topics {
			topics[t] = struct{}{}
		}
	}

	maxShare := 0.0
	if len(selected) > 0 {
		maxShare = roundScore(float64(maxAuthorCount) / float64(len(selected)))
	}

	return DiversityAudit{
		EligibleCandidates:     eligibleTotal,
		DroppedIneligible:      droppedIneligible,
		ReturnedCount:          len(selected),
		UniqueAuthors:          len(authors),
		UniqueTopics:           len(topics),
		MaxAuthorShare:         maxShare,
		AuthorCapDeferrals:     authorDeferrals,
		TopicCapDeferrals:      topicDeferrals,
		ColdStartItems:         coldStartItems,
		AnomalyDiscountedItems: anomalyDiscounted,
	}
}

func normalizeTopics(tags []string) []string {
	if len(tags) == 0 {
		return nil
	}
	out := make([]string, 0, len(tags))
	for _, raw := range tags {
		cleaned := strings.ToLower(strings.TrimSpace(raw))
		if cleaned != "" {
			out = append(out, cleaned)
		}
	}
	slices.Sort(out)
	return slices.Compact(out)
}

func sortedIntsJaccard(a, b []int) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0.0
	}
	i, j, inter := 0, 0, 0
	for i < len(a) && j < len(b) {
		if a[i] == b[j] {
			inter++
			i++
			j++
		} else if a[i] < b[j] {
			i++
		} else {
			j++
		}
	}
	if inter == 0 {
		return 0.0
	}
	union := len(a) + len(b) - inter
	return float64(inter) / float64(union)
}

func trimTailStrings(src []string, limit int) []string {
	if limit <= 0 || len(src) == 0 {
		return nil
	}
	start := max(0, len(src)-limit)
	out := make([]string, len(src)-start)
	copy(out, src[start:])
	return out
}

func trimTailInts(src []int, limit int) []int {
	if limit <= 0 || len(src) == 0 {
		return nil
	}
	start := max(0, len(src)-limit)
	out := make([]int, len(src)-start)
	copy(out, src[start:])
	return out
}

func trimTailTopicSlices(src [][]string, limit int) [][]string {
	if limit <= 0 || len(src) == 0 {
		return nil
	}
	start := max(0, len(src)-limit)
	out := make([][]string, len(src)-start)
	for idx, s := range src[start:] {
		copied := make([]string, len(s))
		copy(copied, s)
		out[idx] = copied
	}
	return out
}

func trimTailIntSlices(src [][]int, limit int) [][]int {
	if limit <= 0 || len(src) == 0 {
		return nil
	}
	start := max(0, len(src)-limit)
	out := make([][]int, len(src)-start)
	for idx, s := range src[start:] {
		copied := make([]int, len(s))
		copy(copied, s)
		out[idx] = copied
	}
	return out
}

func appendBoundedString(list []string, item string, limit int) []string {
	list = append(list, item)
	if len(list) > limit {
		list = list[len(list)-limit:]
	}
	return list
}

func appendBoundedInt(list []int, item int, limit int) []int {
	list = append(list, item)
	if len(list) > limit {
		list = list[len(list)-limit:]
	}
	return list
}

func appendBoundedTopicSlice(list [][]string, item []string, limit int) [][]string {
	list = append(list, item)
	if len(list) > limit {
		list = list[len(list)-limit:]
	}
	return list
}

func appendBoundedIntSlice(list [][]int, item []int, limit int) [][]int {
	list = append(list, item)
	if len(list) > limit {
		list = list[len(list)-limit:]
	}
	return list
}
