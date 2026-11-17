package ranker

import (
	"slices"
	"strings"
)

type candidateEval struct {
	idx               int
	effScore          float64
	penalty           float64
	authorBlocked     bool
	consecutiveAuthor bool
	topicBlocked      bool
	incidentBlocked   bool
}

func (r *Ranker) selectDiversePage(
	pool []scoredCandidate,
	pageSize int,
	numAuthors int,
	numTopics int,
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

	maxAuthorWin := r.cfg.MaxSameAuthorPerWindow
	maxTopicWin := r.cfg.MaxSameTopicPerWindow

	for len(selected) < target {
		bestStrictIdx := -1
		bestStrictScore := 0.0
		bestWindowIdx := -1
		bestWindowScore := 0.0
		bestAuthorOnlyIdx := -1
		bestAuthorOnlyScore := 0.0
		bestNoAuthorStreakIdx := -1
		bestNoAuthorStreakScore := 0.0
		bestFallbackIdx := -1
		bestFallbackScore := 0.0
		evals = evals[:0]

		lastAuthorIdx := -1
		if len(recentAuthorIDs) > 0 {
			lastAuthorIdx = recentAuthorIDs[len(recentAuthorIDs)-1]
		}
		_ = numTopics

		for i := range pool {
			if used[i] {
				continue
			}
			cand := &pool[i]
			if bestStrictIdx != -1 && cand.rawScore < bestStrictScore-1e-12 {
				break
			}

			var penalty float64
			if cand.authorIdx >= 0 && cand.authorIdx < len(authorPageCounts) {
				if repeats := authorPageCounts[cand.authorIdx]; repeats > 0 {
					penalty += r.cfg.MMRAuthorPenalty * float64(repeats)
				}
			}
			maxJaccard, tBlocked, incBlocked := evaluateTopicWindow(cand.topicIDs, recentTopicIDs, maxTopicWin)
			if maxJaccard > 0 {
				penalty += r.cfg.MMRTopicPenalty * maxJaccard
			}
			effScore := cand.rawScore - penalty

			consecAuthor := lastAuthorIdx >= 0 && cand.authorIdx == lastAuthorIdx
			aBlocked := consecAuthor || isAuthorInWindow(cand.authorIdx, recentAuthorIDs, maxAuthorWin)

			evals = append(evals, candidateEval{
				idx:               i,
				effScore:          effScore,
				penalty:           penalty,
				authorBlocked:     aBlocked,
				consecutiveAuthor: consecAuthor,
				topicBlocked:      tBlocked,
				incidentBlocked:   incBlocked,
			})

			if bestFallbackIdx == -1 ||
				compareCandidates(effScore, bestFallbackScore, cand.publishedUnixNano, pool[bestFallbackIdx].publishedUnixNano, cand.candidateID, pool[bestFallbackIdx].candidateID) < 0 {
				bestFallbackIdx = i
				bestFallbackScore = effScore
			}
			if !consecAuthor {
				if bestNoAuthorStreakIdx == -1 ||
					compareCandidates(effScore, bestNoAuthorStreakScore, cand.publishedUnixNano, pool[bestNoAuthorStreakIdx].publishedUnixNano, cand.candidateID, pool[bestNoAuthorStreakIdx].candidateID) < 0 {
					bestNoAuthorStreakIdx = i
					bestNoAuthorStreakScore = effScore
				}
			}
			if !aBlocked {
				if bestAuthorOnlyIdx == -1 ||
					compareCandidates(effScore, bestAuthorOnlyScore, cand.publishedUnixNano, pool[bestAuthorOnlyIdx].publishedUnixNano, cand.candidateID, pool[bestAuthorOnlyIdx].candidateID) < 0 {
					bestAuthorOnlyIdx = i
					bestAuthorOnlyScore = effScore
				}
				if !tBlocked {
					if bestWindowIdx == -1 ||
						compareCandidates(effScore, bestWindowScore, cand.publishedUnixNano, pool[bestWindowIdx].publishedUnixNano, cand.candidateID, pool[bestWindowIdx].candidateID) < 0 {
						bestWindowIdx = i
						bestWindowScore = effScore
					}
					if !incBlocked {
						if bestStrictIdx == -1 ||
							compareCandidates(effScore, bestStrictScore, cand.publishedUnixNano, pool[bestStrictIdx].publishedUnixNano, cand.candidateID, pool[bestStrictIdx].candidateID) < 0 {
							bestStrictIdx = i
							bestStrictScore = effScore
						}
					}
				}
			}
		}

		var chosenIdx int
		var chosenScore float64
		switch {
		case bestStrictIdx != -1:
			chosenIdx = bestStrictIdx
			chosenScore = bestStrictScore
		case bestWindowIdx != -1:
			chosenIdx = bestWindowIdx
			chosenScore = bestWindowScore
		case bestAuthorOnlyIdx != -1:
			chosenIdx = bestAuthorOnlyIdx
			chosenScore = bestAuthorOnlyScore
		case bestNoAuthorStreakIdx != -1:
			chosenIdx = bestNoAuthorStreakIdx
			chosenScore = bestNoAuthorStreakScore
		default:
			chosenIdx = bestFallbackIdx
			chosenScore = bestFallbackScore
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
				if ev.authorBlocked || ev.consecutiveAuthor {
					authorDeferrals++
				} else if ev.topicBlocked || ev.incidentBlocked {
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

func isAuthorInWindow(authorIdx int, recentAuthorIDs []int, maxAllowed int) bool {
	if authorIdx < 0 {
		return false
	}
	count := 0
	for _, a := range recentAuthorIDs {
		if a == authorIdx {
			count++
			if count >= maxAllowed {
				return true
			}
		}
	}
	return false
}

func evaluateTopicWindow(topicIDs []int, recentTopicIDs [][]int, maxAllowed int) (maxJaccard float64, tBlocked bool, incBlocked bool) {
	nCand := len(topicIDs)
	nWin := len(recentTopicIDs)
	if nCand == 0 || nWin == 0 {
		return 0.0, false, false
	}

	var stackCounts [8]int
	var counts []int
	if nCand <= len(stackCounts) {
		counts = stackCounts[:nCand]
	} else {
		counts = make([]int, nCand)
	}

	lastWinIdx := nWin - 1
	for wIdx, prev := range recentTopicIDs {
		nPrev := len(prev)
		if nPrev == 0 {
			continue
		}
		i, j, inter := 0, 0, 0
		for i < nCand && j < nPrev {
			ti, pj := topicIDs[i], prev[j]
			if ti == pj {
				inter++
				counts[i]++
				if counts[i] >= maxAllowed {
					tBlocked = true
				}
				i++
				j++
			} else if ti < pj {
				i++
			} else {
				j++
			}
		}
		if inter > 0 {
			denom := nCand + nPrev - inter
			if jacc := float64(inter) / float64(denom); jacc > maxJaccard {
				maxJaccard = jacc
			}
			if wIdx == lastWinIdx && (inter >= 2 || inter*2 >= denom) {
				incBlocked = true
			}
		}
	}
	return maxJaccard, tBlocked, incBlocked
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
	switch len(out) {
	case 0:
		return nil
	case 1:
		return out
	case 2:
		if out[0] == out[1] {
			return out[:1]
		}
		if out[0] > out[1] {
			out[0], out[1] = out[1], out[0]
		}
		return out
	default:
		slices.Sort(out)
		return slices.Compact(out)
	}
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
