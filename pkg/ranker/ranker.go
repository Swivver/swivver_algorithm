package ranker

import (
	"encoding/base64"
	"encoding/json"
	"math"
	"slices"
	"strings"
	"time"
)

const (
	cursorFormatVersion = 1
	maxCursorBytes      = 65536
)

// Ranker scores, diversifies, and explains eligible candidate batches.
// Ranker holds only immutable configuration and is safe for concurrent use.
type Ranker struct {
	cfg Config
}

// New constructs a Ranker with normalized configuration.
func New(cfg Config) *Ranker {
	return &Ranker{cfg: cfg.normalized()}
}

// NewDefault constructs a Ranker using DefaultConfig.
func NewDefault() *Ranker {
	return New(DefaultConfig())
}

type scoredCandidate struct {
	candidateID          string
	authorID             string
	authorIdx            int
	topics               []string
	topicIDs             []int
	publishedUnixNano    int64
	followedAuthor       bool
	followedTopicMatches int
	languageMatch        bool
	anomalyFlagged       bool
	engagementEligible   bool
	ageHours             float64
	prefAuthor           float64
	prefTopic            float64
	prefLang             float64
	preference           float64
	freshness            float64
	engagement           float64
	signalDelta          float64
	signalPositive       float64
	explorationBonus     float64
	rawScore             float64
	diversityPenalty     float64
	finalScore           float64
	sourceBacked         bool
	highIntensityD01     bool
	isColdStart          bool
}

type cursorPayload struct {
	Version             int        `json:"v"`
	Seen                []string   `json:"seen"`
	RecentAuthors       []string   `json:"recent_authors,omitempty"`
	RecentTopics        [][]string `json:"recent_topics,omitempty"`
	SessionItemsSeen    int        `json:"items_seen"`
	HighIntensityStreak int        `json:"hi_streak"`
}

// Rank filters, scores, diversifies, explains, and paginates the candidate batch.
func (r *Ranker) Rank(req RankRequest) (RankedFeedResult, error) {
	var cur cursorPayload
	seenFromCursor := make(map[string]struct{})
	hasCursor := strings.TrimSpace(req.Cursor) != ""
	if hasCursor {
		decoded, err := decodeCursor(req.Cursor)
		if err != nil {
			return RankedFeedResult{}, err
		}
		cur = decoded
		seenFromCursor = make(map[string]struct{}, len(cur.Seen))
		for _, id := range cur.Seen {
			seenFromCursor[id] = struct{}{}
		}
	}

	now := req.Now
	if now.IsZero() {
		now = time.Now().UTC()
	}

	pageSize := req.PageSize
	if pageSize <= 0 {
		pageSize = r.cfg.DefaultPageSize
	}
	if pageSize > r.cfg.MaxPageSize {
		pageSize = r.cfg.MaxPageSize
	}

	authorIntern := make(map[string]int, len(req.Candidates)+len(cur.RecentAuthors))
	topicIntern := make(map[string]int, len(req.Candidates)*2)
	internAuthor := func(a string) int {
		if a == "" {
			return -1
		}
		if idx, ok := authorIntern[a]; ok {
			return idx
		}
		idx := len(authorIntern)
		authorIntern[a] = idx
		return idx
	}
	internTopics := func(tags []string) []int {
		if len(tags) == 0 {
			return nil
		}
		ids := make([]int, len(tags))
		for i, t := range tags {
			idx, ok := topicIntern[t]
			if !ok {
				idx = len(topicIntern)
				topicIntern[t] = idx
			}
			ids[i] = idx
		}
		slices.Sort(ids)
		return ids
	}

	seenInBatch := make(map[string]struct{}, len(req.Candidates))
	scored := make([]scoredCandidate, 0, len(req.Candidates))
	droppedIneligible := 0
	eligibleTotal := 0
	anomalyDiscounted := 0

	for i := range req.Candidates {
		c := &req.Candidates[i]
		id := strings.TrimSpace(c.CandidateID)
		// Defense-in-depth policy gate: never expose ineligible or blocked items.
		if !c.Eligible || c.PolicyBlocked || id == "" {
			droppedIneligible++
			continue
		}
		if _, dup := seenInBatch[id]; dup {
			continue
		}
		seenInBatch[id] = struct{}{}
		eligibleTotal++
		if c.AnomalyFlagged {
			anomalyDiscounted++
		}
		if _, alreadyReturned := seenFromCursor[id]; alreadyReturned {
			continue
		}
		sc := r.scoreCandidate(c, id, now)
		sc.authorIdx = internAuthor(sc.authorID)
		sc.topicIDs = internTopics(sc.topics)
		scored = append(scored, sc)
	}

	slices.SortFunc(scored, func(a, b scoredCandidate) int {
		return compareCandidates(a.rawScore, b.rawScore, a.publishedUnixNano, b.publishedUnixNano, a.candidateID, b.candidateID)
	})

	seedAuthorIDs := make([]int, len(cur.RecentAuthors))
	for i, a := range cur.RecentAuthors {
		seedAuthorIDs[i] = internAuthor(strings.TrimSpace(a))
	}
	seedTopicIDs := make([][]int, len(cur.RecentTopics))
	for i, ts := range cur.RecentTopics {
		seedTopicIDs[i] = internTopics(normalizeTopics(ts))
	}

	selected, authorDeferrals, topicDeferrals, tailAuthors, tailTopics := r.selectDiversePage(
		scored,
		pageSize,
		len(authorIntern),
		cur.RecentAuthors,
		seedAuthorIDs,
		cur.RecentTopics,
		seedTopicIDs,
	)

	items := make([]RankedItem, len(selected))
	coldStartCount := 0
	for i := range selected {
		sc := &selected[i]
		if sc.isColdStart {
			coldStartCount++
		}
		items[i] = RankedItem{
			CandidateID: sc.candidateID,
			Rank:        i + 1,
			Score:       roundScore(sc.finalScore),
			ScoreComponents: ScoreComponents{
				Preference:       roundScore(sc.preference),
				Freshness:        roundScore(sc.freshness),
				Engagement:       roundScore(sc.engagement),
				SignalDelta:      roundScore(sc.signalDelta),
				ExplorationBonus: roundScore(sc.explorationBonus),
				DiversityPenalty: roundScore(sc.diversityPenalty),
			},
			WhyThis: buildWhyThis(sc),
		}
	}

	priorSeenCount := max(0, req.Session.SessionItemsSeen)
	priorStreak := max(0, req.Session.PriorHighIntensityStreak)
	if hasCursor {
		if cur.SessionItemsSeen > priorSeenCount {
			priorSeenCount = cur.SessionItemsSeen
		}
		priorStreak = cur.HighIntensityStreak
	}

	advisory, endStreak := r.evaluateSessionPacing(selected, priorSeenCount, priorStreak)

	nextCursor := ""
	if len(selected) > 0 && len(selected) < len(scored) {
		nextSeen := make([]string, 0, len(cur.Seen)+len(selected))
		nextSeen = append(nextSeen, cur.Seen...)
		for i := range selected {
			nextSeen = append(nextSeen, selected[i].candidateID)
		}
		nextCursor = encodeCursor(cursorPayload{
			Version:             cursorFormatVersion,
			Seen:                nextSeen,
			RecentAuthors:       tailAuthors,
			RecentTopics:        tailTopics,
			SessionItemsSeen:    priorSeenCount + len(selected),
			HighIntensityStreak: endStreak,
		})
	}

	audit := buildDiversityAudit(
		selected,
		eligibleTotal,
		droppedIneligible,
		authorDeferrals,
		topicDeferrals,
		coldStartCount,
		anomalyDiscounted,
	)

	return RankedFeedResult{
		SchemaVersion:         ResultSchemaVersionV1,
		ModelVersion:          r.cfg.ModelVersion,
		Items:                 items,
		SessionPacingAdvisory: advisory,
		NextCursor:            nextCursor,
		DiversityAudit:        audit,
	}, nil
}

func (r *Ranker) scoreCandidate(c *CandidateFeatureVector, trimmedID string, now time.Time) scoredCandidate {
	var prefAuthor, prefTopic, prefLang float64
	if c.FollowedAuthor {
		prefAuthor = r.cfg.FollowedAuthorWeight
	}
	if c.FollowedTopicMatches == 1 {
		prefTopic = 0.65 * r.cfg.TopicMatchWeight
	} else if c.FollowedTopicMatches >= 2 {
		prefTopic = r.cfg.TopicMatchWeight
	}
	if c.LanguageMatch {
		prefLang = r.cfg.LanguageMatchWeight
	}
	preference := prefAuthor + prefTopic + prefLang

	ageHours := 0.0
	var pubNano int64
	if !c.PublishedAt.IsZero() {
		pubNano = c.PublishedAt.UnixNano()
		diff := now.Sub(c.PublishedAt).Hours()
		if diff > 0 && !math.IsNaN(diff) && !math.IsInf(diff, 0) {
			ageHours = diff
		}
	}
	freshness := r.cfg.FreshnessWeight * math.Pow(2.0, -ageHours/r.cfg.FreshnessHalfLifeHrs)

	engagement := 0.0
	engagementEligible := false
	if !c.AnomalyFlagged {
		articleOpen := clampUnit(c.ArticleOpenRate)
		sourceInspect := clampUnit(c.SourceInspectRate)
		save := clampUnit(c.SaveRate)
		watch := clampUnit(c.ValidWatchRatio)
		engagement = r.cfg.EngagementWeight * (0.40*articleOpen + 0.35*sourceInspect + 0.15*save + 0.10*watch)
		engagementEligible = articleOpen >= 0.15 || sourceInspect >= 0.10 || save >= 0.08
	}

	signalDelta, signalPos, sourceBacked, highIntensity := r.scoreSignal(c.Signal)

	isUnassessed := !r.hasAnyAvailableDimension(c.Signal)
	b0 := 0.0
	impressions := max(0, c.ImpressionCount)
	if r.cfg.ColdStartExplorationBonus > 0 &&
		!c.AnomalyFlagged &&
		ageHours <= r.cfg.ColdStartMaxAgeHrs &&
		impressions <= r.cfg.ColdStartMaxImpressions {
		b0 = r.cfg.ColdStartExplorationBonus
	}

	explorationBonus := 0.0
	if signalDelta >= 0 && b0 > signalDelta {
		explorationBonus = b0 - signalDelta
	}

	rawScore := preference + freshness + engagement + signalDelta + explorationBonus

	return scoredCandidate{
		candidateID:          trimmedID,
		authorID:             strings.TrimSpace(c.AuthorID),
		topics:               normalizeTopics(c.TopicTags),
		publishedUnixNano:    pubNano,
		followedAuthor:       c.FollowedAuthor,
		followedTopicMatches: c.FollowedTopicMatches,
		languageMatch:        c.LanguageMatch,
		anomalyFlagged:       c.AnomalyFlagged,
		engagementEligible:   engagementEligible,
		ageHours:             ageHours,
		prefAuthor:           prefAuthor,
		prefTopic:            prefTopic,
		prefLang:             prefLang,
		preference:           preference,
		freshness:            freshness,
		engagement:           engagement,
		signalDelta:          signalDelta,
		signalPositive:       signalPos,
		explorationBonus:     explorationBonus,
		rawScore:             rawScore,
		finalScore:           rawScore,
		sourceBacked:         sourceBacked,
		highIntensityD01:     highIntensity,
		isColdStart:          isUnassessed && explorationBonus > 0,
	}
}

func (r *Ranker) isSignalActive(sig *ScopedSignalFeatures) bool {
	if sig == nil || sig.Availability != AvailabilityAvailable {
		return false
	}
	if math.IsNaN(sig.FreshnessSec) || math.IsInf(sig.FreshnessSec, 0) || sig.FreshnessSec < 0 {
		return false
	}
	return sig.FreshnessSec <= r.cfg.MaxSignalFreshnessSec
}

func (r *Ranker) isDimAvailable(sig *ScopedSignalFeatures, dim string) bool {
	if !r.isSignalActive(sig) {
		return false
	}
	if len(sig.DimensionAvailability) == 0 {
		return true
	}
	status, exists := sig.DimensionAvailability[dim]
	if !exists {
		return true
	}
	return status == AvailabilityAvailable
}

func (r *Ranker) hasAnyAvailableDimension(sig *ScopedSignalFeatures) bool {
	if !r.isSignalActive(sig) {
		return false
	}
	if r.isDimAvailable(sig, DimEvidenceSupport) &&
		sig.EvidenceSupportTier != "" &&
		sig.EvidenceSupportTier != EvidenceUnknown {
		return true
	}
	if r.isDimAvailable(sig, DimConsistency) &&
		sig.ConsistencyTier != "" &&
		sig.ConsistencyTier != ConsistencyUnknown {
		return true
	}
	if r.isDimAvailable(sig, DimTechnicalLegibility) &&
		sig.TechnicalLegibilityTier != "" &&
		sig.TechnicalLegibilityTier != LegibilityUnknown {
		return true
	}
	if r.isDimAvailable(sig, DimProvenance) &&
		sig.ProvenanceStatus != "" &&
		sig.ProvenanceStatus != ProvenanceUnknown {
		return true
	}
	if r.isDimAvailable(sig, DimIndependentSources) && sig.IndependentSourceClusters > 0 {
		return true
	}
	if r.isDimAvailable(sig, DimAttentionPacing) && clampUnit(sig.AttentionPacingScore) > 0 {
		return true
	}
	if r.isDimAvailable(sig, DimBaitSwitch) && clampUnit(sig.BaitSwitchRisk) > 0 {
		return true
	}
	return false
}

func (r *Ranker) scoreSignal(sig *ScopedSignalFeatures) (delta float64, positive float64, sourceBacked bool, highIntensity bool) {
	if !r.isSignalActive(sig) {
		return 0.0, 0.0, false, false
	}

	if r.isDimAvailable(sig, DimEvidenceSupport) {
		switch sig.EvidenceSupportTier {
		case EvidenceSupported:
			delta += 0.10
			positive += 0.10
			sourceBacked = true
		case EvidenceContradicted:
			delta -= 0.22
		}
	}

	if r.isDimAvailable(sig, DimConsistency) {
		switch sig.ConsistencyTier {
		case ConsistencyConsistent:
			delta += 0.03
			positive += 0.03
		case ConsistencyInconsistent:
			delta -= 0.10
		}
	}

	if r.isDimAvailable(sig, DimTechnicalLegibility) {
		switch sig.TechnicalLegibilityTier {
		case LegibilityClear:
			delta += 0.02
			positive += 0.02
		case LegibilityDegraded:
			delta -= 0.03
		case LegibilityUnusable:
			delta -= 0.12
		}
	}

	if r.isDimAvailable(sig, DimProvenance) {
		switch sig.ProvenanceStatus {
		case ProvenanceVerified:
			delta += 0.03
			positive += 0.03
		case ProvenanceDeclared:
			delta += 0.01
			positive += 0.01
		case ProvenanceInvalid:
			delta -= 0.10
		}
	}

	if r.isDimAvailable(sig, DimIndependentSources) {
		if sig.IndependentSourceClusters == 1 {
			delta += 0.02
			positive += 0.02
		} else if sig.IndependentSourceClusters >= 2 {
			delta += 0.04
			positive += 0.04
			sourceBacked = true
		}
	}

	if r.isDimAvailable(sig, DimAttentionPacing) {
		pacing := clampUnit(sig.AttentionPacingScore)
		if pacing >= r.cfg.HighIntensityPacingThreshold {
			highIntensity = true
		}
		if pacing > 0.70 {
			delta -= 0.08 * ((pacing - 0.70) / 0.30)
		}
	}

	if r.isDimAvailable(sig, DimBaitSwitch) {
		risk := clampUnit(sig.BaitSwitchRisk)
		if risk > 0.35 {
			delta -= 0.20 * ((risk - 0.35) / 0.65)
		}
	}

	if delta > r.cfg.MaxSignalBoost {
		delta = r.cfg.MaxSignalBoost
	}
	if delta < -r.cfg.MaxSignalDemotion {
		delta = -r.cfg.MaxSignalDemotion
	}
	return delta, positive, sourceBacked, highIntensity
}

func compareCandidates(scoreA, scoreB float64, pubA, pubB int64, idA, idB string) int {
	const eps = 1e-12
	if diff := scoreA - scoreB; diff > eps {
		return -1
	} else if diff < -eps {
		return 1
	}
	if pubA != pubB {
		if pubA > pubB {
			return -1
		}
		return 1
	}
	return strings.Compare(idA, idB)
}

func clampUnit(v float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) || v <= 0 {
		return 0.0
	}
	if v >= 1.0 {
		return 1.0
	}
	return v
}

func roundScore(v float64) float64 {
	return math.Round(v*1e6) / 1e6
}

func encodeCursor(p cursorPayload) string {
	raw, _ := json.Marshal(p)
	return base64.RawURLEncoding.EncodeToString(raw)
}

func decodeCursor(token string) (cursorPayload, error) {
	trimmed := strings.TrimSpace(token)
	if len(trimmed) > maxCursorBytes {
		return cursorPayload{}, ErrInvalidCursor
	}
	raw, err := base64.RawURLEncoding.DecodeString(trimmed)
	if err != nil {
		return cursorPayload{}, ErrInvalidCursor
	}
	var payload cursorPayload
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&payload); err != nil {
		return cursorPayload{}, ErrInvalidCursor
	}
	if payload.Version != cursorFormatVersion || payload.SessionItemsSeen < 0 || payload.HighIntensityStreak < 0 {
		return cursorPayload{}, ErrInvalidCursor
	}
	for _, id := range payload.Seen {
		if strings.TrimSpace(id) == "" {
			return cursorPayload{}, ErrInvalidCursor
		}
	}
	return payload, nil
}
