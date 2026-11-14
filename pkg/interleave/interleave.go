package interleave

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"time"

	"github.com/Swivver/swivver_algorithm/pkg/ranker"
)

const (
	// InterleaveSchemaVersionV1 identifies the v1 InterleavedResult wire contract.
	InterleaveSchemaVersionV1 = "interleaved-feed-result/1.0"

	attributionTokenVersion = 1
	attributionTokenPrefix  = "v1"
	maxAttributionTokenLen  = 65536
	defaultPageSize         = 20
	maxPageSize             = 200
	bootstrapResamples      = 2000

	// M01 deliberative usefulness component weights and negative impression penalties.
	WeightCompletedReadOrWatch = 1.0
	WeightSourceInspected      = 1.0
	WeightPositiveUsefulness   = 1.5
	PenaltyRapidSkip           = 1.0
	PenaltyCorrection          = 2.0
)

var (
	// ErrMissingRanker is returned when neither a Ranker instance nor a pre-ranked list is provided for an arm.
	ErrMissingRanker = errors.New("interleave: both arm A and arm B require a Ranker or pre-ranked items")
	// ErrMissingCandidates is returned when pre-ranked lists are supplied without candidate policy attestations.
	ErrMissingCandidates = errors.New("interleave: candidate pool is required to enforce policy-sandwich eligibility")
	// ErrInvalidAttributionToken is returned when an attribution token is malformed or fails HMAC verification.
	ErrInvalidAttributionToken = errors.New("interleave: invalid or tampered attribution token")

	defaultHMACKey = []byte("swivver-algorithm-interleave-hmac-v1")
)

// Team identifies which experimental ranker arm drafted a slot or won a comparison.
type Team string

const (
	TeamA   Team = "team_a"
	TeamB   Team = "team_b"
	TeamTie Team = "tie"
)

// SlotAttribution records which team drafted a 1-indexed feed slot during Team-Draft Interleaving.
type SlotAttribution struct {
	Rank        int    `json:"rank"`
	CandidateID string `json:"candidate_id"`
	Team        Team   `json:"team"`
	DraftRound  int    `json:"draft_round"`
}

type encodedAttributionPayload struct {
	Version      int               `json:"v"`
	ExperimentID string            `json:"exp"`
	ModelA       string            `json:"ma"`
	ModelB       string            `json:"mb"`
	Slots        []SlotAttribution `json:"slots"`
}

// AttributionMap holds authenticated per-slot TeamA/TeamB credit assignments.
// Its JSON representation emits only the opaque HMAC-signed token string so client
// surfaces never observe internal experiment arm names or per-slot assignments.
type AttributionMap struct {
	token        string
	experimentID string
	modelA       string
	modelB       string
	slots        []SlotAttribution
	byCandidate  map[string]SlotAttribution
}

// Token returns the opaque signed attribution token string.
func (a AttributionMap) Token() string {
	return a.token
}

// ExperimentID returns the experiment identifier bound to this attribution map.
func (a AttributionMap) ExperimentID() string {
	return a.experimentID
}

// ModelA returns the control/baseline ranker version recorded in the attribution map.
func (a AttributionMap) ModelA() string {
	return a.modelA
}

// ModelB returns the treatment ranker version recorded in the attribution map.
func (a AttributionMap) ModelB() string {
	return a.modelB
}

// Len returns the number of attributed slots.
func (a AttributionMap) Len() int {
	return len(a.slots)
}

// Slots returns a defensive copy of all per-slot team assignments for server-side audit.
func (a AttributionMap) Slots() []SlotAttribution {
	if len(a.slots) == 0 {
		return nil
	}
	out := make([]SlotAttribution, len(a.slots))
	copy(out, a.slots)
	return out
}

// Lookup retrieves the slot attribution for a candidate ID if it was part of the interleaved page.
func (a AttributionMap) Lookup(candidateID string) (SlotAttribution, bool) {
	if len(a.byCandidate) == 0 {
		return SlotAttribution{}, false
	}
	slot, ok := a.byCandidate[strings.TrimSpace(candidateID)]
	return slot, ok
}

// MarshalJSON serializes AttributionMap strictly as its opaque token string so accidental
// inclusion in client payloads never leaks TeamA/TeamB assignments.
func (a AttributionMap) MarshalJSON() ([]byte, error) {
	return json.Marshal(a.token)
}

// InterleaveRequest configures a deterministic, policy-safe Team-Draft Interleaving run.
type InterleaveRequest struct {
	ExperimentID string                          `json:"experiment_id"`
	Seed         int64                           `json:"seed"`
	RequestKey   string                          `json:"request_key,omitempty"`
	SigningKey   []byte                          `json:"-"`
	Candidates   []ranker.CandidateFeatureVector `json:"candidates"`
	RankerA      *ranker.Ranker                  `json:"-"`
	RankerB      *ranker.Ranker                  `json:"-"`
	RankedA      []ranker.RankedItem             `json:"ranked_a,omitempty"`
	RankedB      []ranker.RankedItem             `json:"ranked_b,omitempty"`
	ModelA       string                          `json:"model_a,omitempty"`
	ModelB       string                          `json:"model_b,omitempty"`
	PageSize     int                             `json:"page_size"`
	Now          time.Time                       `json:"now"`
	Session      ranker.SessionContext           `json:"session,omitempty"`
}

// InterleaveAudit summarizes policy-sandwich filtering and team draft balance.
type InterleaveAudit struct {
	TotalCandidates    int `json:"total_candidates"`
	EligibleInBoth     int `json:"eligible_in_both"`
	DroppedIneligible  int `json:"dropped_ineligible"`
	ExcludedAsymmetric int `json:"excluded_asymmetric"`
	TeamACount         int `json:"team_a_count"`
	TeamBCount         int `json:"team_b_count"`
}

// InterleavedResult contains user-safe ranked items alongside an opaque attribution token.
type InterleavedResult struct {
	SchemaVersion    string              `json:"schema_version"`
	Items            []ranker.RankedItem `json:"items"`
	AttributionToken string              `json:"attribution_token"`
	Attribution      AttributionMap      `json:"-"`
	Audit            InterleaveAudit     `json:"audit"`
}

// OutcomeEvent captures blinded post-impression deliberative reader outcomes for M01 credit attribution.
type OutcomeEvent struct {
	CandidateID          string `json:"candidate_id"`
	CompletedReadOrWatch bool   `json:"completed_read_or_watch"`
	ArticleCompleted     bool   `json:"article_completed,omitempty"`
	WatchCompleted       bool   `json:"watch_completed,omitempty"`
	SourceInspected      bool   `json:"source_inspected"`
	PositiveUsefulness   bool   `json:"positive_usefulness"`
	RapidSkip            bool   `json:"rapid_skip"`
	CorrectionImpression bool   `json:"correction_impression"`
}

// TeamMetrics summarizes deliberative M01 usefulness and negative quality counters for one team.
type TeamMetrics struct {
	AssignedSlots           int     `json:"assigned_slots"`
	EngagedItems            int     `json:"engaged_items"`
	CompletedCount          int     `json:"completed_count"`
	SourceInspectCount      int     `json:"source_inspect_count"`
	PositiveUsefulnessCount int     `json:"positive_usefulness_count"`
	RapidSkipCount          int     `json:"rapid_skip_count"`
	CorrectionImpressions   int     `json:"correction_impressions"`
	M01Score                float64 `json:"m01_score"`
}

// OutcomeReport summarizes credit attribution, paired draft-round wins, and statistical bounds
// for a single interleaved impression page.
type OutcomeReport struct {
	ExperimentID       string      `json:"experiment_id"`
	ModelA             string      `json:"model_a"`
	ModelB             string      `json:"model_b"`
	Winner             Team        `json:"winner"`
	TeamA              TeamMetrics `json:"team_a"`
	TeamB              TeamMetrics `json:"team_b"`
	NetM01Gain         float64     `json:"net_m01_gain"`
	PairWinsA          int         `json:"pair_wins_a"`
	PairWinsB          int         `json:"pair_wins_b"`
	PairTies           int         `json:"pair_ties"`
	UnattributedEvents int         `json:"unattributed_events"`
	SignTestPValue     float64     `json:"sign_test_p_value"`
	Bootstrap95LCB     float64     `json:"bootstrap_95_lcb"`
	Bootstrap95UCB     float64     `json:"bootstrap_95_ucb"`
}

// StudyReport aggregates session-level interleaving outcomes across an online usefulness study.
type StudyReport struct {
	SessionsEvaluated     int     `json:"sessions_evaluated"`
	SessionWinsA          int     `json:"session_wins_a"`
	SessionWinsB          int     `json:"session_wins_b"`
	SessionTies           int     `json:"session_ties"`
	TotalTeamAM01         float64 `json:"total_team_a_m01"`
	TotalTeamBM01         float64 `json:"total_team_b_m01"`
	MeanM01GainPerSession float64 `json:"mean_m01_gain_per_session"`
	RelativeM01Gain       float64 `json:"relative_m01_gain"`
	TeamACorrections      int     `json:"team_a_corrections"`
	TeamBCorrections      int     `json:"team_b_corrections"`
	SignTestPValue        float64 `json:"sign_test_p_value"`
	Bootstrap95LCB        float64 `json:"bootstrap_95_lcb"`
	Bootstrap95UCB        float64 `json:"bootstrap_95_ucb"`
	PassedUsefulnessGate  bool    `json:"passed_usefulness_gate"`
}

// Interleave executes deterministic, seed-reproducible Team-Draft Interleaving over Ranker A and
// Ranker B while enforcing fail-closed Policy-Sandwich eligibility across both arms.
func Interleave(req InterleaveRequest) (InterleavedResult, error) {
	hasArmA := req.RankerA != nil || len(req.RankedA) > 0
	hasArmB := req.RankerB != nil || len(req.RankedB) > 0
	if !hasArmA || !hasArmB {
		return InterleavedResult{}, ErrMissingRanker
	}
	if len(req.Candidates) == 0 && (len(req.RankedA) > 0 || len(req.RankedB) > 0) {
		return InterleavedResult{}, ErrMissingCandidates
	}

	expID := strings.TrimSpace(req.ExperimentID)
	if expID == "" {
		expID = "default-interleave-v1"
	}
	now := req.Now
	if now.IsZero() {
		now = time.Now().UTC()
	}
	pageSize := req.PageSize
	if pageSize <= 0 {
		pageSize = defaultPageSize
	}
	if pageSize > maxPageSize {
		pageSize = maxPageSize
	}

	// Step 1: Fail-closed Policy-Sandwich pass across all supplied candidate records.
	// If any duplicate record for the same ID is !Eligible or PolicyBlocked, that ID is blocked.
	blockedIDs := make(map[string]struct{})
	droppedIneligible := 0
	for i := range req.Candidates {
		c := &req.Candidates[i]
		id := strings.TrimSpace(c.CandidateID)
		if id == "" || !c.Eligible || c.PolicyBlocked {
			droppedIneligible++
			if id != "" {
				blockedIDs[id] = struct{}{}
			}
		}
	}

	admitted := make([]ranker.CandidateFeatureVector, 0, len(req.Candidates))
	admittedSet := make(map[string]struct{}, len(req.Candidates))
	for i := range req.Candidates {
		c := req.Candidates[i]
		id := strings.TrimSpace(c.CandidateID)
		if id == "" || !c.Eligible || c.PolicyBlocked {
			continue
		}
		if _, blocked := blockedIDs[id]; blocked {
			// Earlier/later duplicate carried a policy block: fail closed.
			droppedIneligible++
			continue
		}
		if _, exists := admittedSet[id]; exists {
			continue
		}
		c.CandidateID = id
		admittedSet[id] = struct{}{}
		admitted = append(admitted, c)
	}

	// Step 2 & 3: Obtain ordered candidate lists and enforce bilateral arm eligibility.
	var listA, listB []ranker.RankedItem
	var modelA, modelB string
	var eligibleInBoth, excludedAsymmetric int

	switch {
	case req.RankerA != nil && req.RankerB != nil:
		eligibleInBoth = len(admitted)
		target := min(pageSize, eligibleInBoth)
		listA, modelA = rankTopK(req.RankerA, admitted, target, now, req.Session)
		listB, modelB = rankTopK(req.RankerB, admitted, target, now, req.Session)
		if strings.TrimSpace(req.ModelA) != "" {
			modelA = strings.TrimSpace(req.ModelA)
		}
		if strings.TrimSpace(req.ModelB) != "" {
			modelB = strings.TrimSpace(req.ModelB)
		}

	case req.RankerA != nil && req.RankerB == nil:
		rawB, extraDrops := filterPreRanked(req.RankedB, admittedSet)
		droppedIneligible += extraDrops
		bilateralPool, setB := intersectCandidates(admitted, rawB)
		excludedAsymmetric = len(admitted) - len(bilateralPool)
		eligibleInBoth = len(bilateralPool)
		target := min(pageSize, eligibleInBoth)
		listA, modelA = rankTopK(req.RankerA, bilateralPool, target, now, req.Session)
		listB = filterBySet(rawB, setB)
		if strings.TrimSpace(req.ModelA) != "" {
			modelA = strings.TrimSpace(req.ModelA)
		}
		modelB = firstNonEmpty(strings.TrimSpace(req.ModelB), "ranker-b")

	case req.RankerA == nil && req.RankerB != nil:
		rawA, extraDrops := filterPreRanked(req.RankedA, admittedSet)
		droppedIneligible += extraDrops
		bilateralPool, setA := intersectCandidates(admitted, rawA)
		excludedAsymmetric = len(admitted) - len(bilateralPool)
		eligibleInBoth = len(bilateralPool)
		target := min(pageSize, eligibleInBoth)
		listA = filterBySet(rawA, setA)
		listB, modelB = rankTopK(req.RankerB, bilateralPool, target, now, req.Session)
		modelA = firstNonEmpty(strings.TrimSpace(req.ModelA), "ranker-a")
		if strings.TrimSpace(req.ModelB) != "" {
			modelB = strings.TrimSpace(req.ModelB)
		}

	default:
		rawA, dropsA := filterPreRanked(req.RankedA, admittedSet)
		rawB, dropsB := filterPreRanked(req.RankedB, admittedSet)
		droppedIneligible += dropsA + dropsB

		setA := itemIDSet(rawA)
		setB := itemIDSet(rawB)
		both := make(map[string]struct{}, min(len(setA), len(setB)))
		for id := range setA {
			if _, ok := setB[id]; ok {
				both[id] = struct{}{}
			} else {
				excludedAsymmetric++
			}
		}
		for id := range setB {
			if _, ok := setA[id]; !ok {
				excludedAsymmetric++
			}
		}
		listA = filterBySet(rawA, both)
		listB = filterBySet(rawB, both)
		eligibleInBoth = len(listA)
		modelA = firstNonEmpty(strings.TrimSpace(req.ModelA), "ranker-a")
		modelB = firstNonEmpty(strings.TrimSpace(req.ModelB), "ranker-b")
	}

	// Step 4: Deterministic Team-Draft Interleaving (Radlinski, Kurup & Joachims, 2008).
	target := min(pageSize, eligibleInBoth)
	rng := newDeterministicRNG(expID, req.RequestKey, req.Seed)

	placed := make(map[string]struct{}, target)
	items := make([]ranker.RankedItem, 0, target)
	slots := make([]SlotAttribution, 0, target)
	idxA, idxB := 0, 0
	teamACount, teamBCount := 0, 0

	for len(items) < target {
		for idxA < len(listA) {
			if _, seen := placed[listA[idxA].CandidateID]; !seen {
				break
			}
			idxA++
		}
		for idxB < len(listB) {
			if _, seen := placed[listB[idxB].CandidateID]; !seen {
				break
			}
			idxB++
		}

		var pickTeam Team
		switch {
		case teamACount < teamBCount:
			pickTeam = TeamA
		case teamBCount < teamACount:
			pickTeam = TeamB
		default:
			if rng.Uint64()&1 == 0 {
				pickTeam = TeamA
			} else {
				pickTeam = TeamB
			}
		}

		var chosen ranker.RankedItem
		if pickTeam == TeamA {
			chosen = listA[idxA]
			idxA++
			teamACount++
		} else {
			chosen = listB[idxB]
			idxB++
			teamBCount++
		}

		rank := len(items) + 1
		draftRound := ((rank - 1) / 2) + 1
		placed[chosen.CandidateID] = struct{}{}
		chosen.Rank = rank
		items = append(items, chosen)
		slots = append(slots, SlotAttribution{
			Rank:        rank,
			CandidateID: chosen.CandidateID,
			Team:        pickTeam,
			DraftRound:  draftRound,
		})
	}

	attrMap := buildSignedAttributionMap(expID, modelA, modelB, slots, req.SigningKey)

	return InterleavedResult{
		SchemaVersion:    InterleaveSchemaVersionV1,
		Items:            items,
		AttributionToken: attrMap.Token(),
		Attribution:      attrMap,
		Audit: InterleaveAudit{
			TotalCandidates:    len(req.Candidates),
			EligibleInBoth:     eligibleInBoth,
			DroppedIneligible:  droppedIneligible,
			ExcludedAsymmetric: excludedAsymmetric,
			TeamACount:         teamACount,
			TeamBCount:         teamBCount,
		},
	}, nil
}

// DecodeAttribution verifies the HMAC-SHA256 signature of an opaque attribution token and
// reconstructs the server-side AttributionMap.
func DecodeAttribution(token string, signingKey []byte) (AttributionMap, error) {
	trimmed := strings.TrimSpace(token)
	if trimmed == "" || len(trimmed) > maxAttributionTokenLen {
		return AttributionMap{}, ErrInvalidAttributionToken
	}

	parts := strings.Split(trimmed, ".")
	if len(parts) != 3 || parts[0] != attributionTokenPrefix {
		return AttributionMap{}, ErrInvalidAttributionToken
	}

	signingInput := parts[0] + "." + parts[1]
	gotMAC, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil {
		return AttributionMap{}, ErrInvalidAttributionToken
	}
	wantMAC := computeHMAC(signingInput, signingKey)
	if !hmac.Equal(gotMAC, wantMAC) {
		return AttributionMap{}, ErrInvalidAttributionToken
	}

	rawJSON, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return AttributionMap{}, ErrInvalidAttributionToken
	}

	var payload encodedAttributionPayload
	dec := json.NewDecoder(strings.NewReader(string(rawJSON)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&payload); err != nil {
		return AttributionMap{}, ErrInvalidAttributionToken
	}
	if payload.Version != attributionTokenVersion || strings.TrimSpace(payload.ExperimentID) == "" {
		return AttributionMap{}, ErrInvalidAttributionToken
	}

	byCand := make(map[string]SlotAttribution, len(payload.Slots))
	seenRanks := make(map[int]struct{}, len(payload.Slots))
	cleanSlots := make([]SlotAttribution, len(payload.Slots))
	for i, s := range payload.Slots {
		cid := strings.TrimSpace(s.CandidateID)
		if cid == "" || s.Rank < 1 || s.DraftRound < 1 || (s.Team != TeamA && s.Team != TeamB) {
			return AttributionMap{}, ErrInvalidAttributionToken
		}
		if _, dupCand := byCand[cid]; dupCand {
			return AttributionMap{}, ErrInvalidAttributionToken
		}
		if _, dupRank := seenRanks[s.Rank]; dupRank {
			return AttributionMap{}, ErrInvalidAttributionToken
		}
		seenRanks[s.Rank] = struct{}{}
		s.CandidateID = cid
		cleanSlots[i] = s
		byCand[cid] = s
	}

	return AttributionMap{
		token:        trimmed,
		experimentID: payload.ExperimentID,
		modelA:       payload.ModelA,
		modelB:       payload.ModelB,
		slots:        cleanSlots,
		byCandidate:  byCand,
	}, nil
}

// ScoreTokenOutcomes verifies an opaque attribution token and scores M01 reader outcomes in one step.
func ScoreTokenOutcomes(token string, signingKey []byte, events []OutcomeEvent) (OutcomeReport, error) {
	attr, err := DecodeAttribution(token, signingKey)
	if err != nil {
		return OutcomeReport{}, err
	}
	return ScoreOutcomes(attr, events)
}

type aggregatedItemOutcome struct {
	completed  bool
	source     bool
	useful     bool
	rapidSkip  bool
	correction bool
}

func (o aggregatedItemOutcome) m01Score() float64 {
	var score float64
	if o.completed {
		score += WeightCompletedReadOrWatch
	}
	if o.source {
		score += WeightSourceInspected
	}
	if o.useful {
		score += WeightPositiveUsefulness
	}
	if o.rapidSkip {
		score -= PenaltyRapidSkip
	}
	if o.correction {
		score -= PenaltyCorrection
	}
	return score
}

// ScoreOutcomes attributes reader outcome events to TeamA and TeamB idempotently, computes
// M01 deliberative usefulness scores, and evaluates exact two-sided sign-test p-values and
// deterministic bootstrap 95% confidence bounds across paired draft rounds.
func ScoreOutcomes(attribution AttributionMap, events []OutcomeEvent) (OutcomeReport, error) {
	if attribution.token == "" {
		return OutcomeReport{}, ErrInvalidAttributionToken
	}

	byCandidate := make(map[string]aggregatedItemOutcome, len(events))
	unattributed := 0

	for i := range events {
		ev := &events[i]
		cid := strings.TrimSpace(ev.CandidateID)
		if _, ok := attribution.Lookup(cid); !ok {
			unattributed++
			continue
		}
		agg := byCandidate[cid]
		if ev.CompletedReadOrWatch || ev.ArticleCompleted || ev.WatchCompleted {
			agg.completed = true
		}
		if ev.SourceInspected {
			agg.source = true
		}
		if ev.PositiveUsefulness {
			agg.useful = true
		}
		if ev.RapidSkip {
			agg.rapidSkip = true
		}
		if ev.CorrectionImpression {
			agg.correction = true
		}
		byCandidate[cid] = agg
	}

	var teamA, teamB TeamMetrics
	type roundPair struct {
		hasA   bool
		hasB   bool
		scoreA float64
		scoreB float64
	}
	maxRound := 0
	for _, s := range attribution.slots {
		if s.DraftRound > maxRound {
			maxRound = s.DraftRound
		}
	}
	rounds := make([]roundPair, maxRound)

	for _, s := range attribution.slots {
		agg, engaged := byCandidate[s.CandidateID]
		itemScore := 0.0
		if engaged {
			itemScore = agg.m01Score()
		}

		targetMetrics := &teamA
		if s.Team == TeamB {
			targetMetrics = &teamB
		}
		targetMetrics.AssignedSlots++
		if engaged {
			targetMetrics.EngagedItems++
			if agg.completed {
				targetMetrics.CompletedCount++
			}
			if agg.source {
				targetMetrics.SourceInspectCount++
			}
			if agg.useful {
				targetMetrics.PositiveUsefulnessCount++
			}
			if agg.rapidSkip {
				targetMetrics.RapidSkipCount++
			}
			if agg.correction {
				targetMetrics.CorrectionImpressions++
			}
			targetMetrics.M01Score += itemScore
		}

		rIdx := s.DraftRound - 1
		if s.Team == TeamA {
			rounds[rIdx].hasA = true
			rounds[rIdx].scoreA = itemScore
		} else {
			rounds[rIdx].hasB = true
			rounds[rIdx].scoreB = itemScore
		}
	}

	pairDiffs := make([]float64, 0, len(rounds))
	pairWinsA, pairWinsB, pairTies := 0, 0, 0
	for _, rp := range rounds {
		if !rp.hasA || !rp.hasB {
			continue
		}
		diff := rp.scoreB - rp.scoreA
		pairDiffs = append(pairDiffs, diff)
		switch {
		case diff > 1e-12:
			pairWinsB++
		case diff < -1e-12:
			pairWinsA++
		default:
			pairTies++
		}
	}

	teamA.M01Score = round6(teamA.M01Score)
	teamB.M01Score = round6(teamB.M01Score)
	netGain := round6(teamB.M01Score - teamA.M01Score)

	winner := TeamTie
	if netGain > 1e-12 {
		winner = TeamB
	} else if netGain < -1e-12 {
		winner = TeamA
	}

	pVal := TwoSidedSignTestPValue(pairWinsB, pairWinsA)
	lcb, ucb := bootstrapMeanCI95(pairDiffs, attribution.token)

	return OutcomeReport{
		ExperimentID:       attribution.experimentID,
		ModelA:             attribution.modelA,
		ModelB:             attribution.modelB,
		Winner:             winner,
		TeamA:              teamA,
		TeamB:              teamB,
		NetM01Gain:         netGain,
		PairWinsA:          pairWinsA,
		PairWinsB:          pairWinsB,
		PairTies:           pairTies,
		UnattributedEvents: unattributed,
		SignTestPValue:     round6(pVal),
		Bootstrap95LCB:     round6(lcb),
		Bootstrap95UCB:     round6(ucb),
	}, nil
}

// AggregateStudy combines session-level OutcomeReports across an online interleaved study,
// computing session-level wins, exact two-sided sign-test p-value, bootstrap 95% confidence
// bounds on mean session M01 gain, and the VG-5 online usefulness gate.
func AggregateStudy(sessions []OutcomeReport) StudyReport {
	if len(sessions) == 0 {
		return StudyReport{
			SignTestPValue: 1.0,
		}
	}

	var winsA, winsB, ties int
	var totalA, totalB float64
	var corrA, corrB int
	sessionDeltas := make([]float64, len(sessions))

	for i := range sessions {
		s := &sessions[i]
		totalA += s.TeamA.M01Score
		totalB += s.TeamB.M01Score
		corrA += s.TeamA.CorrectionImpressions
		corrB += s.TeamB.CorrectionImpressions
		delta := s.TeamB.M01Score - s.TeamA.M01Score
		sessionDeltas[i] = delta
		switch {
		case delta > 1e-12:
			winsB++
		case delta < -1e-12:
			winsA++
		default:
			ties++
		}
	}

	meanGain := (totalB - totalA) / float64(len(sessions))
	relGain := 0.0
	if math.Abs(totalA) > 1e-9 {
		relGain = (totalB - totalA) / math.Abs(totalA)
	}

	pVal := TwoSidedSignTestPValue(winsB, winsA)
	lcb, ucb := bootstrapMeanCI95(sessionDeltas, "study:"+sessions[0].ExperimentID)

	return StudyReport{
		SessionsEvaluated:     len(sessions),
		SessionWinsA:          winsA,
		SessionWinsB:          winsB,
		SessionTies:           ties,
		TotalTeamAM01:         round6(totalA),
		TotalTeamBM01:         round6(totalB),
		MeanM01GainPerSession: round6(meanGain),
		RelativeM01Gain:       round6(relGain),
		TeamACorrections:      corrA,
		TeamBCorrections:      corrB,
		SignTestPValue:        round6(pVal),
		Bootstrap95LCB:        round6(lcb),
		Bootstrap95UCB:        round6(ucb),
		PassedUsefulnessGate:  lcb > 0 && corrB <= corrA,
	}
}

// TwoSidedSignTestPValue computes the exact two-sided binomial sign-test p-value under H0: p = 0.5.
func TwoSidedSignTestPValue(winsB, winsA int) float64 {
	if winsB < 0 {
		winsB = 0
	}
	if winsA < 0 {
		winsA = 0
	}
	n := winsA + winsB
	if n == 0 {
		return 1.0
	}
	k := min(winsA, winsB)
	logN := lgammaInt(n + 1)
	nLog2 := float64(n) * math.Ln2

	tail := 0.0
	for j := 0; j <= k; j++ {
		logProb := logN - lgammaInt(j+1) - lgammaInt(n-j+1) - nLog2
		tail += math.Exp(logProb)
	}
	p := 2.0 * tail
	if p > 1.0 {
		return 1.0
	}
	return p
}

func lgammaInt(v int) float64 {
	res, _ := math.Lgamma(float64(v))
	return res
}

func bootstrapMeanCI95(values []float64, seedMaterial string) (lcb float64, ucb float64) {
	n := len(values)
	if n == 0 {
		return 0.0, 0.0
	}
	if n == 1 {
		return values[0], values[0]
	}

	rng := newDeterministicRNG("bootstrap-ci95", seedMaterial, int64(n))
	means := make([]float64, bootstrapResamples)
	invN := 1.0 / float64(n)

	for b := 0; b < bootstrapResamples; b++ {
		sum := 0.0
		for i := 0; i < n; i++ {
			sum += values[rng.IntN(n)]
		}
		means[b] = sum * invN
	}
	slices.Sort(means)
	lowIdx := int(math.Floor(0.025 * float64(bootstrapResamples-1)))
	highIdx := int(math.Ceil(0.975 * float64(bootstrapResamples-1)))
	return means[lowIdx], means[highIdx]
}

func rankTopK(
	r *ranker.Ranker,
	pool []ranker.CandidateFeatureVector,
	k int,
	now time.Time,
	session ranker.SessionContext,
) ([]ranker.RankedItem, string) {
	if k <= 0 || len(pool) == 0 {
		res, _ := r.Rank(ranker.RankRequest{
			Candidates: nil,
			PageSize:   defaultPageSize,
			Now:        now,
			Session:    session,
		})
		return nil, res.ModelVersion
	}

	out := make([]ranker.RankedItem, 0, k)
	cursor := ""
	modelVer := ""
	for len(out) < k {
		remaining := k - len(out)
		res, _ := r.Rank(ranker.RankRequest{
			Candidates: pool,
			PageSize:   remaining,
			Cursor:     cursor,
			Now:        now,
			Session:    session,
		})
		if modelVer == "" {
			modelVer = res.ModelVersion
		}
		out = append(out, res.Items...)
		if res.NextCursor == "" || len(res.Items) == 0 {
			break
		}
		cursor = res.NextCursor
	}
	return out, modelVer
}

func filterPreRanked(items []ranker.RankedItem, admittedSet map[string]struct{}) ([]ranker.RankedItem, int) {
	out := make([]ranker.RankedItem, 0, len(items))
	seen := make(map[string]struct{}, len(items))
	dropped := 0
	for i := range items {
		it := items[i]
		id := strings.TrimSpace(it.CandidateID)
		if id == "" {
			dropped++
			continue
		}
		if _, ok := admittedSet[id]; !ok {
			dropped++
			continue
		}
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		it.CandidateID = id
		out = append(out, it)
	}
	return out, dropped
}

func itemIDSet(items []ranker.RankedItem) map[string]struct{} {
	out := make(map[string]struct{}, len(items))
	for i := range items {
		out[items[i].CandidateID] = struct{}{}
	}
	return out
}

func intersectCandidates(
	admitted []ranker.CandidateFeatureVector,
	preRanked []ranker.RankedItem,
) ([]ranker.CandidateFeatureVector, map[string]struct{}) {
	setPre := itemIDSet(preRanked)
	pool := make([]ranker.CandidateFeatureVector, 0, len(preRanked))
	both := make(map[string]struct{}, len(preRanked))
	for i := range admitted {
		id := admitted[i].CandidateID
		if _, ok := setPre[id]; ok {
			pool = append(pool, admitted[i])
			both[id] = struct{}{}
		}
	}
	return pool, both
}

func filterBySet(items []ranker.RankedItem, allowed map[string]struct{}) []ranker.RankedItem {
	out := make([]ranker.RankedItem, 0, len(allowed))
	for i := range items {
		if _, ok := allowed[items[i].CandidateID]; ok {
			out = append(out, items[i])
		}
	}
	return out
}

func buildSignedAttributionMap(
	expID string,
	modelA string,
	modelB string,
	slots []SlotAttribution,
	signingKey []byte,
) AttributionMap {
	payload := encodedAttributionPayload{
		Version:      attributionTokenVersion,
		ExperimentID: expID,
		ModelA:       modelA,
		ModelB:       modelB,
		Slots:        slots,
	}
	rawJSON, _ := json.Marshal(payload)
	encodedPayload := base64.RawURLEncoding.EncodeToString(rawJSON)
	signingInput := attributionTokenPrefix + "." + encodedPayload
	sig := base64.RawURLEncoding.EncodeToString(computeHMAC(signingInput, signingKey))
	token := signingInput + "." + sig

	byCand := make(map[string]SlotAttribution, len(slots))
	copiedSlots := make([]SlotAttribution, len(slots))
	for i, s := range slots {
		copiedSlots[i] = s
		byCand[s.CandidateID] = s
	}

	return AttributionMap{
		token:        token,
		experimentID: expID,
		modelA:       modelA,
		modelB:       modelB,
		slots:        copiedSlots,
		byCandidate:  byCand,
	}
}

func computeHMAC(signingInput string, signingKey []byte) []byte {
	key := signingKey
	if len(key) == 0 {
		key = defaultHMACKey
	}
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte(signingInput))
	return mac.Sum(nil)
}

func newDeterministicRNG(experimentID, requestKey string, seed int64) *rand.Rand {
	h := sha256.New()
	_, _ = h.Write([]byte(experimentID))
	_, _ = h.Write([]byte{0})
	_, _ = h.Write([]byte(requestKey))
	_, _ = h.Write([]byte{0})
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], uint64(seed))
	_, _ = h.Write(buf[:])
	sum := h.Sum(nil)
	s1 := binary.BigEndian.Uint64(sum[0:8])
	s2 := binary.BigEndian.Uint64(sum[8:16])
	return rand.New(rand.NewPCG(s1, s2))
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

func round6(v float64) float64 {
	return math.Round(v*1e6) / 1e6
}
