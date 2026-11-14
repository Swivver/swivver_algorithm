package ranker

import (
	"errors"
	"time"
)

const (
	// SignalSchemaVersionV1 identifies the v1 ScopedSignalFeatures wire contract.
	SignalSchemaVersionV1 = "scoped-signal-features/1.0"
	// CandidateSchemaVersionV1 identifies the v1 CandidateFeatureVector wire contract.
	CandidateSchemaVersionV1 = "candidate-feature-vector/1.0"
	// ResultSchemaVersionV1 identifies the v1 RankedFeedResult wire contract.
	ResultSchemaVersionV1 = "ranked-feed-result/1.0"
	// DefaultModelVersion identifies the deterministic clean-room ranker release.
	DefaultModelVersion = "swivver-ranker/0.1.0"
)

// ErrInvalidCursor is returned when a caller supplies a malformed or unsupported cursor.
var ErrInvalidCursor = errors.New("ranker: invalid pagination cursor")

// Availability represents the explicit availability state of ScopedSignalFeatures
// or any individual scoped dimension.
type Availability string

const (
	AvailabilityAvailable     Availability = "available"
	AvailabilityUnknown       Availability = "unknown"
	AvailabilityUnsupported   Availability = "unsupported"
	AvailabilityNotApplicable Availability = "not_applicable"
	AvailabilityStale         Availability = "stale"
)

// Scoped dimension keys used in ScopedSignalFeatures.DimensionAvailability.
const (
	DimEvidenceSupport     = "evidence_support_tier"
	DimConsistency         = "consistency_tier"
	DimTechnicalLegibility = "technical_legibility_tier"
	DimAttentionPacing     = "attention_pacing_score"
	DimBaitSwitch          = "bait_switch_risk"
	DimProvenance          = "provenance_status"
	DimIndependentSources  = "independent_source_clusters"
)

// EvidenceSupportTier classifies supplied-claim alignment against attached sources.
type EvidenceSupportTier string

const (
	EvidenceSupported    EvidenceSupportTier = "supported"
	EvidenceMixed        EvidenceSupportTier = "mixed"
	EvidenceContradicted EvidenceSupportTier = "contradicted"
	EvidenceUnknown      EvidenceSupportTier = "unknown"
)

// ConsistencyTier classifies internal alignment across headline, video, and article.
type ConsistencyTier string

const (
	ConsistencyConsistent   ConsistencyTier = "consistent"
	ConsistencyMixed        ConsistencyTier = "mixed"
	ConsistencyInconsistent ConsistencyTier = "inconsistent"
	ConsistencyUnknown      ConsistencyTier = "unknown"
)

// TechnicalLegibilityTier classifies audio/video/caption technical legibility.
type TechnicalLegibilityTier string

const (
	LegibilityClear    TechnicalLegibilityTier = "clear"
	LegibilityDegraded TechnicalLegibilityTier = "degraded"
	LegibilityUnusable TechnicalLegibilityTier = "unusable"
	LegibilityUnknown  TechnicalLegibilityTier = "unknown"
)

// ProvenanceStatus classifies cryptographic or caller-declared provenance state.
type ProvenanceStatus string

const (
	ProvenanceVerified ProvenanceStatus = "verified"
	ProvenanceDeclared ProvenanceStatus = "declared"
	ProvenanceMissing  ProvenanceStatus = "missing"
	ProvenanceInvalid  ProvenanceStatus = "invalid"
	ProvenanceUnknown  ProvenanceStatus = "unknown"
)

// ScopedSignalFeatures contains optional, versioned, normalized media and
// evidence observations. It intentionally excludes universal quality scores,
// editorial merit scores, and author-credibility ranks.
type ScopedSignalFeatures struct {
	SchemaVersion             string                  `json:"schema_version"`
	SnapshotID                string                  `json:"snapshot_id,omitempty"`
	Availability              Availability            `json:"availability"`
	CoverageRatio             float64                 `json:"coverage_ratio"`
	FreshnessSec              float64                 `json:"freshness_sec"`
	EvidenceSupportTier       EvidenceSupportTier     `json:"evidence_support_tier,omitempty"`
	ConsistencyTier           ConsistencyTier         `json:"consistency_tier,omitempty"`
	TechnicalLegibilityTier   TechnicalLegibilityTier `json:"technical_legibility_tier,omitempty"`
	AttentionPacingScore      float64                 `json:"attention_pacing_score,omitempty"`
	BaitSwitchRisk            float64                 `json:"bait_switch_risk,omitempty"`
	ProvenanceStatus          ProvenanceStatus        `json:"provenance_status,omitempty"`
	IndependentSourceClusters int                     `json:"independent_source_clusters,omitempty"`
	DimensionAvailability     map[string]Availability `json:"dimension_availability,omitempty"`
}

// CandidateFeatureVector represents a single candidate supplied to the ranker.
type CandidateFeatureVector struct {
	SchemaVersion        string                `json:"schema_version"`
	CandidateID          string                `json:"candidate_id"`
	AuthorID             string                `json:"author_id"`
	TopicTags            []string              `json:"topic_tags"`
	Language             string                `json:"language"`
	PublishedAt          time.Time             `json:"published_at"`
	Eligible             bool                  `json:"eligible"`
	PolicyBlocked        bool                  `json:"policy_blocked"`
	FollowedAuthor       bool                  `json:"followed_author"`
	FollowedTopicMatches int                   `json:"followed_topic_matches"`
	LanguageMatch        bool                  `json:"language_match"`
	ImpressionCount      int                   `json:"impression_count,omitempty"`
	ValidWatchRatio      float64               `json:"valid_watch_ratio"`
	ArticleOpenRate      float64               `json:"article_open_rate"`
	SourceInspectRate    float64               `json:"source_inspect_rate"`
	SaveRate             float64               `json:"save_rate"`
	AnomalyFlagged       bool                  `json:"anomaly_flagged"`
	Signal               *ScopedSignalFeatures `json:"signal,omitempty"`
}

// SessionContext carries optional viewer session counters for D01/D03 pacing cues.
type SessionContext struct {
	SessionItemsSeen         int `json:"session_items_seen"`
	PriorHighIntensityStreak int `json:"prior_high_intensity_streak"`
}

// RankRequest specifies the candidate batch, pagination state, and reference clock.
type RankRequest struct {
	Candidates []CandidateFeatureVector `json:"candidates"`
	PageSize   int                      `json:"page_size"`
	Cursor     string                   `json:"cursor,omitempty"`
	Now        time.Time                `json:"now"`
	Session    SessionContext           `json:"session,omitempty"`
}

// ReasonCode identifies a structured, inspectable recommendation reason.
type ReasonCode string

const (
	ReasonFollowedAuthor   ReasonCode = "FOLLOWED_AUTHOR"
	ReasonTopicMatch       ReasonCode = "TOPIC_MATCH"
	ReasonSourceBacked     ReasonCode = "SOURCE_BACKED"
	ReasonReaderEngagement ReasonCode = "READER_ENGAGEMENT"
	ReasonLanguageMatch    ReasonCode = "LANGUAGE_MATCH"
	ReasonRecentUpdate     ReasonCode = "RECENT_UPDATE"
	ReasonFreshDiscovery   ReasonCode = "FRESH_DISCOVERY"
	ReasonGeneralDiscovery ReasonCode = "GENERAL_DISCOVERY"
)

// WhyThis contains at most two plain-language reasons explaining why a story was surfaced.
type WhyThis struct {
	ReasonCodes []ReasonCode `json:"reason_codes"`
	Reasons     []string     `json:"reasons"`
	Summary     string       `json:"summary"`
}

// ScoreComponents exposes additive scoring terms for offline audit and debugging.
type ScoreComponents struct {
	Preference       float64 `json:"preference"`
	Freshness        float64 `json:"freshness"`
	Engagement       float64 `json:"engagement"`
	SignalDelta      float64 `json:"signal_delta"`
	ExplorationBonus float64 `json:"exploration_bonus"`
	DiversityPenalty float64 `json:"diversity_penalty"`
}

// RankedItem represents a single ordered recommendation in RankedFeedResult.
type RankedItem struct {
	CandidateID     string          `json:"candidate_id"`
	Rank            int             `json:"rank"`
	Score           float64         `json:"score"`
	ScoreComponents ScoreComponents `json:"score_components"`
	WhyThis         WhyThis         `json:"why_this"`
}

// PacingTriggerCode identifies why a session pacing advisory was emitted.
type PacingTriggerCode string

const (
	TriggerHighIntensityStreak  PacingTriggerCode = "D01_HIGH_INTENSITY_STREAK"
	TriggerSessionDepthBoundary PacingTriggerCode = "D03_SESSION_DEPTH_BOUNDARY"
)

// AdvisoryKind classifies the gentle reader pacing or stopping suggestion.
type AdvisoryKind string

const (
	AdvisoryNone                AdvisoryKind = "none"
	AdvisorySlowerContextBreak  AdvisoryKind = "slower_context_break"
	AdvisoryGentleStoppingPoint AdvisoryKind = "gentle_stopping_point"
)

// SessionPacingAdvisory surfaces non-coercive pacing or stopping cues (D01/D03).
type SessionPacingAdvisory struct {
	Triggered           bool                `json:"triggered"`
	TriggerCodes        []PacingTriggerCode `json:"trigger_codes"`
	AdvisoryKind        AdvisoryKind        `json:"advisory_kind"`
	InsertAfterRank     int                 `json:"insert_after_rank"`
	Message             string              `json:"message"`
	HighIntensityStreak int                 `json:"high_intensity_streak"`
	SessionDepthItems   int                 `json:"session_depth_items"`
}

// DiversityAudit summarizes eligibility filtering, diversity caps, and cold-start metrics.
type DiversityAudit struct {
	EligibleCandidates     int     `json:"eligible_candidates"`
	DroppedIneligible      int     `json:"dropped_ineligible"`
	ReturnedCount          int     `json:"returned_count"`
	UniqueAuthors          int     `json:"unique_authors"`
	UniqueTopics           int     `json:"unique_topics"`
	MaxAuthorShare         float64 `json:"max_author_share"`
	AuthorCapDeferrals     int     `json:"author_cap_deferrals"`
	TopicCapDeferrals      int     `json:"topic_cap_deferrals"`
	ColdStartItems         int     `json:"cold_start_items"`
	AnomalyDiscountedItems int     `json:"anomaly_discounted_items"`
}

// RankedFeedResult is the versioned output of Ranker.Rank.
type RankedFeedResult struct {
	SchemaVersion         string                `json:"schema_version"`
	ModelVersion          string                `json:"model_version"`
	Items                 []RankedItem          `json:"items"`
	SessionPacingAdvisory SessionPacingAdvisory `json:"session_pacing_advisory"`
	NextCursor            string                `json:"next_cursor"`
	DiversityAudit        DiversityAudit        `json:"diversity_audit"`
}

// Config parameterizes Ranker weights, freshness windows, diversity caps, and pacing thresholds.
type Config struct {
	ModelVersion string

	// Pagination bounds.
	DefaultPageSize int
	MaxPageSize     int

	// Base relevance weights.
	FollowedAuthorWeight float64
	TopicMatchWeight     float64
	LanguageMatchWeight  float64
	FreshnessWeight      float64
	FreshnessHalfLifeHrs float64
	EngagementWeight     float64

	// Signal freshness and bounded delta caps.
	MaxSignalFreshnessSec float64
	MaxSignalBoost        float64
	MaxSignalDemotion     float64

	// Cold-start exploration envelope for fresh, low-impression candidates.
	ColdStartExplorationBonus float64
	ColdStartMaxAgeHrs        float64
	ColdStartMaxImpressions   int

	// Sliding-window and MMR diversity parameters.
	DiversityWindowSize    int
	MaxSameAuthorPerWindow int
	MaxSameTopicPerWindow  int
	MMRAuthorPenalty       float64
	MMRTopicPenalty        float64

	// Session pacing thresholds (D01 / D03).
	HighIntensityPacingThreshold  float64
	MaxHighIntensityStreak        int
	SessionDepthAdvisoryThreshold int
}

// DefaultConfig returns the calibrated clean-room v0.1 configuration.
func DefaultConfig() Config {
	return Config{
		ModelVersion:                  DefaultModelVersion,
		DefaultPageSize:               20,
		MaxPageSize:                   100,
		FollowedAuthorWeight:          0.30,
		TopicMatchWeight:              0.22,
		LanguageMatchWeight:           0.15,
		FreshnessWeight:               0.20,
		FreshnessHalfLifeHrs:          18.0,
		EngagementWeight:              0.25,
		MaxSignalFreshnessSec:         86400.0 * 7, // 7 days
		MaxSignalBoost:                0.18,
		MaxSignalDemotion:             0.35,
		ColdStartExplorationBonus:     0.03,
		ColdStartMaxAgeHrs:            12.0,
		ColdStartMaxImpressions:       100,
		DiversityWindowSize:           5,
		MaxSameAuthorPerWindow:        1,
		MaxSameTopicPerWindow:         2,
		MMRAuthorPenalty:              0.12,
		MMRTopicPenalty:               0.06,
		HighIntensityPacingThreshold:  0.75,
		MaxHighIntensityStreak:        3,
		SessionDepthAdvisoryThreshold: 15,
	}
}

func (c Config) normalized() Config {
	def := DefaultConfig()
	if c.ModelVersion == "" {
		c.ModelVersion = def.ModelVersion
	}
	if c.DefaultPageSize <= 0 {
		c.DefaultPageSize = def.DefaultPageSize
	}
	if c.MaxPageSize <= 0 {
		c.MaxPageSize = def.MaxPageSize
	}
	if c.DefaultPageSize > c.MaxPageSize {
		c.DefaultPageSize = c.MaxPageSize
	}
	if c.FreshnessHalfLifeHrs <= 0 {
		c.FreshnessHalfLifeHrs = def.FreshnessHalfLifeHrs
	}
	if c.MaxSignalFreshnessSec <= 0 {
		c.MaxSignalFreshnessSec = def.MaxSignalFreshnessSec
	}
	if c.MaxSignalBoost < 0 {
		c.MaxSignalBoost = 0
	}
	if c.MaxSignalDemotion < 0 {
		c.MaxSignalDemotion = 0
	}
	if c.ColdStartExplorationBonus < 0 {
		c.ColdStartExplorationBonus = 0
	}
	if c.ColdStartMaxAgeHrs <= 0 {
		c.ColdStartMaxAgeHrs = def.ColdStartMaxAgeHrs
	}
	if c.DiversityWindowSize < 2 {
		c.DiversityWindowSize = def.DiversityWindowSize
	}
	if c.MaxSameAuthorPerWindow <= 0 {
		c.MaxSameAuthorPerWindow = def.MaxSameAuthorPerWindow
	}
	if c.MaxSameTopicPerWindow <= 0 {
		c.MaxSameTopicPerWindow = def.MaxSameTopicPerWindow
	}
	if c.MMRAuthorPenalty < 0 {
		c.MMRAuthorPenalty = 0
	}
	if c.MMRTopicPenalty < 0 {
		c.MMRTopicPenalty = 0
	}
	if c.HighIntensityPacingThreshold <= 0 || c.HighIntensityPacingThreshold > 1 {
		c.HighIntensityPacingThreshold = def.HighIntensityPacingThreshold
	}
	if c.MaxHighIntensityStreak <= 0 {
		c.MaxHighIntensityStreak = def.MaxHighIntensityStreak
	}
	if c.SessionDepthAdvisoryThreshold <= 0 {
		c.SessionDepthAdvisoryThreshold = def.SessionDepthAdvisoryThreshold
	}
	return c
}
