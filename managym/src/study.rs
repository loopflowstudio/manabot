//! Versioned, viewer-safe evidence for restoring historical decisions.
//!
//! Study artifacts preserve an already-projected experience boundary. They do
//! not serialize `GameState`, determinization seeds, sampled hidden worlds, or
//! any other authority-private search state.

use std::collections::BTreeSet;

use schemars::JsonSchema;
use serde::{de, Deserialize, Deserializer, Serialize};

use crate::experience::{Command, ExperienceFrame, InteractionOffer, OfferId, PlayerId, PromptId};

#[derive(Clone, Copy, Debug, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(transparent)]
#[schemars(transparent)]
pub struct StudyVersion(#[schemars(range(min = 1, max = 1))] pub u16);

impl<'de> Deserialize<'de> for StudyVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match u16::deserialize(deserializer)? {
            1 => Ok(Self(1)),
            version => Err(de::Error::custom(format!(
                "unsupported study artifact version {version}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeScope {
    /// Only facts visible to the deciding player at the recorded revision.
    HistoricalViewer,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContentPackIdentity {
    pub id: String,
    pub version: String,
    pub content_hash: String,
    pub asset_manifest_sha256: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EngineIdentity {
    pub version: String,
    pub build_sha256: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelIdentity {
    pub id: String,
    pub checkpoint_sha256: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisBudgetIdentity {
    pub id: String,
    pub max_nodes: u64,
    pub sampled_worlds: u32,
    pub rollouts_per_world: u32,
}

/// Pins every producer whose output can change the meaning of an artifact.
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StudyIdentity {
    pub artifact_id: String,
    pub source_replay_id: String,
    pub source_replay_sha256: String,
    pub match_id: String,
    pub content_pack: ContentPackIdentity,
    pub engine: EngineIdentity,
    pub model: ModelIdentity,
    pub analysis_budget: AnalysisBudgetIdentity,
    pub knowledge_scope: KnowledgeScope,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct AlternativeId(pub String);

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionAlternative {
    pub id: AlternativeId,
    pub command: Command,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyMass {
    pub alternative: AlternativeId,
    pub probability: f64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchValue {
    pub alternative: AlternativeId,
    pub perspective: PlayerId,
    pub expected_match_points: f64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VisitCount {
    pub alternative: AlternativeId,
    pub visits: u64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SampledWorldRobustness {
    pub alternative: AlternativeId,
    pub favorable_worlds: u32,
    pub sampled_worlds: u32,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UncertaintyEvidence {
    pub alternative: AlternativeId,
    pub standard_error: f64,
    pub method: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceProvenance {
    pub producer: String,
    pub producer_version: String,
    pub generated_at: String,
    pub evidence_sha256: String,
}

/// Quantities remain separate so clients cannot accidentally present visits,
/// probability, robustness, or uncertainty as interchangeable confidence.
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionEvidence {
    pub policy_mass: Vec<PolicyMass>,
    pub search_value: Vec<SearchValue>,
    pub visits: Vec<VisitCount>,
    pub sampled_world_robustness: Vec<SampledWorldRobustness>,
    pub uncertainty: Vec<UncertaintyEvidence>,
    pub provenance: EvidenceProvenance,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StudyLandmark {
    pub id: String,
    pub decision_id: String,
    pub match_state_hash: String,
    pub viewer: PlayerId,
    pub prompt_id: PromptId,
    pub offer_id: OfferId,
    /// Exact historical projection, not a later reconstruction.
    pub frame: ExperienceFrame,
    /// Exact offer selected from `frame.offers` at the historical revision.
    pub offer: InteractionOffer,
    pub played: Command,
    pub alternatives: Vec<DecisionAlternative>,
    pub evidence: DecisionEvidence,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StudyArtifact {
    pub version: StudyVersion,
    pub identity: StudyIdentity,
    pub landmarks: Vec<StudyLandmark>,
}

impl StudyArtifact {
    /// Enforce cross-field identity bindings and the historical-viewer privacy
    /// boundary that ordinary JSON Schema cannot express.
    pub fn validate(&self) -> Result<(), String> {
        if self.identity.knowledge_scope != KnowledgeScope::HistoricalViewer {
            return Err("study v1 requires historical_viewer knowledge".into());
        }
        if self.landmarks.is_empty() {
            return Err("study artifact must contain at least one landmark".into());
        }

        for landmark in &self.landmarks {
            self.validate_landmark(landmark)?;
        }
        Ok(())
    }

    fn validate_landmark(&self, landmark: &StudyLandmark) -> Result<(), String> {
        let context = format!("landmark {}", landmark.id);
        let frame = &landmark.frame;
        if frame.match_id.0 != self.identity.match_id {
            return Err(format!(
                "{context}: frame match does not match study identity"
            ));
        }
        if frame.content_hash.0 != self.identity.content_pack.content_hash
            || frame.asset_manifest_hash.0 != self.identity.content_pack.asset_manifest_sha256
        {
            return Err(format!("{context}: frame content pack hashes drifted"));
        }
        if let Some(asset_pack) = frame.asset_pack.as_ref() {
            if asset_pack.id != self.identity.content_pack.id
                || asset_pack.version != self.identity.content_pack.version
                || asset_pack.manifest_sha256 != self.identity.content_pack.asset_manifest_sha256
            {
                return Err(format!("{context}: frame asset pack identity drifted"));
            }
        }
        if !frame.projection.opponent.hand.is_empty() {
            return Err(format!(
                "{context}: opponent-private hand identities are forbidden"
            ));
        }
        let prompt = frame
            .prompt
            .as_ref()
            .ok_or_else(|| format!("{context}: decision frame has no prompt"))?;
        if prompt.id != landmark.prompt_id
            || prompt.actor != landmark.viewer
            || landmark.offer.id != landmark.offer_id
            || landmark.offer.actor != landmark.viewer
        {
            return Err(format!(
                "{context}: viewer, prompt, or offer binding drifted"
            ));
        }
        let frame_offer = frame
            .offers
            .iter()
            .find(|offer| offer.id == landmark.offer_id)
            .ok_or_else(|| format!("{context}: selected offer is absent from frame"))?;
        if serde_json::to_value(frame_offer).map_err(|error| error.to_string())?
            != serde_json::to_value(&landmark.offer).map_err(|error| error.to_string())?
        {
            return Err(format!(
                "{context}: selected offer differs from frame offer"
            ));
        }

        validate_command_binding(&landmark.played, landmark, true, &context)?;
        if landmark.alternatives.is_empty() {
            return Err(format!("{context}: no decision alternatives recorded"));
        }
        let mut alternatives = BTreeSet::new();
        for alternative in &landmark.alternatives {
            validate_command_binding(&alternative.command, landmark, false, &context)?;
            if !alternatives.insert(alternative.id.0.clone()) {
                return Err(format!(
                    "{context}: duplicate alternative {}",
                    alternative.id.0
                ));
            }
        }

        validate_metric_cover(
            &alternatives,
            landmark
                .evidence
                .policy_mass
                .iter()
                .map(|row| &row.alternative),
            "policy mass",
            &context,
        )?;
        validate_metric_cover(
            &alternatives,
            landmark
                .evidence
                .search_value
                .iter()
                .map(|row| &row.alternative),
            "search value",
            &context,
        )?;
        validate_metric_cover(
            &alternatives,
            landmark.evidence.visits.iter().map(|row| &row.alternative),
            "visits",
            &context,
        )?;
        validate_metric_cover(
            &alternatives,
            landmark
                .evidence
                .sampled_world_robustness
                .iter()
                .map(|row| &row.alternative),
            "sampled-world robustness",
            &context,
        )?;
        validate_metric_cover(
            &alternatives,
            landmark
                .evidence
                .uncertainty
                .iter()
                .map(|row| &row.alternative),
            "uncertainty",
            &context,
        )?;

        let probability_sum = landmark
            .evidence
            .policy_mass
            .iter()
            .try_fold(0.0, |sum, row| {
                if row.probability.is_finite() && (0.0..=1.0).contains(&row.probability) {
                    Ok(sum + row.probability)
                } else {
                    Err(format!("{context}: invalid policy probability"))
                }
            })?;
        if (probability_sum - 1.0_f64).abs() > 1e-9_f64 {
            return Err(format!("{context}: policy mass must sum to one"));
        }
        for row in &landmark.evidence.search_value {
            if row.perspective != landmark.viewer || !row.expected_match_points.is_finite() {
                return Err(format!("{context}: invalid search value"));
            }
        }
        for row in &landmark.evidence.sampled_world_robustness {
            if row.sampled_worlds == 0 || row.favorable_worlds > row.sampled_worlds {
                return Err(format!("{context}: invalid sampled-world robustness"));
            }
        }
        for row in &landmark.evidence.uncertainty {
            if !row.standard_error.is_finite() || row.standard_error < 0.0 {
                return Err(format!("{context}: invalid uncertainty"));
            }
        }
        Ok(())
    }
}

fn validate_command_binding(
    command: &Command,
    landmark: &StudyLandmark,
    require_selected_offer: bool,
    context: &str,
) -> Result<(), String> {
    if command.match_id != landmark.frame.match_id
        || command.expected_revision != landmark.frame.revision
        || command.prompt_id != landmark.prompt_id
        || (require_selected_offer && command.offer_id != landmark.offer_id)
        || !landmark
            .frame
            .offers
            .iter()
            .any(|offer| offer.id == command.offer_id)
    {
        return Err(format!("{context}: command identity drifted"));
    }
    Ok(())
}

fn validate_metric_cover<'a>(
    alternatives: &BTreeSet<String>,
    ids: impl Iterator<Item = &'a AlternativeId>,
    label: &str,
    context: &str,
) -> Result<(), String> {
    let mut actual = BTreeSet::new();
    for id in ids {
        if !actual.insert(id.0.clone()) {
            return Err(format!("{context}: duplicate {label} for {}", id.0));
        }
    }
    if &actual != alternatives {
        return Err(format!("{context}: {label} does not cover alternatives"));
    }
    Ok(())
}
