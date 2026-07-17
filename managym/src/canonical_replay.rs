//! Canonical replay decision chronology and viewer-safe projections.
//!
//! `CanonicalReplayV1` is authority-private because its rows are oriented to
//! both players. `CanonicalReplayProjectionV1` is the only portable/client
//! shape and contains one viewer's rows plus one semantic event track.

use std::collections::{BTreeMap, BTreeSet};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use schemars::JsonSchema;
use serde::{de, Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::experience::{
    AssetManifestHash, Command, CommandId, ContentHash, ExperienceFrame, InteractionOffer, MatchId,
    OfferId, PlayerId, PresentationEvent, PresentationSeq, PromptId, Revision,
};

#[derive(Clone, Copy, Debug, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(transparent)]
#[schemars(transparent)]
pub struct CanonicalReplayVersion(#[schemars(range(min = 1, max = 1))] pub u16);

impl<'de> Deserialize<'de> for CanonicalReplayVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match u16::deserialize(deserializer)? {
            1 => Ok(Self(1)),
            version => Err(de::Error::custom(format!(
                "unsupported canonical replay version {version}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayDecisionSource {
    Client,
    Policy,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayDecision {
    pub ordinal: u64,
    pub viewer: PlayerId,
    pub source: ReplayDecisionSource,
    pub revision: Revision,
    pub prompt_id: PromptId,
    pub offer_id: OfferId,
    pub command_id: CommandId,
    pub presentation_cursor: PresentationSeq,
    pub frame: ExperienceFrame,
    pub offer: InteractionOffer,
    pub command: Command,
}

impl ReplayDecision {
    pub fn validate(&self) -> Result<(), String> {
        let prompt = self
            .frame
            .prompt
            .as_ref()
            .ok_or_else(|| "decision frame has no prompt".to_string())?;
        if prompt.id != self.prompt_id
            || prompt.actor != self.viewer
            || self.frame.revision != self.revision
            || self.offer.id != self.offer_id
            || self.offer.actor != self.viewer
            || self.command.command_id.0 != self.command_id.0
            || self.command.match_id.0 != self.frame.match_id.0
            || self.command.expected_revision != self.revision
            || self.command.prompt_id != self.prompt_id
            || self.command.offer_id != self.offer_id
        {
            return Err("decision frame, offer, and command bindings drifted".into());
        }
        if self.frame.projection.agent.player_index != self.viewer.0 {
            return Err("decision frame is not oriented to its acting viewer".into());
        }
        if !self.frame.projection.opponent.hand.is_empty() {
            return Err("decision frame exposes opponent-private hand identities".into());
        }
        let selected = self
            .frame
            .offers
            .iter()
            .find(|offer| offer.id == self.offer_id)
            .ok_or_else(|| "decision offer is absent from captured frame".to_string())?;
        if serde_json::to_value(selected).map_err(|error| error.to_string())?
            != serde_json::to_value(&self.offer).map_err(|error| error.to_string())?
        {
            return Err("decision offer differs from captured frame offer".into());
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, String> {
        decision_payload_sha256(
            &self.frame,
            &self.offer,
            &self.command,
            self.presentation_cursor.0,
        )
    }
}

pub fn decision_payload_sha256(
    frame: &ExperienceFrame,
    offer: &InteractionOffer,
    command: &Command,
    presentation_cursor: u64,
) -> Result<String, String> {
    let payload = json!({
        "frame": frame,
        "offer": offer,
        "command": command,
        "presentation_cursor": presentation_cursor,
    });
    let bytes = serde_json::to_vec(&payload).map_err(|error| error.to_string())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ViewerPresentationTrack {
    pub viewer: PlayerId,
    pub head: PresentationSeq,
    pub events: Vec<PresentationEvent>,
}

impl ViewerPresentationTrack {
    fn validate(&self) -> Result<(), String> {
        for (expected, event) in self.events.iter().enumerate() {
            if event.seq.0 != expected as u64 {
                return Err(format!(
                    "viewer {} presentation sequence gap at {expected}",
                    self.viewer.0
                ));
            }
        }
        if self.head.0 != self.events.len() as u64 {
            return Err(format!(
                "viewer {} presentation head drifted",
                self.viewer.0
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalReplayV1 {
    pub version: CanonicalReplayVersion,
    pub replay_id: String,
    pub match_id: MatchId,
    pub content_hash: ContentHash,
    pub asset_manifest_hash: AssetManifestHash,
    pub decisions: Vec<ReplayDecision>,
    pub presentation_tracks: Vec<ViewerPresentationTrack>,
}

impl CanonicalReplayV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.replay_id.is_empty() {
            return Err("canonical replay requires replay_id".into());
        }
        let mut tracks = BTreeMap::new();
        for track in &self.presentation_tracks {
            track.validate()?;
            if tracks.insert(track.viewer.0, track).is_some() {
                return Err("canonical replay has duplicate viewer tracks".into());
            }
        }
        let mut commands = BTreeSet::new();
        let mut prompts = BTreeSet::new();
        let mut previous_cursor = BTreeMap::new();
        for (expected, row) in self.decisions.iter().enumerate() {
            row.validate()?;
            if row.ordinal != expected as u64 {
                return Err(format!("canonical replay ordinal gap at {expected}"));
            }
            if row.frame.match_id.0 != self.match_id.0
                || row.frame.content_hash.0 != self.content_hash.0
                || row.frame.asset_manifest_hash.0 != self.asset_manifest_hash.0
            {
                return Err("decision authority identity differs from replay".into());
            }
            if !commands.insert(row.command_id.0.clone()) {
                return Err("canonical replay has duplicate command id".into());
            }
            if !prompts.insert((row.revision.0, row.prompt_id.0)) {
                return Err("canonical replay has duplicate revision/prompt identity".into());
            }
            let track = tracks
                .get(&row.viewer.0)
                .ok_or_else(|| "decision has no viewer presentation track".to_string())?;
            if row.presentation_cursor.0 > track.head.0 {
                return Err("decision cursor is outside its viewer track".into());
            }
            if previous_cursor
                .insert(row.viewer.0, row.presentation_cursor.0)
                .is_some_and(|previous| previous > row.presentation_cursor.0)
            {
                return Err("decision cursor moves backwards".into());
            }
        }
        Ok(())
    }

    pub fn project(&self, viewer: PlayerId) -> Result<CanonicalReplayProjectionV1, String> {
        let track = self
            .presentation_tracks
            .iter()
            .find(|track| track.viewer == viewer)
            .ok_or_else(|| "decision not found".to_string())?;
        let projection = CanonicalReplayProjectionV1 {
            version: self.version,
            replay_id: self.replay_id.clone(),
            match_id: self.match_id.clone(),
            content_hash: self.content_hash.clone(),
            asset_manifest_hash: self.asset_manifest_hash.clone(),
            viewer,
            decisions: self
                .decisions
                .iter()
                .filter(|row| row.viewer == viewer)
                .cloned()
                .collect(),
            presentation_head: track.head,
            presentation: track.events.clone(),
        };
        projection.validate()?;
        Ok(projection)
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalReplayProjectionV1 {
    pub version: CanonicalReplayVersion,
    pub replay_id: String,
    pub match_id: MatchId,
    pub content_hash: ContentHash,
    pub asset_manifest_hash: AssetManifestHash,
    pub viewer: PlayerId,
    pub decisions: Vec<ReplayDecision>,
    pub presentation_head: PresentationSeq,
    pub presentation: Vec<PresentationEvent>,
}

impl CanonicalReplayProjectionV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.replay_id.is_empty() {
            return Err("canonical projection requires replay_id".into());
        }
        let mut previous_ordinal = None;
        let mut previous_cursor = 0;
        let mut commands = BTreeSet::new();
        let mut prompts = BTreeSet::new();
        for row in &self.decisions {
            row.validate()?;
            if row.viewer != self.viewer {
                return Err("canonical projection mixes viewer decision rows".into());
            }
            if row.frame.match_id.0 != self.match_id.0
                || row.frame.content_hash.0 != self.content_hash.0
                || row.frame.asset_manifest_hash.0 != self.asset_manifest_hash.0
            {
                return Err("decision identity differs from canonical projection".into());
            }
            if previous_ordinal.is_some_and(|previous| previous >= row.ordinal) {
                return Err("canonical projection ordinals are not increasing".into());
            }
            if row.presentation_cursor.0 < previous_cursor
                || row.presentation_cursor.0 > self.presentation_head.0
            {
                return Err("canonical projection decision cursor drifted".into());
            }
            if !commands.insert(row.command_id.0.clone())
                || !prompts.insert((row.revision.0, row.prompt_id.0))
            {
                return Err("canonical projection has duplicate identity".into());
            }
            previous_ordinal = Some(row.ordinal);
            previous_cursor = row.presentation_cursor.0;
        }
        for (expected, event) in self.presentation.iter().enumerate() {
            if event.seq.0 != expected as u64 {
                return Err(format!(
                    "canonical projection presentation gap at {expected}"
                ));
            }
        }
        if self.presentation_head.0 != self.presentation.len() as u64 {
            return Err("canonical projection presentation head drifted".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayDecisionAddress {
    pub version: CanonicalReplayVersion,
    pub replay_id: String,
    pub match_id: String,
    pub ordinal: u64,
    pub viewer: u8,
    pub revision: u64,
    pub prompt_id: u64,
    pub offer_id: u32,
    pub command_id: String,
    pub presentation_cursor: u64,
    pub decision_sha256: String,
}

impl ReplayDecisionAddress {
    pub fn from_decision(
        replay: &CanonicalReplayProjectionV1,
        row: &ReplayDecision,
    ) -> Result<Self, String> {
        Ok(Self {
            version: CanonicalReplayVersion(1),
            replay_id: replay.replay_id.clone(),
            match_id: replay.match_id.0.clone(),
            ordinal: row.ordinal,
            viewer: row.viewer.0,
            revision: row.revision.0,
            prompt_id: row.prompt_id.0,
            offer_id: row.offer_id.0,
            command_id: row.command_id.0.clone(),
            presentation_cursor: row.presentation_cursor.0,
            decision_sha256: row.sha256()?,
        })
    }

    pub fn encode(&self) -> Result<String, String> {
        let payload = json!([
            1,
            self.replay_id,
            self.match_id,
            self.ordinal.to_string(),
            self.viewer.to_string(),
            self.revision.to_string(),
            self.prompt_id.to_string(),
            self.offer_id.to_string(),
            self.command_id,
            self.presentation_cursor.to_string(),
            self.decision_sha256,
        ]);
        let bytes = serde_json::to_vec(&payload).map_err(|error| error.to_string())?;
        Ok(format!("erd1.{}", URL_SAFE_NO_PAD.encode(bytes)))
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        let encoded = value
            .strip_prefix("erd1.")
            .ok_or_else(|| "invalid replay decision address".to_string())?;
        if encoded.is_empty() || encoded.contains('=') {
            return Err("invalid replay decision address".into());
        }
        let bytes = URL_SAFE_NO_PAD
            .decode(encoded)
            .map_err(|_| "invalid replay decision address".to_string())?;
        let payload: Vec<Value> = serde_json::from_slice(&bytes)
            .map_err(|_| "invalid replay decision address".to_string())?;
        if payload.len() != 11 || payload[0] != json!(1) {
            return Err("invalid replay decision address".into());
        }
        let text = |index: usize| {
            payload[index]
                .as_str()
                .ok_or_else(|| "invalid replay decision address".to_string())
        };
        let decimal = |index: usize| -> Result<u64, String> {
            let raw = text(index)?;
            if raw.is_empty()
                || (raw.len() > 1 && raw.starts_with('0'))
                || !raw.bytes().all(|byte| byte.is_ascii_digit())
            {
                return Err("invalid replay decision address".into());
            }
            raw.parse()
                .map_err(|_| "invalid replay decision address".to_string())
        };
        let address = Self {
            version: CanonicalReplayVersion(1),
            replay_id: text(1)?.to_string(),
            match_id: text(2)?.to_string(),
            ordinal: decimal(3)?,
            viewer: u8::try_from(decimal(4)?).map_err(|_| "invalid replay decision address")?,
            revision: decimal(5)?,
            prompt_id: decimal(6)?,
            offer_id: u32::try_from(decimal(7)?).map_err(|_| "invalid replay decision address")?,
            command_id: text(8)?.to_string(),
            presentation_cursor: decimal(9)?,
            decision_sha256: text(10)?.to_string(),
        };
        if address.replay_id.is_empty()
            || address.match_id.is_empty()
            || address.command_id.is_empty()
            || address.decision_sha256.len() != 64
            || !address
                .decision_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            || address.encode()? != value
        {
            return Err("invalid replay decision address".into());
        }
        Ok(address)
    }
}
