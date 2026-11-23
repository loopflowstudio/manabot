//! Private replay encoding, execution and verification for a saved game boundary.
use super::*;
use serde::{Deserialize, Serialize};

impl GameSnapshot {
    /// Export private replay inputs, or fail if a mutation broke the prefix.
    pub fn to_replay_json(&self) -> Result<String, AgentError> {
        let provenance = self.provenance.as_ref().ok_or_else(|| {
            AgentError("snapshot has no replayable prefix after an unjournaled mutation".into())
        })?;
        SnapshotReplay {
            replay_digest: String::new(),
            schema: Schema::Snapshot,
            object_candidate_frames: self
                .candidates
                .keys()
                .map(|addr| addr.decision_fingerprint.clone())
                .collect(),
            terminal: self.state.is_game_over(),
            managym_version: env!("CARGO_PKG_VERSION").into(),
            card_digest: default_content_pack().content_digest(),
            seed: self.reset_seed,
            state_digest: state_digest(
                &self.state,
                &self.action_space,
                &self.pending_choice,
                self.skip_trivial,
                self.skipped,
            ),
            provenance: provenance.clone(),
        }
        .into_json()
    }

    /// Rebuild the position from private inputs and verify its rules digest.
    pub fn from_replay_json(raw: &str) -> Result<Self, AgentError> {
        let replay = SnapshotReplay::from_json(raw)?;
        let provenance = replay.provenance;
        let mut remaining_frames = replay.object_candidate_frames;

        let mut env = Env::new(replay.seed, provenance.skip_trivial, false, false)
            .with_player_policies(provenance.player_policies)
            .with_decision_memory(provenance.decision_memory)
            .with_decision_memory_include_own(provenance.decision_memory_include_own);
        env.reset(Arc::unwrap_or_clone(provenance.setup))?;
        let mut interventions = provenance.interventions.iter().peekable();
        for action_index in 0..=provenance.actions.len() {
            loop {
                if !remaining_frames.is_empty() {
                    env.restore_frame_candidates(|key| Ok(remaining_frames.remove(key)))
                        .map_err(|e| {
                            AgentError(format!("snapshot frame at action {action_index}: {e}"))
                        })?;
                }
                let Some(item) = interventions.next_if(|item| item.at_action == action_index)
                else {
                    break;
                };
                env.apply_intervention(item.op).map_err(|error| {
                    AgentError(format!(
                        "snapshot intervention at action {action_index}: {error}"
                    ))
                })?;
            }
            if let Some(action) = provenance.actions.get(action_index) {
                let result = match action {
                    RecordedAction::Index(i) => env.step(*i).map(|_| ()),
                    RecordedAction::Command(raw) => serde_json::from_str(raw)
                        .map_err(|e| AgentError(e.to_string()))
                        .and_then(|c| env.execute_semantic_command(&c).map(|_| ())),
                };
                result.map_err(|e| AgentError(format!("snapshot action {action_index}: {e}")))?;
            }
        }
        if !remaining_frames.is_empty() {
            return Err(AgentError("snapshot object candidate frame missing".into()));
        }
        if env.is_game_over() != replay.terminal {
            return Err(AgentError("snapshot terminal status mismatch".into()));
        }
        if env.state_digest()? != replay.state_digest {
            return Err(AgentError("snapshot state_digest mismatch".into()));
        }
        env.snapshot()
    }

    #[cfg(feature = "python")]
    pub(crate) fn check_replay(raw: &str) -> Result<(), AgentError> {
        SnapshotReplay::from_json(raw).map(|_| ())
    }
}

#[derive(Serialize, Deserialize)]
enum Schema {
    #[serde(rename = "managym-snapshot-v1")]
    Snapshot,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SnapshotReplay {
    replay_digest: String,
    schema: Schema,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    object_candidate_frames: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    terminal: bool,
    managym_version: String,
    seed: u64,
    card_digest: String,
    state_digest: String,
    #[serde(flatten)]
    provenance: GameProvenance,
}

/// The rules digest intentionally excludes chooser policies and memory capture.
/// Bind all replay fields as well so editing those cannot silently change the
/// next step. Canonical object key order makes this independent of formatting.
impl SnapshotReplay {
    fn into_json(mut self) -> Result<String, AgentError> {
        self.replay_digest = self.digest()?;
        serde_json::to_string(&self).map_err(|e| AgentError(e.to_string()))
    }

    fn from_json(raw: &str) -> Result<Self, AgentError> {
        let replay: Self = serde_json::from_str(raw)
            .map_err(|e| AgentError(format!("invalid snapshot replay: {e}")))?;
        if replay.card_digest != default_content_pack().content_digest() {
            return Err(AgentError("snapshot card_digest mismatch".into()));
        }
        if replay.digest()? != replay.replay_digest {
            return Err(AgentError("snapshot replay_digest mismatch".into()));
        }
        let provenance = &replay.provenance;
        if provenance.decision_memory_include_own && !provenance.decision_memory {
            return Err(AgentError(
                "invalid initial decision memory settings".into(),
            ));
        }
        if provenance
            .interventions
            .last()
            .is_some_and(|item| item.at_action > provenance.actions.len())
            || provenance
                .interventions
                .windows(2)
                .any(|pair| pair[0].at_action > pair[1].at_action)
        {
            return Err(AgentError("invalid snapshot intervention offset".into()));
        }
        Ok(replay)
    }

    fn digest(&self) -> Result<String, AgentError> {
        let mut canonical = serde_json::to_value(self).map_err(|e| AgentError(e.to_string()))?;
        canonical
            .as_object_mut()
            .expect("snapshot replay is an object")
            .remove("replay_digest");
        let bytes = serde_json::to_vec(&canonical).map_err(|e| AgentError(e.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_replay_reports_the_first_divergent_boundary() {
        let mut source = Env::new(42, true, false, false);
        source
            .reset(
                ["a", "b"]
                    .map(|name| PlayerConfig::new(name, BTreeMap::from([("Forest".into(), 20)])))
                    .to_vec(),
            )
            .unwrap();
        let raw = source.snapshot().unwrap().to_replay_json().unwrap();
        let fields: serde_json::Value = serde_json::from_str(&raw).unwrap();
        for (key, bad, reason) in [
            ("actions", serde_json::json!([0, 999]), "action 1"),
            (
                "interventions",
                serde_json::json!([{"at_action": 1, "op":"reseed", "seed":9}]),
                "offset",
            ),
            ("state_digest", serde_json::json!("bad"), "state_digest"),
        ] {
            let mut invalid = fields.clone();
            invalid[key] = bad;
            let replay: SnapshotReplay = serde_json::from_value(invalid).unwrap();
            let raw = replay.into_json().unwrap();
            let error = GameSnapshot::from_replay_json(&raw)
                .err()
                .expect("invalid replay");
            assert!(error.to_string().contains(reason), "{error}");
        }
    }
}
