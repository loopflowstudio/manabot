//! Atomic, representation-neutral input for one acting decision. Schema 6 stays
//! frozen; capture v2 shares native continuation, content and public-history
//! authorities. The digest covers every serialized fact, excluding itself.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::{
    agent::policy_history::{HistoryReference, HISTORY_LIMIT},
    decision::{Observation, SemanticError},
    experience::DecisionContextView,
    flow::game::Game,
    state::{card::CardDefId, game_object::PlayerId},
};

/// Changes to capture field inclusion, ordering or digest semantics require a
/// new version. Embedded Observation remains SEMANTIC_DECISION_VERSION (6).
pub const DECISION_CAPTURE_VERSION: u16 = 2;

#[derive(Clone, Debug, Serialize)]
pub struct DecisionCapture {
    pub capture_version: u16,
    pub world_version: String,
    pub content_digest: String,
    pub observation: Observation,
    pub continuation: Option<DecisionContextView>,
    pub frames: Vec<crate::agent::continuation::ContinuationFrame>,
    pub definitions: BTreeMap<u32, Value>,
    /// Visible physical-card/display-object joins; IDs never confer authorization.
    pub objects: BTreeMap<i32, Value>,
    pub history: PublicHistory,
    pub digest: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct PublicHistory {
    pub version: u16,
    /// Zero declares history off, otherwise the admitted chronological suffix.
    pub capacity: usize,
    pub rows: Vec<Value>,
}

impl Game {
    pub fn capture_decision(
        &self,
        viewer: PlayerId,
        history_capacity: usize,
    ) -> Result<DecisionCapture, SemanticError> {
        if history_capacity > HISTORY_LIMIT {
            return Err(SemanticError::IllegalCommand(
                "public history v1 requires 0..32 rows".into(),
            ));
        }
        let (observation, agent) = self.semantic_observation_with_input(viewer)?;
        if observation.decision.is_none() {
            return Err(SemanticError::NoActiveDecision);
        }
        // Capture admits only policy history, not the presentation's prose suffix.
        let continuation = self.decision_context_from_frames(viewer, &agent.continuation);
        let mut objects = BTreeMap::new();
        for visible in agent.agent_cards.iter().chain(&agent.opponent_cards) {
            let (index, card) = self
                .state
                .cards
                .iter()
                .enumerate()
                .find(|(_, card)| card.id.0 as i32 == visible.id)
                .ok_or(SemanticError::ViewerSafetyViolation)?;
            let reference = self
                .current_object_ref(crate::state::game_object::CardId(index))
                .ok_or(SemanticError::ViewerSafetyViolation)?;
            let permanent = self
                .state
                .card_to_permanent
                .get(index)
                .and_then(|id| *id)
                .and_then(|id| self.state.permanents[id].as_ref())
                .map(|permanent| permanent.id.0);
            objects.insert(
                card.id.0 as i32,
                json!({"permanent_id": permanent, "incarnation": reference.incarnation.0}),
            );
        }
        let mut ids = BTreeSet::new();
        for card in agent.agent_cards.iter().chain(&agent.opponent_cards) {
            ids.insert(card.registry_key as u32);
        }
        for card in &agent.agent_sideboard {
            ids.insert(card.registry_key as u32);
        }
        for stack in &agent.stack_objects {
            ids.insert(stack.source_card_registry_key as u32);
        }
        for player in [&agent.agent, &agent.opponent] {
            for counts in [
                &player.known_hand,
                &player.sideboard_counts,
                &player.remaining_sideboard_counts,
            ] {
                ids.extend(counts.keys());
            }
        }
        let mut reference = |reference: &HistoryReference| -> Value {
            match reference {
                HistoryReference::Unavailable => json!({"kind": "unavailable"}),
                HistoryReference::Player(player) => json!({"kind": "player", "seat": player.0}),
                HistoryReference::Card {
                    definition,
                    owner,
                    zone,
                    context_id,
                } => {
                    ids.insert(*definition);
                    // Same visibility/link admission as the native tensor encoder.
                    let visible = context_id.filter(|id| {
                        agent
                            .agent_cards
                            .iter()
                            .chain(&agent.opponent_cards)
                            .any(|card| card.id == *id)
                    });
                    json!({"kind": "card", "definition": definition, "owner": owner.0,
                        "zone": zone.map(|zone| zone as i32), "context_id": visible})
                }
            }
        };
        let rows = agent
            .policy_history
            .iter()
            .skip(agent.policy_history.len().saturating_sub(history_capacity))
            .map(|event| {
                json!({"kind": event.kind as u8, "amount": event.amount,
                "source": reference(&event.source), "target": reference(&event.target)})
            })
            .collect();
        for frame in &agent.continuation {
            ids.extend(frame.source.definition);
            for target in &frame.targets {
                ids.extend(target.reference.definition);
            }
            ids.extend(frame.context.and_then(|r| r.definition));
            ids.extend(frame.sacrifice.and_then(|r| r.definition));
        }
        let mut definitions = BTreeMap::new();
        for id in ids {
            let definition = self
                .state
                .content
                .public_definition(CardDefId(id))
                .ok_or_else(|| {
                    SemanticError::IllegalCommand(format!("missing public definition {id}"))
                })?;
            definitions.insert(id, definition.clone());
        }
        let mut capture = DecisionCapture {
            capture_version: DECISION_CAPTURE_VERSION,
            world_version: crate::WORLD_VERSION.into(),
            content_digest: self.state.content.content_digest(),
            observation,
            objects,
            continuation,
            frames: agent.continuation,
            definitions,
            history: PublicHistory {
                version: 1,
                capacity: history_capacity,
                rows,
            },
            digest: String::new(),
        };
        let mut payload = serde_json::to_value(&capture).expect("capture serializes");
        payload.as_object_mut().unwrap().remove("digest");
        capture.digest = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&payload).unwrap())
        );
        Ok(capture)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{flow::event::GameEvent, state::player::PlayerConfig};

    fn game() -> Game {
        let deck = [("Mountain".into(), 20), ("Lightning Bolt".into(), 20)].into();
        Game::new(
            vec![
                PlayerConfig::new("a", deck),
                PlayerConfig::new("b", [("Island".into(), 40)].into()),
            ],
            23,
            false,
        )
    }

    #[test]
    fn capture_preserves_schema_six_and_declares_history() {
        let mut game = game();
        let actor = PlayerId(game.semantic_decision_frame().unwrap().actor as usize);
        let before = serde_json::to_value(game.semantic_observation(actor).unwrap()).unwrap();
        let snapshot = game.capture_decision(actor, 0).unwrap();
        assert_eq!(serde_json::to_value(&snapshot.observation).unwrap(), before);
        assert_eq!(snapshot.observation.identity.schema_version, 6);
        assert_eq!(
            snapshot.digest,
            game.capture_decision(actor, 0).unwrap().digest
        );
        assert!(game.capture_decision(actor, 33).is_err());
        assert!(game.capture_decision(PlayerId(1 - actor.0), 32).is_err());
        for amount in 0..40 {
            game.emit(GameEvent::LifeChanged {
                player: actor,
                old: 20,
                new: 20 + amount,
            });
        }
        // Fixture events leave the present observation unchanged; the suffix is
        // still a different declared model input and is included in its digest.
        let history = game.capture_decision(actor, 3).unwrap();
        assert_eq!(serde_json::to_value(&history.observation).unwrap(), before);
        assert_eq!(history.history.rows.len(), 3);
        assert_eq!(history.history.rows[0]["amount"], 37);
        assert_ne!(history.digest, snapshot.digest);
        assert_eq!(
            game.capture_decision(actor, 0).unwrap().digest,
            snapshot.digest
        );
    }
}
