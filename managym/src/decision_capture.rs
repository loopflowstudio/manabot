//! Atomic, representation-neutral input for one acting decision. Capture v6
//! combines schema-8 observations with native continuation, content and public-history
//! authorities. The digest covers every serialized fact, excluding itself.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::{
    agent::policy_history::{HistoryReference, HISTORY_LIMIT},
    cards::content_pack::default_content_pack,
    decision::{Observation, SemanticError},
    experience::DecisionContextView,
    flow::game::Game,
    state::{card::CardDefId, game_object::PlayerId},
};

/// Changes to capture field inclusion, ordering or digest semantics require a
/// new version. Embedded Observation follows SEMANTIC_DECISION_VERSION.
/// Version 6 dropped the world label; the content digest identifies the cards.
pub const DECISION_CAPTURE_VERSION: u16 = 6;

#[derive(Clone, Debug, Serialize)]
pub struct DecisionCapture {
    pub capture_version: u16,
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
        let (observation, mut agent) = self.semantic_observation_with_input(viewer)?;
        if observation.decision.is_none() {
            return Err(SemanticError::NoActiveDecision);
        }
        // Captured frames carry the effect execution text the LLM view and
        // pending summaries read; the policy observation itself never does.
        self.describe_effect_execution(viewer, &mut agent.continuation);
        // Capture admits only policy history, not the presentation's prose suffix.
        let continuation = self.decision_context_from_frames(viewer, &agent.continuation);
        let mut objects = BTreeMap::new();
        let live_by_alias: BTreeMap<_, _> = self
            .state
            .latest_objects()
            .filter_map(|(id, _)| {
                self.state
                    .live(id)
                    .map(|_| (self.observed_object_id(id).0 as i32, id))
            })
            .collect();
        for visible in agent.object_records() {
            let reference = *live_by_alias
                .get(&visible.id)
                .ok_or(SemanticError::ViewerSafetyViolation)?;
            let permanent = self
                .state
                .permanent_of(reference)
                .and_then(|id| self.state.permanents[id].as_ref())
                .map(|permanent| permanent.id.0);
            objects.insert(
                visible.id,
                json!({"permanent_id": permanent, "incarnation": reference.generation()}),
            );
        }
        let mut ids = BTreeSet::new();
        for object in agent.object_records() {
            if let Ok(id) = u32::try_from(object.registry_key) {
                ids.insert(id);
            }
        }
        for object in &agent.agent_sideboard {
            ids.insert(object.registry_key as u32);
        }
        // A face-down or token source has no catalog key (negative).
        ids.extend(
            agent
                .stack_objects
                .iter()
                .filter_map(|stack| u32::try_from(stack.source_card_registry_key).ok()),
        );
        for player in [&agent.agent, &agent.opponent] {
            ids.extend(player.known_hand.keys());
            ids.extend(player.known_library.values());
            for counts in [
                &player.main_counts,
                &player.unseen_counts,
                &player.sideboard_counts,
                &player.remaining_sideboard_counts,
            ]
            .into_iter()
            .flatten()
            {
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
                    ids.extend(*definition);
                    // Same visibility/link admission as the native tensor encoder.
                    let visible = context_id
                        .filter(|id| agent.object_records().any(|object| object.id == *id));
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
        for copy in agent
            .agent_permanents
            .iter()
            .chain(&agent.opponent_permanents)
            .filter_map(|permanent| permanent.copy.as_ref())
        {
            ids.extend(u32::try_from(copy.registry_key).ok());
            ids.extend(
                copy.exceptions
                    .iter()
                    .filter_map(|exception| exception.source_registry_key)
                    .filter_map(|key| u32::try_from(key).ok()),
            );
        }
        for effects in [&agent.agent.effects, &agent.opponent.effects]
            .into_iter()
            .chain(
                agent
                    .agent_permanents
                    .iter()
                    .chain(&agent.opponent_permanents)
                    .map(|permanent| &permanent.effects),
            )
        {
            ids.extend(
                effects
                    .iter()
                    .filter_map(|effect| effect.source_registry_key)
                    .filter_map(|key| u32::try_from(key).ok()),
            );
        }
        for frame in &agent.continuation {
            ids.extend(frame.source.definition);
            for target in &frame.targets {
                ids.extend(target.reference.definition);
            }
            ids.extend(frame.context.and_then(|r| r.definition));
            ids.extend(frame.remembered.iter().filter_map(|r| r.definition));
            ids.extend(
                frame
                    .effect_execution_references
                    .iter()
                    .filter_map(|r| r.definition),
            );
            ids.extend(frame.inherited_from.and_then(|r| r.definition));
            ids.extend(
                frame
                    .program
                    .iter()
                    .chain(&frame.effect_execution)
                    .filter_map(|&(kind, value)| {
                        (kind == crate::cards::tokens::TokenKind::DefinitionRef as u16)
                            .then_some(value as u32)
                    }),
            );
            ids.extend(frame.sacrifice.and_then(|r| r.definition));
        }
        for record in agent.decision_memory.iter() {
            for reference in record
                .context
                .mana_sources
                .iter()
                .chain(&record.response)
                .chain(&record.choice.references)
            {
                ids.extend(reference.definition);
            }
        }
        let mut definitions = BTreeMap::new();
        for id in ids {
            let definition = default_content_pack()
                .public_definition(CardDefId(id))
                .ok_or_else(|| {
                    SemanticError::IllegalCommand(format!("missing public definition {id}"))
                })?;
            definitions.insert(id, definition.clone());
        }
        let mut capture = DecisionCapture {
            capture_version: DECISION_CAPTURE_VERSION,
            content_digest: default_content_pack().content_digest(),
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
        assert_eq!(
            snapshot.observation.identity.schema_version,
            crate::decision::SEMANTIC_DECISION_VERSION
        );
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

    /// Player 0 manifests `hidden` and `other`, grants its face-down
    /// permanent flying, bounces the other manifest after one of its
    /// abilities is countered, and player 1 must choose a creature to
    /// destroy. Returns the four projections player 1 sees.
    fn manifest_root(hidden: &str, other: &str) -> [Value; 4] {
        use crate::{
            flow::decision::EffectFrame,
            state::{
                ability::{Effect, ObjectSelector},
                card::{CardType, Keyword, Keywords},
                effect::{Affected, EffectDuration, Modification},
                game_object::PublicId,
                predicate::CardPredicate,
                zone::ZoneType,
            },
        };
        let deck = BTreeMap::from([
            ("Gray Ogre".into(), 4),
            ("Forest".into(), 4),
            ("Mountain".into(), 32),
        ]);
        let mut game = Game::new(
            vec![
                PlayerConfig::new("a", deck.clone()),
                PlayerConfig::new("b", deck),
            ],
            708_2,
            false,
        );
        game.keep_opening_hands();
        game.scenario_force_battlefield(PlayerId(1), "Gray Ogre", true)
            .unwrap();
        let owner = PlayerId(0);
        let library = |game: &Game, name: &str| {
            *game
                .state
                .zones
                .zone_objects(ZoneType::Library, owner)
                .iter()
                .find(|object| game.state[**object].name == name)
                .unwrap()
        };
        let face_down = game.manifest(library(&game, hidden), owner);
        let bounced = game.manifest(library(&game, other), owner);
        game.apply_continuous(
            Some(face_down),
            owner,
            [Affected::Object(face_down)],
            Modification::AddKeywords(Keywords::of(&[Keyword::Flying])),
            EffectDuration::EndOfTurn,
        );
        game.emit(GameEvent::AbilityCountered {
            ability: PublicId(0),
            source_object: bounced,
            controller: owner,
            by: None,
        });
        game.move_object(bounced, ZoneType::Hand);
        game.run_frame(EffectFrame::new(
            PlayerId(1),
            vec![Effect::destroy(ObjectSelector::chosen(CardPredicate {
                card_type: Some(CardType::Creature),
                ..Default::default()
            }))],
        ));
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
        let viewer = PlayerId(1);
        assert_eq!(game.action_space().unwrap().player, Some(viewer));
        let context = game.decision_context(viewer).unwrap();
        assert!(context
            .recent_facts
            .iter()
            .any(|fact| fact == "An ability of a face-down permanent was countered."));
        let capture = serde_json::to_value(game.capture_decision(viewer, 8).unwrap()).unwrap();
        [
            serde_json::to_value(game.decision_offer_subjects(viewer)).unwrap(),
            serde_json::to_value(context).unwrap(),
            serde_json::to_value(game.presentation_subjects()).unwrap(),
            capture,
        ]
    }

    /// Two roots that differ only in which card backs each manifest give the
    /// choosing opponent identical offer subjects, context, presentation
    /// subjects and capture (CR 708.2).
    #[test]
    fn swapped_manifests_project_alike_to_the_choosing_opponent() {
        let ogre_first = manifest_root("Gray Ogre", "Forest");
        let forest_first = manifest_root("Forest", "Gray Ogre");
        for (index, (left, right)) in ogre_first.iter().zip(&forest_first).enumerate() {
            assert_eq!(left, right, "projection {index}");
        }
    }
}
