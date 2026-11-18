// card_capture.rs
// Private game facts at setup and committed boundaries; no rule events or RNG.

use crate::agent::{
    action::{Action, ActionSpace, ActionType},
    observation::Observation,
};
use crate::flow::decision::EffectFrame;
use crate::{
    cards::content_pack::default_content_pack,
    flow::game::Game,
    state::{
        game_object::{ObjectId, PlayerId},
        zone::ZoneType,
    },
};
use serde::{Deserialize, Serialize};

/// One consumed, nontrivial native choice. Focus is the shared action projection,
/// not a claim that each focused card caused or benefited from the action.
#[derive(Clone, Debug, Serialize)]
pub struct ExposureChoice {
    pub revision: u64,
    pub actor: usize,
    pub kind: crate::agent::action::ActionSpaceKind,
    pub options: Vec<ExposureOption>,
    pub selected: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExposureOption {
    pub action_type: ActionType,
    pub objects: Vec<CardSnapshot>,
}

/// A completed effect frame, including untargeted and suspended resolutions.
/// Completion does not assert an effect changed state or a strategy succeeded.
#[derive(Clone, Debug, Serialize)]
pub struct ExposureResolution {
    pub revision: u64,
    pub controller: usize,
    #[serde(serialize_with = "crate::state::game_object::exact")]
    pub source: Option<ObjectId>,
    pub definition: Option<u32>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ExposureCapture {
    pub choices: Vec<ExposureChoice>,
    pub resolutions: Vec<ExposureResolution>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CardSnapshot {
    pub copy: usize,
    pub definition: u32,
    pub name: String,
    pub incarnation: u32,
    pub owner: usize,
    pub zone: Option<ZoneType>,
    pub origin: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct CardMove {
    pub sequence: usize,
    #[serde(rename = "card")]
    pub object: CardSnapshot,
    pub from: Option<ZoneType>,
    pub controller: usize,
    pub cause: &'static str,
    pub turn: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct CardCapture {
    pub version: u32,
    pub content_digest: String,
    pub initial: Vec<CardSnapshot>,
    pub opening: [Vec<usize>; 2],
    pub moves: Vec<CardMove>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exposure: Option<ExposureCapture>,
}

fn snapshot(game: &Game, object: ObjectId) -> CardSnapshot {
    let (definition, data) = game.state.source_base(object);
    let (_, owner, _) = game
        .state
        .objects
        .provenance(object)
        .expect("captured provenance");
    let player = &game.state.players[owner.0];
    let original = game.capture_originals.get(&object.slot());
    CardSnapshot {
        copy: object.slot(),
        definition: original.map_or(definition.0, |object| object.definition),
        name: original.map_or_else(|| data.name.clone(), |object| object.name.clone()),
        incarnation: object.generation(),
        owner: owner.0,
        zone: game.state.zones.zone_of(object),
        origin: match game.state.card_id(object) {
            Some(card) if player.deck.contains(&card) => "deck",
            Some(card) if player.sideboard.contains(&card) => "outside",
            _ => "generated",
        }
        .into(),
    }
}

pub(crate) fn snapshots(game: &Game) -> Vec<CardSnapshot> {
    game.state
        .zones
        .latest()
        .map(|object| snapshot(game, object))
        .collect()
}

impl CardCapture {
    pub(crate) fn new(game: &Game) -> Self {
        Self {
            version: 1,
            content_digest: default_content_pack().content_digest(),
            initial: snapshots(game),
            opening: std::array::from_fn(|seat| {
                game.state
                    .zones
                    .zone_objects(ZoneType::Hand, PlayerId(seat))
                    .iter()
                    .map(|c| c.slot())
                    .collect()
            }),
            moves: Vec::new(),
            exposure: None,
        }
    }
}

impl Game {
    /// Transactions preserve diagnostic sinks; search clones deliberately omit them.
    pub(crate) fn rollback_snapshot(&self) -> Self {
        let mut snapshot = self.clone();
        snapshot.decision_memory_enabled = self.decision_memory_enabled;
        snapshot.card_capture = self.card_capture.clone();
        snapshot.statistics = self.statistics.clone();
        snapshot.capture_originals = self.capture_originals.clone();
        snapshot.capture_drawing = self.capture_drawing;
        snapshot
    }

    /// Bind an atomic action to the same published choice as indexed stepping.
    pub(crate) fn exposure_for_action(&self, action: &Action) -> Option<ExposureChoice> {
        let space = self.current_action_space.as_ref()?;
        let index = space
            .actions
            .iter()
            .position(|candidate| candidate == action)?;
        self.exposure_choice(space, index)
    }

    pub(crate) fn exposure_choice(
        &self,
        space: &ActionSpace,
        selected: usize,
    ) -> Option<ExposureChoice> {
        if self.statistics.is_none()
            && self
                .card_capture
                .as_ref()
                .and_then(|c| c.exposure.as_ref())
                .is_none()
        {
            return None;
        }
        if space.actions.len() <= 1 {
            return None;
        }
        Some(ExposureChoice {
            revision: self.decision_epoch,
            actor: space.player.expect("published chooser").0,
            kind: space.kind,
            selected,
            options: space
                .actions
                .iter()
                .map(|action| {
                    let objects = Observation::action_focus(self, action)
                        .into_iter()
                        .filter_map(|id| {
                            self.state
                                .latest_objects()
                                .find(|(_, object)| object.id == id)
                                .map(|(object, _)| object)
                                .or_else(|| {
                                    self.state
                                        .permanents
                                        .iter()
                                        .flatten()
                                        .find(|p| p.id == id)
                                        .map(|p| p.object)
                                })
                        })
                        .map(|object_id| snapshot(self, object_id))
                        .collect();
                    ExposureOption {
                        action_type: action.action_type(),
                        objects,
                    }
                })
                .collect(),
        })
    }

    pub(crate) fn commit_capture_choice(&mut self, choice: ExposureChoice) {
        if let Some(stats) = &mut self.statistics {
            stats.choice(&choice);
        }
        if let Some(exposure) = self.card_capture.as_mut().and_then(|c| c.exposure.as_mut()) {
            exposure.choices.push(choice);
        }
    }

    pub(crate) fn capture_resolution(&mut self, frame: &EffectFrame) {
        // Attribute to the captured physical copy's initial definition. Copy/face
        // changes do not turn this into a count of executed mechanic programs.
        let definition = frame.source().and_then(|source| {
            self.capture_originals
                .get(&source.slot())
                .map(|card| card.definition)
        });
        if let Some(stats) = &mut self.statistics {
            stats.resolution(frame.controller.0, definition);
        }
        if let Some(exposure) = self
            .card_capture
            .as_mut()
            .and_then(|capture| capture.exposure.as_mut())
        {
            exposure.resolutions.push(ExposureResolution {
                revision: self.decision_epoch,
                controller: frame.controller.0,
                source: frame.source_ref(),
                definition,
            });
        }
    }

    pub(crate) fn capture_object_move(
        &mut self,
        object: ObjectId,
        from: Option<ZoneType>,
        to: Option<ZoneType>,
        controller: PlayerId,
    ) {
        if self.card_capture.is_none() && self.statistics.is_none() {
            return;
        }
        let data = snapshot(self, object);
        self.capture_originals
            .entry(data.copy)
            .or_insert_with(|| data.clone());
        let zones = &self.state.zones;

        // Only the actual committed movement caused by draw_cards consumes this
        // marker: `object` is what the marked library card became (CR 400.7).
        // Triggered movements during its processing cannot inherit it.
        let drawn = self
            .capture_drawing
            .and_then(|drawing| zones.successor(drawing))
            == Some(object)
            && from == Some(ZoneType::Library)
            && to == Some(ZoneType::Hand);
        if drawn {
            self.capture_drawing = None;
        }
        let cause = if drawn {
            "draw"
        } else if to == Some(ZoneType::Hand) {
            match from {
                Some(ZoneType::Library) => "selection",
                Some(_) => "return",
                None if data.origin == "outside" => "outside",
                None if data.origin == "generated" => "generated",
                None => "unknown",
            }
        } else {
            "move"
        };
        if let Some(stats) = &mut self.statistics {
            stats.movement(&data, drawn);
        }
        let Some(capture) = self.card_capture.as_mut() else {
            return;
        };
        capture.moves.push(CardMove {
            sequence: capture.moves.len(),
            object: data,
            from,
            controller: controller.0,
            cause,
            turn: self.state.turn.turn_number,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::player::PlayerConfig;
    use std::collections::BTreeMap;

    fn configs() -> Vec<PlayerConfig> {
        ["a", "b"]
            .into_iter()
            .map(|name| PlayerConfig::new(name, BTreeMap::from([("Mountain".into(), 20)])))
            .collect()
    }

    #[test]
    fn capture_preserves_rules_and_excludes_forks() {
        let plain = Game::new(configs(), 17, true);
        let mut captured = Game::new_capturing(configs(), 17, true);
        assert_eq!(
            plain.state.deterministic_hash(),
            captured.state.deterministic_hash()
        );
        let capture = captured.card_capture.as_ref().unwrap();
        assert_eq!(capture.opening[0].len(), 7);
        assert_eq!(capture.opening[1].len(), 7);
        assert!(captured.clone().card_capture.is_none());
        let mut fork = captured.clone();
        fork.draw_cards(PlayerId(0), 2);
        captured.draw_cards(PlayerId(0), 2);
        assert_eq!(
            fork.state.deterministic_hash(),
            captured.state.deterministic_hash()
        );
        assert_eq!(
            captured
                .card_capture
                .as_ref()
                .unwrap()
                .moves
                .iter()
                .filter(|m| m.cause == "draw")
                .count(),
            2
        );
    }

    #[test]
    fn selection_bounce_and_redraw_keep_distinct_causes() {
        let mut game = Game::new_capturing(configs(), 18, true);
        let object = game
            .state
            .zones
            .top(ZoneType::Library, PlayerId(0))
            .unwrap();
        let object = game.move_object(object, ZoneType::Hand); // selection, not draw
        let object = game.move_object(object, ZoneType::Battlefield);
        let object = game.move_object(object, ZoneType::Hand); // return, not draw
        let object = game.move_object(object, ZoneType::Library);
        // ZoneManager appends to the library's top.
        game.draw_cards(PlayerId(0), 1);
        let moves = &game.card_capture.as_ref().unwrap().moves;
        assert_eq!(
            moves.iter().map(|m| m.cause).collect::<Vec<_>>(),
            ["selection", "move", "return", "move", "draw"]
        );
        assert_eq!(moves.last().unwrap().object.copy, object.slot());
        assert_eq!(moves.last().unwrap().object.incarnation, 5);
    }
    #[test]
    fn exposure_distinguishes_declined_countered_and_completed() {
        use crate::agent::action::{Action, ActionSpace, ActionSpaceKind};
        for outcome in ["declined", "countered", "completed"] {
            let deck = BTreeMap::from([("Mountain".into(), 20), ("Gray Ogre".into(), 4)]);
            let mut game = Game::new_capturing(
                vec![
                    PlayerConfig::new("a", deck.clone()),
                    PlayerConfig::new("b", deck),
                ],
                17,
                false,
            );
            game.scenario_force_card_in_hand(PlayerId(0), "Gray Ogre")
                .unwrap();
            let object_id = *game
                .state
                .zones
                .zone_objects(ZoneType::Hand, PlayerId(0))
                .iter()
                .find(|object_id| game.state[**object_id].name == "Gray Ogre")
                .unwrap();
            // Synthetic free, untargeted spell isolates capture from card rules.
            game.state[object_id].mana_cost = None;
            game.state[object_id]
                .keywords
                .insert(crate::state::card::Keyword::Flash);
            game.card_capture = Some(CardCapture::new(&game));
            game.statistics = Some(crate::flow::statistics::Statistics::new(
                &snapshots(&game),
                true,
            ));
            game.card_capture.as_mut().unwrap().exposure = Some(ExposureCapture::default());
            game.publish_action_space(ActionSpace {
                player: Some(PlayerId(0)),
                kind: ActionSpaceKind::Priority,
                actions: vec![
                    Action::CastSpell {
                        player: PlayerId(0),
                        object: object_id,
                    },
                    Action::PassPriority {
                        player: PlayerId(0),
                    },
                ],
                focus: Vec::new(),
            });
            let source = game.state.live(object_id).unwrap();
            game.step(usize::from(outcome == "declined")).unwrap();
            if outcome == "countered" {
                game.counter_spell(object_id, None, ZoneType::Graveyard);
            }
            if outcome == "completed" {
                game.resolve_top_of_stack();
            }
            let exposure = game
                .card_capture
                .as_ref()
                .unwrap()
                .exposure
                .as_ref()
                .unwrap();
            assert_eq!(exposure.choices.len(), 1);
            assert_eq!(exposure.choices[0].actor, 0);
            assert_eq!(
                exposure.choices[0].options[0].objects[0].copy,
                source.slot()
            );
            assert_eq!(
                exposure.choices[0].options[0].objects[0].incarnation,
                source.generation()
            );
            assert_eq!(
                exposure.choices[0].selected,
                usize::from(outcome == "declined")
            );
            assert_eq!(
                exposure.resolutions.len(),
                usize::from(outcome == "completed")
            );
            let rows = game.statistics.as_ref().unwrap().rows();
            assert_eq!(
                rows.iter().map(|r| r.resolutions.unwrap()).sum::<u64>(),
                u64::from(outcome == "completed")
            );
            assert_eq!(
                rows.iter().map(|r| r.selections.unwrap()).sum::<u64>(),
                u64::from(outcome != "declined")
            );

            if let Some(resolution) = exposure.resolutions.first() {
                // CR 400.7g: the spell is the object the hand card became.
                let spell = resolution.source.unwrap();
                assert_eq!(game.state.card_id(spell), game.state.card_id(source));
                assert_eq!(spell.generation(), source.generation() + 1);
                assert_eq!(resolution.controller, 0);
            }
        }
    }
    #[test]
    fn exposure_capture_keeps_physical_identity_across_face_changes() {
        use std::sync::Arc;
        let mut game = Game::new_capturing(configs(), 17, true);
        let object = game.state.zones.zone_objects(ZoneType::Hand, PlayerId(0))[0];
        let before = snapshot(&game, object);
        // A controlled face switch uses two real definitions without asserting
        // that this synthetic double-faced land is a playable card.
        let content = default_content_pack();
        let back = content.definition_id("Forest").unwrap();
        game.state[object]
            .turn_back_face_up((back, Arc::clone(&content.definition(back).unwrap().printed)));
        let after = snapshot(&game, object);
        assert_eq!(game.state[object].name, "Forest");
        assert_eq!(after.name, before.name);
        assert_eq!(after.definition, before.definition);
        assert_eq!(after.incarnation, before.incarnation);
        game.move_object(object, ZoneType::Battlefield);
        let moved = &game
            .card_capture
            .as_ref()
            .unwrap()
            .moves
            .last()
            .unwrap()
            .object;
        assert_eq!(moved.name, before.name);
        assert_eq!(moved.definition, before.definition);
        assert_eq!(moved.incarnation, before.incarnation + 1);
    }
}
