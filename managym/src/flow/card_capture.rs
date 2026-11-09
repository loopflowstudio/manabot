// card_capture.rs
// Private game facts at setup and committed boundaries; no rule events or RNG.

use crate::agent::{
    action::{ActionSpace, ActionType},
    observation::Observation,
};
use crate::flow::decision::EffectFrame;
use crate::state::game_object::ObjectRef;
use crate::{
    cards::content_pack::default_content_pack,
    flow::game::Game,
    state::{
        game_object::{CardId, PlayerId},
        zone::ZoneType,
    },
};
use serde::Serialize;

/// One consumed, nontrivial native choice. Focus is the shared action projection,
/// not a claim that each focused card caused or benefited from the action.
#[derive(Debug, Serialize)]
pub struct ExposureChoice {
    pub revision: u64,
    pub actor: usize,
    pub kind: crate::agent::action::ActionSpaceKind,
    pub options: Vec<ExposureOption>,
    pub selected: usize,
}

#[derive(Debug, Serialize)]
pub struct ExposureOption {
    pub action_type: ActionType,
    pub objects: Vec<CardSnapshot>,
}

/// A completed effect frame, including untargeted and suspended resolutions.
/// Completion does not assert an effect changed state or a strategy succeeded.
#[derive(Debug, Serialize)]
pub struct ExposureResolution {
    pub revision: u64,
    pub controller: usize,
    pub source: Option<ObjectRef>,
    pub definition: Option<u32>,
}

#[derive(Debug, Default, Serialize)]
pub struct ExposureCapture {
    pub choices: Vec<ExposureChoice>,
    pub resolutions: Vec<ExposureResolution>,
}

#[derive(Debug, Serialize)]
pub struct CardSnapshot {
    pub copy: usize,
    pub definition: u32,
    pub name: String,
    pub incarnation: u32,
    pub owner: usize,
    pub zone: Option<ZoneType>,
    pub origin: &'static str,
}

#[derive(Debug, Serialize)]
pub struct CardMove {
    pub sequence: usize,
    pub card: CardSnapshot,
    pub from: Option<ZoneType>,
    pub controller: usize,
    pub cause: &'static str,
    pub turn: u32,
}

#[derive(Debug, Serialize)]
pub struct CardCapture {
    pub version: u32,
    pub content_digest: String,
    pub initial: Vec<CardSnapshot>,
    pub opening: [Vec<usize>; 2],
    pub moves: Vec<CardMove>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exposure: Option<ExposureCapture>,
    #[serde(skip)]
    pub(crate) drawing: Option<CardId>,
}

fn snapshot(game: &Game, card: CardId) -> CardSnapshot {
    let data = &game.state.cards[card];
    let player = &game.state.players[data.owner.0];
    let original = game
        .card_capture
        .as_ref()
        .and_then(|capture| capture.original(card));
    CardSnapshot {
        copy: card.0,
        definition: original.map_or(data.definition_id.0, |card| card.definition),
        name: original.map_or_else(|| data.name.clone(), |card| card.name.clone()),
        incarnation: game.state.object_incarnations[card].0,
        owner: data.owner.0,
        zone: game.state.zones.zone_of(card),
        origin: if player.deck.contains(&card) {
            "deck"
        } else if player.sideboard.contains(&card) {
            "outside"
        } else {
            "generated"
        },
    }
}

pub(crate) fn snapshots(game: &Game) -> Vec<CardSnapshot> {
    (0..game.state.cards.len())
        .map(|i| snapshot(game, CardId(i)))
        .collect()
}

impl CardCapture {
    /// Physical-copy identity is fixed at setup/creation, not by its current face.
    fn original(&self, card: CardId) -> Option<&CardSnapshot> {
        self.initial
            .iter()
            .chain(self.moves.iter().map(|m| &m.card))
            .find(|snapshot| snapshot.copy == card.0)
    }

    pub(crate) fn new(game: &Game) -> Self {
        Self {
            version: 1,
            content_digest: default_content_pack().content_digest(),
            initial: snapshots(game),
            opening: std::array::from_fn(|seat| {
                game.state
                    .zones
                    .zone_cards(ZoneType::Hand, PlayerId(seat))
                    .iter()
                    .map(|c| c.0)
                    .collect()
            }),
            moves: Vec::new(),
            drawing: None,
            exposure: None,
        }
    }
}

impl Game {
    pub(crate) fn exposure_choice(
        &self,
        space: &ActionSpace,
        selected: usize,
    ) -> Option<ExposureChoice> {
        self.card_capture.as_ref()?.exposure.as_ref()?;
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
                                .cards
                                .iter()
                                .enumerate()
                                .find(|(_, card)| card.id == id)
                                .map(|(index, _)| CardId(index))
                                .or_else(|| {
                                    self.state
                                        .permanents
                                        .iter()
                                        .flatten()
                                        .find(|p| p.id == id)
                                        .map(|p| p.card)
                                })
                        })
                        .map(|card| snapshot(self, card))
                        .collect();
                    ExposureOption {
                        action_type: action.action_type(),
                        objects,
                    }
                })
                .collect(),
        })
    }

    pub(crate) fn capture_resolution(&mut self, frame: &EffectFrame) {
        // Attribute to the captured physical copy's initial definition. Copy/face
        // changes do not turn this into a count of executed mechanic programs.
        let definition = frame.source.and_then(|source| {
            let capture = self.card_capture.as_ref()?;
            capture.original(source).map(|card| card.definition)
        });
        if let Some(exposure) = self
            .card_capture
            .as_mut()
            .and_then(|capture| capture.exposure.as_mut())
        {
            exposure.resolutions.push(ExposureResolution {
                revision: self.decision_epoch,
                controller: frame.controller.0,
                source: frame.source_ref,
                definition,
            });
        }
    }

    pub(crate) fn capture_card_move(
        &mut self,
        card: CardId,
        from: Option<ZoneType>,
        to: Option<ZoneType>,
        controller: PlayerId,
    ) {
        if self.card_capture.is_none() {
            return;
        }
        let data = snapshot(self, card);
        let capture = self.card_capture.as_mut().expect("capture enabled");
        // Only the actual committed movement caused by draw_cards consumes this
        // marker. Triggered movements during its processing cannot inherit it.
        let drawn = capture.drawing == Some(card)
            && from == Some(ZoneType::Library)
            && to == Some(ZoneType::Hand);
        if drawn {
            capture.drawing = None;
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
        capture.moves.push(CardMove {
            sequence: capture.moves.len(),
            card: data,
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
        let card = game
            .state
            .zones
            .top(ZoneType::Library, PlayerId(0))
            .unwrap();
        game.move_card(card, ZoneType::Hand); // selection, not draw
        game.move_card(card, ZoneType::Battlefield);
        game.move_card(card, ZoneType::Hand); // return, not draw
        game.move_card(card, ZoneType::Library);
        // ZoneManager appends to the library's top.
        game.draw_cards(PlayerId(0), 1);
        let moves = &game.card_capture.as_ref().unwrap().moves;
        assert_eq!(
            moves.iter().map(|m| m.cause).collect::<Vec<_>>(),
            ["selection", "move", "return", "move", "draw"]
        );
        assert_eq!(moves.last().unwrap().card.copy, card.0);
        assert_eq!(moves.last().unwrap().card.incarnation, 5);
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
            let card = *game
                .state
                .zones
                .zone_cards(ZoneType::Hand, PlayerId(0))
                .iter()
                .find(|card| game.state.cards[**card].name == "Gray Ogre")
                .unwrap();
            // Synthetic free, untargeted spell isolates capture from card rules.
            game.state.cards[card].mana_cost = None;
            game.state.cards[card]
                .keywords
                .insert(crate::state::card::Keyword::Flash);
            game.card_capture = Some(CardCapture::new(&game));
            game.card_capture.as_mut().unwrap().exposure = Some(ExposureCapture::default());
            game.publish_action_space(ActionSpace {
                player: Some(PlayerId(0)),
                kind: ActionSpaceKind::Priority,
                actions: vec![
                    Action::CastSpell {
                        player: PlayerId(0),
                        card,
                    },
                    Action::PassPriority {
                        player: PlayerId(0),
                    },
                ],
                focus: Vec::new(),
            });
            let source = game.current_object_ref(card).unwrap();
            game.step(usize::from(outcome == "declined")).unwrap();
            if outcome == "countered" {
                game.counter_spell(card, None);
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
                source.entity.0
            );
            assert_eq!(
                exposure.choices[0].options[0].objects[0].incarnation,
                source.incarnation.0
            );
            assert_eq!(
                exposure.choices[0].selected,
                usize::from(outcome == "declined")
            );
            assert_eq!(
                exposure.resolutions.len(),
                usize::from(outcome == "completed")
            );
            if let Some(resolution) = exposure.resolutions.first() {
                assert_eq!(resolution.source.unwrap().entity, source.entity);
                assert_eq!(resolution.controller, 0);
            }
        }
    }
    #[test]
    fn exposure_capture_keeps_physical_identity_across_face_changes() {
        use std::sync::Arc;
        let mut game = Game::new_capturing(configs(), 17, true);
        let card = game.state.zones.zone_cards(ZoneType::Hand, PlayerId(0))[0];
        let before = snapshot(&game, card);
        // A controlled face switch uses two real definitions without asserting
        // that this synthetic double-faced land is a playable card.
        let content = default_content_pack();
        let back = content.definition_id("Forest").unwrap();
        game.state.cards[card]
            .turn_back_face_up((back, Arc::new(content.definition(back).unwrap().clone())));
        let after = snapshot(&game, card);
        assert_eq!(game.state.cards[card].name, "Forest");
        assert_eq!(after.name, before.name);
        assert_eq!(after.definition, before.definition);
        assert_eq!(after.incarnation, before.incarnation);
        game.move_card(card, ZoneType::Battlefield);
        let moved = &game
            .card_capture
            .as_ref()
            .unwrap()
            .moves
            .last()
            .unwrap()
            .card;
        assert_eq!(moved.name, before.name);
        assert_eq!(moved.definition, before.definition);
        assert_eq!(moved.incarnation, before.incarnation + 1);
    }
}
