// card_capture.rs
// Private evaluation facts at setup and committed zone moves; no rule events or RNG.

use crate::{
    flow::game::Game,
    state::{
        game_object::{CardId, PlayerId},
        zone::ZoneType,
    },
};
use serde::Serialize;

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
    #[serde(skip)]
    pub(crate) drawing: Option<CardId>,
}

fn snapshot(game: &Game, card: CardId) -> CardSnapshot {
    let data = &game.state.cards[card];
    let player = &game.state.players[data.owner.0];
    CardSnapshot {
        copy: card.0,
        definition: data.definition_id.0,
        name: data.name.clone(),
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
    pub(crate) fn new(game: &Game) -> Self {
        Self {
            version: 1,
            content_digest: game.state.content.content_digest(),
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
        }
    }
}

impl Game {
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
}
