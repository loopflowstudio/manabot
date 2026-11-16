//! Optional per-game counters. Live and retained facts use the same reducer;
//! no action, movement or exposure history is stored here.
use std::collections::{BTreeMap, BTreeSet};

use super::card_capture::{CardSnapshot, ExposureChoice};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CardCounts {
    pub seat: usize,
    pub definition: u32,
    pub name: String,
    pub deck_copies: u64,
    pub opening_copies: u64,
    pub drawn_copies: u64,
    pub opportunities: Option<u64>,
    pub selections: Option<u64>,
    pub resolutions: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub struct Statistics {
    counts: BTreeMap<(usize, u32), CardCounts>,
    exposure: bool,
}

impl Statistics {
    pub fn new(initial: &[CardSnapshot], exposure: bool) -> Self {
        let mut stats = Self {
            exposure,
            ..Self::default()
        };
        for card in initial {
            let row = stats.row(card.owner, card.definition, &card.name);
            row.deck_copies += u64::from(card.origin == "deck");
            row.opening_copies += u64::from(card.zone == Some(crate::state::zone::ZoneType::Hand));
        }
        stats
    }

    fn row(&mut self, seat: usize, definition: u32, name: &str) -> &mut CardCounts {
        self.counts
            .entry((seat, definition))
            .or_insert_with(|| CardCounts {
                seat,
                definition,
                name: name.into(),
                opportunities: self.exposure.then_some(0),
                selections: self.exposure.then_some(0),
                resolutions: self.exposure.then_some(0),
                ..CardCounts::default()
            })
    }

    pub fn movement(&mut self, card: &CardSnapshot, draw: bool) {
        self.row(card.owner, card.definition, &card.name)
            .drawn_copies += u64::from(draw);
    }

    pub fn choice(&mut self, choice: &ExposureChoice) {
        self.choice_objects(
            choice.actor,
            choice.selected,
            choice.options.iter().map(|o| o.objects.as_slice()),
        );
    }

    fn choice_objects<'a>(
        &mut self,
        actor: usize,
        selected_index: usize,
        options: impl Iterator<Item = &'a [CardSnapshot]>,
    ) {
        let mut offered = BTreeSet::new();
        let mut selected = BTreeSet::new();
        for (index, option) in options.enumerate() {
            for card in option {
                self.row(actor, card.definition, &card.name);
                offered.insert(card.definition);
                if index == selected_index {
                    selected.insert(card.definition);
                }
            }
        }
        for definition in offered {
            *self
                .counts
                .get_mut(&(actor, definition))
                .unwrap()
                .opportunities
                .as_mut()
                .unwrap() += 1;
        }
        for definition in selected {
            *self
                .counts
                .get_mut(&(actor, definition))
                .unwrap()
                .selections
                .as_mut()
                .unwrap() += 1;
        }
    }

    pub fn resolution(&mut self, seat: usize, definition: Option<u32>) {
        if let Some(definition) = definition {
            if let Some(card) = self.counts.values().find(|c| c.definition == definition) {
                let name = card.name.clone();
                *self
                    .row(seat, definition, &name)
                    .resolutions
                    .as_mut()
                    .unwrap() += 1;
            }
        }
    }

    pub fn rows(&self) -> Vec<CardCounts> {
        self.counts.values().cloned().collect()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct GameStatistics {
    pub seed: u64,
    pub generation: u64,
    pub stream: usize,
    pub terminal: bool,
    pub winner: Option<usize>,
    pub cards: Vec<CardCounts>,
}

/// Reduce validated retained captures without reconstructing a game. Python
/// validates physical-copy continuity before crossing this serialization edge.
pub fn reduce_capture(raw: &str) -> Result<Vec<CardCounts>, serde_json::Error> {
    #[derive(Deserialize)]
    struct Move {
        card: CardSnapshot,
        cause: String,
    }
    #[derive(Deserialize)]
    struct OptionFact {
        objects: Vec<CardSnapshot>,
    }
    #[derive(Deserialize)]
    struct Choice {
        actor: usize,
        options: Vec<OptionFact>,
        selected: usize,
    }
    #[derive(Deserialize)]
    struct Resolution {
        controller: usize,
        definition: Option<u32>,
    }
    #[derive(Deserialize)]
    struct Exposure {
        choices: Vec<Choice>,
        resolutions: Vec<Resolution>,
    }
    #[derive(Deserialize)]
    struct Capture {
        initial: Vec<CardSnapshot>,
        moves: Vec<Move>,
        exposure: Option<Exposure>,
    }
    let capture: Capture = serde_json::from_str(raw)?;
    let mut stats = Statistics::new(&capture.initial, capture.exposure.is_some());
    for movement in capture.moves {
        stats.movement(&movement.card, movement.cause == "draw");
    }
    if let Some(exposure) = capture.exposure {
        for choice in exposure.choices {
            stats.choice_objects(
                choice.actor,
                choice.selected,
                choice.options.iter().map(|o| o.objects.as_slice()),
            );
        }
        for resolution in exposure.resolutions {
            stats.resolution(resolution.controller, resolution.definition);
        }
    }
    Ok(stats.rows())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        flow::{card_capture::ExposureCapture, game::Game},
        state::{game_object::PlayerId, player::PlayerConfig, zone::ZoneType},
    };

    fn game(capture: bool) -> Game {
        let configs = ["a", "b"]
            .into_iter()
            .map(|name| PlayerConfig::new(name, BTreeMap::from([("Mountain".into(), 20)])))
            .collect();
        Game::construct(configs, 17, true, capture, true)
    }

    #[test]
    fn statistics_count_draws_not_selection_or_return_and_match_retained() {
        let mut game = game(true);
        game.card_capture.as_mut().unwrap().exposure = Some(ExposureCapture::default());
        let object = game
            .state
            .zones
            .top(ZoneType::Library, PlayerId(0))
            .unwrap();
        let object = game.move_object(object, ZoneType::Hand);
        let object = game.move_object(object, ZoneType::Battlefield);
        let object = game.move_object(object, ZoneType::Hand);
        game.move_object(object, ZoneType::Library);
        game.draw_cards(PlayerId(0), 1);
        game.step(0).unwrap();
        let live = game.statistics.as_ref().unwrap().rows();
        let retained =
            reduce_capture(&serde_json::to_string(game.card_capture.as_ref().unwrap()).unwrap())
                .unwrap();
        assert_eq!(
            serde_json::to_value(&live).unwrap(),
            serde_json::to_value(retained).unwrap()
        );
        assert_eq!(
            live.iter()
                .filter(|r| r.seat == 0)
                .map(|r| r.drawn_copies)
                .sum::<u64>(),
            1
        );
        assert!(game.clone().statistics.is_none());
    }

    #[test]
    fn statistics_without_capture_keeps_rules_and_rejected_choices() {
        let mut stats = game(false);
        assert!(stats.card_capture.is_none());
        let mut plain = stats.clone();
        let before = serde_json::to_value(stats.statistics.as_ref().unwrap().rows()).unwrap();
        assert!(stats.step(usize::MAX).is_err());
        assert_eq!(
            before,
            serde_json::to_value(stats.statistics.as_ref().unwrap().rows()).unwrap()
        );
        for _ in 0..8 {
            stats.step(0).unwrap();
            plain.step(0).unwrap();
            assert_eq!(
                stats.state.deterministic_hash(),
                plain.state.deterministic_hash()
            );
        }
    }
    #[test]
    fn statistics_command_counts_once_and_rejects_stale_binding() {
        use crate::agent::structured_offer::OfferSubmission;
        let mut command_game = game(false);
        let mut indexed = game(false);
        let offers = command_game.structured_search_offers().unwrap();
        let command = offers
            .decode(&OfferSubmission {
                offer_id: offers.projection().offers[0].id,
                answers: Vec::new(),
            })
            .unwrap();
        command_game.apply_atomic_command(&command).unwrap();
        indexed.step(0).unwrap();
        let expected = serde_json::to_value(indexed.statistics.as_ref().unwrap().rows()).unwrap();
        assert_eq!(
            serde_json::to_value(command_game.statistics.as_ref().unwrap().rows()).unwrap(),
            expected
        );
        assert!(command_game.apply_atomic_command(&command).is_err());
        // The legacy transaction rejects the same stale choice after taking a
        // rollback snapshot. Neither rejection may discard accumulated counts.
        assert!(command_game
            .apply_legacy_offer_submission(
                &offers,
                &OfferSubmission {
                    offer_id: offers.projection().offers[0].id,
                    answers: Vec::new(),
                },
            )
            .is_err());
        assert_eq!(
            serde_json::to_value(command_game.statistics.as_ref().unwrap().rows()).unwrap(),
            expected
        );
    }
}
