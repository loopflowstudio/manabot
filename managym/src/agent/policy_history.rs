//! Bounded public learning projection maintained at the authoritative event append.
//! Observation reads share the immutable suffix: no ledger replay, no permanent
//! slot reconstruction. Native zone moves expire links before recording arrivals.

use crate::{
    agent::observation::Observation,
    flow::{
        event::{DamageTarget, GameEvent},
        game::Game,
    },
    state::{
        game_object::{CardId, PlayerId, Target},
        zone::ZoneType,
    },
};
use std::{collections::VecDeque, sync::Arc};

pub const HISTORY_DIM: usize = 12;
pub const HISTORY_LIMIT: usize = 32;
pub type HistoryWindow = Arc<VecDeque<HistoryEvent>>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HistoryReference {
    Unavailable,
    Player(PlayerId),
    Card {
        definition: u32,
        owner: PlayerId,
        zone: Option<ZoneType>,
        context_id: Option<i32>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HistoryKind {
    Arrival,
    Damage,
    Life,
    Cast,
    Resolved,
    Countered,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryEvent {
    pub kind: HistoryKind,
    pub amount: i32,
    pub source: HistoryReference,
    pub target: HistoryReference,
}

fn public(zone: ZoneType) -> bool {
    matches!(
        zone,
        ZoneType::Battlefield
            | ZoneType::Stack
            | ZoneType::Graveyard
            | ZoneType::Exile
            | ZoneType::Command
    )
}

fn card_reference(game: &Game, card: CardId) -> HistoryReference {
    let Some(zone) = game.state.zones.zone_of(card).filter(|zone| public(*zone)) else {
        return HistoryReference::Unavailable;
    };
    let data = &game.state.cards[card];
    HistoryReference::Card {
        definition: data.definition_id.0,
        owner: data.owner,
        zone: Some(zone),
        context_id: Some(data.id.0 as i32),
    }
}

// Damage and counter-source payloads name a physical card, not the exact
// originating incarnation. A delayed effect must not route to a returned object.
// Retain only a definition currently public at emission; hidden sources and
// causal source zone/incarnation remain unavailable.
fn source_reference(game: &Game, source: Option<CardId>) -> HistoryReference {
    let mut reference = source
        .map(|card| card_reference(game, card))
        .unwrap_or(HistoryReference::Unavailable);
    if let HistoryReference::Card {
        zone, context_id, ..
    } = &mut reference
    {
        *zone = None;
        *context_id = None;
    }
    reference
}

fn target_reference(game: &Game, target: Target) -> HistoryReference {
    match target {
        Target::Player(player) => HistoryReference::Player(player),
        Target::StackSpell(card) => card_reference(game, card),
        Target::Permanent(id) => game
            .permanent_object_ref(id)
            .map(|exact| card_reference(game, CardId::from(exact.entity)))
            .unwrap_or(HistoryReference::Unavailable),
    }
}

/// Update the derived suffix at the same append that owns the committed ledger.
/// Work is bounded by 4*HISTORY_LIMIT reference comparisons plus the current
/// event's target count, independent of game length. Clones and undo marks share
/// this copy-on-write projection; it never executes or reconstructs rules.
pub(crate) fn record(game: &mut Game, event: &GameEvent) {
    use HistoryKind::*;
    if let GameEvent::CardMoved { card, .. } = event {
        let id = game.state.cards[*card].id.0 as i32;
        // Every real zone move creates a new incarnation. The old reference
        // keeps its public definition/zone but cannot follow the physical card.
        let matches = |reference: &HistoryReference| {
            matches!(reference,
            HistoryReference::Card { context_id: Some(context), .. } if *context == id)
        };
        if game
            .policy_history
            .iter()
            .any(|row| matches(&row.source) || matches(&row.target))
        {
            for row in Arc::make_mut(&mut game.policy_history) {
                for reference in [&mut row.source, &mut row.target] {
                    if matches(reference) {
                        if let HistoryReference::Card { context_id, .. } = reference {
                            *context_id = None;
                        }
                    }
                }
            }
        }
    }
    let unknown = HistoryReference::Unavailable;
    let (kind, amount, source, targets) = match event {
        GameEvent::CardMoved { card, to, .. } if public(*to) => {
            (Arrival, 0, card_reference(game, *card), vec![unknown])
        }
        GameEvent::DamageDealt {
            source,
            target,
            amount,
        } => {
            let target = match target {
                DamageTarget::Player(id) => Target::Player(*id),
                DamageTarget::Permanent(id) => Target::Permanent(*id),
            };
            (
                Damage,
                *amount as i32,
                source_reference(game, *source),
                vec![target_reference(game, target)],
            )
        }
        GameEvent::LifeChanged { player, old, new } => (
            Life,
            new - old,
            unknown,
            vec![HistoryReference::Player(*player)],
        ),
        GameEvent::SpellCast { card, targets } => (
            Cast,
            0,
            card_reference(game, *card),
            if targets.is_empty() {
                vec![unknown]
            } else {
                targets
                    .iter()
                    .map(|target| target_reference(game, *target))
                    .collect()
            },
        ),
        GameEvent::SpellResolved { card } => {
            (Resolved, 0, card_reference(game, *card), vec![unknown])
        }
        GameEvent::SpellCountered { card, by } => (
            Countered,
            0,
            source_reference(game, *by),
            vec![card_reference(game, *card)],
        ),
        _ => return,
    };
    let history = Arc::make_mut(&mut game.policy_history);
    for target in targets {
        if history.len() == HISTORY_LIMIT {
            history.pop_front();
        }
        history.push_back(HistoryEvent {
            kind,
            amount,
            source: source.clone(),
            target,
        });
    }
}

/// v1 rows: kind, signed amount; two references each containing definition+1,
/// owner role (0 unknown, 1 viewer, 2 opponent), zone+1, context index (-1 absent),
/// availability (0 unknown, 1 exact current object/player, 2 definition without a current link).
pub fn encode(
    obs: &Observation,
    capacity: usize,
    cards_per_player: usize,
) -> Vec<[f32; HISTORY_DIM]> {
    let reference = |value: &HistoryReference| -> [f32; 5] {
        let role = |owner: PlayerId| {
            if owner.0 as i32 == obs.turn.agent_player_id {
                1.0
            } else {
                2.0
            }
        };
        match value {
            HistoryReference::Unavailable => [0.0, 0.0, 0.0, -1.0, 0.0],
            HistoryReference::Player(player) => [0.0, role(*player), 0.0, role(*player) - 1.0, 1.0],
            HistoryReference::Card {
                definition,
                owner,
                zone,
                context_id,
            } => {
                let context = context_id.and_then(|id| {
                    obs.agent_cards
                        .iter()
                        .position(|card| card.id == id)
                        .map(|slot| 2 + slot)
                        .or_else(|| {
                            obs.opponent_cards
                                .iter()
                                .position(|card| card.id == id)
                                .map(|slot| 2 + cards_per_player + slot)
                        })
                });
                [
                    (*definition + 1) as f32,
                    role(*owner),
                    zone.map(|zone| (zone as usize + 1) as f32).unwrap_or(0.0),
                    context.map(|v| v as f32).unwrap_or(-1.0),
                    if context.is_some() { 1.0 } else { 2.0 },
                ]
            }
        }
    };
    obs.policy_history
        .iter()
        .skip(obs.policy_history.len().saturating_sub(capacity))
        .map(|event| {
            let mut row = [0.0; HISTORY_DIM];
            row[0] = event.kind as u8 as f32;
            row[1] = event.amount as f32;
            row[2..7].copy_from_slice(&reference(&event.source));
            row[7..12].copy_from_slice(&reference(&event.target));
            row
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{game_object::ObjectId, player::PlayerConfig};

    fn game() -> Game {
        Game::new(
            vec![
                PlayerConfig::new(
                    "a",
                    [("Mountain".into(), 20), ("Lightning Bolt".into(), 20)].into(),
                ),
                PlayerConfig::new(
                    "b",
                    [("Forest".into(), 20), ("Llanowar Elves".into(), 20)].into(),
                ),
            ],
            111,
            false,
        )
    }

    #[test]
    fn history_survives_drain_departure_reentry_and_hidden_reassignment() {
        let mut game = game();
        let card = CardId(0);
        let other = CardId(40);
        game.move_card(card, ZoneType::Stack);
        game.move_card(other, ZoneType::Battlefield);
        let target = game.state.card_to_permanent[other].unwrap();
        game.emit(GameEvent::SpellCast {
            card,
            targets: vec![Target::Permanent(target)],
        });
        let initial = Observation::for_player(&game, PlayerId(0));
        let cast = initial.policy_history.back().unwrap();
        assert!(matches!(
            cast.source,
            HistoryReference::Card {
                context_id: Some(_),
                ..
            }
        ));
        assert!(matches!(
            cast.target,
            HistoryReference::Card {
                context_id: Some(_),
                ..
            }
        ));
        let expected = encode(&initial, 32, 60);
        game.take_observation_events();
        assert_eq!(
            expected,
            encode(&Observation::for_player(&game, PlayerId(0)), 32, 60)
        );
        // Consistent presentation-ID renaming cannot alter learned rows.
        for card in game.state.cards.iter_mut() {
            card.id = ObjectId(card.id.0 + 1000);
        }
        for row in Arc::make_mut(&mut game.policy_history) {
            for reference in [&mut row.source, &mut row.target] {
                if let HistoryReference::Card {
                    context_id: Some(id),
                    ..
                } = reference
                {
                    *id += 1000;
                }
            }
        }
        assert_eq!(
            expected,
            encode(&Observation::for_player(&game, PlayerId(0)), 32, 60)
        );
        game.move_card(card, ZoneType::Hand);
        game.move_card(other, ZoneType::Graveyard);
        game.move_card(other, ZoneType::Battlefield);
        let projected = Observation::for_player(&game, PlayerId(0));
        let cast = projected
            .policy_history
            .iter()
            .find(|event| event.kind == HistoryKind::Cast)
            .unwrap();
        assert!(matches!(
            cast.source,
            HistoryReference::Card {
                context_id: None,
                ..
            }
        ));
        assert!(matches!(
            cast.target,
            HistoryReference::Card {
                context_id: None,
                ..
            }
        ));
        let before = encode(&projected, 32, 60);
        // Future hidden placement cannot rewrite a previously public identity.
        game.state
            .zones
            .move_card(card, PlayerId(0), ZoneType::Library);
        assert_eq!(
            before,
            encode(&Observation::for_player(&game, PlayerId(0)), 32, 60)
        );
        assert!(self::game().policy_history.is_empty());
    }

    #[test]
    fn history_is_bounded_public_and_viewer_relative() {
        let mut game = game();
        // Private hand movements must not consume the public history budget.
        for index in 0..40 {
            game.emit(GameEvent::LifeChanged {
                player: PlayerId(1),
                old: 20,
                new: 20 - index,
            });
            game.move_card(CardId(index as usize), ZoneType::Hand);
        }
        let first = Observation::for_player(&game, PlayerId(0));
        let second = Observation::for_player(&game, PlayerId(1));
        assert_eq!(first.policy_history.len(), HISTORY_LIMIT);
        let a = encode(&first, 3, 60);
        let b = encode(&second, 3, 60);
        assert_eq!(
            a.iter().map(|row| row[1]).collect::<Vec<_>>(),
            vec![-37.0, -38.0, -39.0]
        );
        assert_eq!(a[0][8], 2.0);
        assert_eq!(b[0][8], 1.0);
        assert_eq!(a[0][10], 1.0);
        assert_eq!(b[0][10], 0.0);
        game.scenario_set_life(PlayerId(0), 12);
        assert!(game.policy_history.is_empty());
        assert_eq!(a[0][6], 0.0); // unavailable source
    }
    #[test]
    fn observations_share_bounded_suffix_and_forks_restore_it() {
        use crate::flow::{event_log::CowStats, undo::JournalStats};
        let mut game = game();
        game.create_token("Ally", PlayerId(0), false);
        let token = CardId(game.state.cards.len() - 1);
        let permanent = game.state.card_to_permanent[token].unwrap();
        game.emit(GameEvent::DamageDealt {
            source: Some(token),
            target: DamageTarget::Permanent(permanent),
            amount: 1,
        });
        let root = game.policy_history.clone();
        for _ in 0..2048 {
            game.emit(GameEvent::TurnStarted {
                player: PlayerId(0),
                turn_number: 1,
            });
        }
        assert!(game.state.events.len() > 2048);
        for _ in 0..8 {
            let obs = Observation::for_player(&game, PlayerId(0));
            // An observation is an O(1) shared-suffix read, regardless of ledger
            // length. No history-off encoding walks these rows.
            assert!(Arc::ptr_eq(&root, &obs.policy_history));
            assert_eq!(encode(&obs, 32, 60).len(), 2);
        }
        assert!(matches!(
            root.back().unwrap().source,
            HistoryReference::Card {
                zone: None,
                context_id: None,
                ..
            }
        ));
        let target = &root.back().unwrap().target;
        assert!(matches!(
            target,
            HistoryReference::Card {
                context_id: Some(_),
                ..
            }
        ));
        game.admit_page_cow_root();
        let mut fork = game.page_cow_fork(Arc::new(CowStats::default()));
        assert!(Arc::ptr_eq(&root, &fork.policy_history));
        fork.enable_undo(Arc::new(JournalStats::default()), 1);
        let mark = fork.undo_mark();
        fork.move_card(token, ZoneType::Graveyard);
        assert_ne!(fork.policy_history, root);
        assert_eq!(game.policy_history, root);
        fork.undo_rollback(mark);
        assert!(Arc::ptr_eq(&root, &fork.policy_history));
        assert_eq!(
            encode(&Observation::for_player(&fork, PlayerId(0)), 32, 60),
            encode(&Observation::for_player(&game, PlayerId(0)), 32, 60)
        );
    }
    #[test]
    fn physical_card_slots_are_only_routing() {
        let root = |source: CardId, destination: CardId| {
            let mut game = game();
            game.scenario_clear_hand(PlayerId(0));
            game.scenario_clear_hand(PlayerId(1));
            game.push_spell_to_stack(source, PlayerId(0), vec![], vec![], false);
            game.move_card(destination, ZoneType::Battlefield);
            let target = game.state.card_to_permanent[destination].unwrap();
            game.emit(GameEvent::SpellCast {
                card: source,
                targets: vec![Target::Permanent(target)],
            });
            Observation::for_player(&game, PlayerId(0))
        };
        // Identical public definitions instantiated in different physical slots.
        let first = root(CardId(0), CardId(40));
        let second = root(CardId(1), CardId(41));
        assert_ne!(first.agent_cards[0].id, second.agent_cards[0].id);
        assert_eq!(encode(&first, 32, 60), encode(&second, 32, 60));
    }
}
