//! Bounded viewer history, recorded once and projected before any consumer reads it.
//! Observation reads share the immutable suffix: no ledger replay, no permanent
//! slot reconstruction. Native zone moves expire links before recording arrivals.

use super::observation_layout::history;

use crate::{
    agent::observation::Observation,
    flow::{
        event::{DamageTarget, GameEvent},
        game::Game,
    },
    state::{
        game_object::{ObjectId, PlayerId, Target},
        zone::ZoneType,
    },
};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

pub const HISTORY_DIM: usize = history::WIDTH;
pub const HISTORY_LIMIT: usize = 32;
pub type HistoryWindow = Arc<VecDeque<HistoryEvent>>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HistoryReference {
    Unavailable,
    Player(PlayerId),
    Card {
        definition: Option<u32>,
        seen: u8,
        owner: PlayerId,
        zone: Option<ZoneType>,
        context_id: Option<i32>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HistoryKind {
    Moved,
    Damage,
    Life,
    Cast,
    Resolved,
    Countered,
    Draw,
    Reveal,
    Look,
    Shuffle,
    Declined,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryEvent {
    pub kind: HistoryKind,
    pub amount: i32,
    pub source: HistoryReference,
    pub target: HistoryReference,
    pub from_zone: Option<ZoneType>,
    /// 0 none, 1 top, 2 bottom, 3 second from top.
    pub library_end: u8,
}

/// One immutable window with a viewer; raw identities never leave this boundary.
#[derive(Clone)]
pub struct ViewerHistory {
    window: HistoryWindow,
    viewer: PlayerId,
}

impl std::fmt::Debug for ViewerHistory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl Default for ViewerHistory {
    fn default() -> Self {
        Self::new(Default::default(), PlayerId(0))
    }
}
impl PartialEq for ViewerHistory {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}
impl Eq for ViewerHistory {}

impl ViewerHistory {
    pub(crate) fn new(window: HistoryWindow, viewer: PlayerId) -> Self {
        Self { window, viewer }
    }
    pub fn len(&self) -> usize {
        self.window.len()
    }
    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = HistoryEvent> + ExactSizeIterator + '_ {
        self.window.iter().cloned().map(|mut row| {
            for reference in [&mut row.source, &mut row.target] {
                project(reference, self.viewer);
            }
            row
        })
    }
    pub fn back(&self) -> Option<HistoryEvent> {
        self.iter().next_back()
    }
}

fn project(reference: &mut HistoryReference, viewer: PlayerId) {
    if let HistoryReference::Card {
        definition,
        seen,
        context_id,
        ..
    } = reference
    {
        if *seen & (1 << viewer.0) == 0 {
            *definition = None;
            *context_id = None;
        }
        // The reader receives no other seat's authorization metadata.
        *seen = 0;
    }
}

/// A possible world must not retain private evidence unavailable at its root.
pub(crate) fn scrub(game: &mut Game, perspective: PlayerId) {
    for row in Arc::make_mut(&mut game.policy_history) {
        for reference in [&mut row.source, &mut row.target] {
            if let HistoryReference::Card {
                definition,
                seen,
                context_id,
                ..
            } = reference
            {
                if *seen & (1 << perspective.0) == 0 {
                    *definition = None;
                    *context_id = None;
                    *seen = 0;
                }
            }
        }
    }
}

fn private_viewers(game: &Game, owner: PlayerId) -> u8 {
    (0..game.state.players.len())
        .filter(|&seat| game.shares_information(PlayerId(seat), owner))
        .fold(0, |mask, seat| mask | (1 << seat))
}

fn push(game: &mut Game, row: HistoryEvent) {
    #[cfg(feature = "benchmark")]
    measurement::record(&row);
    let window = Arc::make_mut(&mut game.policy_history);
    if window.len() == HISTORY_LIMIT {
        window.pop_front();
    }
    window.push_back(row);
}

/// Retain the inspected multiset, never the zone's hidden order or live slots.
/// The looker adds entitlement; owning a library never grants its visibility.
pub(crate) fn look(
    game: &mut Game,
    viewer: PlayerId,
    owner: PlayerId,
    zone: ZoneType,
    objects: &[ObjectId],
) {
    let seen = private_viewers(game, viewer)
        | if zone == ZoneType::Hand {
            private_viewers(game, owner)
        } else if public(zone) {
            3
        } else {
            0
        };
    let mut definitions: Vec<_> = objects
        .iter()
        .map(|&object| game.state[object].definition_id.0)
        .collect();
    definitions.sort_unstable();
    // An empty inspection still records who looked and whose zone it was.
    if definitions.is_empty() {
        push(
            game,
            HistoryEvent {
                kind: HistoryKind::Look,
                amount: 0,
                source: HistoryReference::Player(viewer),
                target: HistoryReference::Player(owner),
                from_zone: Some(zone),
                library_end: 0,
            },
        );
    }
    for definition in definitions {
        push(
            game,
            HistoryEvent {
                kind: HistoryKind::Look,
                amount: 1,
                source: HistoryReference::Player(viewer),
                target: HistoryReference::Card {
                    definition: Some(definition),
                    seen,
                    owner,
                    zone: Some(zone),
                    context_id: None,
                },
                from_zone: Some(zone),
                library_end: 0,
            },
        );
    }
}

pub(crate) fn shuffle(game: &mut Game, owner: PlayerId) {
    push(
        game,
        HistoryEvent {
            kind: HistoryKind::Shuffle,
            amount: 0,
            source: HistoryReference::Unavailable,
            target: HistoryReference::Player(owner),
            from_zone: None,
            library_end: 0,
        },
    );
}

pub(crate) fn declined(game: &mut Game, player: PlayerId, source: Option<ObjectId>) {
    let target = source_reference(game, source);
    push(
        game,
        HistoryEvent {
            kind: HistoryKind::Declined,
            amount: 0,
            source: HistoryReference::Player(player),
            target,
            from_zone: None,
            library_end: 0,
        },
    );
}

/// Amend this move's destination, or record an in-library reorder. Matching the
/// live incarnation prevents a later move of the same physical card being edited.
pub(crate) fn library_end(game: &mut Game, object: ObjectId, end: u8, moved: bool) {
    let id = game.state[object].id.0 as i32;
    if moved {
        if let Some(row) = Arc::make_mut(&mut game.policy_history)
            .iter_mut()
            .rev()
            .find(|row| {
                row.kind == HistoryKind::Moved
                    && matches!(row.source,
            HistoryReference::Card { context_id: Some(context), .. } if context == id)
            })
        {
            row.library_end = end;
            return;
        }
    }
    let data = &game.state[object];
    let source = HistoryReference::Card {
        definition: Some(data.definition_id.0),
        seen: private_viewers(game, data.owner),
        owner: data.owner,
        zone: Some(ZoneType::Library),
        context_id: Some(id),
    };
    push(
        game,
        HistoryEvent {
            kind: HistoryKind::Moved,
            amount: 0,
            source,
            target: HistoryReference::Unavailable,
            from_zone: Some(ZoneType::Library),
            library_end: end,
        },
    );
}

/// Chronological identities in a randomly ordered batch would disclose the
/// hidden library order, even when every member was public before placement.
/// Call after placing each card, before returning control to any observer.
pub(crate) fn conceal_random_placement(game: &mut Game, object: ObjectId) {
    let id = game.state[object].id.0 as i32;
    if let Some(row) = Arc::make_mut(&mut game.policy_history).iter_mut().rev().find(|row|
        row.kind == HistoryKind::Moved && matches!(row.source,
            HistoryReference::Card { context_id: Some(context), zone: Some(ZoneType::Library), .. } if context == id)) {
        if let HistoryReference::Card { seen, .. } = &mut row.source {
            *seen = 0;
        }
    }
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

fn object_reference(game: &Game, object: ObjectId) -> HistoryReference {
    let Some(zone) = game
        .state
        .zones
        .zone_of(object)
        .filter(|zone| public(*zone))
    else {
        return HistoryReference::Unavailable;
    };
    if game.is_foretold(object) {
        return HistoryReference::Unavailable;
    }
    let data = &game.state[object];
    HistoryReference::Card {
        definition: Some(data.definition_id.0),
        seen: 3,
        owner: data.owner,
        zone: Some(zone),
        context_id: Some(data.id.0 as i32),
    }
}

// Damage and counter-source payloads name a physical card, not the exact
// originating incarnation. A delayed effect must not route to a returned object.
// Retain only a definition currently public at emission; hidden sources and
// causal source zone/incarnation remain unavailable.
fn source_reference(game: &Game, source: Option<ObjectId>) -> HistoryReference {
    let mut reference = source
        .map(|object| object_reference(game, object))
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
        Target::StackSpell(object_id) => object_reference(game, object_id),
        Target::StackAbility(id) => game
            .stack_ability_source(id)
            .map(|object_id| object_reference(game, object_id))
            .unwrap_or(HistoryReference::Unavailable),
        Target::GraveyardCard(object) => object_reference(game, object),
        Target::Permanent(id) => game
            .permanent_object(id)
            .map(|exact| object_reference(game, exact))
            .unwrap_or(HistoryReference::Unavailable),
    }
}

/// Update the derived suffix at the same append that owns the committed ledger.
/// Work is bounded by 4*HISTORY_LIMIT reference comparisons plus the current
/// event's target count, independent of game length. Clones share
/// this copy-on-write projection; it never executes or reconstructs rules.
pub(crate) fn record(game: &mut Game, event: &GameEvent) {
    use HistoryKind::*;
    if let GameEvent::CardMoved {
        departed: Some(object),
        ..
    } = event
    {
        let id = game
            .state
            .objects
            .provenance(*object)
            .expect("departing object")
            .0
             .0 as i32;
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
    let mut from_zone = None;
    let mut library_end = 0;
    let (kind, amount, source, targets) = match event {
        GameEvent::CardMoved {
            object_id: object,
            from,
            to,
            face_down,
            departed,
            ..
        } => {
            let data = &game.state[*object];
            from_zone = *from;
            library_end = if *to == ZoneType::Library { 1 } else { 0 };
            let was_public = from.is_some_and(public)
                && !departed.is_some_and(|object| {
                    game.state
                        .cast_permissions
                        .exile_grants
                        .iter()
                        .any(|grant| grant.object == object && grant.foretold)
                });
            // Reveal-then-move instructions show the definition before the
            // hidden destination is committed. Do not consult future knowledge.
            let revealed = matches!(game.state.events.len().checked_sub(1).and_then(|index| game.state.events.get(index)), Some(GameEvent::CardRevealed { owner, definition })
                if *owner == data.owner && *definition == data.definition_id);
            let seen = if was_public || revealed || (public(*to) && !face_down) {
                3
            } else {
                private_viewers(game, data.owner)
            };
            let source = HistoryReference::Card {
                definition: Some(data.definition_id.0),
                seen,
                owner: data.owner,
                zone: Some(*to),
                context_id: Some(data.id.0 as i32),
            };
            (Moved, 0, source, vec![unknown])
        }
        GameEvent::CardDrawn { player, .. } => {
            if let Some(row) = Arc::make_mut(&mut game.policy_history).iter_mut().rev().find(|row|
                row.kind == Moved && row.from_zone == Some(ZoneType::Library) &&
                matches!(row.source, HistoryReference::Card { owner, zone: Some(ZoneType::Hand), .. } if owner == *player)) {
                #[cfg(feature = "benchmark")]
                measurement::draw();
                row.kind = Draw;
                row.amount = 1;
                row.target = HistoryReference::Player(*player);
            }
            return;
        }
        GameEvent::CardRevealed { owner, definition } => (
            Reveal,
            0,
            HistoryReference::Card {
                definition: Some(definition.0),
                seen: 3,
                owner: *owner,
                zone: None,
                context_id: None,
            },
            vec![unknown],
        ),
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
        GameEvent::SpellCast {
            object_id: object,
            targets,
        } => (
            Cast,
            0,
            object_reference(game, *object),
            if targets.is_empty() {
                vec![unknown]
            } else {
                targets
                    .iter()
                    .map(|target| target_reference(game, *target))
                    .collect()
            },
        ),
        GameEvent::SpellResolved { object_id: object } => {
            (Resolved, 0, object_reference(game, *object), vec![unknown])
        }
        GameEvent::SpellCountered {
            object_id: object,
            by,
        } => (
            Countered,
            0,
            source_reference(game, *by),
            vec![object_reference(game, *object)],
        ),
        // The countered ability is gone; its source card stands for it.
        GameEvent::AbilityCountered {
            source_object, by, ..
        } => (
            Countered,
            0,
            source_reference(game, *by),
            vec![source_reference(game, Some(*source_object))],
        ),
        _ => return,
    };
    for target in targets {
        push(
            game,
            HistoryEvent {
                kind,
                amount,
                source: source.clone(),
                target,
                from_zone,
                library_end,
            },
        );
    }
}

/// Viewer rows: kind, signed amount; two references each containing definition+1,
/// owner role (0 unknown, 1 viewer, 2 opponent), zone+1, context index (-1 absent),
/// availability (0 unknown, 1 exact current object/player, 2 definition only, 3 hidden).
/// A context is the zero-based object-table index of the object's one row
/// (`object_to_index`): a card, a permanent or a stack row.
pub fn encode(
    obs: &Observation,
    capacity: usize,
    object_to_index: &HashMap<i32, i32>,
) -> Vec<[f32; HISTORY_DIM]> {
    encode_rows(obs, capacity, |id| {
        object_to_index.get(&id).map(|index| *index as usize)
    })
}

/// Rows whose object contexts `context` resolves from an object id.
pub(crate) fn encode_rows(
    obs: &Observation,
    capacity: usize,
    context: impl Fn(i32) -> Option<usize>,
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
                ..
            } => {
                let context = context_id.and_then(&context);
                [
                    definition.map(|id| (id + 1) as f32).unwrap_or(0.0),
                    role(*owner),
                    zone.map(|zone| (zone as usize + 1) as f32).unwrap_or(0.0),
                    context.map(|v| v as f32).unwrap_or(-1.0),
                    if definition.is_none() {
                        3.0
                    } else if context.is_some() {
                        1.0
                    } else {
                        2.0
                    },
                ]
            }
        }
    };
    obs.policy_history
        .iter()
        .skip(obs.policy_history.len().saturating_sub(capacity))
        .map(|event| {
            let mut row = [0.0; HISTORY_DIM];
            row[history::KIND] = event.kind as u8 as f32;
            row[history::AMOUNT] = event.amount as f32;
            row[history::SOURCE_DEFINITION..history::TARGET_DEFINITION]
                .copy_from_slice(&reference(&event.source));
            row[history::TARGET_DEFINITION..history::FROM_ZONE]
                .copy_from_slice(&reference(&event.target));
            row[history::FROM_ZONE] =
                event.from_zone.map(|zone| zone as usize + 1).unwrap_or(0) as f32;
            row[history::LIBRARY_END] = event.library_end as f32;
            row
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{game_object::PublicId, player::PlayerConfig};

    /// The v11 encoder's valid history rows, at `capacity` rows.
    fn rows(obs: &Observation, capacity: usize) -> Vec<Vec<f32>> {
        let config = crate::agent::observation_encoder::ObservationEncoderConfig {
            policy_history: true,
            max_events: capacity,
            ..Default::default()
        };
        let encoded = crate::agent::observation_encoder::encode(obs, &config).unwrap();
        encoded
            .events
            .chunks(HISTORY_DIM)
            .zip(&encoded.events_valid)
            .filter(|(_, valid)| **valid > 0.0)
            .map(|(row, _)| row.to_vec())
            .collect()
    }

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
    fn policy_history_look_multiset_conceals_order_and_keeps_empty_count() {
        let mut root = game();
        let viewer = PlayerId(0);
        let owner = PlayerId(1);
        let inspected = root.library_top_objects(owner, 4);
        root.policy_history = Default::default();
        let mut other = root.clone();
        look(&mut root, viewer, owner, ZoneType::Library, &inspected);
        let reversed: Vec<_> = inspected.into_iter().rev().collect();
        look(&mut other, viewer, owner, ZoneType::Library, &reversed);
        for seat in [viewer, owner] {
            assert_eq!(
                rows(&Observation::for_player(&root, seat), 32),
                rows(&Observation::for_player(&other, seat), 32)
            );
        }
        look(&mut root, viewer, owner, ZoneType::Library, &[]);
        let empty = root.policy_history.back().unwrap();
        assert_eq!(empty.amount, 0);
        assert_eq!(empty.source, HistoryReference::Player(viewer));
        assert_eq!(empty.target, HistoryReference::Player(owner));
        let object = root.move_object(reversed[0], ZoneType::Hand);
        look(&mut root, viewer, owner, ZoneType::Hand, &[object]);
        for seat in [viewer, owner] {
            assert!(matches!(
                Observation::for_player(&root, seat)
                    .policy_history
                    .back()
                    .unwrap()
                    .target,
                HistoryReference::Card {
                    definition: Some(_),
                    ..
                }
            ));
        }
    }

    #[test]
    fn history_survives_drain_departure_reentry_and_hidden_reassignment() {
        let mut game = game();
        let objects: Vec<_> = game.state.zones.latest().collect();
        let (object, other) = (objects[0], objects[40]);
        let object = game.move_object(object, ZoneType::Stack);
        let other = game.move_object(other, ZoneType::Battlefield);
        let target = game.state.permanent_of(other).unwrap();
        game.emit(GameEvent::SpellCast {
            object_id: object,
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
        let expected = rows(&initial, 32);
        game.take_observation_events();
        assert_eq!(
            expected,
            rows(&Observation::for_player(&game, PlayerId(0)), 32)
        );
        // Consistent presentation-ID renaming cannot alter learned rows.
        for object in game.state.objects.iter_mut() {
            object.id = PublicId(object.id.0 + 1000);
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
            rows(&Observation::for_player(&game, PlayerId(0)), 32)
        );
        let object = game.move_object(object, ZoneType::Hand);
        let other = game.move_object(other, ZoneType::Graveyard);
        game.move_object(other, ZoneType::Battlefield);
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
        let before = rows(&projected, 32);
        // Future hidden placement cannot rewrite a previously public identity.
        game.state
            .place_object(object, PlayerId(0), ZoneType::Library);
        assert_eq!(
            before[..3],
            rows(&Observation::for_player(&game, PlayerId(0)), 32)[..3]
        );
        assert!(self::game().policy_history.is_empty());
    }

    #[test]
    fn policy_history_is_bounded_and_viewer_relative() {
        let mut game = game();
        // Private moves compete for the same history budget as public events.
        for index in 0..40 {
            game.emit(GameEvent::LifeChanged {
                player: PlayerId(1),
                old: 20,
                new: 20 - index,
            });
            let object = game.state.zones.latest().nth(index as usize).unwrap();
            game.move_object(object, ZoneType::Hand);
        }
        let first = Observation::for_player(&game, PlayerId(0));
        let second = Observation::for_player(&game, PlayerId(1));
        assert_eq!(first.policy_history.len(), HISTORY_LIMIT);
        let a: Vec<_> = rows(&first, 6)
            .into_iter()
            .filter(|row| row[0] == HistoryKind::Life as u8 as f32)
            .collect();
        let b: Vec<_> = rows(&second, 6)
            .into_iter()
            .filter(|row| row[0] == HistoryKind::Life as u8 as f32)
            .collect();
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
    fn observations_share_bounded_suffix_and_forks_isolate_it() {
        let mut game = game();
        game.create_token("token.ally", PlayerId(0), false);
        let token = game.state.zones.latest().last().unwrap();
        let permanent = game.state.permanent_of(token).unwrap();
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
            assert!(Arc::ptr_eq(&root, &obs.policy_history.window));
            assert_eq!(rows(&obs, 32).len(), 2);
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
        let mut fork = game.clone();
        assert!(Arc::ptr_eq(&root, &fork.policy_history));
        fork.move_object(token, ZoneType::Graveyard);
        assert_ne!(fork.policy_history, root);
        assert!(Arc::ptr_eq(&root, &game.policy_history));
    }
    #[test]
    fn policy_history_moves_reveal_and_fixed_visibility() {
        let mut game = game();
        let player = PlayerId(1);
        game.draw_cards(player, 1);
        let drawn = game
            .state
            .zones
            .zone_objects(ZoneType::Hand, player)
            .last()
            .copied()
            .unwrap();
        let definition = game.state[drawn].definition_id;
        game.emit(GameEvent::CardRevealed {
            owner: player,
            definition,
        });
        let obs = Observation::for_player(&game, PlayerId(0));
        let encoded = rows(&obs, 32);
        assert_eq!(encoded[0][0], HistoryKind::Draw as u8 as f32);
        assert_eq!(encoded[0][2], 0.0);
        assert_eq!(encoded[0][6], 3.0);
        assert_eq!(encoded[1][2], (definition.0 + 1) as f32);
        let grave = game.move_object(drawn, ZoneType::Graveyard);
        let field = game.move_object(grave, ZoneType::Battlefield);
        game.move_object(field, ZoneType::Hand);
        let obs = Observation::for_player(&game, PlayerId(0));
        let encoded = rows(&obs, 32);
        let bounced = encoded.last().unwrap();
        assert_eq!(bounced[2], (definition.0 + 1) as f32);
        assert_eq!(bounced[12], ZoneType::Battlefield as u8 as f32 + 1.0);
        assert_eq!(bounced[4], ZoneType::Hand as u8 as f32 + 1.0);
        assert_eq!(encoded[2][12], ZoneType::Hand as u8 as f32 + 1.0);
        let milled = game.state.zones.top(ZoneType::Library, player).unwrap();
        game.move_object(milled, ZoneType::Graveyard);
        let obs = Observation::for_player(&game, PlayerId(0));
        let encoded = rows(&obs, 32);
        assert_eq!(
            encoded.last().unwrap()[12],
            ZoneType::Library as u8 as f32 + 1.0
        );
        assert!(encoded.last().unwrap()[2] > 0.0);
    }

    #[test]
    fn policy_history_random_library_placement_never_discloses_order() {
        let mut root = game();
        let mut objects = Vec::new();
        for name in ["Mountain", "Lightning Bolt"] {
            let object = root
                .state
                .zones
                .zone_objects(ZoneType::Library, PlayerId(0))
                .iter()
                .copied()
                .find(|&object| root.state[object].name == name)
                .unwrap();
            objects.push(root.move_object(object, ZoneType::Exile));
        }
        let mut orders = std::collections::BTreeSet::new();
        let mut expected = None;
        for seed in 0..8 {
            let mut game = root.clone();
            game.reseed(seed);
            game.put_exiled_on_bottom_in_random_order(objects.clone());
            orders.insert(
                game.state
                    .zones
                    .zone_objects(ZoneType::Library, PlayerId(0))
                    .iter()
                    .take(2)
                    .map(|&object| game.state[object].definition_id)
                    .collect::<Vec<_>>(),
            );
            let projected: Vec<_> = [PlayerId(0), PlayerId(1)]
                .into_iter()
                .map(|viewer| rows(&Observation::for_player(&game, viewer), 32))
                .collect();
            for view in &projected {
                assert_eq!(view.len(), 4);
                for moved in &view[2..] {
                    assert_eq!(moved[2], 0.0);
                    assert_eq!(moved[6], 3.0);
                    assert_eq!(moved[13], 2.0);
                }
            }
            if let Some(expected) = &expected {
                assert_eq!(&projected, expected);
            } else {
                expected = Some(projected);
            }
        }
        assert_eq!(
            orders.len(),
            2,
            "fixture must actually vary the hidden order"
        );
    }

    #[test]
    fn physical_card_slots_are_only_routing() {
        let root = |source: usize, destination: usize| {
            let mut game = game();
            let objects: Vec<_> = game.state.zones.latest().collect();
            let (source, destination) = (objects[source], objects[destination]);
            game.scenario_clear_hand(PlayerId(0));
            game.scenario_clear_hand(PlayerId(1));
            let source =
                game.push_spell_to_stack(source, PlayerId(0), vec![], vec![], Default::default());
            let destination = game.move_object(destination, ZoneType::Battlefield);
            let target = game.state.permanent_of(destination).unwrap();
            game.emit(GameEvent::SpellCast {
                object_id: source,
                targets: vec![Target::Permanent(target)],
            });
            Observation::for_player(&game, PlayerId(0))
        };
        // Identical public definitions instantiated in different physical slots.
        let first = root(0, 40);
        let second = root(1, 41);
        let spell = |obs: &Observation| obs.stack_objects[0].spell.as_ref().unwrap().object_id;
        assert_ne!(spell(&first), spell(&second));
        assert_eq!(rows(&first, 32), rows(&second, 32));
    }
}

/// Opt-in accounting for the ordinary seeded `bench_steps` workload. It never
/// enters game state, hashes, observations, or release wheels.
#[cfg(feature = "benchmark")]
pub mod measurement {
    use super::{public, HistoryEvent, HistoryKind, HistoryReference, HISTORY_LIMIT};
    use std::{cell::RefCell, collections::VecDeque};

    #[derive(Default)]
    struct Counts {
        enabled: bool,
        rows: [u64; 11],
        retained: VecDeque<bool>,
        original_rows: usize,
        decisions: u64,
        displaced_decisions: u64,
        displaced_rows: u64,
    }
    thread_local! { static COUNTS: RefCell<Counts> = RefCell::new(Counts::default()); }

    pub fn reset() {
        COUNTS.with(|counts| {
            *counts.borrow_mut() = Counts {
                enabled: true,
                ..Default::default()
            }
        });
    }
    pub fn start_game() {
        COUNTS.with(|counts| {
            let mut counts = counts.borrow_mut();
            counts.retained.clear();
            counts.original_rows = 0;
        });
    }
    pub(super) fn record(row: &HistoryEvent) {
        COUNTS.with(|counts| {
            let mut counts = counts.borrow_mut();
            if !counts.enabled { return; }
            counts.rows[row.kind as usize] += 1;
            let original = match row.kind {
                HistoryKind::Moved => matches!(row.source, HistoryReference::Card { zone: Some(zone), .. } if public(zone)),
                HistoryKind::Damage | HistoryKind::Life | HistoryKind::Cast |
                HistoryKind::Resolved | HistoryKind::Countered => true,
                HistoryKind::Draw | HistoryKind::Reveal | HistoryKind::Look |
                HistoryKind::Shuffle | HistoryKind::Declined => false,
            };
            counts.original_rows = (counts.original_rows + usize::from(original)).min(HISTORY_LIMIT);
            if counts.retained.len() == HISTORY_LIMIT { counts.retained.pop_front(); }
            counts.retained.push_back(original);
        });
    }
    pub(super) fn draw() {
        COUNTS.with(|counts| {
            let mut counts = counts.borrow_mut();
            if !counts.enabled {
                return;
            }
            counts.rows[HistoryKind::Moved as usize] -= 1;
            counts.rows[HistoryKind::Draw as usize] += 1;
        });
    }
    pub fn decision() {
        COUNTS.with(|counts| {
            let mut counts = counts.borrow_mut();
            let retained = counts.retained.iter().filter(|&&old| old).count();
            let displaced = counts.original_rows - retained;
            counts.decisions += 1;
            counts.displaced_decisions += u64::from(displaced > 0);
            counts.displaced_rows += displaced as u64;
        });
    }
    pub fn snapshot(games: u64) -> serde_json::Value {
        COUNTS.with(|counts| {
            let counts = counts.borrow();
            serde_json::json!({
                "games": games, "decisions": counts.decisions,
                "kind_order": ["moved", "damage", "life", "cast", "resolved", "countered", "draw", "reveal", "look", "shuffle", "declined"],
                "mean_rows_per_game": counts.rows.map(|count| count as f64 / games.max(1) as f64),
                "displaced_decision_share": counts.displaced_decisions as f64 / counts.decisions.max(1) as f64,
                "mean_displaced_rows_per_decision": counts.displaced_rows as f64 / counts.decisions.max(1) as f64,
            })
        })
    }
}
