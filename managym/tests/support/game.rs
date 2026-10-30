// game.rs
// Build a test game: decks, the `Scenario` wrapper, and silent or
// trigger-firing placement of cards into hand, library, battlefield and
// graveyard.
//
// Placement helpers differ in two ways that tests rely on, so each variant has
// its own name: which copy they find (any zone, library only, hand or library,
// or off the battlefield), and whether they move through the game's zone-change
// authority (firing enter triggers) or edit the zone tables silently.

use std::collections::BTreeMap;

use managym::{
    flow::turn::StepKind,
    state::{
        ability::Ability,
        effect::{Affected, EffectDuration, Modification},
        game_object::{CardId, PermanentId, PlayerId},
        mana::Mana,
        permanent::{CounterKind, Permanent},
        player::PlayerConfig,
        zone::ZoneType,
    },
    Game,
};

use crate::support::inspect::{
    find_owned_card, find_owned_card_off_battlefield, library_card, owned_card, permanent_named,
};

// ---------------------------------------------------------------------------
// Decks
// ---------------------------------------------------------------------------

pub fn deck(entries: &[(&str, usize)]) -> BTreeMap<String, usize> {
    entries
        .iter()
        .map(|(name, qty)| ((*name).to_string(), *qty))
        .collect()
}

/// The one-card deck: 24 basic lands and 16 copies of `card`.
pub fn deck_of(land: &str, card: &str) -> BTreeMap<String, usize> {
    deck(&[(land, 24), (card, 16)])
}

// ---------------------------------------------------------------------------
// Scenario
// ---------------------------------------------------------------------------

/// A two-player game driven by tests. Construction and setup live here;
/// directed play is in `play.rs` and reads are in `inspect.rs`.
pub struct Scenario {
    pub(super) game: Game,
}

impl Scenario {
    pub fn new(
        player0_deck: BTreeMap<String, usize>,
        player1_deck: BTreeMap<String, usize>,
        seed: u64,
    ) -> Scenario {
        let p0 = PlayerConfig::new("p0", player0_deck);
        let p1 = PlayerConfig::new("p1", player1_deck);
        Scenario {
            game: Game::new(vec![p0, p1], seed, false),
        }
    }

    pub fn with_sideboards(
        player0_deck: BTreeMap<String, usize>,
        player1_deck: BTreeMap<String, usize>,
        player0_sideboard: BTreeMap<String, usize>,
        player1_sideboard: BTreeMap<String, usize>,
        seed: u64,
    ) -> Self {
        Self {
            game: Game::new(
                vec![
                    PlayerConfig::new("p0", player0_deck).with_sideboard(player0_sideboard),
                    PlayerConfig::new("p1", player1_deck).with_sideboard(player1_sideboard),
                ],
                seed,
                false,
            ),
        }
    }

    pub fn game(&self) -> &Game {
        &self.game
    }

    pub fn game_mut(&mut self) -> &mut Game {
        &mut self.game
    }

    /// Silently move the first owned card with this name, from any zone, into
    /// the hand. Calling it twice finds the same copy.
    pub fn force_card_in_hand(&mut self, player: usize, card_name: &str) {
        let card = owned_card(self, player, card_name);
        self.game
            .state
            .zones
            .move_card(card, PlayerId(player), ZoneType::Hand);
    }

    pub fn force_cards_in_hand(&mut self, player: usize, card_name: &str, count: usize) {
        for _ in 0..count {
            self.force_card_in_hand(player, card_name);
        }
    }

    /// Silently put a copy onto the battlefield without enter triggers. Prefers a
    /// copy not yet on the battlefield; returns an existing permanent only when
    /// every copy is already there.
    pub fn force_permanent_on_battlefield(
        &mut self,
        player: usize,
        card_name: &str,
    ) -> PermanentId {
        let mut chosen_card_id = None;
        let mut existing_battlefield_permanent = None;
        for (index, card) in self.game.state.cards.iter().enumerate() {
            if card.owner != PlayerId(player) || card.name != card_name {
                continue;
            }
            let card_id = CardId(index);
            if self.game.state.zones.zone_of(card_id) == Some(ZoneType::Battlefield) {
                if let Some(permanent_id) = self.game.state.card_to_permanent[card_id] {
                    if existing_battlefield_permanent.is_none() {
                        existing_battlefield_permanent = Some(permanent_id);
                    }
                    continue;
                }
            }
            chosen_card_id = Some(card_id);
            break;
        }

        let Some(card_id) = chosen_card_id else {
            if let Some(permanent_id) = existing_battlefield_permanent {
                return permanent_id;
            }
            panic!("card {card_name} not found for player {player}");
        };

        if self.game.state.zones.zone_of(card_id) == Some(ZoneType::Battlefield) {
            if let Some(permanent_id) = self.game.state.card_to_permanent[card_id] {
                return permanent_id;
            }
        } else {
            self.game
                .state
                .zones
                .move_card(card_id, PlayerId(player), ZoneType::Battlefield);
        }

        let permanent_id = PermanentId(self.game.state.permanents.len());
        let permanent = Permanent::new(
            self.game.state.id_gen.next_id(),
            card_id,
            &self.game.state.cards[card_id],
        );
        self.game.state.permanents.push(Some(permanent));
        if self.game.state.card_to_permanent.len() <= card_id.0 {
            self.game
                .state
                .card_to_permanent
                .resize(card_id.0 + 1, None);
        }
        self.game.state.card_to_permanent[card_id] = Some(permanent_id);
        permanent_id
    }

    pub fn set_player_mana_pool(&mut self, player: usize, mana: &str) {
        self.game.state.players[player].mana_pool = Mana::parse(mana);
    }
}

// ---------------------------------------------------------------------------
// Prepared games
// ---------------------------------------------------------------------------

/// Both players hold 16 of each basic plus 4 of each named card (seed 161).
/// Advances to player 0's main phase, empties both hands, and gives each
/// player twenty generic and four of each colored mana.
pub fn five_color_setup(names: &[&str]) -> Scenario {
    let mut s = five_color_unfunded_setup(names);
    s.set_player_mana_pool(0, "20WWWWUUUUBBBBRRRRGGGG");
    s.set_player_mana_pool(1, "20WWWWUUUUBBBBRRRRGGGG");
    s
}

/// [`five_color_setup`] without floating mana: tests pay costs from lands
/// they play.
pub fn five_color_unfunded_setup(names: &[&str]) -> Scenario {
    let mut deck: BTreeMap<String, usize> = ["Plains", "Island", "Swamp", "Mountain", "Forest"]
        .into_iter()
        .map(|name| (name.into(), 16))
        .collect();
    for name in names {
        deck.insert((*name).into(), 4);
    }
    let mut s = Scenario::new(deck.clone(), deck, 161);
    s.advance_to_active_step(0, StepKind::Main);
    s.game_mut().scenario_clear_hand(PlayerId(0));
    s.game_mut().scenario_clear_hand(PlayerId(1));
    s
}

// ---------------------------------------------------------------------------
// Hand, library and graveyard
// ---------------------------------------------------------------------------

/// Move every card in a player's hand back to their library (test setup —
/// removes castable noise so `PriorityCastSpell` is unambiguous).
pub fn clear_hand(s: &mut Scenario, player: usize) {
    let hand: Vec<CardId> = s
        .game()
        .state
        .zones
        .zone_cards(ZoneType::Hand, PlayerId(player))
        .clone();
    for card in hand {
        s.game_mut()
            .state
            .zones
            .move_card(card, PlayerId(player), ZoneType::Library);
    }
}

/// Move player 0's library copy into the hand through the zone-change
/// authority and refresh priority.
pub fn move_to_hand(s: &mut Scenario, name: &str) -> CardId {
    let id = library_card(s, 0, name);
    s.game_mut().move_card(id, ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    id
}

/// Put a fresh library copy into the hand — `force_card_in_hand` re-finds the
/// same copy when called twice. Copies are searched in card-id order.
pub fn force_new_card_in_hand(s: &mut Scenario, player: usize, card_name: &str) -> CardId {
    let card = s
        .game()
        .state
        .cards
        .iter()
        .enumerate()
        .find_map(|(index, card)| {
            (card.owner == PlayerId(player)
                && card.name == card_name
                && s.game().state.zones.zone_of(CardId(index)) == Some(ZoneType::Library))
            .then_some(CardId(index))
        })
        .unwrap_or_else(|| panic!("no library copy of {card_name} for player {player}"));
    s.game_mut()
        .state
        .zones
        .move_card(card, PlayerId(player), ZoneType::Hand);
    card
}

/// Move a library card into the hand, searching in library order (avoids
/// colliding with same-named cards already on the battlefield or in hand).
pub fn force_from_library(s: &mut Scenario, player: usize, name: &str) -> CardId {
    let game = s.game();
    let card_id = game
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(player))
        .iter()
        .copied()
        .find(|card_id| game.state.cards[card_id].name == name)
        .unwrap_or_else(|| panic!("no {name} in player {player}'s library"));
    s.game_mut()
        .state
        .zones
        .move_card(card_id, PlayerId(player), ZoneType::Hand);
    card_id
}

/// Arrange a known prefix on top of player 0's library without generating
/// unrelated zone-change events. Returns the pinned cards, first on top.
pub fn pin_top(s: &mut Scenario, names: &[&str]) -> Vec<CardId> {
    let mut cards = Vec::new();
    for name in names {
        let next = s
            .game()
            .state
            .zones
            .zone_cards(ZoneType::Library, PlayerId(0))
            .iter()
            .copied()
            .find(|id| s.game().state.cards[*id].name == *name && !cards.contains(id))
            .unwrap();
        cards.push(next);
    }
    let library = s
        .game_mut()
        .state
        .zones
        .zone_cards_mut(ZoneType::Library, PlayerId(0));
    library.retain(|id| !cards.contains(id));
    library.extend(cards.iter().rev());
    cards
}

/// Move player `player`'s library copy of `name` to `zone` through the
/// zone-change authority (firing any enter or leave triggers).
pub fn move_from_library(s: &mut Scenario, player: usize, name: &str, zone: ZoneType) -> CardId {
    let card = library_card(s, player, name);
    s.game_mut().move_card(card, zone);
    card
}

/// Silently move an owned hand or library card into the graveyard (e.g.
/// planting a Lesson).
pub fn plant_in_graveyard(s: &mut Scenario, player: usize, card_name: &str) -> CardId {
    let card = find_owned_card(s, player, card_name);
    s.game_mut()
        .state
        .zones
        .move_card(card, PlayerId(player), ZoneType::Graveyard);
    card
}

// ---------------------------------------------------------------------------
// Battlefield
// ---------------------------------------------------------------------------

pub fn force_lands(s: &mut Scenario, player: usize, land: &str, count: usize) {
    for _ in 0..count {
        s.force_permanent_on_battlefield(player, land);
    }
}

/// Put player 0's library copy onto the battlefield through the zone-change
/// authority (enter triggers fire) and refresh priority.
pub fn enter_battlefield(s: &mut Scenario, name: &str) -> PermanentId {
    let id = library_card(s, 0, name);
    s.game_mut().move_card(id, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    s.game().state.card_to_permanent[id].unwrap()
}

/// Put an owned hand or library copy onto the battlefield through the
/// zone-change authority, without refreshing priority.
pub fn put_onto_battlefield(s: &mut Scenario, player: usize, name: &str) -> PermanentId {
    let card = find_owned_card(s, player, name);
    s.game_mut().move_card(card, ZoneType::Battlefield);
    s.game().state.card_to_permanent[card].unwrap()
}

/// Move an off-battlefield copy onto the battlefield through the real
/// zone-change path so its enters-the-battlefield triggers fire.
pub fn enters_battlefield_with_triggers(
    s: &mut Scenario,
    player: usize,
    name: &str,
) -> PermanentId {
    let card_id = find_owned_card_off_battlefield(s, player, name);
    s.game_mut().move_card(card_id, ZoneType::Battlefield);
    s.game().state.card_to_permanent[card_id].expect("permanent should exist")
}

pub fn clear_summoning_sickness(s: &mut Scenario, id: PermanentId) {
    s.game_mut().state.permanents[id]
        .as_mut()
        .expect("permanent")
        .summoning_sick = false;
}

/// Clear summoning sickness on player 0's first permanent with this name and
/// refresh priority so its tap abilities are offered.
pub fn ready(s: &mut Scenario, name: &str) {
    let source = permanent_named(s, name);
    s.game_mut().state.permanents[source]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    s.game_mut().scenario_refresh_priority().unwrap();
}

/// Record an until-end-of-turn continuous effect on a permanent, as a
/// resolving spell of player 0 would.
pub fn until_end_of_turn(s: &mut Scenario, id: PermanentId, modification: Modification) {
    let affected = s
        .game()
        .affected_permanent(id)
        .expect("permanent on the battlefield");
    s.game_mut().apply_continuous(
        None,
        PlayerId(0),
        [affected],
        modification,
        EffectDuration::EndOfTurn,
    );
}

/// Record a continuous effect on a player.
pub fn player_effect(
    s: &mut Scenario,
    player: usize,
    modification: Modification,
    duration: EffectDuration,
) {
    s.game_mut().apply_continuous(
        None,
        PlayerId(player),
        [Affected::Player(PlayerId(player))],
        modification,
        duration,
    );
}

/// Set a permanent's counters of one kind, without events (scenario setup).
pub fn set_counters(s: &mut Scenario, id: PermanentId, kind: CounterKind, count: i32) {
    let permanent = s.game_mut().state.permanents[id]
        .as_mut()
        .expect("permanent on the battlefield");
    let current = permanent.counter_count(kind);
    permanent.change_counters(kind, count - current);
}

pub fn inject_ability(s: &mut Scenario, card_id: CardId, ability: Ability) {
    s.game_mut().state.cards[card_id].abilities.push(ability);
}

/// The UR Lessons list both players use for Gran-Gran, Proft's Eidetic Memory
/// and Combustion Technique tests.
pub fn ur_lessons_deck() -> BTreeMap<String, usize> {
    [
        ("Island", 16),
        ("Mountain", 8),
        ("Gran-Gran", 3),
        ("Proft's Eidetic Memory", 3),
        ("Combustion Technique", 4),
        ("Accumulate Wisdom", 4),
        ("Pop Quiz", 2),
        ("Otter-Penguin", 2),
        ("Water Tribe Rallier", 2),
        ("Lightning Bolt", 2),
    ]
    .into_iter()
    .map(|(name, n)| (name.into(), n))
    .collect()
}

/// Both players on `ur_lessons_deck` (seed 88), at player 0's main phase.
pub fn ur_lessons_setup() -> Scenario {
    let mut s = Scenario::new(ur_lessons_deck(), ur_lessons_deck(), 88);
    s.advance_to_active_step(0, StepKind::Main);
    s
}
