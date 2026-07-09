// tla_cards.rs
// Trace tests for the Stage-1 TLA proof cards and the trigger substrate they
// stand on: trigger sources, +1/+1 counters, tokens, type predicates, flash,
// and conditional unblockability.

use std::collections::BTreeMap;

use managym::{
    agent::{
        action::{Action, ActionSpaceKind, ActionType},
        observation::Observation,
    },
    flow::turn::StepKind,
    state::{
        ability::{Ability, Effect, TriggerCondition, TriggerSubject},
        card::CardType,
        game_object::{CardId, PermanentId, PlayerId},
        predicate::CardPredicate,
        zone::ZoneType,
    },
};

use super::helpers::*;

fn deck(entries: &[(&str, usize)]) -> BTreeMap<String, usize> {
    entries
        .iter()
        .map(|(name, qty)| ((*name).to_string(), *qty))
        .collect()
}

fn find_owned_card_off_battlefield(s: &Scenario, player: usize, name: &str) -> CardId {
    let game = s.game();
    for (index, card) in game.state.cards.iter().enumerate() {
        if card.owner != PlayerId(player) || card.name != name {
            continue;
        }
        let card_id = CardId(index);
        if game.state.zones.zone_of(card_id) != Some(ZoneType::Battlefield) {
            return card_id;
        }
    }
    panic!("no off-battlefield {name} found for player {player}");
}

/// Move a card onto the battlefield through the real zone-change path so its
/// enters-the-battlefield triggers fire.
fn enters_battlefield_with_triggers(s: &mut Scenario, player: usize, name: &str) -> PermanentId {
    let card_id = find_owned_card_off_battlefield(s, player, name);
    s.game_mut().move_card(card_id, ZoneType::Battlefield);
    s.game().state.card_to_permanent[card_id].expect("permanent should exist")
}

fn permanent(s: &Scenario, id: PermanentId) -> &managym::state::permanent::Permanent {
    s.game().state.permanents[id]
        .as_ref()
        .expect("permanent should exist")
}

fn counters(s: &Scenario, id: PermanentId) -> i32 {
    permanent(s, id).plus1_counters
}

/// Advance with default actions until the stack, pending events, and pending
/// triggers are empty (all triggered abilities have resolved).
fn resolve_all(s: &mut Scenario) {
    s.advance_until(
        |s| {
            s.game().state.stack_objects.is_empty()
                && s.game().state.pending_events.is_empty()
                && s.game().state.pending_triggers.is_empty()
                && s.action_space().kind == ActionSpaceKind::Priority
        },
        "stack should empty out".to_string(),
    );
}

/// Move every card in a player's hand back to their library (test setup —
/// removes castable noise so `PriorityCastSpell` is unambiguous).
fn clear_hand(s: &mut Scenario, player: usize) {
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

/// Move a specific card from the library into the hand (avoids colliding
/// with same-named cards already on the battlefield or in hand).
fn force_from_library(s: &mut Scenario, player: usize, name: &str) -> CardId {
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

fn force_lands(s: &mut Scenario, player: usize, land: &str, count: usize) {
    for _ in 0..count {
        s.force_permanent_on_battlefield(player, land);
    }
}

fn cast_and_resolve(s: &mut Scenario) {
    assert!(
        s.take_action_by_type(ActionType::PriorityCastSpell),
        "cast action should be available"
    );
    s.pass_priority();
    s.pass_priority();
}

// ---------------------------------------------------------------------------
// Kyoshi Warriors — ETB token creation, trigger uses the stack with responses.
// ---------------------------------------------------------------------------

/// Kyoshi Warriors: "When this creature enters, create a 1/1 white Ally
/// creature token." The targetless trigger still uses the stack and both
/// players receive priority before it resolves (CR 603.3, 117.3d).
#[test]
fn kyoshi_warriors_etb_creates_ally_token_via_stack() {
    let mut s = Scenario::new(
        deck(&[("Plains", 24), ("Kyoshi Warriors", 16)]),
        deck(&[("Plains", 40)]),
        7,
    );
    force_lands(&mut s, 0, "Plains", 4);
    s.force_card_in_hand(0, "Kyoshi Warriors");
    s.advance_to_active_step(0, StepKind::Main);

    cast_and_resolve(&mut s);

    // The ETB trigger is on the stack; both players get a response window.
    assert_eq!(s.game().state.stack_objects.len(), 1);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    s.pass_priority();
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    assert_eq!(s.game().state.stack_objects.len(), 1);
    s.pass_priority();

    let tokens = s.battlefield_permanents_named(0, "Ally");
    assert_eq!(tokens.len(), 1, "one Ally token should have been created");
    let token = permanent(&s, tokens[0]);
    let token_card = &s.game().state.cards[token.card];
    assert!(token_card.is_token);
    assert!(token_card.has_subtype("Ally"));
    assert!(token_card.types.is_creature());
    assert_eq!(token_card.power, Some(1));
    assert_eq!(token_card.toughness, Some(1));
    assert_eq!(
        token_card.colors,
        managym::state::mana::Colors::from([managym::state::mana::Color::White])
    );
    assert_eq!(token.controller, PlayerId(0));
}

// ---------------------------------------------------------------------------
// Avatar Enthusiasts — "another Ally you control enters" filtering.
// ---------------------------------------------------------------------------

fn enthusiasts_scenario() -> (Scenario, PermanentId) {
    let list = deck(&[
        ("Plains", 20),
        ("Avatar Enthusiasts", 8),
        ("South Pole Voyager", 8),
        ("Grey Ogre", 4),
    ]);
    let mut s = Scenario::new(list.clone(), list, 11);
    let enthusiasts = s.force_permanent_on_battlefield(0, "Avatar Enthusiasts");
    (s, enthusiasts)
}

/// Avatar Enthusiasts gets a +1/+1 counter when another Ally enters under
/// its controller's control...
#[test]
fn avatar_enthusiasts_counter_when_another_ally_enters() {
    let (mut s, enthusiasts) = enthusiasts_scenario();
    assert_eq!(counters(&s, enthusiasts), 0);

    enters_battlefield_with_triggers(&mut s, 0, "South Pole Voyager");
    resolve_all(&mut s);

    assert_eq!(counters(&s, enthusiasts), 1);
    // Counters contribute to P/T: 2/2 base + 1 counter = 3/3.
    assert_eq!(s.game().effective_power(enthusiasts), 3);
    assert_eq!(s.game().effective_toughness(enthusiasts), 3);
}

/// ...but NOT when a non-Ally creature enters...
#[test]
fn avatar_enthusiasts_no_counter_for_non_ally() {
    let (mut s, enthusiasts) = enthusiasts_scenario();

    enters_battlefield_with_triggers(&mut s, 0, "Grey Ogre");
    resolve_all(&mut s);

    assert_eq!(counters(&s, enthusiasts), 0);
}

/// ...nor when an opponent's Ally enters...
#[test]
fn avatar_enthusiasts_no_counter_for_opponents_ally() {
    let (mut s, enthusiasts) = enthusiasts_scenario();

    enters_battlefield_with_triggers(&mut s, 1, "South Pole Voyager");
    resolve_all(&mut s);

    assert_eq!(counters(&s, enthusiasts), 0);
}

/// ...nor for itself entering ("another").
#[test]
fn avatar_enthusiasts_does_not_count_itself() {
    let list = deck(&[("Plains", 24), ("Avatar Enthusiasts", 16)]);
    let mut s = Scenario::new(list.clone(), list, 12);

    let entered = enters_battlefield_with_triggers(&mut s, 0, "Avatar Enthusiasts");
    resolve_all(&mut s);

    assert_eq!(counters(&s, entered), 0);
}

/// A created Ally token is "another Ally you control entering" too.
#[test]
fn avatar_enthusiasts_counts_token_allies() {
    let (mut s, enthusiasts) = enthusiasts_scenario();

    s.game_mut().create_token("Ally", PlayerId(0), false);
    resolve_all(&mut s);

    assert_eq!(counters(&s, enthusiasts), 1);
}

// ---------------------------------------------------------------------------
// Invasion Reinforcements — flash.
// ---------------------------------------------------------------------------

/// Invasion Reinforcements has flash: castable during the opponent's turn,
/// and its ETB still makes an Ally token.
#[test]
fn invasion_reinforcements_flash_cast_on_opponents_turn() {
    let mut s = Scenario::new(
        deck(&[("Plains", 24), ("Invasion Reinforcements", 16)]),
        deck(&[("Island", 40)]),
        13,
    );
    force_lands(&mut s, 0, "Plains", 2);
    s.force_card_in_hand(0, "Invasion Reinforcements");

    // Opponent's turn, opponent holds priority first; pass to player 0.
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    s.pass_priority();
    assert_eq!(s.action_space().player, Some(PlayerId(0)));

    s.assert_action_available(ActionType::PriorityCastSpell);
    cast_and_resolve(&mut s);
    resolve_all(&mut s);

    assert_eq!(s.battlefield_permanents_named(0, "Invasion Reinforcements").len(), 1);
    assert_eq!(s.battlefield_permanents_named(0, "Ally").len(), 1);
}

// ---------------------------------------------------------------------------
// Jeong Jeong's Deserters — targeted +1/+1 counter, SBA interaction.
// ---------------------------------------------------------------------------

/// Jeong Jeong's Deserters puts a +1/+1 counter on target creature; the
/// counter raises toughness for the lethal-damage check (CR 704.5g).
#[test]
fn jeong_jeongs_deserters_targeted_counter_and_lethal_sba() {
    let mut s = Scenario::new(
        deck(&[("Plains", 24), ("Jeong Jeong's Deserters", 16)]),
        deck(&[("Plains", 40)]),
        17,
    );
    force_lands(&mut s, 0, "Plains", 2);
    s.force_card_in_hand(0, "Jeong Jeong's Deserters");
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);

    // The ETB trigger needs a target.
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    s.choose_target_named("Jeong Jeong's Deserters");
    resolve_all(&mut s);

    let deserters = s.battlefield_permanents_named(0, "Jeong Jeong's Deserters")[0];
    assert_eq!(counters(&s, deserters), 1);

    // 1/2 with a +1/+1 counter is 2/3: 2 damage is not lethal...
    s.game_mut().state.permanents[deserters]
        .as_mut()
        .unwrap()
        .damage = 2;
    s.pass_priority();
    assert!(!s.battlefield_permanents_named(0, "Jeong Jeong's Deserters").is_empty());

    // ...but 3 damage is.
    s.game_mut().state.permanents[deserters]
        .as_mut()
        .unwrap()
        .damage = 3;
    s.pass_priority();
    assert!(s.battlefield_permanents_named(0, "Jeong Jeong's Deserters").is_empty());
    assert_eq!(
        s.game()
            .state
            .zones
            .zone_cards(ZoneType::Graveyard, PlayerId(0))
            .len(),
        1
    );
}

// ---------------------------------------------------------------------------
// Tiger-Seal — upkeep trigger, second-card-drawn trigger, per-turn reset.
// ---------------------------------------------------------------------------

#[test]
fn tiger_seal_taps_at_upkeep_and_untaps_on_second_draw() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Tiger-Seal", 16)]),
        deck(&[("Island", 40)]),
        19,
    );
    force_lands(&mut s, 0, "Island", 1);
    s.force_card_in_hand(0, "Tiger-Seal");
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);
    let seal = s.battlefield_permanents_named(0, "Tiger-Seal")[0];
    assert!(!permanent(&s, seal).tapped);

    // Next upkeep of its controller: "At the beginning of your upkeep, tap
    // this creature." (It does not fire on the opponent's upkeep.)
    s.advance_to_active_step(1, StepKind::Draw);
    assert!(!permanent(&s, seal).tapped, "opponent's upkeep must not tap it");
    s.advance_to_active_step(0, StepKind::Draw);
    assert!(
        permanent(&s, seal).tapped,
        "should be tapped by its upkeep trigger"
    );

    // The draw-step draw was this turn's first card; the second draw untaps.
    assert_eq!(s.game().state.turn.cards_drawn_this_turn[0], 1);
    s.game_mut().draw_cards(PlayerId(0), 1);
    resolve_all(&mut s);
    assert!(!permanent(&s, seal).tapped, "second draw should untap");

    // A third draw the same turn does not trigger again (it stays untapped,
    // and no new trigger goes on the stack).
    s.game_mut().draw_cards(PlayerId(0), 1);
    s.pass_priority();
    assert!(s.game().state.stack_objects.is_empty());
    assert!(s.game().state.pending_triggers.is_empty());
}

// ---------------------------------------------------------------------------
// Otter-Penguin — second-draw buff + unblockable, resets next turn.
// ---------------------------------------------------------------------------

#[test]
fn otter_penguin_buffs_on_second_draw_only_and_resets() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Otter-Penguin", 16)]),
        deck(&[("Island", 40)]),
        23,
    );
    let otter = s.force_permanent_on_battlefield(0, "Otter-Penguin");
    // Player 0 skips their draw on turn 1; run the test from their second
    // turn, where the draw step provides the first card of the turn.
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);

    // Draw-step draw was the first card this turn — no buff yet.
    assert_eq!(s.game().state.turn.cards_drawn_this_turn[0], 1);
    assert_eq!(permanent(&s, otter).temp_power, 0);

    // Second draw: +1/+2 and can't be blocked this turn.
    s.game_mut().draw_cards(PlayerId(0), 1);
    resolve_all(&mut s);
    assert_eq!(permanent(&s, otter).temp_power, 1);
    assert_eq!(permanent(&s, otter).temp_toughness, 2);
    assert!(permanent(&s, otter).cant_be_blocked_this_turn);

    // Third draw: no further trigger.
    s.game_mut().draw_cards(PlayerId(0), 1);
    resolve_all(&mut s);
    assert_eq!(permanent(&s, otter).temp_power, 1);

    // Cleanup clears the until-EOT buff and the unblockable flag; the
    // per-turn draw count resets.
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(permanent(&s, otter).temp_power, 0);
    assert_eq!(permanent(&s, otter).temp_toughness, 0);
    assert!(!permanent(&s, otter).cant_be_blocked_this_turn);
    assert_eq!(s.game().state.turn.cards_drawn_this_turn[0], 0);

    // Next turn the "second card each turn" trigger works again.
    s.advance_to_active_step(0, StepKind::Main);
    s.game_mut().draw_cards(PlayerId(0), 1);
    resolve_all(&mut s);
    assert_eq!(permanent(&s, otter).temp_power, 1);
}

/// The opponent drawing their second card must not trigger player 0's
/// Otter-Penguin.
#[test]
fn otter_penguin_ignores_opponent_draws() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Otter-Penguin", 16)]),
        deck(&[("Island", 40)]),
        29,
    );
    let otter = s.force_permanent_on_battlefield(0, "Otter-Penguin");
    s.advance_to_active_step(1, StepKind::Main);

    s.game_mut().draw_cards(PlayerId(1), 1);
    resolve_all(&mut s);
    assert_eq!(s.game().state.turn.cards_drawn_this_turn[1], 2);
    assert_eq!(permanent(&s, otter).temp_power, 0);
}

// ---------------------------------------------------------------------------
// Forecasting Fortune Teller — Clue token, sacrifice-to-draw, token SBA.
// ---------------------------------------------------------------------------

#[test]
fn fortune_teller_clue_sacrifices_to_draw_and_ceases_to_exist() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Forecasting Fortune Teller", 16)]),
        deck(&[("Island", 40)]),
        31,
    );
    force_lands(&mut s, 0, "Island", 4);
    s.force_card_in_hand(0, "Forecasting Fortune Teller");
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);
    resolve_all(&mut s);

    let clues = s.battlefield_permanents_named(0, "Clue");
    assert_eq!(clues.len(), 1, "Clue token should be created");
    let clue_card = permanent(&s, clues[0]).card;
    assert!(s.game().state.cards[clue_card].is_token);
    assert!(s.game().state.cards[clue_card].types.is_artifact());

    // "{2}, Sacrifice this token: Draw a card."
    let hand_before = s.zone_size(0, ZoneType::Hand);
    s.assert_action_available(ActionType::PriorityActivateAbility);
    assert!(s.take_action_by_type(ActionType::PriorityActivateAbility));
    // The sacrifice is paid on activation: the Clue is already gone.
    assert!(s.battlefield_permanents_named(0, "Clue").is_empty());
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand_before + 1);

    // CR 704.5d — the token ceased to exist instead of staying in a zone.
    assert_eq!(s.game().state.zones.zone_of(clue_card), None);
    assert_eq!(s.zone_size(0, ZoneType::Graveyard), 0);
}

/// A creature token that dies goes to the graveyard (firing zone-change
/// events) and then ceases to exist via state-based actions.
#[test]
fn creature_token_ceases_to_exist_after_dying() {
    let mut s = Scenario::new(
        deck(&[("Plains", 40)]),
        deck(&[("Plains", 40)]),
        37,
    );
    s.game_mut().create_token("Ally", PlayerId(0), false);
    resolve_all(&mut s);
    let token = s.battlefield_permanents_named(0, "Ally")[0];
    let token_card = permanent(&s, token).card;

    s.game_mut().state.permanents[token]
        .as_mut()
        .unwrap()
        .damage = 1;
    s.pass_priority();

    assert!(s.battlefield_permanents_named(0, "Ally").is_empty());
    assert_eq!(s.game().state.zones.zone_of(token_card), None);
    assert_eq!(s.zone_size(0, ZoneType::Graveyard), 0);
}

// ---------------------------------------------------------------------------
// South Pole Voyager — this-or-another subject + Nth-resolution gating.
// ---------------------------------------------------------------------------

#[test]
fn south_pole_voyager_second_resolution_draws() {
    let mut s = Scenario::new(
        deck(&[("Plains", 24), ("South Pole Voyager", 16)]),
        deck(&[("Plains", 40)]),
        41,
    );
    s.force_permanent_on_battlefield(0, "South Pole Voyager");
    s.advance_to_active_step(0, StepKind::Main);

    // First Ally entering this turn: gain 1 life, no draw.
    let hand_before = s.zone_size(0, ZoneType::Hand);
    let life_before = s.life(0);
    s.game_mut().create_token("Ally", PlayerId(0), false);
    resolve_all(&mut s);
    assert_eq!(s.life(0), life_before + 1);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand_before);

    // Second time the ability resolves this turn: gain 1 life AND draw.
    s.game_mut().create_token("Ally", PlayerId(0), false);
    resolve_all(&mut s);
    assert_eq!(s.life(0), life_before + 2);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand_before + 1);

    // Third time: life only.
    s.game_mut().create_token("Ally", PlayerId(0), false);
    resolve_all(&mut s);
    assert_eq!(s.life(0), life_before + 3);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand_before + 1);

    // The per-turn resolution count resets: next turn's first resolution
    // does not draw.
    s.advance_to_active_step(1, StepKind::Main);
    let hand_now = s.zone_size(0, ZoneType::Hand);
    s.game_mut().create_token("Ally", PlayerId(0), false);
    resolve_all(&mut s);
    assert_eq!(s.life(0), life_before + 4);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand_now);
}

// ---------------------------------------------------------------------------
// Attack triggers, tapped-and-attacking tokens, vigilance & haste.
// ---------------------------------------------------------------------------

fn inject_ability(s: &mut Scenario, card_id: CardId, ability: Ability) {
    s.game_mut().state.cards[card_id].abilities.push(ability);
}

fn battlefield_card(s: &Scenario, permanent_id: PermanentId) -> CardId {
    permanent(s, permanent_id).card
}

/// "Whenever this creature attacks" fires after the declare-attackers batch;
/// a token created tapped-and-attacking joins combat and deals damage.
#[test]
fn attack_trigger_creates_tapped_and_attacking_token() {
    let mut s = Scenario::new(ogre_only_deck(), deck(&[("Plains", 40)]), 43);
    let ogre = s.force_permanent_on_battlefield(0, "Grey Ogre");
    s.game_mut().state.permanents[ogre].as_mut().unwrap().summoning_sick = false;
    let ogre_card = battlefield_card(&s, ogre);
    inject_ability(
        &mut s,
        ogre_card,
        Ability::Triggered {
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CreateToken {
                token_name: "Ally".to_string(),
                count: 1,
                tapped_and_attacking: true,
            }],
        },
    );

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    resolve_all(&mut s);

    let tokens = s.battlefield_permanents_named(0, "Ally");
    assert_eq!(tokens.len(), 1, "attack trigger should create one token");
    assert!(permanent(&s, tokens[0]).tapped);
    assert!(permanent(&s, tokens[0]).attacking);

    // Unblocked: Grey Ogre (2) + token (1) = 3 damage.
    s.advance_to_active_step(0, StepKind::EndOfCombat);
    s.assert_life(1, 17);
}

/// "Whenever one or more creatures you control attack" fires once per
/// combat, not once per attacker.
#[test]
fn one_or_more_attack_trigger_fires_once() {
    let mut s = Scenario::new(ogre_only_deck(), deck(&[("Plains", 40)]), 47);
    let ogre_a = s.force_permanent_on_battlefield(0, "Grey Ogre");
    let ogre_b = s.force_permanent_on_battlefield(0, "Grey Ogre");
    for ogre in [ogre_a, ogre_b] {
        s.game_mut().state.permanents[ogre].as_mut().unwrap().summoning_sick = false;
    }
    let ogre_a_card = battlefield_card(&s, ogre_a);
    inject_ability(
        &mut s,
        ogre_a_card,
        Ability::Triggered {
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::AnyYouControl(CardPredicate::creature()),
            },
            effects: vec![Effect::GainLife { amount: 1 }],
        },
    );

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.declare_attack();
    resolve_all(&mut s);

    s.assert_life(0, 21);
}

/// A creature with haste can attack the turn it enters and its attack
/// trigger fires.
#[test]
fn haste_attacker_fires_attack_trigger_on_entry_turn() {
    let mut s = Scenario::new(raging_goblin_deck(), deck(&[("Plains", 40)]), 53);
    force_lands(&mut s, 0, "Mountain", 1);
    clear_hand(&mut s, 0);
    // Inject on the card before it enters; abilities are per-card-instance.
    let goblin_card = force_from_library(&mut s, 0, "Raging Goblin");
    inject_ability(
        &mut s,
        goblin_card,
        Ability::Triggered {
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::GainLife { amount: 1 }],
        },
    );

    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    resolve_all(&mut s);

    s.assert_life(0, 21);
}

/// Attacking taps a non-vigilance creature and fires "becomes tapped".
#[test]
fn becomes_tapped_fires_when_attacker_taps() {
    let mut s = Scenario::new(ogre_only_deck(), deck(&[("Plains", 40)]), 59);
    let ogre = s.force_permanent_on_battlefield(0, "Grey Ogre");
    s.game_mut().state.permanents[ogre].as_mut().unwrap().summoning_sick = false;
    let ogre_card = battlefield_card(&s, ogre);
    inject_ability(
        &mut s,
        ogre_card,
        Ability::Triggered {
            condition: TriggerCondition::BecomesTapped {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::GainLife { amount: 1 }],
        },
    );

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    resolve_all(&mut s);

    assert!(permanent(&s, ogre).tapped);
    s.assert_life(0, 21);
}

/// A vigilance attacker does not tap, so "becomes tapped" must NOT fire.
#[test]
fn vigilance_attacker_does_not_become_tapped() {
    let mut s = Scenario::new(serra_angel_deck(), deck(&[("Plains", 40)]), 61);
    let angel = s.force_permanent_on_battlefield(0, "Serra Angel");
    s.game_mut().state.permanents[angel].as_mut().unwrap().summoning_sick = false;
    let angel_card = battlefield_card(&s, angel);
    inject_ability(
        &mut s,
        angel_card,
        Ability::Triggered {
            condition: TriggerCondition::BecomesTapped {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::GainLife { amount: 1 }],
        },
    );

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::EndOfCombat);

    assert!(!permanent(&s, angel).tapped, "vigilance attacker stays untapped");
    s.assert_life(0, 20);
}

/// Tapping a creature for mana fires "tapped for mana" (and it does not
/// fire for combat taps).
#[test]
fn tapped_for_mana_trigger() {
    let mut s = Scenario::new(forest_elves_deck(), deck(&[("Plains", 40)]), 67);
    let elves = s.force_permanent_on_battlefield(0, "Llanowar Elves");
    s.game_mut().state.permanents[elves].as_mut().unwrap().summoning_sick = false;
    let elves_card = battlefield_card(&s, elves);
    inject_ability(
        &mut s,
        elves_card,
        Ability::Triggered {
            condition: TriggerCondition::TappedForMana {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::GainLife { amount: 1 }],
        },
    );

    // Cast another Llanowar Elves paying {G} with the battlefield Elves.
    clear_hand(&mut s, 0);
    force_from_library(&mut s, 0, "Llanowar Elves");
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);
    resolve_all(&mut s);

    assert!(permanent(&s, elves).tapped);
    s.assert_life(0, 21);
}

/// A combat tap is not "tapped for mana".
#[test]
fn attack_tap_is_not_tapped_for_mana() {
    let mut s = Scenario::new(ogre_only_deck(), deck(&[("Plains", 40)]), 71);
    let ogre = s.force_permanent_on_battlefield(0, "Grey Ogre");
    s.game_mut().state.permanents[ogre].as_mut().unwrap().summoning_sick = false;
    let ogre_card = battlefield_card(&s, ogre);
    inject_ability(
        &mut s,
        ogre_card,
        Ability::Triggered {
            condition: TriggerCondition::TappedForMana {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::GainLife { amount: 1 }],
        },
    );

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::EndOfCombat);

    assert!(permanent(&s, ogre).tapped);
    s.assert_life(0, 20);
}

// ---------------------------------------------------------------------------
// Death triggers.
// ---------------------------------------------------------------------------

/// "When this creature dies" fires on battlefield → graveyard.
#[test]
fn dies_trigger_fires_on_death() {
    let mut s = Scenario::new(ogre_only_deck(), deck(&[("Plains", 40)]), 73);
    let ogre = s.force_permanent_on_battlefield(0, "Grey Ogre");
    let ogre_card = battlefield_card(&s, ogre);
    inject_ability(
        &mut s,
        ogre_card,
        Ability::Triggered {
            condition: TriggerCondition::Dies {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::GainLife { amount: 1 }],
        },
    );

    s.game_mut().state.permanents[ogre].as_mut().unwrap().damage = 2;
    s.pass_priority();
    resolve_all(&mut s);

    assert_eq!(s.zone_size(0, ZoneType::Graveyard), 1);
    s.assert_life(0, 21);
}

/// A creature going battlefield → hand did not die.
#[test]
fn bounce_is_not_death() {
    let mut s = Scenario::new(ogre_only_deck(), deck(&[("Plains", 40)]), 79);
    let ogre = s.force_permanent_on_battlefield(0, "Grey Ogre");
    let ogre_card = battlefield_card(&s, ogre);
    inject_ability(
        &mut s,
        ogre_card,
        Ability::Triggered {
            condition: TriggerCondition::Dies {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::GainLife { amount: 1 }],
        },
    );

    s.game_mut().move_card(ogre_card, ZoneType::Hand);
    s.pass_priority();
    resolve_all(&mut s);

    s.assert_life(0, 20);
}

// ---------------------------------------------------------------------------
// Conditional unblockability.
// ---------------------------------------------------------------------------

/// "Can't be blocked by creatures with power 2 or less" — the predicate is
/// checked against effective power, so +1/+1 counters can re-enable a block.
#[test]
fn block_restriction_respects_effective_power() {
    let mut s = Scenario::new(ogre_only_deck(), ogre_only_deck(), 83);
    let attacker = s.force_permanent_on_battlefield(0, "Grey Ogre");
    s.game_mut().state.permanents[attacker].as_mut().unwrap().summoning_sick = false;
    let attacker_card = battlefield_card(&s, attacker);
    s.game_mut().state.cards[attacker_card].block_restriction =
        Some(CardPredicate {
            card_type: Some(CardType::Creature),
            max_power: Some(2),
            ..Default::default()
        });

    let weak = s.force_permanent_on_battlefield(1, "Grey Ogre");
    let strong = s.force_permanent_on_battlefield(1, "Grey Ogre");
    // The "strong" ogre has a +1/+1 counter: effective power 3 > 2.
    s.game_mut().state.permanents[strong].as_mut().unwrap().plus1_counters = 1;

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();

    // Two blocker prompts follow; the countered ogre may block, the plain
    // one may not.
    for _ in 0..2 {
        s.advance_until(
            |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
            "expected blocker prompt".to_string(),
        );
        let actions = s.action_space().actions.clone();
        let blocker = actions
            .iter()
            .find_map(|action| match action {
                Action::DeclareBlocker { blocker, .. } => Some(*blocker),
                _ => None,
            })
            .expect("blocker action should reference a blocker");
        let can_block = actions.iter().any(
            |action| matches!(action, Action::DeclareBlocker { attacker: Some(_), .. }),
        );
        if blocker == weak {
            assert!(!can_block, "power-2 creature must not be able to block");
        } else {
            assert_eq!(blocker, strong);
            assert!(can_block, "power-3 creature may block");
        }
        s.decline_block();
    }
}

/// A permanent flagged "can't be blocked this turn" cannot be blocked.
#[test]
fn cant_be_blocked_this_turn_blocks_all_blocks() {
    let mut s = Scenario::new(ogre_only_deck(), ogre_only_deck(), 89);
    let attacker = s.force_permanent_on_battlefield(0, "Grey Ogre");
    s.game_mut().state.permanents[attacker].as_mut().unwrap().summoning_sick = false;
    s.game_mut().state.permanents[attacker]
        .as_mut()
        .unwrap()
        .cant_be_blocked_this_turn = true;
    s.force_permanent_on_battlefield(1, "Grey Ogre");

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
        "expected blocker prompt".to_string(),
    );
    let can_block = s.action_space().actions.iter().any(
        |action| matches!(action, Action::DeclareBlocker { attacker: Some(_), .. }),
    );
    assert!(!can_block);
}

// ---------------------------------------------------------------------------
// Counters on lands, upkeep trigger scoping, observation encoding.
// ---------------------------------------------------------------------------

/// +1/+1 counters sit on lands without killing them (earthbend prereq —
/// Stage 3 animates lands; the counter state must not assume creature-ness).
#[test]
fn counters_on_lands_are_inert_and_persistent() {
    let mut s = Scenario::new(deck(&[("Forest", 40)]), deck(&[("Forest", 40)]), 97);
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    s.game_mut().state.permanents[forest].as_mut().unwrap().plus1_counters = 1;

    s.pass_priority();
    s.pass_priority();

    let forest_perm = permanent(&s, forest);
    assert_eq!(forest_perm.plus1_counters, 1);
    assert!(s
        .game()
        .state
        .zones
        .zone_cards(ZoneType::Battlefield, PlayerId(0))
        .contains(&forest_perm.card));
}

/// Observation encoding: counter counts, token flag, Ally/Lesson type tags,
/// and per-player graveyard Lesson counts are all agent-visible.
#[test]
fn observation_encodes_counters_tokens_and_type_tags() {
    let mut s = Scenario::new(
        deck(&[("Mountain", 24), ("Lightning Bolt", 16)]),
        deck(&[("Plains", 40)]),
        101,
    );
    // A token with counters on the battlefield.
    s.game_mut().create_token("Ally", PlayerId(0), false);
    resolve_all(&mut s);
    let token = s.battlefield_permanents_named(0, "Ally")[0];
    s.game_mut().state.permanents[token].as_mut().unwrap().plus1_counters = 3;

    // A Lesson in the graveyard (subtype injected — no Lesson spell is
    // registered until Stage 2 machinery lands).
    let bolt = find_owned_card_off_battlefield(&s, 0, "Lightning Bolt");
    s.game_mut().state.cards[bolt].subtypes.push("Lesson".to_string());
    s.game_mut().move_card(bolt, ZoneType::Graveyard);
    s.pass_priority();

    let events = s.game_mut().take_observation_events();
    let obs = Observation::new(s.game(), &events);
    let (me, them) = if obs.agent.player_index == 0 {
        (&obs.agent, &obs.opponent)
    } else {
        (&obs.opponent, &obs.agent)
    };
    assert_eq!(me.graveyard_lessons, 1);
    assert_eq!(them.graveyard_lessons, 0);

    let all_cards = obs.agent_cards.iter().chain(obs.opponent_cards.iter());
    let token_card = all_cards
        .clone()
        .find(|card| card.name == "Ally")
        .expect("token card should be observable");
    assert!(token_card.is_token);
    assert!(token_card.is_ally);
    assert!(!token_card.is_lesson);

    let lesson_card = all_cards
        .clone()
        .find(|card| card.name == "Lightning Bolt")
        .expect("graveyard lesson should be observable");
    assert!(lesson_card.is_lesson);
    assert!(!lesson_card.is_token);

    let token_perm = obs
        .agent_permanents
        .iter()
        .chain(obs.opponent_permanents.iter())
        .find(|perm| perm.plus1_counters == 3)
        .expect("counter count should be observable");
    assert!(!token_perm.cant_be_blocked_this_turn);
}
