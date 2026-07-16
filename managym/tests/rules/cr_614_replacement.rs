use managym::{
    agent::action::ActionType,
    flow::{
        event::{DamageTarget, GameEvent},
        turn::StepKind,
    },
    state::{
        card::ReplacementEffect,
        game_object::{CardId, PermanentId, PlayerId, Target},
        zone::ZoneType,
    },
};

use super::helpers::*;

fn card_for_permanent(s: &Scenario, permanent: PermanentId) -> CardId {
    s.game().state.permanents[permanent]
        .as_ref()
        .expect("driver permanent")
        .card
}

fn card_named_outside_battlefield(s: &Scenario, player: usize, name: &str) -> CardId {
    s.game()
        .state
        .cards
        .iter()
        .enumerate()
        .find_map(|(index, card)| {
            let card_id = CardId(index);
            (card.owner == PlayerId(player)
                && card.name == name
                && s.game().state.zones.zone_of(card_id) != Some(ZoneType::Battlefield))
            .then_some(card_id)
        })
        .unwrap_or_else(|| panic!("missing {name} outside battlefield"))
}

fn resolve_bolt_with_drivers(
    seed: u64,
    driver_effects: Vec<Vec<ReplacementEffect>>,
    target: PlayerId,
) -> (i32, Vec<GameEvent>) {
    let mut s = Scenario::new(ogre_only_deck(), bolt_deck(), seed);

    for effects in driver_effects {
        let permanent = s.force_permanent_on_battlefield(0, "Gray Ogre");
        let card = card_for_permanent(&s, permanent);
        // Scenario-local COW definitions make these rule drivers without
        // expanding the production card pack.
        s.game_mut().state.cards[card].replacement_effects = effects;
    }

    s.force_card_in_hand(1, "Mountain");
    s.force_card_in_hand(1, "Lightning Bolt");
    s.advance_to_active_step(1, StepKind::Main);
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    let _ = s.drain_events();

    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    assert!(s.choose_target(Target::Player(target)));
    s.pass_priority();
    s.pass_priority();

    let life = s.life(target.0);
    let events = s.drain_events();
    (life, events)
}

fn committed_damage(events: &[GameEvent], player: PlayerId) -> Option<u32> {
    events.iter().find_map(|event| match event {
        GameEvent::DamageDealt {
            target: DamageTarget::Player(target),
            amount,
            ..
        } if *target == player => Some(*amount),
        _ => None,
    })
}

/// CR 615.1 — prevention modifies damage before the event commits.
#[test]
fn cr_615_prevented_damage_emits_no_committed_damage_or_life_event() {
    let (life, events) = resolve_bolt_with_drivers(
        614_001,
        vec![vec![ReplacementEffect::PreventDamageToController {
            amount: 3,
        }]],
        PlayerId(0),
    );

    assert_eq!(life, 20);
    assert_eq!(committed_damage(&events, PlayerId(0)), None);
    assert!(!events.iter().any(|event| matches!(
        event,
        GameEvent::LifeChanged {
            player: PlayerId(0),
            ..
        }
    )));
}

/// CR 616.1 is broader than this slice. The curated contract uses exact
/// source identity then definition order, making noncommuting effects replay
/// deterministically until an affected-player choice ABI exists.
#[test]
fn cr_616_curated_replacement_order_is_deterministic_and_observable() {
    let prevent_then_double = vec![
        vec![ReplacementEffect::PreventDamageToController { amount: 1 }],
        vec![ReplacementEffect::DoubleDamageToController],
    ];
    let first = resolve_bolt_with_drivers(614_002, prevent_then_double.clone(), PlayerId(0));
    let replay = resolve_bolt_with_drivers(614_002, prevent_then_double, PlayerId(0));

    assert_eq!(
        first, replay,
        "same seed and definitions must replay exactly"
    );
    assert_eq!(first.0, 16, "(3 - 1) * 2 damage");
    assert_eq!(committed_damage(&first.1, PlayerId(0)), Some(4));
    assert!(first.1.contains(&GameEvent::LifeChanged {
        player: PlayerId(0),
        old: 20,
        new: 16,
    }));

    let (reverse_life, reverse_events) = resolve_bolt_with_drivers(
        614_002,
        vec![
            vec![ReplacementEffect::DoubleDamageToController],
            vec![ReplacementEffect::PreventDamageToController { amount: 1 }],
        ],
        PlayerId(0),
    );
    assert_eq!(reverse_life, 15, "(3 * 2) - 1 damage");
    assert_eq!(committed_damage(&reverse_events, PlayerId(0)), Some(5));
}

#[test]
fn nonmatching_replacement_preserves_the_legacy_committed_trace() {
    let baseline = resolve_bolt_with_drivers(614_003, vec![Vec::new()], PlayerId(1));
    let nonmatching = resolve_bolt_with_drivers(
        614_003,
        vec![vec![ReplacementEffect::PreventDamageToController {
            amount: 3,
        }]],
        PlayerId(1),
    );

    assert_eq!(baseline, nonmatching);
    assert_eq!(baseline.0, 17);
    assert_eq!(committed_damage(&baseline.1, PlayerId(1)), Some(3));
}

/// CR 614.1c — ETB replacements modify the permanent before the move event is
/// committed and observed by triggers.
#[test]
fn cr_614_1c_enters_tapped_and_with_counters_before_card_moved_commits() {
    let mut s = Scenario::new(ogre_deck(), mountain_deck(), 614_004);
    let card = card_named_outside_battlefield(&s, 0, "Gray Ogre");
    let from = s.game().state.zones.zone_of(card);
    let old_ref = s.game().current_object_ref(card).expect("departing object");
    s.game_mut().state.cards[card].replacement_effects = vec![
        ReplacementEffect::EntersTapped,
        ReplacementEffect::EntersWithPlusOneCounters { count: 2 },
    ];
    let _ = s.drain_events();

    s.game_mut().move_card(card, ZoneType::Battlefield);

    let permanent_id = s.game().state.card_to_permanent[card].expect("entered permanent");
    let permanent = s.game().state.permanents[permanent_id]
        .as_ref()
        .expect("entered permanent facts");
    assert!(permanent.tapped);
    assert_eq!(permanent.plus1_counters, 2);
    assert_ne!(s.game().current_object_ref(card), Some(old_ref));
    assert!(s.drain_events().contains(&GameEvent::CardMoved {
        card,
        from,
        to: ZoneType::Battlefield,
        controller: PlayerId(0),
    }));
}

#[test]
fn entry_replacements_do_not_modify_other_objects_or_nonbattlefield_moves() {
    let mut s = Scenario::new(ogre_deck(), mountain_deck(), 614_005);
    let driver = card_named_outside_battlefield(&s, 0, "Gray Ogre");
    s.game_mut().state.cards[driver].replacement_effects = vec![
        ReplacementEffect::EntersTapped,
        ReplacementEffect::EntersWithPlusOneCounters { count: 2 },
    ];
    s.game_mut().move_card(driver, ZoneType::Graveyard);
    assert_eq!(s.game().state.card_to_permanent[driver], None);

    let mountain = card_named_outside_battlefield(&s, 0, "Mountain");
    s.game_mut().move_card(mountain, ZoneType::Battlefield);
    let permanent_id = s.game().state.card_to_permanent[mountain].expect("mountain permanent");
    let permanent = s.game().state.permanents[permanent_id]
        .as_ref()
        .expect("mountain facts");
    assert!(!permanent.tapped);
    assert_eq!(permanent.plus1_counters, 0);
}
