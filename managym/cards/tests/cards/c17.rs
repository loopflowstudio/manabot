//! Tests for cards filed in `src/c17.rs`, in card order.

use crate::support::{
    game::{
        clear_summoning_sickness, five_color_setup, force_new_card_in_hand, set_counters, Scenario,
    },
    inspect::zone_of,
    play::{attack_with, cast, cast_card, offers_ability, pass_until, settle},
};
use managym::{
    agent::{action::Action, action::ActionSpaceKind, observation::Observation},
    flow::turn::StepKind,
    state::{
        effect::{Affected, EffectDuration, Modification},
        game_object::{PlayerId, Target},
        permanent::CounterKind,
        zone::ZoneType,
    },
};

fn offered_targets(s: &Scenario) -> Vec<Target> {
    s.action_space()
        .actions
        .iter()
        .filter_map(|action| match action {
            Action::ChooseTarget { target, .. } => Some(*target),
            _ => None,
        })
        .collect()
}

fn tokens(s: &Scenario, player: usize) -> usize {
    s.game()
        .state
        .zones
        .zone_objects(ZoneType::Battlefield, PlayerId(player))
        .iter()
        .filter(|object| !s.game().state[**object].is_card())
        .count()
}

/// The caster's permanents phase out and the spell exiles itself. While they
/// are out, they keep their observation rows but nothing can target them,
/// Pyroclasm misses them, and neither Lightning Bolt nor an attack can reach
/// the caster. They phase in on the caster's next untap step, before it
/// untaps, with their counters, and with no enters trigger (Bastion of
/// Remembrance makes no Soldier).
#[test]
fn teferis_protection_phases_out_the_casters_permanents_until_their_untap_step() {
    let mut s = five_color_setup(&[
        "Teferi's Protection",
        "Gray Ogre",
        "Hill Giant",
        "Bastion of Remembrance",
        "Pyroclasm",
        "Lightning Bolt",
    ]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    set_counters(&mut s, ogre, CounterKind::PlusOnePlusOne, 1);
    s.game_mut().state.permanents[ogre].as_mut().unwrap().tapped = true;
    let bastion = s.force_permanent_on_battlefield(0, "Bastion of Remembrance");
    let soldiers = tokens(&s, 0);
    let rows = Observation::for_player(s.game(), PlayerId(1))
        .opponent_permanents
        .len();

    let protection = cast(&mut s, "Teferi's Protection");
    settle(&mut s);
    assert_eq!(zone_of(&s, protection), Some(ZoneType::Exile));
    assert!(s.game().is_phased_out(ogre));
    assert!(s.game().is_phased_out(bastion));
    assert_eq!(
        Observation::for_player(s.game(), PlayerId(1))
            .opponent_permanents
            .len(),
        rows,
        "phased-out permanents keep their public rows"
    );

    cast(&mut s, "Lightning Bolt");
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    let targets = offered_targets(&s);
    assert!(!targets.contains(&Target::Permanent(ogre)));
    assert!(
        !targets.contains(&Target::Player(PlayerId(0))),
        "protection from everything stops even the caster's own spell"
    );
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    settle(&mut s);

    s.advance_to_active_step(1, StepKind::Main);
    let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
    clear_summoning_sickness(&mut s, giant);
    s.set_player_mana_pool(1, "RR");
    let pyroclasm = force_new_card_in_hand(&mut s, 1, "Pyroclasm");
    cast_card(&mut s, pyroclasm);
    settle(&mut s);
    assert_eq!(s.game().state.permanents[giant].as_ref().unwrap().damage, 2);
    assert_eq!(s.game().state.permanents[ogre].as_ref().unwrap().damage, 0);
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, giant);
    s.advance_to_active_step(1, StepKind::End);
    assert_eq!(s.life(0), 20, "the attack's damage is prevented");
    assert!(
        s.game().is_phased_out(ogre),
        "still out on the opponent's turn"
    );

    s.advance_to_active_step(0, StepKind::Upkeep);
    assert!(!s.game().is_phased_out(ogre));
    assert!(!s.game().is_phased_out(bastion));
    let permanent = s.game().state.permanents[ogre].as_ref().unwrap();
    assert!(!permanent.tapped, "it phased in before the untap");
    assert_eq!(permanent.counter_count(CounterKind::PlusOnePlusOne), 1);
    assert_eq!(tokens(&s, 0), soldiers, "phasing in is no zone change");
    assert_eq!(s.game().player_effects(PlayerId(0)).count(), 0);
}

/// While the caster's life total can't change, a life payment is refused;
/// the record alone lets damage be dealt without moving the total, and it
/// ends as the caster's next turn begins.
#[test]
fn teferis_protection_locks_the_casters_life_total_until_their_next_turn() {
    let mut s = five_color_setup(&["Teferi's Protection", "Prismatic Vista"]);
    cast(&mut s, "Teferi's Protection");
    settle(&mut s);
    let vista = s.force_permanent_on_battlefield(0, "Prismatic Vista");
    assert!(!offers_ability(&mut s, vista, 0), "no life can be paid");
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    assert!(!s.game().life_total_cant_change(PlayerId(0)));
    assert!(offers_ability(&mut s, vista, 0));

    let mut s = five_color_setup(&["Lightning Bolt"]);
    s.game_mut().apply_continuous(
        None,
        PlayerId(0),
        [Affected::Player(PlayerId(0))],
        Modification::LifeTotalCantChange,
        EffectDuration::UntilYourNextTurn,
    );
    cast(&mut s, "Lightning Bolt");
    assert!(s.choose_target(Target::Player(PlayerId(0))));
    settle(&mut s);
    assert_eq!(s.life(0), 20);
}
