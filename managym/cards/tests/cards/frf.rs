//! Tests for cards filed in `src/frf.rs`, in card order.

use crate::support::{
    game::{clear_summoning_sickness, five_color_setup},
    inspect::{can_activate, permanent},
    play::{activate, settle},
};
use managym::{
    flow::turn::StepKind,
    state::{game_object::PlayerId, zone::ZoneType},
};

#[test]
fn humble_defector_draws_two_then_its_target_opponent_gains_control_only_on_your_turn() {
    let mut s = five_color_setup(&["Humble Defector"]);
    let defector = s.force_permanent_on_battlefield(0, "Humble Defector");
    clear_summoning_sickness(&mut s, defector);
    s.game_mut().scenario_refresh_priority().unwrap();
    let hand = s.zone_size(0, ZoneType::Hand);
    activate(&mut s, defector, 0);
    settle(&mut s);
    s.assert_zone_size(0, ZoneType::Hand, hand + 2);
    assert_eq!(permanent(&s, defector).controller, PlayerId(1));
    s.advance_to_active_step(0, StepKind::Upkeep);
    while s.action_space().player != Some(PlayerId(1)) {
        s.advance_default_action();
    }
    assert!(
        !can_activate(&s, defector),
        "only during its controller's turn"
    );
}

#[test]
fn foundations_5e_valorous_stance() {
    use crate::support::game::{five_color_setup, until_end_of_turn};
    use crate::support::inspect::{card_of, keywords, mode_labels, offered_targets, zone_of};
    use crate::support::play::{cast, choose_mode, settle};
    use managym::state::card::Keyword;
    use managym::state::effect::Modification;
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Valorous Stance", "Craw Wurm", "Hill Giant"]);
    let wurm = s.force_permanent_on_battlefield(1, "Craw Wurm");
    let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
    let grown = s.force_permanent_on_battlefield(1, "Hill Giant");
    until_end_of_turn(&mut s, grown, Modification::pump(0, 1));
    cast(&mut s, "Valorous Stance");
    assert_eq!(mode_labels(&s).len(), 2);
    let mut shield = s.clone();
    choose_mode(&mut shield, 0);
    assert!(shield.choose_target(Target::Permanent(giant)));
    settle(&mut shield);
    assert!(keywords(&shield, giant).has(Keyword::Indestructible));
    choose_mode(&mut s, 1);
    let offers = offered_targets(&s);
    assert!(offers.contains(&Target::Permanent(wurm)));
    assert!(
        offers.contains(&Target::Permanent(grown)),
        "effective toughness 4"
    );
    assert!(!offers.contains(&Target::Permanent(giant)));
    let card = card_of(&s, wurm);
    assert!(s.choose_target(Target::Permanent(wurm)));
    settle(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Graveyard));
}

#[test]
fn flamewake_phoenix_must_attack_and_returns_from_graveyard_on_ferocious() {
    use crate::support::{
        game::plant_in_graveyard,
        inspect::zone_of,
        play::{choose, pass_until},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::player::ManaPayment,
    };

    // Without a 4+ power creature: graveyard trigger does not fire.
    let mut s = five_color_setup(&["Flamewake Phoenix", "Gray Ogre", "Craw Wurm"]);
    let phoenix = plant_in_graveyard(&mut s, 0, "Flamewake Phoenix");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert_eq!(zone_of(&s, phoenix), Some(ZoneType::Graveyard));

    // With a 4+ power creature: triggers at beginning of combat, pays {R}, returns with haste and must attack.
    let mut s = five_color_setup(&["Flamewake Phoenix", "Craw Wurm"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let phoenix = plant_in_graveyard(&mut s, 0, "Flamewake Phoenix");
    s.force_permanent_on_battlefield(0, "Craw Wurm");
    s.force_permanent_on_battlefield(0, "Mountain");
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    pass_until(&mut s, ActionSpaceKind::PayOrNot);
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    settle(&mut s);
    assert_eq!(zone_of(&s, phoenix), Some(ZoneType::Battlefield));
    let phoenix_perm = s.battlefield_permanents_named(0, "Flamewake Phoenix")[0];
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    assert!(
        !s.action_space().actions.iter().any(|a| matches!(
            a,
            Action::DeclareAttacker {
                permanent,
                attack: false,
                ..
            } if *permanent == phoenix_perm
        )),
        "Flamewake Phoenix must attack if able"
    );
}
