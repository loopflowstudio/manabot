//! Tests for cards filed in `src/dom.rs`, in card order.

use crate::support::{
    game::{clear_summoning_sickness, five_color_setup, move_to_hand},
    inspect::{keywords, offered_targets, pt_modifications},
    play::{attack_with, cast_card, pass_until, settle},
};
use managym::{
    agent::action::ActionSpaceKind,
    flow::turn::StepKind,
    state::{card::Keyword, game_object::Target},
};

#[test]
fn run_amok_pumps_only_an_attacking_creature() {
    let mut s = five_color_setup(&["Run Amok", "Gray Ogre"]);
    let attacker = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let idle = s.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut s, attacker);
    let run_amok = move_to_hand(&mut s, "Run Amok");
    s.game_mut().scenario_refresh_priority().unwrap();
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, attacker);
    s.advance_to_step(StepKind::DeclareAttackers);
    while s.action_space().kind != ActionSpaceKind::Priority {
        s.advance_default_action();
    }
    s.set_player_mana_pool(0, "1R");
    cast_card(&mut s, run_amok);
    let targets = offered_targets(&s);
    assert!(targets.contains(&Target::Permanent(attacker)));
    assert!(!targets.contains(&Target::Permanent(idle)));
    assert!(s.choose_target(Target::Permanent(attacker)));
    settle(&mut s);
    assert_eq!(pt_modifications(&s, attacker), (3, 3));
    assert!(keywords(&s, attacker).has(Keyword::Trample));
    s.advance_to_step(StepKind::EndOfCombat);
    s.assert_life(1, 15);
}

#[test]
fn foundations_permanent_tatyova_benthic_druid() {
    use crate::support::game::five_color_setup;
    use crate::support::play::enter_and_settle;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Tatyova, Benthic Druid"]);
    s.force_permanent_on_battlefield(0, "Tatyova, Benthic Druid");
    let hand = s.zone_size(0, ZoneType::Hand);
    enter_and_settle(&mut s, "Forest");
    s.assert_life(0, 21);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand + 1);
}

#[test]
fn foundations_grant_lyra_dawnbringer() {
    use crate::support::game::{five_color_setup, move_card};
    use crate::support::inspect::{card_of, keywords};
    use managym::state::card::Keyword;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Lyra Dawnbringer", "Serra Angel", "Gray Ogre"]);
    let ally = s.force_permanent_on_battlefield(0, "Serra Angel");
    let enemy = s.force_permanent_on_battlefield(1, "Serra Angel");
    let other = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let base = s.game().characteristics(ally).pt();
    let source = s.force_permanent_on_battlefield(0, "Lyra Dawnbringer");
    assert_eq!(
        s.game().characteristics(ally).pt(),
        (base.0 + 1, base.1 + 1)
    );
    assert_eq!(s.game().characteristics(source).power, 5);
    assert_eq!(s.game().characteristics(other).pt(), (2, 2));
    assert!(keywords(&s, ally).has(Keyword::Lifelink));
    assert!(!keywords(&s, enemy).has(Keyword::Lifelink));
    assert!(!keywords(&s, other).has(Keyword::Lifelink));
    assert_eq!(s.game().characteristics(enemy).pt(), base);
    let card = card_of(&s, source);
    move_card(&mut s, card, ZoneType::Graveyard);
    assert_eq!(s.game().characteristics(ally).pt(), base);
    assert!(!keywords(&s, ally).has(Keyword::Lifelink));
}

#[test]
fn foundations_5d_ghitu_lavarunner() {
    use crate::support::game::{five_color_setup, move_card, move_from_library};
    use crate::support::inspect::keywords;
    use managym::state::card::Keyword;
    use managym::state::zone::ZoneType;

    let mut s = five_color_setup(&["Ghitu Lavarunner", "Lightning Bolt", "Day of Judgment"]);
    let id = s.force_permanent_on_battlefield(0, "Ghitu Lavarunner");
    move_from_library(&mut s, 0, "Lightning Bolt", ZoneType::Graveyard);
    move_from_library(&mut s, 1, "Day of Judgment", ZoneType::Graveyard);
    assert!(!keywords(&s, id).has(Keyword::Haste));
    let second = move_from_library(&mut s, 0, "Day of Judgment", ZoneType::Graveyard);
    assert!(keywords(&s, id).has(Keyword::Haste));
    assert_eq!(s.game().characteristics(id).pt(), (2, 2));
    move_card(&mut s, second, ZoneType::Exile);
    assert_eq!(s.game().characteristics(id).pt(), (1, 2));
    assert!(!keywords(&s, id).has(Keyword::Haste));
}

#[test]
fn foundations_5e_adamant_will() {
    use crate::support::play::{cast, settle};
    use managym::state::card::Keywords;
    let mut s = five_color_setup(&["Adamant Will", "Day of Judgment", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let other = s.force_permanent_on_battlefield(1, "Gray Ogre");
    cast(&mut s, "Adamant Will");
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(s.game().characteristics(ogre).pt(), (4, 4));
    assert_eq!(keywords(&s, ogre), Keywords::of(&[Keyword::Indestructible]));
    s.assert_life(0, 20);
    s.assert_zone_size(0, managym::state::zone::ZoneType::Hand, 0);

    cast(&mut s, "Day of Judgment");
    settle(&mut s);
    assert!(s.game().state.permanents[other].is_none());
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().characteristics(ogre).pt(), (2, 2));
    assert_eq!(keywords(&s, ogre), Keywords::default());
}
