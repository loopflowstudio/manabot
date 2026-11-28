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

#[test]
fn grow_from_the_ashes_fetches_one_or_two_untapped_basics_when_kicked() {
    use crate::support::{
        inspect::{library_card, permanent_of, tapped},
        play::{cast, choose, select_card},
    };
    use managym::{agent::action::Action, state::player::ManaPayment};

    for kicked in [false, true] {
        let mut s = five_color_setup(&["Grow from the Ashes"]);
        s.game_mut().mana_payment[0] = ManaPayment::Auto;
        cast(&mut s, "Grow from the Ashes");
        if kicked {
            choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        } else {
            choose(&mut s, |a| matches!(a, Action::Decline { .. }));
        }
        pass_until(&mut s, ActionSpaceKind::LookAndSelect);
        let first = library_card(&s, 0, "Forest");
        select_card(&mut s, first);
        let second = if kicked {
            let island = library_card(&s, 0, "Island");
            select_card(&mut s, island);
            Some(island)
        } else {
            None
        };
        settle(&mut s);
        assert!(!tapped(&s, permanent_of(&s, first).unwrap()));
        if let Some(island) = second {
            assert!(!tapped(&s, permanent_of(&s, island).unwrap()));
        }
    }
}

#[test]
fn knights_restrict_opposing_sources_and_count_either_battlefield() {
    use crate::support::{game::five_color_setup, inspect::offered_targets, play::cast};
    use managym::state::game_object::Target;
    for (spell, protected, other) in [
        ("Stab", "Knight of Grace", "Knight of Malice"),
        ("Luminous Rebuke", "Knight of Malice", "Knight of Grace"),
    ] {
        let mut s = five_color_setup(&[spell, protected, other]);
        let theirs = s.force_permanent_on_battlefield(1, protected);
        assert_eq!(s.game().characteristics(theirs).pt(), (2, 2));
        let mine = s.force_permanent_on_battlefield(0, protected);
        let other = s.force_permanent_on_battlefield(0, other);
        assert_eq!(s.game().characteristics(theirs).pt(), (3, 2));
        assert_eq!(s.game().characteristics(other).pt(), (3, 2));
        cast(&mut s, spell);
        let targets = offered_targets(&s);
        assert!(!targets.contains(&Target::Permanent(theirs)));
        assert!(targets.contains(&Target::Permanent(mine)));
        assert!(targets.contains(&Target::Permanent(other)));
        let view = managym::agent::observation::Observation::for_player(
            s.game(),
            managym::state::game_object::PlayerId(0),
        );
        let quality = if spell == "Stab" { "black" } else { "white" };
        assert_eq!(view.opponent_permanents[0].hexproof_from, [quality]);
    }
}

#[test]
fn knight_checks_activated_source_colors_on_the_battlefield_and_after_sacrifice() {
    use crate::support::{
        game::{five_color_setup, until_end_of_turn},
        inspect::{offered_targets, permanent},
        play::{activate, settle},
    };
    use managym::state::{
        card::{SourceQuality, StaticAbility},
        effect::Modification,
        game_object::Target,
        mana::Color,
    };
    let mut s = five_color_setup(&["Fanatical Firebrand", "Knight of Grace", "Gray Ogre"]);
    let firebrand = s.force_permanent_on_battlefield(0, "Fanatical Firebrand");
    let knight = s.force_permanent_on_battlefield(1, "Knight of Grace");
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    until_end_of_turn(
        &mut s,
        firebrand,
        Modification::SetColors([Color::Black].into()),
    );
    activate(&mut s, firebrand, 0);
    assert!(!offered_targets(&s).contains(&Target::Permanent(knight)));
    assert!(s.choose_target(Target::Permanent(ogre)));
    assert!(s.game().state.permanents[firebrand].is_none());
    let mut unprotected = s.clone();
    settle(&mut unprotected);
    assert_eq!(permanent(&unprotected, ogre).damage, 1);
    until_end_of_turn(
        &mut s,
        ogre,
        Modification::GrantStatic(Box::new(StaticAbility::HexproofFrom(SourceQuality::Color(
            Color::Black,
        )))),
    );
    settle(&mut s);
    assert_eq!(permanent(&s, ogre).damage, 0);
}

#[test]
fn knights_check_the_announced_omen_face_and_live_spell_on_resolution() {
    use crate::support::{
        game::{five_color_setup, move_to_hand},
        inspect::{now, offered_targets, permanent},
        play::{cast, cast_profile, settle},
    };
    use managym::state::{card::SpellProfile, game_object::Target, mana::Color};
    let mut s = five_color_setup(&[
        "Twinmaw Stormbrood",
        "Knight of Malice",
        "Knight of Grace",
        "Lightning Bolt",
    ]);
    let malice = s.force_permanent_on_battlefield(1, "Knight of Malice");
    let grace = s.force_permanent_on_battlefield(1, "Knight of Grace");
    let mut omen = s.clone();
    let dragon = move_to_hand(&mut omen, "Twinmaw Stormbrood");
    cast_profile(&mut omen, dragon, SpellProfile::Omen);
    assert!(
        offered_targets(&omen).contains(&Target::Permanent(malice)),
        "Charring Bite is red although the Dragon is white"
    );
    let bolt = cast(&mut s, "Lightning Bolt");
    assert!(s.choose_target(Target::Permanent(grace)));
    let stack_object = now(&s, bolt);
    s.game_mut().state[stack_object].color_override = Some([Color::Black].into());
    settle(&mut s);
    assert_eq!(permanent(&s, grace).damage, 0);
}

#[test]
fn tempest_djinn_counts_only_its_controllers_basic_islands() {
    use crate::support::game::until_end_of_turn;
    use managym::state::effect::Modification;
    let mut s = five_color_setup(&["Tempest Djinn", "Soulstone Sanctuary"]);
    let djinn = s.force_permanent_on_battlefield(0, "Tempest Djinn");
    let nonbasic = s.force_permanent_on_battlefield(0, "Soulstone Sanctuary");
    until_end_of_turn(
        &mut s,
        nonbasic,
        Modification::AddSubtypes(vec!["Island".into()]),
    );
    s.force_permanent_on_battlefield(1, "Island");
    assert_eq!(s.game().characteristics(djinn).pt(), (0, 4));
    s.force_permanent_on_battlefield(0, "Island");
    assert_eq!(s.game().characteristics(djinn).pt(), (1, 4));
}
