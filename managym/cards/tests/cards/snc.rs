//! Streets of New Capenna triggers through the public decision interface.
use crate::support::game::{five_color_setup, move_card};
use crate::support::inspect::{card_of, keywords, observed};
use crate::support::play::{cast, command, resolve_until_priority, settle};
use managym::agent::action::Action;
use managym::agent::observation::Observation;
use managym::state::card::Keyword;
use managym::state::game_object::{PlayerId, Target};
use managym::state::mana::Color;
use managym::state::zone::ZoneType;

#[test]
fn venom_connoisseur_commands_replay_old_and_new_resolutions() {
    let mut s = five_color_setup(&["Venom Connoisseur"]);
    let source = s.force_permanent_on_battlefield(0, "Venom Connoisseur");
    let player = PlayerId(0);
    let ally = s
        .game_mut()
        .create_token("token.ally", player, false)
        .unwrap();
    resolve_until_priority(&mut s);
    assert!(keywords(&s, source).has(Keyword::Deathtouch));
    assert!(!keywords(&s, ally).has(Keyword::Deathtouch));

    s.game_mut().create_token("token.ally", player, false);
    s.advance_until(
        |s| s.game().state.stack_objects.len() == 1,
        "original trigger on stack".into(),
    );
    let card = card_of(&s, source);
    let exiled = move_card(&mut s, card, ZoneType::Exile);
    let returned = move_card(&mut s, exiled, ZoneType::Battlefield);
    s.game_mut().create_token("token.ally", player, false);
    s.advance_until(
        |s| s.game().state.stack_objects.len() == 2,
        "both triggers on stack".into(),
    );
    let pass = |action: &Action| matches!(action, Action::PassPriority { .. });
    for remaining in [1, 0] {
        for _ in 0..2 {
            command(&mut s, pass);
            if s.game().state.stack_objects.len() == remaining {
                break;
            }
        }
        assert_eq!(s.game().state.stack_objects.len(), remaining);
        let observation = Observation::for_player(s.game(), player);
        let public = s.game().state.permanents[s.game().state.permanent_of(returned).unwrap()]
            .as_ref()
            .unwrap()
            .id;
        let live = observation
            .agent_permanents
            .iter()
            .find(|p| p.id == public.0 as i32)
            .unwrap();
        assert_eq!(live.ability_resolutions, [(0, 1)]);
        assert_eq!(keywords(&s, ally).has(Keyword::Deathtouch), remaining == 0);
    }
}

/// An Offer You Can't Refuse counters a noncreature spell, and that spell's
/// controller, not the Offer's caster, creates two Treasures; the Treasures
/// come even when the spell can't be countered. A creature spell is no
/// legal target.
#[test]
fn an_offer_you_cant_refuse_pays_the_targeted_spells_controller_two_treasures() {
    use crate::support::{
        game::force_new_card_in_hand,
        inspect::{top_spell, zone_of},
        play::{cast_card, castable, settle},
    };
    use managym::state::{game_object::Target, player::ManaPayment};
    let mut s = five_color_setup(&[
        "An Offer You Can't Refuse",
        "Lightning Bolt",
        "Rending Volley",
        "Spectral Sailor",
        "Gray Ogre",
    ]);
    s.game_mut().mana_payment = [ManaPayment::Auto; 2];
    let sailor = s.force_permanent_on_battlefield(1, "Spectral Sailor");
    let ogre = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
    cast_card(&mut s, ogre);
    s.pass_priority();
    let offer = force_new_card_in_hand(&mut s, 1, "An Offer You Can't Refuse");
    assert!(!castable(&mut s, offer));
    settle(&mut s);
    for (name, target, countered, treasures) in [
        ("Lightning Bolt", Target::Player(PlayerId(1)), true, 2),
        ("Rending Volley", Target::Permanent(sailor), false, 4),
    ] {
        let card = force_new_card_in_hand(&mut s, 0, name);
        cast_card(&mut s, card);
        assert!(s.choose_target(target));
        let spell = top_spell(&s).object;
        s.pass_priority();
        let offer = force_new_card_in_hand(&mut s, 1, "An Offer You Can't Refuse");
        cast_card(&mut s, offer);
        assert!(s.choose_target(Target::StackSpell(spell)));
        settle(&mut s);
        assert_eq!(zone_of(&s, spell), Some(ZoneType::Graveyard));
        assert_eq!(
            s.battlefield_permanents_named(0, "Treasure").len(),
            treasures
        );
        assert!(s.battlefield_permanents_named(1, "Treasure").is_empty());
        if countered {
            s.assert_life(1, 20);
        }
    }
    assert!(
        s.game().state.permanents[sailor].is_none(),
        "Volley resolved"
    );
}

#[test]
fn foundations_5d_involuntary_employment() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{keywords, permanent, tapped};
    use crate::support::play::{cast, settle};
    use managym::flow::turn::StepKind;
    use managym::state::card::Keyword;
    use managym::state::game_object::{PlayerId, Target};

    let mut s = five_color_setup(&["Involuntary Employment", "Gray Ogre"]);
    let id = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().state.permanents[id].as_mut().unwrap().tapped = true;
    cast(&mut s, "Involuntary Employment");
    assert!(s.choose_target(Target::Permanent(id)));
    settle(&mut s);
    assert!(!tapped(&s, id));
    assert_eq!(permanent(&s, id).controller, PlayerId(0));
    assert!(keywords(&s, id).has(Keyword::Haste));
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(permanent(&s, id).controller, PlayerId(1));
    assert!(!keywords(&s, id).has(Keyword::Haste));
    assert_eq!(s.battlefield_permanents_named(0, "Treasure").len(), 1);
}

#[test]
fn foundations_5f_inspiring_overseer() {
    use crate::support::game::five_color_setup;
    use crate::support::play::enter_and_settle;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Inspiring Overseer"]);
    let hand = s.zone_size(0, ZoneType::Hand);
    enter_and_settle(&mut s, "Inspiring Overseer");
    s.assert_life(0, 21);
    s.assert_zone_size(0, ZoneType::Hand, hand + 1);
}

/// Witness Protection on Juggernaut, an artifact creature that attacks each
/// combat: it is a green and white 1/1 Citizen creature named Legitimate
/// Businessperson with no abilities, no longer an artifact; removing the Aura
/// restores every printed characteristic.
#[test]
fn foundations_5f_witness_protection() {
    crate::assert_sets_juggernaut_characteristics(
        "Witness Protection",
        "Legitimate Businessperson",
        "Citizen",
        &[
            managym::state::mana::Color::Green,
            managym::state::mana::Color::White,
        ],
    );
}

/// The Citizen keeps no other creature type, color, ability or name: a
/// flying Drake becomes a green and white 1/1 Legitimate Businessperson.
#[test]
fn witness_protection_renames_a_creature_a_green_white_citizen() {
    let mut s = five_color_setup(&["Witness Protection", "Wind Drake"]);
    let drake = s.force_permanent_on_battlefield(1, "Wind Drake");
    cast(&mut s, "Witness Protection");
    assert!(s.choose_target(Target::Permanent(drake)));
    settle(&mut s);
    let view = s.game().characteristics(drake);
    assert_eq!(&*view.name, "Legitimate Businessperson");
    assert_eq!(view.subtypes, vec!["Citizen".to_string()]);
    assert_eq!(
        view.colors.iter().copied().collect::<Vec<_>>(),
        vec![Color::White, Color::Green]
    );
    assert_eq!(view.pt(), (1, 1));
    assert!(
        !view.keywords.has(managym::state::card::Keyword::Flying),
        "no flying"
    );
    let row = observed(&s, drake);
    assert_eq!(row.name.as_deref(), Some("Legitimate Businessperson"));
    assert_eq!(row.creature_types, Some(vec!["Citizen".to_string()]));
    assert!(row.temp_subtypes.is_empty());
}

/// A copy takes copiable values only (CR 707.2): Clone copying a protected
/// Drake enters as a flying Wind Drake, not a Legitimate Businessperson.
#[test]
fn clone_copies_the_drake_under_witness_protection_not_the_citizen() {
    use crate::support::{
        inspect::permanent_of,
        play::{resolve_top, spend_colorless},
    };
    use managym::state::card::Keyword;
    let mut s = five_color_setup(&["Witness Protection", "Wind Drake", "Clone"]);
    let drake = s.force_permanent_on_battlefield(1, "Wind Drake");
    cast(&mut s, "Witness Protection");
    assert!(s.choose_target(Target::Permanent(drake)));
    settle(&mut s);
    let clone = cast(&mut s, "Clone");
    spend_colorless(&mut s);
    resolve_top(&mut s);
    assert!(s.choose_target(Target::Permanent(drake)));
    settle(&mut s);
    let copy = permanent_of(&s, clone).expect("entered");
    let view = s.game().characteristics(copy);
    assert_eq!(&*view.name, "Wind Drake");
    assert!(view.keywords.has(Keyword::Flying));
    assert_eq!(view.pt(), (2, 2));
    assert_eq!(observed(&s, copy).name, None);
    assert_eq!(
        &*s.game().characteristics(drake).name,
        "Legitimate Businessperson"
    );
}
