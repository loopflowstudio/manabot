//! Streets of New Capenna triggers through the public decision interface.
use crate::support::{
    game::{five_color_setup, move_card},
    inspect::{card_of, keywords},
    play::{command, resolve_until_priority},
};
use managym::{
    agent::{action::Action, observation::Observation},
    state::{card::Keyword, game_object::PlayerId, zone::ZoneType},
};

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
