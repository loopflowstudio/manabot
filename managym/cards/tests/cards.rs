//! Card behavior tests, one module per set module in `src/`: a test lives
//! with the set that first printed the card it is about. Engine rules that
//! are not about one card stay in the engine's `rules_tests`. Tests never
//! decide whether a card is supported; see `cards/README.md` for coverage.
//! The scenario helpers are the engine's `tests/support`, mounted by path so
//! each primitive stays defined once.

#[path = "../../tests/support/mod.rs"]
mod support;

#[path = "cards/aer.rs"]
mod aer;
#[path = "cards/afr.rs"]
mod afr;
#[path = "cards/akh.rs"]
mod akh;
#[path = "cards/ala.rs"]
mod ala;
#[path = "cards/apc.rs"]
mod apc;
#[path = "cards/atq.rs"]
mod atq;
#[path = "cards/avr.rs"]
mod avr;
#[path = "cards/bfz.rs"]
mod bfz;
#[path = "cards/blb.rs"]
mod blb;
#[path = "cards/bng.rs"]
mod bng;
#[path = "cards/bro.rs"]
mod bro;
#[path = "cards/c13.rs"]
mod c13;
#[path = "cards/c17.rs"]
mod c17;
#[path = "cards/c19.rs"]
mod c19;
#[path = "cards/chk.rs"]
mod chk;
#[path = "cards/clb.rs"]
mod clb;
#[path = "cards/cmr.rs"]
mod cmr;
#[path = "cards/csp.rs"]
mod csp;
#[path = "cards/definitions.rs"]
mod definitions;
#[path = "cards/dft.rs"]
mod dft;
#[path = "cards/dgm.rs"]
mod dgm;
#[path = "cards/dis.rs"]
mod dis;
#[path = "cards/dka.rs"]
mod dka;
#[path = "cards/dmu.rs"]
mod dmu;
#[path = "cards/dom.rs"]
mod dom;
#[path = "cards/dtk.rs"]
mod dtk;
#[path = "cards/ecl.rs"]
mod ecl;
#[path = "cards/eld.rs"]
mod eld;
#[path = "cards/emn.rs"]
mod emn;
#[path = "cards/eoe.rs"]
mod eoe;
#[path = "cards/fdn.rs"]
mod fdn;
#[path = "cards/frf.rs"]
mod frf;
#[path = "cards/fut.rs"]
mod fut;
#[path = "cards/gpt.rs"]
mod gpt;
#[path = "cards/grn.rs"]
mod grn;
#[path = "cards/gtc.rs"]
mod gtc;
#[path = "cards/hob.rs"]
mod hob;
#[path = "cards/hou.rs"]
mod hou;
#[path = "cards/ice.rs"]
mod ice;
#[path = "cards/iko.rs"]
mod iko;
#[path = "cards/inv.rs"]
mod inv;
#[path = "cards/isd.rs"]
mod isd;
#[path = "cards/j22.rs"]
mod j22;
#[path = "cards/jmp.rs"]
mod jmp;
#[path = "cards/jou.rs"]
mod jou;
#[path = "cards/khm.rs"]
mod khm;
#[path = "cards/kld.rs"]
mod kld;
#[path = "cards/ktk.rs"]
mod ktk;
#[path = "cards/lea.rs"]
mod lea;
#[path = "cards/lrw.rs"]
mod lrw;
#[path = "cards/m10.rs"]
mod m10;
#[path = "cards/m12.rs"]
mod m12;
#[path = "cards/m14.rs"]
mod m14;
#[path = "cards/m15.rs"]
mod m15;
#[path = "cards/m19.rs"]
mod m19;
#[path = "cards/m20.rs"]
mod m20;
#[path = "cards/m21.rs"]
mod m21;
#[path = "cards/mbs.rs"]
mod mbs;
#[path = "cards/mh1.rs"]
mod mh1;
#[path = "cards/mh2.rs"]
mod mh2;
#[path = "cards/mid.rs"]
mod mid;
#[path = "cards/mir.rs"]
mod mir;
#[path = "cards/mkm.rs"]
mod mkm;
#[path = "cards/mmq.rs"]
mod mmq;
#[path = "cards/mom.rs"]
mod mom;
#[path = "cards/mor.rs"]
mod mor;
#[path = "cards/mrd.rs"]
mod mrd;
#[path = "cards/msh.rs"]
mod msh;
#[path = "cards/nec.rs"]
mod nec;
#[path = "cards/observability.rs"]
mod observability;
#[path = "cards/ody.rs"]
mod ody;
#[path = "cards/ogw.rs"]
mod ogw;
#[path = "cards/one.rs"]
mod one;
#[path = "cards/ons.rs"]
mod ons;
#[path = "cards/ori.rs"]
mod ori;
#[path = "cards/p02.rs"]
mod p02;
#[path = "cards/pls.rs"]
mod pls;
#[path = "cards/pm15.rs"]
mod pm15;
#[path = "cards/por.rs"]
mod por;
#[path = "cards/psal.rs"]
mod psal;
#[path = "cards/ptk.rs"]
mod ptk;
#[path = "cards/rav.rs"]
mod rav;
#[path = "cards/rix.rs"]
mod rix;
#[path = "cards/roe.rs"]
mod roe;
#[path = "cards/rtr.rs"]
mod rtr;
#[path = "cards/scg.rs"]
mod scg;
#[path = "cards/scryfall.rs"]
mod scryfall;
#[path = "cards/shm.rs"]
mod shm;
#[path = "cards/snc.rs"]
mod snc;
#[path = "cards/soi.rs"]
mod soi;
#[path = "cards/spm.rs"]
mod spm;
#[path = "cards/sth.rs"]
mod sth;
#[path = "cards/stx.rs"]
mod stx;
#[path = "cards/tdm.rs"]
mod tdm;
#[path = "cards/thb.rs"]
mod thb;
#[path = "cards/ths.rs"]
mod ths;
#[path = "cards/tla.rs"]
mod tla;
#[path = "cards/tle.rs"]
mod tle;
#[path = "cards/tmp.rs"]
mod tmp;
#[path = "cards/tsp.rs"]
mod tsp;
#[path = "cards/ulg.rs"]
mod ulg;
#[path = "cards/usg.rs"]
mod usg;
#[path = "cards/vis.rs"]
mod vis;
#[path = "cards/war.rs"]
mod war;
#[path = "cards/woe.rs"]
mod woe;
#[path = "cards/wth.rs"]
mod wth;
#[path = "cards/wwk.rs"]
mod wwk;
#[path = "cards/xln.rs"]
mod xln;
#[path = "cards/zen.rs"]
mod zen;
#[path = "cards/znr.rs"]
mod znr;

/// Play an unblocked combat with just the printed creature on the battlefield.
fn printed_creature_combat(
    name: &str,
    power: i32,
    toughness: i32,
    keywords: &[managym::state::card::Keyword],
) {
    use managym::{flow::turn::StepKind, state::card::Keyword};
    use support::{
        game::five_color_setup,
        inspect::permanent_named,
        play::{attack_with, play_and_settle},
    };
    let mut s = five_color_setup(&[name]);
    s.set_player_mana_pool(0, "20WWWWWWUUUUUUBBBBBBRRRRRRGGGGGG");
    play_and_settle(&mut s, name);
    let id = permanent_named(&s, name);
    let face = s.game().characteristics(id);
    assert_eq!(face.pt(), (power, toughness));
    let keywords = managym::state::card::Keywords::of(keywords);
    assert_eq!(face.keywords, keywords);
    if !keywords.has(Keyword::Haste) {
        s.advance_to_active_step(1, StepKind::Main);
    }
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, id);
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    let damage = power
        * if keywords.has(Keyword::DoubleStrike) {
            2
        } else {
            1
        };
    s.assert_life(1, 20 - damage);
    s.assert_life(
        0,
        20 + if keywords.has(Keyword::Lifelink) {
            damage
        } else {
            0
        },
    );
    assert_eq!(
        support::inspect::tapped(&s, id),
        !keywords.has(Keyword::Vigilance)
    );
}

/// Check the controller's tokens in the resolved position, including any active team bonus.
fn assert_creature_tokens(
    s: &support::game::Scenario,
    name: &str,
    count: usize,
    pt: (i32, i32),
    colors: &[managym::state::mana::Color],
    keywords: &[managym::state::card::Keyword],
) {
    use managym::state::{card::Keywords, game_object::ObjectKind, mana::Colors};
    let tokens = s.battlefield_permanents_named(0, name);
    assert_eq!(tokens.len(), count);
    let subtypes: Vec<_> = name.split_whitespace().collect();
    for id in tokens {
        assert_eq!(
            s.game().state[support::inspect::card_of(s, id)].kind(),
            ObjectKind::Token
        );
        let face = s.game().characteristics(id);
        assert_eq!(face.pt(), pt);
        assert_eq!(face.creature_types, subtypes);
        assert_eq!(face.colors, Colors::from_iter(colors.iter().copied()));
        assert_eq!(face.keywords, Keywords::of(keywords));
    }
}

fn check_counterspell(counter: &str, eligible: &str, ineligible: &str) {
    use managym::state::{game_object::Target, zone::ZoneType};
    use support::{
        game::{five_color_setup, move_to_hand},
        inspect::{now, zone_of},
        play::{cast, cast_card, castable, settle, spend_colorless},
    };
    let mut s = five_color_setup(&[counter, eligible, ineligible]);
    let counter = move_to_hand(&mut s, counter);
    cast(&mut s, ineligible);
    spend_colorless(&mut s);
    assert!(!castable(&mut s, counter));
    settle(&mut s);
    let spell = cast(&mut s, eligible);
    spend_colorless(&mut s);
    cast_card(&mut s, counter);
    assert!(s.choose_target(Target::StackSpell(now(&s, spell))));
    settle(&mut s);
    assert_eq!(zone_of(&s, spell), Some(ZoneType::Graveyard));
}

fn check_targeted_removal(
    spell: &str,
    eligible: &str,
    ineligible: &str,
    destination: managym::state::zone::ZoneType,
) {
    use managym::state::game_object::Target;
    use support::{
        game::five_color_setup,
        inspect::{card_of, zone_of},
        play::{cast, settle},
    };
    let mut s = five_color_setup(&[spell, eligible, ineligible]);
    let target = s.force_permanent_on_battlefield(1, eligible);
    let card = card_of(&s, target);
    let excluded = s.force_permanent_on_battlefield(1, ineligible);
    cast(&mut s, spell);
    assert!(!s.choose_target(Target::Permanent(excluded)));
    assert!(s.choose_target(Target::Permanent(target)));
    settle(&mut s);
    assert_eq!(zone_of(&s, card), Some(destination));
}

/// Cast the Aura or pay the Equipment's equip cost, then remove the attachment.
fn check_attachment_grant(
    name: &str,
    equip_cost: Option<&str>,
    wearer_pt: (i32, i32),
    granted: &[managym::state::card::Keyword],
) {
    use managym::state::{card::Keywords, game_object::Target, zone::ZoneType};
    use support::{
        game::{five_color_setup, move_card},
        inspect::{card_of, offered_targets, permanent_named},
        play::{activate, cast, settle},
    };

    let mut s = five_color_setup(&[name, "Gray Ogre"]);
    let wearer = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let ally = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let enemy = s.force_permanent_on_battlefield(1, "Gray Ogre");
    if let Some(cost) = equip_cost {
        let equipment = s.force_permanent_on_battlefield(0, name);
        s.set_player_mana_pool(0, cost);
        activate(&mut s, equipment, 0);
    } else {
        cast(&mut s, name);
    }
    let offered = offered_targets(&s);
    assert!(offered.contains(&Target::Permanent(ally)));
    assert_eq!(
        offered.contains(&Target::Permanent(enemy)),
        equip_cost.is_none()
    );
    assert!(s.choose_target(Target::Permanent(wearer)));
    settle(&mut s);

    let face = s.game().characteristics(wearer);
    assert_eq!(face.pt(), wearer_pt);
    assert_eq!(face.keywords, Keywords::of(granted));
    for other in [ally, enemy] {
        let face = s.game().characteristics(other);
        assert_eq!(face.pt(), (2, 2));
        assert_eq!(face.keywords, Keywords::default());
    }

    let source = card_of(&s, permanent_named(&s, name));
    move_card(&mut s, source, ZoneType::Graveyard);
    let face = s.game().characteristics(wearer);
    assert_eq!(face.pt(), (2, 2));
    assert_eq!(face.keywords, Keywords::default());
}

fn assert_surveil_one(s: &mut support::game::Scenario) {
    use crate::support::inspect::{library_top, offered_cards, zone_of};
    use crate::support::play::{choose, pass_until, select_card, settle};
    use managym::agent::action::{Action, ActionSpaceKind};
    use managym::state::game_object::PlayerId;
    use managym::state::zone::ZoneType;

    pass_until(s, ActionSpaceKind::LookAndSelect);
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    let top = library_top(s, 0);
    assert_eq!(offered_cards(s), [top]);
    let mut keep = s.clone();
    choose(&mut keep, |a| matches!(a, Action::Decline { .. }));
    settle(&mut keep);
    assert_eq!(library_top(&keep, 0), top);
    select_card(s, top);
    settle(s);
    assert_eq!(zone_of(s, top), Some(ZoneType::Graveyard));
}

fn assert_draw_then_discard(
    s: &mut support::game::Scenario,
    before: managym::state::game_object::ObjectId,
    drawn: managym::state::game_object::ObjectId,
) {
    use crate::support::inspect::{offered_cards, zone_of};
    use crate::support::play::{fork_selecting, pass_until, settle};
    use managym::agent::action::ActionSpaceKind;
    use managym::state::game_object::PlayerId;
    use managym::state::zone::ZoneType;

    pass_until(s, ActionSpaceKind::Discard);
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    let choices = offered_cards(s);
    assert_eq!(choices.len(), 2);
    for (discard, keep) in [(before, drawn), (drawn, before)] {
        let discard = support::inspect::now(s, discard);
        assert!(choices.contains(&discard));
        let mut fork = fork_selecting(s, discard);
        settle(&mut fork);
        assert_eq!(zone_of(&fork, discard), Some(ZoneType::Graveyard));
        assert_eq!(zone_of(&fork, keep), Some(ZoneType::Hand));
        fork.assert_zone_size(0, ZoneType::Hand, 1);
    }
}

/// Check the target's characteristics, the caster's gains and expiry at cleanup.
fn assert_targeted_pump(
    spell: &str,
    creature: &str,
    controller: usize,
    pt: (i32, i32),
    granted: &[managym::state::card::Keyword],
    life: i32,
    drawn: usize,
) {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{hand, keywords};
    use crate::support::play::{cast, settle};
    use managym::flow::turn::StepKind;
    use managym::state::game_object::Target;

    let mut s = five_color_setup(&[spell, creature]);
    let subject = s.force_permanent_on_battlefield(controller, creature);
    let printed_pt = s.game().characteristics(subject).pt();
    let printed_keywords = keywords(&s, subject);
    let (life_before, hand_before) = (s.life(0), hand(&s, 0).len());
    cast(&mut s, spell);
    assert!(s.choose_target(Target::Permanent(subject)));
    settle(&mut s);
    assert_eq!(s.game().characteristics(subject).pt(), pt, "{spell}");
    assert_eq!(
        keywords(&s, subject),
        printed_keywords.union(managym::state::card::Keywords::of(granted)),
        "{spell}"
    );
    assert_eq!(s.life(0), life_before + life, "{spell}");
    assert_eq!(hand(&s, 0).len(), hand_before + drawn, "{spell}");
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(
        s.game().characteristics(subject).pt(),
        printed_pt,
        "{spell}"
    );
    assert_eq!(keywords(&s, subject), printed_keywords, "{spell}");
}

/// Makes the permanent a planeswalker: the catalog defines none, so target
/// legality is checked on a constructed one.
fn make_planeswalker(
    s: &mut support::game::Scenario,
    id: managym::state::game_object::PermanentId,
) {
    use managym::state::card::{CardType, CardTypes};
    let object = s.game().state.permanents[id].as_ref().unwrap().object;
    s.game_mut().state[object].types = CardTypes::new([CardType::Planeswalker]);
}

fn assert_optional_revealed_search(name: &str, eligible: &[&str], excluded: &[&str]) {
    use crate::support::game::{five_color_setup, put_onto_battlefield};
    use crate::support::inspect::{library, offered_cards, zone_of};
    use crate::support::play::{choose_mode, fork_selecting, pass_until, settle};
    use managym::agent::action::ActionSpaceKind;
    use managym::flow::event::GameEvent;
    use managym::state::game_object::PlayerId;
    use managym::state::zone::ZoneType;

    let mut deck = vec![name];
    deck.extend_from_slice(eligible);
    deck.extend_from_slice(excluded);
    let mut s = five_color_setup(&deck);
    put_onto_battlefield(&mut s, 0, name);
    pass_until(&mut s, ActionSpaceKind::Modal);
    let order = library(&s, 0);
    let mut decline = s.clone();
    choose_mode(&mut decline, 1);
    settle(&mut decline);
    assert_eq!(library(&decline, 0), order);
    choose_mode(&mut s, 0);
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    let offered = offered_cards(&s);
    assert!(offered
        .iter()
        .all(|id| eligible.contains(&s.game().state[*id].name.as_str())));
    for name in eligible {
        let chosen = *offered
            .iter()
            .find(|id| s.game().state[**id].name == *name)
            .unwrap();
        let definition = s.game().state[chosen].definition_id;
        let event_cursor = s.game().state.events.len();
        let mut fork = fork_selecting(&s, chosen);
        settle(&mut fork);
        assert_eq!(zone_of(&fork, chosen), Some(ZoneType::Hand));
        assert!(fork.game().state.events.iter().skip(event_cursor).any(|event| matches!(
            event,
            GameEvent::CardRevealed { owner: PlayerId(0), definition: shown } if *shown == definition
        )));
        assert_eq!(
            fork.game().state.players[0].known_hand.get(&definition),
            Some(&1)
        );
    }

    // Finding nothing still shuffles, without asking for a choice.
    let mut deck = vec![name];
    deck.extend_from_slice(excluded);
    let mut s = five_color_setup(&deck);
    put_onto_battlefield(&mut s, 0, name);
    pass_until(&mut s, ActionSpaceKind::Modal);
    let mut order = library(&s, 0);
    choose_mode(&mut s, 0);
    assert_ne!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
    settle(&mut s);
    let mut shuffled = library(&s, 0);
    assert_ne!(shuffled, order);
    order.sort();
    shuffled.sort();
    assert_eq!(shuffled, order);
}

fn assert_required_library_search(name: &str, destination: managym::state::zone::ZoneType) {
    use crate::support::game::{five_color_setup, put_onto_battlefield};
    use crate::support::inspect::{has_action, library, library_card, offered_cards, zone_of};
    use crate::support::play::{fork_selecting, pass_until, settle};
    use managym::agent::action::{Action, ActionSpaceKind};
    use managym::flow::event::GameEvent;
    use managym::state::game_object::PlayerId;
    use managym::state::zone::ZoneType;

    let mut s = five_color_setup(&[name, "Gray Ogre"]);
    put_onto_battlefield(&mut s, 0, name);
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    assert!(!has_action(&s, |a| matches!(a, Action::Decline { .. })));
    let offered = offered_cards(&s);
    assert_eq!(offered.len(), library(&s, 0).len());
    for name in ["Gray Ogre", "Forest"] {
        let chosen = library_card(&s, 0, name);
        let definition = s.game().state[chosen].definition_id;
        let event_cursor = s.game().state.events.len();
        let mut fork = fork_selecting(&s, chosen);
        settle(&mut fork);
        assert_eq!(zone_of(&fork, chosen), Some(destination));
        if destination == ZoneType::Hand {
            assert!(!fork.game().state.events.iter().skip(event_cursor).any(|event|
                matches!(event,GameEvent::CardRevealed {definition:shown,..} if *shown==definition)
            ));
            assert!(!fork.game().state.players[0]
                .known_hand
                .contains_key(&definition));
        }
    }
}

fn assert_revealed_discard(spell: &str, eligible: &[&str]) {
    use managym::{
        agent::action::ActionSpaceKind,
        state::{
            game_object::{PlayerId, Target},
            zone::ZoneType,
        },
    };
    use support::{
        game::{five_color_setup, move_from_library},
        inspect::{offered_cards, zone_of},
        play::{cast, fork_selecting, pass_until, settle, spend_colorless},
    };

    let mut s = five_color_setup(&[spell, "Gray Ogre", "Lightning Bolt", "Giant Growth"]);
    let hand = ["Gray Ogre", "Forest", "Lightning Bolt", "Giant Growth"]
        .map(|name| move_from_library(&mut s, 1, name, ZoneType::Hand));
    cast(&mut s, spell);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::Discard);
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    let offered = offered_cards(&s);
    assert_eq!(offered.len(), eligible.len());
    for id in hand {
        let face = &s.game().state[id];
        assert_eq!(
            s.game().state.players[1]
                .known_hand
                .get(&face.definition_id),
            Some(&1)
        );
        assert_eq!(
            offered.contains(&id),
            eligible.contains(&face.name.as_str())
        );
    }
    for selected in offered {
        let mut branch = fork_selecting(&s, selected);
        settle(&mut branch);
        for id in hand {
            assert_eq!(
                zone_of(&branch, id),
                Some(if id == selected {
                    ZoneType::Graveyard
                } else {
                    ZoneType::Hand
                })
            );
        }
    }
}
