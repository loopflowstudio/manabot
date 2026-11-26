//! Card behavior tests, one module per set module in `src/`: a test lives
//! with the set that first printed the card it is about. Engine rules that
//! are not about one card stay in the engine's `rules_tests`. Tests never
//! decide whether a card is supported; see `cards/README.md` for coverage.
//! The scenario helpers are the engine's `tests/support`, mounted by path so
//! each primitive stays defined once.

#[path = "../../tests/support/mod.rs"]
mod support;

use managym::{agent::env::GameSnapshot, state::game_object::ObjectId};
use support::game::{deck, Scenario};

// Constructed card positions use the `fixtures` access this crate enables.
impl Scenario {
    /// Capture this constructed decision for exact native forks.
    pub fn snapshot(&self) -> GameSnapshot {
        use managym::{state::player::PlayerConfig, Env};
        let mut env = Env::new(1, false, false, false);
        env.reset(
            ["a", "b"]
                .map(|name| PlayerConfig::new(name, deck(&[("Island", 40)])))
                .to_vec(),
        )
        .unwrap();
        *env.fixture_game_mut() = self.game().clone();
        env.snapshot().unwrap()
    }

    fn restore(&mut self, snapshot: &GameSnapshot) {
        let mut env = snapshot.restore(Default::default()).unwrap();
        *self.game_mut() = env.fixture_game_mut().clone();
    }
}

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
#[path = "cards/arb.rs"]
mod arb;
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
#[path = "cards/c15.rs"]
mod c15;
#[path = "cards/c17.rs"]
mod c17;
#[path = "cards/c19.rs"]
mod c19;
#[path = "cards/chk.rs"]
mod chk;
#[path = "cards/clb.rs"]
mod clb;
#[path = "cards/cmd.rs"]
mod cmd;
#[path = "cards/cmr.rs"]
mod cmr;
#[path = "cards/con.rs"]
mod con;
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
#[path = "cards/drk.rs"]
mod drk;
#[path = "cards/dsk.rs"]
mod dsk;
#[path = "cards/dst.rs"]
mod dst;
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
#[path = "cards/fin.rs"]
mod fin;
#[path = "cards/fra.rs"]
mod fra;
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
#[path = "cards/khc.rs"]
mod khc;
#[path = "cards/khm.rs"]
mod khm;
#[path = "cards/kld.rs"]
mod kld;
#[path = "cards/ktk.rs"]
mod ktk;
#[path = "cards/lci.rs"]
mod lci;
#[path = "cards/lea.rs"]
mod lea;
#[path = "cards/lrw.rs"]
mod lrw;
#[path = "cards/m10.rs"]
mod m10;
#[path = "cards/m12.rs"]
mod m12;
#[path = "cards/m13.rs"]
mod m13;
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
#[path = "cards/pjou.rs"]
mod pjou;
#[path = "cards/pls.rs"]
mod pls;
#[path = "cards/pm15.rs"]
mod pm15;
#[path = "cards/por.rs"]
mod por;
#[path = "cards/prtr.rs"]
mod prtr;
#[path = "cards/psal.rs"]
mod psal;
#[path = "cards/psnc.rs"]
mod psnc;
#[path = "cards/ptk.rs"]
mod ptk;
#[path = "cards/rav.rs"]
mod rav;
#[path = "cards/rix.rs"]
mod rix;
#[path = "cards/rna.rs"]
mod rna;
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
#[path = "cards/sok.rs"]
mod sok;
#[path = "cards/som.rs"]
mod som;
#[path = "cards/sos.rs"]
mod sos;
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
#[path = "cards/tmt.rs"]
mod tmt;
#[path = "cards/tsp.rs"]
mod tsp;
#[path = "cards/ulg.rs"]
mod ulg;
#[path = "cards/usg.rs"]
mod usg;
#[path = "cards/vis.rs"]
mod vis;
#[path = "cards/voc.rs"]
mod voc;
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
        assert_eq!(face.creature_types.listed(), subtypes);
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

/// The ETB-targeting `name` enters from player 0's library, offers exactly
/// `offers` for its trigger's target, and is handed `choice`.
fn enter_choosing(
    s: &mut support::game::Scenario,
    name: &str,
    offers: &[managym::state::game_object::Target],
    choice: managym::state::game_object::Target,
) -> managym::state::game_object::ObjectId {
    use support::{
        inspect::{now, offered_targets},
        play::{enter_until_target, settle},
    };
    let card = enter_until_target(s, name);
    let mut offered = offered_targets(s);
    offered.sort_by_key(|target| format!("{target:?}"));
    let mut wanted = offers.to_vec();
    wanted.sort_by_key(|target| format!("{target:?}"));
    assert_eq!(offered, wanted, "{name} offers");
    assert!(s.choose_target(choice));
    settle(s);
    now(s, card)
}

/// "When this creature enters, return target creature an opponent controls
/// to its owner's hand": only the opposing creature is offered.
fn bounces_an_opposing_creature(name: &str) {
    use managym::state::{game_object::Target, zone::ZoneType};
    use support::{
        game::five_color_setup,
        inspect::{card_of, zone_of},
    };
    let mut s = five_color_setup(&[name, "Gray Ogre"]);
    let ours = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let card = card_of(&s, theirs);
    enter_choosing(
        &mut s,
        name,
        &[Target::Permanent(theirs)],
        Target::Permanent(theirs),
    );
    assert_eq!(zone_of(&s, card), Some(ZoneType::Hand));
    assert_eq!(s.game().state[support::inspect::now(&s, card)].owner.0, 1);
    assert!(s.game().state.permanents[ours].is_some());
}

/// "Whenever another creature you control enters, you gain 1 life": the
/// source's own entry and an opposing creature's entry gain nothing.
fn gains_life_when_another_creature_enters(name: &str) {
    use support::{
        game::five_color_setup,
        inspect::library_card,
        play::{enter_and_settle, move_and_settle},
    };
    let mut s = five_color_setup(&[name, "Gray Ogre"]);
    enter_and_settle(&mut s, name);
    s.assert_life(0, 20);
    enter_and_settle(&mut s, "Gray Ogre");
    s.assert_life(0, 21);
    let enemy = library_card(&s, 1, "Gray Ogre");
    move_and_settle(&mut s, enemy, managym::state::zone::ZoneType::Battlefield);
    s.assert_life(0, 21);
}

/// `name` is ready to attack alone, beside an idle Gray Ogre of each player;
/// the attack trigger resolves before blockers.
fn attack_and_resolve(
    name: &str,
) -> (
    support::game::Scenario,
    managym::state::game_object::PermanentId,
    managym::state::game_object::PermanentId,
    managym::state::game_object::PermanentId,
) {
    use managym::flow::turn::StepKind;
    use support::{
        game::{clear_summoning_sickness, five_color_setup},
        play::{attack_with, settle},
    };
    let mut s = five_color_setup(&[name, "Gray Ogre"]);
    let source = s.force_permanent_on_battlefield(0, name);
    let ally = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let enemy = s.force_permanent_on_battlefield(1, "Gray Ogre");
    clear_summoning_sickness(&mut s, source);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, source);
    settle(&mut s);
    assert_eq!(s.current_step(), StepKind::DeclareAttackers);
    (s, source, ally, enemy)
}

/// "Whenever this creature attacks, [scope] creatures you control get +1/+1
/// until end of turn": the bonus reaches the controller's creatures named by
/// the scope, never the opponent's, and ends at cleanup.
fn attack_pumps_the_team(name: &str, printed: (i32, i32), includes_self: bool) {
    use managym::flow::turn::StepKind;
    let (mut s, source, ally, enemy) = attack_and_resolve(name);
    let bonus = i32::from(includes_self);
    assert_eq!(
        s.game().characteristics(source).pt(),
        (printed.0 + bonus, printed.1 + bonus)
    );
    assert_eq!(s.game().characteristics(ally).pt(), (3, 3));
    assert_eq!(s.game().characteristics(enemy).pt(), (2, 2));
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().characteristics(source).pt(), printed);
    assert_eq!(s.game().characteristics(ally).pt(), (2, 2));
}

/// Walk the active player's attack declaration, declaring only `attackers`
/// (or the one offered choice), and return each asked creature with the
/// attack values it was offered.
fn walk_attack_declaration(
    s: &mut support::game::Scenario,
    attackers: &[managym::state::game_object::PermanentId],
) -> Vec<(managym::state::game_object::PermanentId, Vec<bool>)> {
    use managym::agent::action::{Action, ActionSpaceKind};
    let mut asked = Vec::new();
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        let offered: Vec<_> = s
            .action_space()
            .actions
            .iter()
            .filter_map(|action| match action {
                Action::DeclareAttacker {
                    permanent, attack, ..
                } => Some((*permanent, attack.is_some())),
                _ => None,
            })
            .collect();
        let permanent = offered[0].0;
        let wanted = attackers.contains(&permanent);
        let index = offered
            .iter()
            .position(|(_, attack)| *attack == wanted)
            .unwrap_or(0);
        asked.push((
            permanent,
            offered.iter().map(|(_, attack)| *attack).collect(),
        ));
        s.step_action(index);
    }
    asked
}

/// Walk the defending player's block declaration, declining every block,
/// and return each asked blocker with the attackers it was offered.
fn walk_block_declaration(
    s: &mut support::game::Scenario,
) -> Vec<(
    managym::state::game_object::PermanentId,
    Vec<managym::state::game_object::PermanentId>,
)> {
    use managym::agent::action::{Action, ActionSpaceKind};
    s.advance_until(
        |s| {
            s.action_space().kind == ActionSpaceKind::DeclareBlocker
                || s.current_step() as i32 > managym::flow::turn::StepKind::DeclareBlockers as i32
        },
        "blockers".into(),
    );
    let mut asked = Vec::new();
    while s.action_space().kind == ActionSpaceKind::DeclareBlocker {
        let actions = s.action_space().actions.clone();
        let mut blocker = None;
        let mut offered = Vec::new();
        let mut decline = 0;
        for (index, action) in actions.iter().enumerate() {
            if let Action::DeclareBlocker {
                blocker: who,
                attacker,
                ..
            } = action
            {
                blocker = Some(*who);
                match attacker {
                    Some(attacker) => offered.push(*attacker),
                    None => decline = index,
                }
            }
        }
        asked.push((blocker.unwrap(), offered));
        s.step_action(decline);
    }
    asked
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

/// A characteristic-setting Aura cast on Juggernaut (a colorless 5/3
/// artifact creature that attacks each combat) leaves exactly `name`, the
/// creature type `subtype`, `colors`, base 1/1 and no abilities; moving the
/// Aura away restores the printed values.
fn assert_sets_juggernaut_characteristics(
    aura: &str,
    name: &str,
    subtype: &str,
    colors: &[managym::state::mana::Color],
) {
    use managym::{
        agent::action::ActionSpaceKind,
        state::{card::CardType, game_object::Target, zone::ZoneType},
    };
    use support::{
        game::{five_color_setup, move_card},
        inspect::permanent_named,
        play::{cast, pass_until, settle},
    };
    let mut s = five_color_setup(&[aura, "Juggernaut"]);
    let juggernaut = s.force_permanent_on_battlefield(1, "Juggernaut");
    let printed = s.game().characteristics(juggernaut);
    assert!(printed.types.types.contains(&CardType::Artifact));
    cast(&mut s, aura);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(juggernaut)));
    settle(&mut s);
    let set = s.game().characteristics(juggernaut);
    assert_eq!(set.name, name, "{aura}");
    assert_eq!(
        set.types.types.iter().copied().collect::<Vec<_>>(),
        vec![CardType::Creature],
        "{aura}: no longer an artifact"
    );
    assert_eq!(set.subtypes, vec![subtype.to_string()], "{aura}");
    assert_eq!(set.colors, colors.iter().copied().collect(), "{aura}");
    assert_eq!(set.pt(), (1, 1), "{aura}");
    assert!(
        set.static_abilities.is_empty(),
        "{aura}: loses attacks each combat"
    );

    let attached = permanent_named(&s, aura);
    let card = support::inspect::card_of(&s, attached);
    move_card(&mut s, card, ZoneType::Graveyard);
    let restored = s.game().characteristics(juggernaut);
    assert_eq!(restored.name, printed.name);
    assert_eq!(restored.types, printed.types);
    assert_eq!(restored.subtypes, printed.subtypes);
    assert_eq!(restored.colors, printed.colors);
    assert_eq!(restored.pt(), printed.pt());
    assert_eq!(restored.static_abilities, printed.static_abilities);
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

/// `name` deals `damage` to target creature or planeswalker, and that
/// permanent is exiled if it would die this turn (CR 614.1a): from lethal
/// damage, from a planeswalker's zero loyalty, and from a later sacrifice. A
/// survivor's replacement ends in the cleanup step (CR 514.2).
fn assert_damage_exiles_creature_or_planeswalker(name: &str, damage: i32) {
    use managym::{
        flow::turn::StepKind,
        state::{game_object::Target, permanent::CounterKind, zone::ZoneType},
    };
    use support::{
        game::{five_color_setup, move_card, set_counters},
        inspect::{card_of, permanent_of, zone_of},
        play::{cast, settle},
    };
    let setup = || five_color_setup(&[name, "Gray Ogre", "Wall of Stone", "Blasphemous Edict"]);
    let strike = |s: &mut support::game::Scenario, victim| {
        s.set_player_mana_pool(0, "1R");
        cast(s, name);
        assert!(s.choose_target(Target::Permanent(victim)));
        settle(s);
    };

    // Lethal damage: state-based actions exile the Ogre.
    let mut s = setup();
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let card = card_of(&s, ogre);
    strike(&mut s, ogre);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Exile), "{name}: creature");

    // A planeswalker left with no loyalty is exiled.
    let mut s = setup();
    let walker = s.force_permanent_on_battlefield(1, "Wall of Stone");
    make_planeswalker(&mut s, walker);
    set_counters(&mut s, walker, CounterKind::Loyalty, damage);
    let card = card_of(&s, walker);
    strike(&mut s, walker);
    assert_eq!(
        zone_of(&s, card),
        Some(ZoneType::Exile),
        "{name}: planeswalker"
    );

    // A survivor sacrificed later this turn is exiled.
    let mut s = setup();
    let wall = s.force_permanent_on_battlefield(1, "Wall of Stone");
    let card = card_of(&s, wall);
    strike(&mut s, wall);
    assert_eq!(support::inspect::permanent(&s, wall).damage, damage);
    assert!(s.game().exiled_if_it_would_die(wall));
    s.set_player_mana_pool(0, "3BB");
    cast(&mut s, "Blasphemous Edict");
    settle(&mut s);
    assert_eq!(
        zone_of(&s, card),
        Some(ZoneType::Exile),
        "{name}: sacrifice"
    );

    // The replacement ends in the cleanup step.
    let mut s = setup();
    let wall = s.force_permanent_on_battlefield(1, "Wall of Stone");
    let card = card_of(&s, wall);
    strike(&mut s, wall);
    s.advance_to_active_step(1, StepKind::Upkeep);
    let wall = permanent_of(&s, card).unwrap();
    assert!(!s.game().exiled_if_it_would_die(wall));
    let card = move_card(&mut s, card, ZoneType::Graveyard);
    assert_eq!(
        zone_of(&s, card),
        Some(ZoneType::Graveyard),
        "{name}: expired"
    );
}

/// Both Clerics share a graveyard and a competing restricted spend. Only
/// their second legal target and their kicker price differ.
fn check_kicked_return(name: &str, mana: &str, second_target: &str, excluded: &str) {
    use managym::{
        agent::{
            action::{Action, ActionSpaceKind},
            observation::Observation,
        },
        state::{
            game_object::{PlayerId, Target},
            mana::{Color, Mana, ManaRestriction},
            zone::ZoneType,
        },
    };
    use support::{
        game::{five_color_unfunded_setup, move_card, move_from_library},
        inspect::{offered_targets, permanent_of, top_spell, zone_of},
        play::{
            cast, cast_card, command, pass_until, resolve_top, settle, spend_colorless, spend_named,
        },
    };

    let mut s = five_color_unfunded_setup(&[
        name,
        "Llanowar Elves",
        "Sol Ring",
        "Gray Ogre",
        "Lightning Bolt",
    ]);
    let elves = move_from_library(&mut s, 0, "Llanowar Elves", ZoneType::Graveyard);
    let other = move_from_library(&mut s, 0, second_target, ZoneType::Graveyard);
    for excluded in ["Forest", "Lightning Bolt", excluded] {
        move_from_library(&mut s, 0, excluded, ZoneType::Graveyard);
    }
    move_from_library(&mut s, 1, "Llanowar Elves", ZoneType::Graveyard);
    let bolt = move_from_library(&mut s, 1, "Lightning Bolt", ZoneType::Hand);
    s.set_player_mana_pool(1, "R");
    s.set_player_mana_pool(0, mana);
    s.game_mut().state.players[0].mana_pool.add(
        &Mana::restricted_to(ManaRestriction::spell(["Cleric"]), [0, 0, 0, 0, 1, 0]),
        None,
    );

    let card = cast(&mut s, name);
    let kicker = s.snapshot();
    command(&mut s, |a| matches!(a, Action::Decline { .. }));
    spend_colorless(&mut s);
    assert_eq!(top_spell(&s).spent.total(), 2);
    resolve_top(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert!(s.game().state.stack_objects.is_empty());

    s.restore(&kicker);
    command(&mut s, |a| matches!(a, Action::PayCost { .. }));
    spend_named(&mut s, |label| label.contains("{G}"));
    spend_colorless(&mut s);
    let spent = top_spell(&s).spent.clone();
    assert_eq!(spent.total(), Mana::parse(mana).total());
    assert_eq!(spent.amounts()[Color::Green as usize], 1);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    let targets = offered_targets(&s);
    assert_eq!(targets.len(), 2);
    let source = permanent_of(&s, card).unwrap();
    let entry = s.game().state.permanents[source]
        .as_ref()
        .unwrap()
        .spell_entry
        .as_deref()
        .unwrap();
    assert!(entry.kicked);
    assert_eq!(entry.payment.as_ref(), Some(&spent));
    let chooser = Observation::for_player(s.game(), PlayerId(0));
    assert!(chooser.continuation.iter().any(|frame| frame.kicked));
    let targeting = s.snapshot();
    for (chosen, stays) in [(elves, other), (other, elves)] {
        s.restore(&targeting);
        let target = Target::GraveyardCard(chosen);
        assert!(targets.contains(&target));
        command(
            &mut s,
            |a| matches!(a, Action::ChooseTarget { target: t, .. } if *t == target),
        );
        s.pass_priority();
        cast_card(&mut s, bolt);
        assert!(s.choose_target(Target::Permanent(source)));
        resolve_top(&mut s);
        assert_eq!(zone_of(&s, card), Some(ZoneType::Graveyard));
        assert_eq!(zone_of(&s, chosen), Some(ZoneType::Graveyard));
        for viewer in [PlayerId(0), PlayerId(1)] {
            let obs = Observation::for_player(s.game(), viewer);
            assert_eq!(obs.stack_objects.len(), 1);
            assert!(obs.stack_objects[0].kicked);
            assert!(obs.continuation[0].kicked);
        }
        settle(&mut s);
        assert_eq!(zone_of(&s, chosen), Some(ZoneType::Battlefield));
        assert_eq!(zone_of(&s, stays), Some(ZoneType::Graveyard));
    }

    // Blink the kicked incarnation while its original trigger waits.
    s.restore(&targeting);
    command(
        &mut s,
        |a| matches!(a, Action::ChooseTarget {target: Target::GraveyardCard(id), ..} if *id == elves),
    );
    let exile = move_card(&mut s, card, ZoneType::Exile);
    let returned = move_card(&mut s, exile, ZoneType::Battlefield);
    let id = permanent_of(&s, returned).unwrap();
    assert!(s.game().state.permanents[id]
        .as_ref()
        .unwrap()
        .spell_entry
        .is_none());
    assert!(s.game().state.pending_triggers.is_empty());
    settle(&mut s);
    assert_eq!(zone_of(&s, elves), Some(ZoneType::Battlefield));
    assert_eq!(zone_of(&s, other), Some(ZoneType::Graveyard));
}

/// The objects in both graveyards after a mass return resolves, followed to
/// their current appearances.
struct GraveyardReturn {
    momo: ObjectId,
    sailor: ObjectId,
    drake: ObjectId,
    bolt: ObjectId,
    their_sailor: ObjectId,
}

/// Cast `spell` over two graveyards: the caster's Momo, Spectral Sailor
/// (mana value 1, flying), Wind Drake (mana value 3, flying) and Lightning
/// Bolt, and the opponent's Spectral Sailor. Momo triggers once for each
/// other flier entering under its controller's control together with it.
fn return_from_two_graveyards(spell: &str) -> (Scenario, GraveyardReturn) {
    use support::{
        game::{five_color_setup, plant_in_graveyard},
        inspect::now,
        play::{cast, settle, spend_colorless},
    };
    let mut s = five_color_setup(&[
        spell,
        "Momo, Friendly Flier",
        "Spectral Sailor",
        "Wind Drake",
        "Lightning Bolt",
    ]);
    let momo = plant_in_graveyard(&mut s, 0, "Momo, Friendly Flier");
    let sailor = plant_in_graveyard(&mut s, 0, "Spectral Sailor");
    let drake = plant_in_graveyard(&mut s, 0, "Wind Drake");
    let bolt = plant_in_graveyard(&mut s, 0, "Lightning Bolt");
    let their_sailor = plant_in_graveyard(&mut s, 1, "Spectral Sailor");
    cast(&mut s, spell);
    spend_colorless(&mut s);
    settle(&mut s);
    let moved = GraveyardReturn {
        momo: now(&s, momo),
        sailor: now(&s, sailor),
        drake: now(&s, drake),
        bolt: now(&s, bolt),
        their_sailor: now(&s, their_sailor),
    };
    (s, moved)
}
