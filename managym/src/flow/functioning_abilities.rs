//! Single authority for which abilities function right now across statics,
//! triggers, and replacements (CR 113.6, CR 114.4, CR 702.26b).
//!
//! Every rules consumer that gathers functioning static abilities, triggered
//! abilities, or replacement effects routes through this module instead of
//! re-checking phasing, command-zone emblem status, or zone-of-function rules.

use crate::{
    flow::game::Game,
    state::{
        ability::TriggerCondition,
        card::{EntryScope, GraveyardMover, ReplacementEffect, StaticAbility},
        game_object::{ObjectId, ObjectKind, PermanentId, PlayerId},
        zone::ZoneType,
    },
};

/// Where an ability functions under CR 113.6 and CR 114.4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FunctionZones {
    /// CR 113.6a, CR 604.3: characteristic-defining abilities function in all
    /// zones and even outside the game (except while phased out, CR 702.26b).
    AllZones,
    /// CR 113.6 default + CR 114.4: battlefield permanents (while phased in)
    /// and emblems in the command zone.
    Battlefield,
    /// Explicit opt-in for abilities that function in the command zone even on
    /// non-emblem objects (CR 113.6, CR 114.4).
    #[allow(dead_code)]
    Command,
    /// CR 113.6b: functions only while the object is a spell on the stack
    /// (`CantBeCountered`, `YouCastThis`, stack-resolution replacements).
    Stack,
    /// CR 113.6m, CR 702.143a: functions in hand (`BeginOnBattlefield`, `Foretell`).
    Hand,
    /// CR 113.6k: functions in its owner's graveyard (`Flashback`,
    /// `CastFromGraveyardRemovingCounters`, `TriggerCondition::InGraveyard`).
    Graveyard,
    /// CR 113.6f: spell cost modifiers and `FlashIf` function in any zone from
    /// which the spell can be cast and on the stack.
    Casting,
    /// CR 614.1c, CR 614.12: self-replacement effects that apply as the object
    /// enters the battlefield, never as ambient battlefield replacements.
    EnteringBattlefield,
}

/// Traversal order for functioning replacement-effect sources.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReplacementSourceOrder {
    /// Battlefield permanents of seat 0 then seat 1, then command-zone emblems.
    BySeat,
    /// Battlefield permanents in permanent-slot order, then command-zone emblems.
    SlotOrder,
    /// Battlefield permanents controlled by `player`, then `player`'s command-zone emblems.
    ControlledBy(PlayerId),
}

/// Single CR 113.6 / CR 114.4 / CR 702.26b admission predicate.
pub(crate) fn ability_functions_in(
    kind: ObjectKind,
    zone: Option<ZoneType>,
    phased_out: bool,
    zones: FunctionZones,
) -> bool {
    // CR 702.26b: phased-out permanents are treated as though they don't exist;
    // none of their abilities function.
    if zone == Some(ZoneType::Battlefield) && phased_out {
        return false;
    }
    match zones {
        FunctionZones::AllZones => true,
        FunctionZones::Battlefield => match zone {
            Some(ZoneType::Battlefield) => true,
            // CR 114.4: command-zone objects function by default only when
            // they are emblems.
            Some(ZoneType::Command) => kind == ObjectKind::Emblem,
            _ => false,
        },
        FunctionZones::Command => zone == Some(ZoneType::Command),
        FunctionZones::Stack => zone == Some(ZoneType::Stack),
        FunctionZones::Hand => zone == Some(ZoneType::Hand),
        FunctionZones::Graveyard => zone == Some(ZoneType::Graveyard),
        FunctionZones::Casting => matches!(
            zone,
            Some(
                ZoneType::Hand
                    | ZoneType::Graveyard
                    | ZoneType::Exile
                    | ZoneType::Library
                    | ZoneType::Stack
                    | ZoneType::Command
            )
        ),
        FunctionZones::EnteringBattlefield => false,
    }
}

/// Exhaustive CR 113.6 zone-of-function classification for static abilities.
pub(crate) fn static_ability_zones(ability: &StaticAbility) -> FunctionZones {
    match ability {
        StaticAbility::Continuous(effect) if effect.characteristic_defining => {
            FunctionZones::AllZones
        }
        StaticAbility::CantBeCountered => FunctionZones::Stack,
        StaticAbility::BeginOnBattlefield | StaticAbility::Foretell(_) => FunctionZones::Hand,
        StaticAbility::Flashback(_) | StaticAbility::CastFromGraveyardRemovingCounters(_) => {
            FunctionZones::Graveyard
        }
        StaticAbility::CostsLessPerPermanent(_)
        | StaticAbility::CostsLessPerPermanentOnBattlefield(_)
        | StaticAbility::CostsLessPerGraveyardCard(_)
        | StaticAbility::CostsLessByGreatestPower(_)
        | StaticAbility::CostsLess { .. }
        | StaticAbility::CostsLessIfTargets(_)
        | StaticAbility::FlashIf(_) => FunctionZones::Casting,
        StaticAbility::Continuous(_)
        | StaticAbility::HexproofFrom(_)
        | StaticAbility::Protection(_)
        | StaticAbility::ControllerHas(_)
        | StaticAbility::SpellsCantBeCountered(_)
        | StaticAbility::GraveyardSpellsHaveFlashback
        | StaticAbility::CastFromHandWithoutManaCost
        | StaticAbility::CastSpellsAsThoughFlash(_)
        | StaticAbility::GrantsAnyColorManaAbility(_)
        | StaticAbility::CantBeBlockedIf(_)
        | StaticAbility::PlayersCantGainLife
        | StaticAbility::CantAttackOrBlockUnlessYouControlAnother(_)
        | StaticAbility::BlockRestriction(_)
        | StaticAbility::SpellsCostLess(_)
        | StaticAbility::NoMaximumHandSize
        | StaticAbility::OpponentsCastOnlyFromHand
        | StaticAbility::AdditionalLandPlayEachPlayer
        | StaticAbility::AdditionalLandPlay
        | StaticAbility::CantLoseAndOpponentsCantWin
        | StaticAbility::ActivatedAbilitiesCostLess { .. }
        | StaticAbility::Ward(_)
        | StaticAbility::LookAtLibraryTop
        | StaticAbility::CastFromLibraryTop(_)
        | StaticAbility::SpendAnyManaToCast(_)
        | StaticAbility::PlayGraveyardByPermanentType
        | StaticAbility::PlayUnownedExiledWithCounter(_)
        | StaticAbility::AttachedHas(_)
        | StaticAbility::AttachedSetsCharacteristics(_)
        | StaticAbility::KeywordGrant(_)
        | StaticAbility::Restriction(_)
        | StaticAbility::TypeGrant(_)
        | StaticAbility::AttacksDespiteDefender(_)
        | StaticAbility::EntersTapped(_)
        | StaticAbility::OpponentsEnterTapped(_)
        | StaticAbility::SetsLandType(_)
        | StaticAbility::TriggersAdditionalTime(_)
        | StaticAbility::TriggeredGrant(_)
        | StaticAbility::SpellsCostLessMana(_)
        | StaticAbility::SpellsFromOutsideHandCostLess(_)
        | StaticAbility::UntapsDuringOtherUntapSteps
        | StaticAbility::SpellsHave { .. }
        | StaticAbility::StateBasedActionDoesntApply { .. } => FunctionZones::Battlefield,
    }
}

/// Exhaustive CR 113.6 zone-of-function classification for triggered abilities.
pub(crate) fn trigger_condition_zones(condition: &TriggerCondition) -> FunctionZones {
    match condition {
        TriggerCondition::InGraveyard { .. } => FunctionZones::Graveyard,
        TriggerCondition::YouCastThis => FunctionZones::Stack,
        TriggerCondition::WasCast { condition }
        | TriggerCondition::OnceEachTurn { condition }
        | TriggerCondition::WhileSourceAttacking { condition }
        | TriggerCondition::FromZone { condition, .. }
        | TriggerCondition::Prepared { condition }
        | TriggerCondition::ActiveIf { condition, .. } => trigger_condition_zones(condition),
        TriggerCondition::Sacrificed { .. }
        | TriggerCondition::NoCounters { .. }
        | TriggerCondition::State(_)
        | TriggerCondition::Expend(_)
        | TriggerCondition::SpellTargetsOnlyThis { .. }
        | TriggerCondition::EntersTheBattlefield { .. }
        | TriggerCondition::Dies { .. }
        | TriggerCondition::LeavesTheBattlefield { .. }
        | TriggerCondition::Attacks { .. }
        | TriggerCondition::BecomesTapped { .. }
        | TriggerCondition::BecomesUntapped { .. }
        | TriggerCondition::TappedForMana { .. }
        | TriggerCondition::BecomesTargeted { .. }
        | TriggerCondition::YourAttackingCreatureTriggers
        | TriggerCondition::Chapter { .. }
        | TriggerCondition::NthCounter { .. }
        | TriggerCondition::PutsCounters { .. }
        | TriggerCondition::PutsCountersOn { .. }
        | TriggerCondition::Casts { .. }
        | TriggerCondition::Draws { .. }
        | TriggerCondition::GainsLife { .. }
        | TriggerCondition::LosesLife { .. }
        | TriggerCondition::BeginningOf { .. }
        | TriggerCondition::Performs { .. }
        | TriggerCondition::ActivatesLoyaltyAbility { .. }
        | TriggerCondition::PutIntoGraveyard { .. }
        | TriggerCondition::DealsCombatDamage { .. }
        | TriggerCondition::CardsLeaveYourGraveyard
        | TriggerCondition::SourceDealsDamage
        | TriggerCondition::SourceDealsCombatDamage
        | TriggerCondition::DealsDamage { .. } => FunctionZones::Battlefield,
    }
}

/// Exhaustive CR 113.6 / CR 614.12 zone-of-function classification for replacement effects.
pub(crate) fn replacement_effect_zones(effect: &ReplacementEffect) -> FunctionZones {
    match effect {
        ReplacementEffect::AdventureExile => FunctionZones::Stack,
        ReplacementEffect::GraveyardInstead {
            what: GraveyardMover::This,
            ..
        } => FunctionZones::AllZones,
        ReplacementEffect::DiscardedInstead { .. } => FunctionZones::Hand,
        ReplacementEffect::EntersPrepared
        | ReplacementEffect::AsEntersChoose(_)
        | ReplacementEffect::EntersTapped
        | ReplacementEffect::EntersTappedUnlessControls { .. }
        | ReplacementEffect::EntersWithCounters {
            on: EntryScope::This,
            ..
        }
        | ReplacementEffect::EntersAsCopy { .. } => FunctionZones::EnteringBattlefield,
        ReplacementEffect::ExileIfWouldDie { .. }
        | ReplacementEffect::GraveyardInstead {
            what: GraveyardMover::Matching(_),
            ..
        }
        | ReplacementEffect::EntersWithCounters {
            on: EntryScope::OtherYouControl(_),
            ..
        }
        | ReplacementEffect::ModifyLifeGain { .. }
        | ReplacementEffect::PreventCombatDamage
        | ReplacementEffect::ModifyDamage { .. }
        | ReplacementEffect::UnspentManaBecomesRed
        | ReplacementEffect::ModifyCounters { .. }
        | ReplacementEffect::MultiplyTokens { .. } => FunctionZones::Battlefield,
    }
}

impl Game {
    /// Whether an ability with `zones` on `object` functions right now.
    pub(crate) fn object_ability_functions(&self, object: ObjectId, zones: FunctionZones) -> bool {
        if !self.state.zones.is_latest(object) {
            return false;
        }
        let Some(record) = self.state.objects.get(object.slot()) else {
            return false;
        };
        let zone = self.state.zones.zone_of(object);
        let phased_out = zone == Some(ZoneType::Battlefield)
            && self
                .state
                .permanent_of(object)
                .is_some_and(|id| self.is_phased_out(id));
        ability_functions_in(record.kind(), zone, phased_out, zones)
    }

    /// Whether a live battlefield permanent's abilities function (CR 702.26b).
    pub(crate) fn permanent_abilities_function(&self, id: PermanentId) -> bool {
        self.state
            .permanents
            .get(id.0)
            .and_then(Option::as_ref)
            .is_some_and(|permanent| {
                self.object_ability_functions(permanent.object, FunctionZones::Battlefield)
            })
    }

    /// Whether an object in the command zone functions under CR 114.4 (emblems only by default).
    pub(crate) fn command_object_functions(&self, object: ObjectId) -> bool {
        self.state.zones.zone_of(object) == Some(ZoneType::Command)
            && self.object_ability_functions(object, FunctionZones::Battlefield)
    }

    pub(crate) fn static_ability_functions(
        &self,
        object: ObjectId,
        ability: &StaticAbility,
    ) -> bool {
        self.object_ability_functions(object, static_ability_zones(ability))
    }

    pub(crate) fn trigger_condition_functions(
        &self,
        object: ObjectId,
        condition: &TriggerCondition,
    ) -> bool {
        self.object_ability_functions(object, trigger_condition_zones(condition))
    }

    pub(crate) fn replacement_effect_functions(
        &self,
        object: ObjectId,
        effect: &ReplacementEffect,
    ) -> bool {
        self.object_ability_functions(object, replacement_effect_zones(effect))
    }

    /// Every live battlefield permanent whose abilities currently function, in slot order.
    pub(crate) fn functioning_permanents(&self) -> impl Iterator<Item = PermanentId> + '_ {
        (0..self.state.permanents.len())
            .map(PermanentId)
            .filter(|&id| self.permanent_abilities_function(id))
    }

    /// Functioning command-zone objects controlled by `controller` (CR 114.4).
    pub(crate) fn functioning_command_objects_for(
        &self,
        controller: PlayerId,
    ) -> impl Iterator<Item = ObjectId> + '_ {
        self.state
            .zones
            .zone_objects(ZoneType::Command, controller)
            .iter()
            .copied()
            .filter(|&object| self.command_object_functions(object))
    }

    /// Every functioning command-zone object across both players (CR 114.4).
    pub(crate) fn functioning_command_objects(
        &self,
    ) -> impl Iterator<Item = (PlayerId, ObjectId)> + '_ {
        [PlayerId(0), PlayerId(1)]
            .into_iter()
            .flat_map(|controller| {
                self.functioning_command_objects_for(controller)
                    .map(move |object| (controller, object))
            })
    }

    /// Visit every functioning static ability on the battlefield and command
    /// zone (for `controller`, or across both seats when `controller` is `None`),
    /// iterating only the `LayerBoard`'s condition-independent `StaticSourceIndex`
    /// (`O(|generators|)` instead of `O(|battlefield|)`).
    pub(crate) fn for_each_functioning_static(
        &self,
        controller: Option<PlayerId>,
        mut visit: impl FnMut(ObjectId, Option<PermanentId>, PlayerId, usize, &StaticAbility),
    ) {
        let board = self.layer_board();
        for id in board.static_sources.battlefield_ids(controller) {
            let Some(permanent) = self.state.permanents[id].as_ref() else {
                continue;
            };
            let Some(view) = board.objects.get(id.0).and_then(Option::as_ref) else {
                continue;
            };
            let source = permanent.object;
            let owner = permanent.controller;
            for (index, ability) in view.static_abilities.iter().enumerate() {
                if self.static_ability_functions(source, ability) {
                    visit(source, Some(id), owner, index, ability);
                }
            }
        }
        for (seat, object) in board.static_sources.command_objects(controller) {
            for (index, ability) in self.state[object].statics.iter().enumerate() {
                if self.static_ability_functions(object, ability) {
                    visit(object, None, seat, index, ability);
                }
            }
        }
    }

    /// Whether any functioning static ability (battlefield or command zone,
    /// optionally scoped to `controller`) matches `predicate`.
    pub(crate) fn has_functioning_static(
        &self,
        controller: Option<PlayerId>,
        predicate: impl Fn(&StaticAbility) -> bool,
    ) -> bool {
        let board = self.layer_board();
        for id in board.static_sources.battlefield_ids(controller) {
            let Some(permanent) = self.state.permanents[id].as_ref() else {
                continue;
            };
            let Some(view) = board.objects.get(id.0).and_then(Option::as_ref) else {
                continue;
            };
            for ability in &view.static_abilities {
                if self.static_ability_functions(permanent.object, ability) && predicate(ability) {
                    return true;
                }
            }
        }
        for (_, object) in board.static_sources.command_objects(controller) {
            for ability in &self.state[object].statics {
                if self.static_ability_functions(object, ability) && predicate(ability) {
                    return true;
                }
            }
        }
        false
    }

    /// Visit every functioning ambient replacement effect on the battlefield
    /// and command zone, respecting phasing (CR 702.26b), ability loss,
    /// command-zone emblem admission (CR 114.4), and zone-of-function (CR 113.6).
    pub(crate) fn for_each_functioning_replacement(
        &self,
        order: ReplacementSourceOrder,
        mut visit: impl FnMut(ObjectId, PlayerId, usize, &ReplacementEffect),
    ) {
        let mut visit_object = |source: ObjectId, controller: PlayerId| {
            for (index, effect) in self.state[source].replacement_effects.iter().enumerate() {
                if self.replacement_effect_functions(source, effect) {
                    visit(source, controller, index, effect);
                }
            }
        };
        match order {
            ReplacementSourceOrder::BySeat => {
                let ids = [PlayerId(0), PlayerId(1)]
                    .into_iter()
                    .flat_map(|seat| self.iter_battlefield_permanents(seat));
                for (id, view) in self.layer_views(ids) {
                    if view.abilities_lost {
                        continue;
                    }
                    let permanent = self.state.permanents[id].as_ref().expect("battlefield");
                    visit_object(permanent.object, permanent.controller);
                }
                for (controller, object) in self.functioning_command_objects() {
                    visit_object(object, controller);
                }
            }
            ReplacementSourceOrder::SlotOrder => {
                for (id, view) in self.layer_views(self.functioning_permanents()) {
                    if view.abilities_lost {
                        continue;
                    }
                    let permanent = self.state.permanents[id].as_ref().expect("battlefield");
                    visit_object(permanent.object, permanent.controller);
                }
                for (controller, object) in self.functioning_command_objects() {
                    visit_object(object, controller);
                }
            }
            ReplacementSourceOrder::ControlledBy(seat) => {
                for (id, view) in self.layer_views(self.iter_battlefield_permanents(seat)) {
                    if view.abilities_lost {
                        continue;
                    }
                    let permanent = self.state.permanents[id].as_ref().expect("battlefield");
                    visit_object(permanent.object, permanent.controller);
                }
                for object in self.functioning_command_objects_for(seat) {
                    visit_object(object, seat);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::{
        flow::{
            decision::EffectFrame,
            game::CastChoices,
            proposed_event::{ProposedDamageTarget, ProposedEvent},
        },
        state::{
            ability::{
                Ability, AbilityProvenance, Actor, Effect, LinkedAbilitySource, LinkedExile,
                ObjectSelector, TriggerSubject,
            },
            card::{
                ActivatedAbilityDefinition, DamageRecipient, DamageSource, DamageTransform,
                LayerCount, LayerValue, StaticScope,
            },
            effect::{EffectDuration, Modification},
            player::PlayerConfig,
            predicate::CardPredicate,
        },
    };

    fn test_game() -> Game {
        let mut deck = BTreeMap::new();
        deck.insert("Mountain".to_string(), 16);
        deck.insert("Gray Ogre".to_string(), 8);
        deck.insert("Lightning Bolt".to_string(), 8);
        deck.insert("Counterspell".to_string(), 4);
        deck.insert("Koh, the Face Stealer".to_string(), 2);
        deck.insert("Boiling Rock Rioter".to_string(), 2);
        deck.insert("Chandra, Flameshaper".to_string(), 2);
        let mut game = Game::new(
            vec![
                PlayerConfig::new("hero", deck.clone()),
                PlayerConfig::new("villain", deck),
            ],
            42,
            false,
        );
        game.state.turn.active_player = PlayerId(0);
        game
    }

    fn phase_out(game: &mut Game, permanent: PermanentId) {
        let affected = game.affected_permanent(permanent).unwrap();
        game.apply_continuous(
            None,
            PlayerId(0),
            [affected],
            Modification::PhasedOut,
            EffectDuration::EndOfTurn,
        );
    }

    #[test]
    fn phased_out_permanent_suppresses_statics_triggers_and_replacements() {
        let mut game = test_game();
        let source = game
            .scenario_force_battlefield(PlayerId(0), "Gray Ogre", false)
            .unwrap();
        let peer = game
            .scenario_force_battlefield(PlayerId(0), "Gray Ogre", false)
            .unwrap();
        let source_obj = game.permanent_object(source).unwrap();

        // Install a layer static (+1/+1), a non-layer static (PlayersCantGainLife),
        // a triggered ability (whenever you draw a card, gain 1 life), and a
        // damage-doubling replacement effect on `source`.
        game.state[source_obj].statics = vec![
            StaticAbility::continuous(
                StaticScope::YouControl(CardPredicate::creature()),
                None,
                vec![Modification::ModifyPt {
                    power: LayerValue::Fixed(1),
                    toughness: LayerValue::Fixed(1),
                }],
            ),
            StaticAbility::PlayersCantGainLife,
        ];
        game.state[source_obj].abilities = vec![Ability::Triggered {
            condition: TriggerCondition::Draws {
                who: Actor::You,
                nth: None,
            },
            intervening_if: None,
            effects: vec![Effect::gain_life(1)],
        }];
        game.state[source_obj].replacement_effects = vec![ReplacementEffect::ModifyDamage {
            source: DamageSource::any(Actor::You),
            recipient: DamageRecipient::any(Actor::AnyPlayer),
            combat: None,
            transform: DamageTransform::Double,
        }];

        // Baseline: while phased in, every ability functions.
        assert_eq!(game.characteristics(peer).pt(), (3, 3));
        assert!(game.has_functioning_static(None, |a| {
            matches!(a, StaticAbility::PlayersCantGainLife)
        }));
        assert!(!game.change_life(PlayerId(0), 2));
        assert_eq!(game.state.players[0].life, 20);
        game.draw_cards(PlayerId(0), 1);
        game.collect_waiting_triggers();
        assert_eq!(game.state.pending_triggers.len(), 1);
        game.state.pending_triggers_mut().clear();

        let peer_obj = game.permanent_object(peer).unwrap();
        game.apply_immediate_event(ProposedEvent::Damage {
            source: Some(peer_obj),
            target: ProposedDamageTarget::Player(PlayerId(1)),
            amount: 2,
            combat: false,
        });
        assert_eq!(game.state.players[1].life, 16);

        // CR 702.26b: once phased out, statics, triggers, and replacements all stop.
        phase_out(&mut game, source);
        assert!(!game.permanent_abilities_function(source));
        assert_eq!(game.characteristics(peer).pt(), (2, 2));
        assert!(!game.has_functioning_static(None, |a| {
            matches!(a, StaticAbility::PlayersCantGainLife)
        }));
        assert!(game.change_life(PlayerId(0), 2));
        assert_eq!(game.state.players[0].life, 22);

        game.draw_cards(PlayerId(0), 1);
        game.collect_waiting_triggers();
        assert!(
            game.state.pending_triggers.is_empty(),
            "phased-out permanent's trigger must not fire"
        );

        game.apply_immediate_event(ProposedEvent::Damage {
            source: Some(peer_obj),
            target: ProposedDamageTarget::Player(PlayerId(1)),
            amount: 2,
            combat: false,
        });
        assert_eq!(
            game.state.players[1].life, 14,
            "phased-out permanent's damage replacement must not double damage"
        );
    }

    #[test]
    fn command_zone_rejects_non_emblem_and_admits_emblem() {
        let mut game = test_game();
        let peer = game
            .scenario_force_battlefield(PlayerId(0), "Gray Ogre", false)
            .unwrap();
        let peer_obj = game.permanent_object(peer).unwrap();

        // Put a regular card (ObjectKind::Card) in the command zone with a static
        // buff, a trigger, and a damage-doubling replacement effect.
        let card_in_cmd = game
            .state
            .zones
            .top(ZoneType::Library, PlayerId(0))
            .unwrap();
        game.state
            .place_object(card_in_cmd, PlayerId(0), ZoneType::Command);
        game.state[card_in_cmd].statics = vec![StaticAbility::continuous(
            StaticScope::YouControl(CardPredicate::creature()),
            None,
            vec![Modification::ModifyPt {
                power: LayerValue::Fixed(2),
                toughness: LayerValue::Fixed(2),
            }],
        )];
        game.state[card_in_cmd].abilities = vec![Ability::Triggered {
            condition: TriggerCondition::Draws {
                who: Actor::You,
                nth: None,
            },
            intervening_if: None,
            effects: vec![Effect::gain_life(1)],
        }];
        game.state[card_in_cmd].replacement_effects = vec![ReplacementEffect::ModifyDamage {
            source: DamageSource::any(Actor::You),
            recipient: DamageRecipient::any(Actor::AnyPlayer),
            combat: None,
            transform: DamageTransform::Double,
        }];

        // CR 114.4: non-emblem in Command zone is rejected by default, admitted only
        // for explicit `FunctionZones::Command` opt-in.
        assert!(!game.command_object_functions(card_in_cmd));
        assert!(game.functioning_command_objects().next().is_none());
        assert!(game.emblems().next().is_none());
        assert!(game.object_ability_functions(card_in_cmd, FunctionZones::Command));
        assert!(!game.object_ability_functions(card_in_cmd, FunctionZones::Battlefield));
        assert_eq!(game.characteristics(peer).pt(), (2, 2));

        game.draw_cards(PlayerId(0), 1);
        game.collect_waiting_triggers();
        assert!(game.state.pending_triggers.is_empty());

        game.apply_immediate_event(ProposedEvent::Damage {
            source: Some(peer_obj),
            target: ProposedDamageTarget::Player(PlayerId(1)),
            amount: 2,
            combat: false,
        });
        assert_eq!(game.state.players[1].life, 18);

        // Now create a real emblem in the command zone and give it the same abilities.
        let emblem = game.create_emblem("emblem.chandra_torch_of_defiance", PlayerId(0));
        game.state[emblem].statics = vec![StaticAbility::continuous(
            StaticScope::YouControl(CardPredicate::creature()),
            None,
            vec![Modification::ModifyPt {
                power: LayerValue::Fixed(2),
                toughness: LayerValue::Fixed(2),
            }],
        )];
        game.state[emblem].abilities = vec![Ability::Triggered {
            condition: TriggerCondition::Draws {
                who: Actor::You,
                nth: None,
            },
            intervening_if: None,
            effects: vec![Effect::gain_life(1)],
        }];
        game.state[emblem].replacement_effects = vec![ReplacementEffect::ModifyDamage {
            source: DamageSource::any(Actor::You),
            recipient: DamageRecipient::any(Actor::AnyPlayer),
            combat: None,
            transform: DamageTransform::Double,
        }];

        assert!(game.command_object_functions(emblem));
        assert_eq!(
            game.functioning_command_objects().collect::<Vec<_>>(),
            vec![(PlayerId(0), emblem)]
        );
        assert_eq!(game.characteristics(peer).pt(), (4, 4));

        game.draw_cards(PlayerId(0), 1);
        game.collect_waiting_triggers();
        assert_eq!(game.state.pending_triggers.len(), 1);
        assert_eq!(game.state.pending_triggers[0].source_object(), emblem);

        game.apply_immediate_event(ProposedEvent::Damage {
            source: Some(peer_obj),
            target: ProposedDamageTarget::Player(PlayerId(1)),
            amount: 2,
            combat: false,
        });
        assert_eq!(
            game.state.players[1].life, 14,
            "emblem damage replacement must apply from the command zone"
        );
    }

    #[test]
    fn off_zone_cda_stack_and_graveyard_admission() {
        let mut game = test_game();
        let _bf_creature = game
            .scenario_force_battlefield(PlayerId(0), "Gray Ogre", false)
            .unwrap();
        let _bf_creature_2 = game
            .scenario_force_battlefield(PlayerId(0), "Gray Ogre", false)
            .unwrap();

        // 1. CDA functions in every zone (Hand, Graveyard, Exile, Library, Stack).
        let count = LayerValue::count(LayerCount::YouControl(CardPredicate::creature()));
        let cda_static = StaticAbility::cda(vec![Modification::SetBasePt {
            power: Some(count.clone()),
            toughness: Some(count),
        }]);
        let cda_obj = game
            .state
            .zones
            .top(ZoneType::Library, PlayerId(0))
            .unwrap();
        game.state[cda_obj].statics = vec![cda_static.clone()];
        for zone in [
            ZoneType::Library,
            ZoneType::Hand,
            ZoneType::Graveyard,
            ZoneType::Exile,
        ] {
            game.state.place_object(cda_obj, PlayerId(0), zone);
            assert!(
                game.static_ability_functions(cda_obj, &cda_static),
                "CDA must function in {zone:?}"
            );
            assert_eq!(game.object_pt(cda_obj), (2, 2));
        }

        // 2. Stack exception: CantBeCountered functions only on the stack.
        let spell_card = game
            .state
            .zones
            .top(ZoneType::Library, PlayerId(0))
            .unwrap();
        game.state[spell_card].statics = vec![StaticAbility::CantBeCountered];
        game.state
            .place_object(spell_card, PlayerId(0), ZoneType::Hand);
        assert!(!game.static_ability_functions(spell_card, &StaticAbility::CantBeCountered));
        let spell_on_stack = game.push_spell_to_stack(
            spell_card,
            PlayerId(0),
            Vec::new(),
            Vec::new(),
            CastChoices::default(),
        );
        assert!(game.static_ability_functions(spell_on_stack, &StaticAbility::CantBeCountered));
        let mut batch = game.counter_spells([spell_on_stack], None, ZoneType::Graveyard);
        let _ = game.advance_event_batch(&mut batch, None);
        assert_eq!(
            game.state.zones.zone_of(spell_on_stack),
            Some(ZoneType::Stack),
            "CantBeCountered must function on the stack and prevent countering"
        );

        // 3. Graveyard opt-in: InGraveyard trigger functions in graveyard, not on battlefield.
        let gy_trigger = TriggerCondition::InGraveyard {
            condition: Box::new(TriggerCondition::Draws {
                who: Actor::You,
                nth: None,
            }),
        };
        let gy_bf = game
            .scenario_force_battlefield(PlayerId(0), "Gray Ogre", false)
            .unwrap();
        let gy_bf_obj = game.permanent_object(gy_bf).unwrap();
        game.state[gy_bf_obj].abilities = vec![Ability::Triggered {
            condition: gy_trigger.clone(),
            intervening_if: None,
            effects: vec![Effect::gain_life(1)],
        }];
        assert!(!game.trigger_condition_functions(gy_bf_obj, &gy_trigger));
        game.draw_cards(PlayerId(0), 1);
        game.collect_waiting_triggers();
        assert!(
            game.state.pending_triggers.is_empty(),
            "InGraveyard trigger must not fire from the battlefield"
        );

        let gy_obj = game.move_object(gy_bf_obj, ZoneType::Graveyard);
        game.state[gy_obj].abilities = vec![Ability::Triggered {
            condition: gy_trigger.clone(),
            intervening_if: None,
            effects: vec![Effect::gain_life(1)],
        }];
        assert!(game.trigger_condition_functions(gy_obj, &gy_trigger));
        game.draw_cards(PlayerId(0), 1);
        game.collect_waiting_triggers();
        assert_eq!(game.state.pending_triggers.len(), 1);
        assert_eq!(game.state.pending_triggers[0].source_object(), gy_obj);
    }

    #[test]
    fn layer6_granted_ability_fails_closed_on_linked_exile_pairing() {
        let mut game = test_game();
        let koh = game
            .scenario_force_battlefield(PlayerId(0), "Koh, the Face Stealer", false)
            .unwrap();
        let koh_obj = game.permanent_object(koh).unwrap();
        let victim_a = game
            .scenario_force_battlefield(PlayerId(1), "Gray Ogre", false)
            .unwrap();
        let victim_b = game
            .scenario_force_battlefield(PlayerId(1), "Gray Ogre", false)
            .unwrap();

        // Step 1: Koh's printed (Characteristic) ability exiles `victim_a` linked with Koh.
        let mut printed_exile = EffectFrame::new(
            PlayerId(0),
            vec![Effect::exile_linked(
                ObjectSelector::Target(crate::state::ability::TargetSpec::Creature),
                LinkedExile::WithSource,
            )],
        );
        printed_exile.bind_source(Some(koh_obj), Some(koh_obj));
        printed_exile.ability_provenance = AbilityProvenance::Characteristic;
        printed_exile.targets = vec![crate::flow::decision::ResolutionTarget {
            target: crate::state::game_object::Target::Permanent(victim_a),
            requirement: 0,
            legal: true,
            division: 0,
        }];
        game.run_frame(printed_exile);
        assert_eq!(game.state.exile_links.len(), 1);
        let exiled_a = game.state.exile_links[0].exiled_object;
        assert_eq!(
            game.state.exile_links[0].characteristic_set,
            Some(game.state[koh_obj].definition_id)
        );

        // Step 2: Grant Koh a Layer-6 activated ability and a Layer-6 triggered ability.
        // Verify layers mark both with `AbilityProvenance::Granted`.
        let affected_koh = game.affected_permanent(koh).unwrap();
        game.apply_continuous(
            None,
            PlayerId(0),
            [affected_koh],
            Modification::GrantActivated(Box::new(ActivatedAbilityDefinition {
                effects: vec![Effect::return_to_hand(ObjectSelector::ExiledWithSource)],
                ..Default::default()
            })),
            EffectDuration::Indefinite,
        );
        game.apply_continuous(
            None,
            PlayerId(0),
            [affected_koh],
            Modification::GrantTriggered(Box::new(Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![Effect::CastExiledWithSource {
                    predicate: CardPredicate::creature(),
                }],
            })),
            EffectDuration::Indefinite,
        );
        let view = game.characteristics(koh);
        assert_eq!(
            view.activated_abilities.first().unwrap().provenance,
            AbilityProvenance::Characteristic
        );
        assert_eq!(
            view.activated_abilities.last().unwrap().provenance,
            AbilityProvenance::Granted
        );
        assert_eq!(
            view.triggered_abilities.last().unwrap().provenance,
            AbilityProvenance::Granted
        );

        // Step 3: A Layer-6 granted read on Koh fails closed and cannot return `exiled_a`.
        let mut granted_read = EffectFrame::new(
            PlayerId(0),
            vec![Effect::return_to_hand(ObjectSelector::ExiledWithSource)],
        );
        granted_read.bind_source(Some(koh_obj), Some(koh_obj));
        granted_read.ability_provenance = AbilityProvenance::Granted;
        assert!(game.frame_linked_ability_source(&granted_read).is_none());
        game.run_frame(granted_read);
        assert_eq!(
            game.state.zones.zone_of(exiled_a),
            Some(ZoneType::Exile),
            "Layer-6 granted ability must fail closed and not touch printed linked exile"
        );

        // Step 4: A Layer-6 granted exile on Koh latches `characteristic_set: None`.
        let mut granted_exile = EffectFrame::new(
            PlayerId(0),
            vec![Effect::exile_linked(
                ObjectSelector::Target(crate::state::ability::TargetSpec::Creature),
                LinkedExile::WithSource,
            )],
        );
        granted_exile.bind_source(Some(koh_obj), Some(koh_obj));
        granted_exile.ability_provenance = AbilityProvenance::Granted;
        granted_exile.targets = vec![crate::flow::decision::ResolutionTarget {
            target: crate::state::game_object::Target::Permanent(victim_b),
            requirement: 0,
            legal: true,
            division: 0,
        }];
        game.run_frame(granted_exile);
        assert_eq!(game.state.exile_links.len(), 2);
        let exiled_b = game.state.exile_links[1].exiled_object;
        assert_eq!(game.state.exile_links[1].characteristic_set, None);

        // Step 5: Koh's printed ability sees ONLY `exiled_a`, never `exiled_b`.
        let mut printed_read = EffectFrame::new(PlayerId(0), vec![]);
        printed_read.bind_source(Some(koh_obj), Some(koh_obj));
        printed_read.ability_provenance = AbilityProvenance::Characteristic;
        let linked_source = game
            .frame_linked_ability_source(&printed_read)
            .expect("characteristic ability carries linked source");
        assert_eq!(
            game.state.exiled_with(linked_source).collect::<Vec<_>>(),
            vec![exiled_a]
        );
        assert_ne!(exiled_a, exiled_b);

        // Step 6: Layer-6 inheritance (`HasAbilitiesOfSource`, Koh copying
        // Boiling Rock Rioter) also marks inherited abilities `Granted`, and
        // `AttachedHas` marks both `granted_by` and `AbilityProvenance::Granted`.
        let rioter_def = crate::cards::content_pack::default_content_pack()
            .definition_id("Boiling Rock Rioter")
            .unwrap();
        let timestamp = game.state.id_gen_mut().next_id();
        game.state
            .effects
            .push(crate::state::effect::ContinuousEffect {
                timestamp,
                source: Some(rioter_def),
                controller: PlayerId(0),
                affected: affected_koh,
                modification: Modification::HasAbilitiesOfSource,
                duration: EffectDuration::Indefinite,
                source_object: None,
            });
        let aura = game
            .scenario_force_battlefield(PlayerId(0), "Gray Ogre", false)
            .unwrap();
        let aura_obj = game.permanent_object(aura).unwrap();
        game.state[aura_obj].statics =
            vec![StaticAbility::AttachedHas(ActivatedAbilityDefinition {
                effects: vec![Effect::gain_life(1)],
                ..Default::default()
            })];
        game.state.permanents[aura].as_mut().unwrap().attached_to = Some(koh_obj);

        let view = game.characteristics(koh);
        // Every non-printed activated ability (GrantActivated, HasAbilitiesOfSource,
        // AttachedHas) carries `AbilityProvenance::Granted`.
        let printed_activated = game.state[koh_obj].activated_abilities.len();
        assert!(view.activated_abilities.len() > printed_activated + 2);
        for ability in &view.activated_abilities[printed_activated..] {
            assert_eq!(ability.provenance, AbilityProvenance::Granted);
        }
        assert!(view
            .activated_abilities
            .iter()
            .any(|a| a.granted_by.is_some() && a.provenance.is_granted()));
        // Every non-printed triggered ability (GrantTriggered, HasAbilitiesOfSource)
        // carries `AbilityProvenance::Granted`.
        let printed_triggered = game.state[koh_obj].abilities.len();
        assert!(view.triggered_abilities.len() > printed_triggered + 1);
        for trigger in &view.triggered_abilities[printed_triggered..] {
            assert_eq!(trigger.provenance, AbilityProvenance::Granted);
        }
    }

    #[test]
    fn layer6_granted_ability_fails_closed_on_emblem_pairing() {
        let mut game = test_game();
        let chandra = game
            .scenario_force_battlefield(PlayerId(0), "Chandra, Flameshaper", false)
            .unwrap();
        let chandra_obj = game.permanent_object(chandra).unwrap();
        let chandra_def = game.state[chandra_obj].definition_id;

        // 1. Printed (Characteristic) loyalty ability creates an emblem latching
        // `{ creator: chandra_obj, characteristic_set: Some(chandra_def) }`.
        let mut printed_frame = EffectFrame::new(
            PlayerId(0),
            vec![Effect::CreateEmblem {
                key: "emblem.chandra_torch_of_defiance".into(),
            }],
        );
        printed_frame.bind_source(Some(chandra_obj), Some(chandra_obj));
        printed_frame.ability_provenance = AbilityProvenance::Characteristic;
        game.run_frame(printed_frame.clone());

        // 2. Layer-6 granted ability on Chandra creates an emblem latching
        // `{ creator: chandra_obj, characteristic_set: None }`.
        let mut granted_frame = EffectFrame::new(
            PlayerId(0),
            vec![Effect::CreateEmblem {
                key: "emblem.chandra_torch_of_defiance".into(),
            }],
        );
        granted_frame.bind_source(Some(chandra_obj), Some(chandra_obj));
        granted_frame.ability_provenance = AbilityProvenance::Granted;
        game.run_frame(granted_frame.clone());

        let emblems: Vec<ObjectId> = game.emblems().map(|(_, obj)| obj).collect();
        assert_eq!(emblems.len(), 2);
        let printed_emblem = emblems[0];
        let granted_emblem = emblems[1];
        assert_eq!(
            game.state[printed_emblem].emblem_source(),
            Some(LinkedAbilitySource::characteristic(
                chandra_obj,
                chandra_def
            ))
        );
        assert_eq!(
            game.state[granted_emblem].emblem_source(),
            Some(LinkedAbilitySource::granted(chandra_obj))
        );

        // 3. Printed ability pairs ONLY with its own characteristic emblem;
        // Layer-6 granted ability fails closed (`None`).
        let printed_source = game
            .frame_linked_ability_source(&printed_frame)
            .expect("characteristic source");
        assert_eq!(
            game.emblems_linked_to(printed_source).collect::<Vec<_>>(),
            vec![printed_emblem]
        );
        assert!(game.frame_linked_ability_source(&granted_frame).is_none());
    }
}
