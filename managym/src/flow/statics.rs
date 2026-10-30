// statics.rs
// Continuous-effect queries: effective power/toughness, effective keywords,
// animation-aware type checks, and static-condition evaluation.
//
// Characteristics apply static abilities and continuous-effect records
// (`state::effect`) together, by layer and then by timestamp (CR 613.7). A
// static ability's timestamp is its permanent's (`Permanent::timestamp`); a
// record's is its own. Supported subset of CR 613 (September 25, 2026):
//   layer 4: printed types, static type grants, records that add creature
//     subtypes or make an artifact creature; a characteristic-setting Aura
//     (`SetCharacteristics`) replaces creature types, so subtypes added
//     before its timestamp are gone;
//   layer 6: printed keywords, keyword counters and animation haste, then
//     static grants and records in timestamp order; losing all abilities
//     (a record, or a characteristic-setting Aura) removes everything with
//     an earlier timestamp, so a later grant survives;
//   layer 7a: printed or characteristic-defining P/T (Suki, Dragonfly
//     Swarm); earthbend animation sets base 0/0 (CR 701.66a);
//   layer 7b: setters in timestamp order, the latest winning (the Aura's
//     set characteristics and records);
//   layer 7c: static buffs, records and +1/+1 counters, which commute.
// Layer 7d (switching) and dependency order (CR 613.8) are not represented.
// "Equipped/enchanted creature" effects follow the exact attached object.
// Static restrictions (can't attack or block, doesn't untap, blocker limits)
// use the same scopes.
// A scope reads what the layers before its own determine (`ScopeView`):
// type grants read printed types; keyword grants add granted types; P/T
// scopes add effective keywords; restrictions, which no layer reads, also
// read effective power. No computation can recurse.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    flow::{decision::EffectFrame, game::Game},
    state::{
        ability::Condition,
        card::{
            Card, CardType, Keywords, PowerCda, Restriction, SetCharacteristics, StaticAbility,
            StaticScope,
        },
        effect::Modification,
        game_object::{CardId, ObjectId, ObjectRef, PermanentId, PlayerId, Target},
        permanent::CounterKind,
        predicate::CardPredicate,
        zone::ZoneType,
    },
};

/// Subtypes that permanents in the current pools can carry which are not
/// creature types (land and artifact types; CR 205.3).
const NONCREATURE_SUBTYPES: [&str; 10] = [
    "Plains",
    "Island",
    "Swamp",
    "Mountain",
    "Forest",
    "Clue",
    "Food",
    "Treasure",
    "Equipment",
    "Vehicle",
];

/// Which characteristics a static scope's predicate reads (see the header).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScopeView {
    /// Layer 4: printed types, printed power, non-static keywords.
    Types,
    /// Layer 6: adds granted types.
    Keywords,
    /// Layer 7: adds effective keywords.
    PowerToughness,
    /// Rules effects and ordinary predicates: adds effective power.
    Rules,
}

impl Game {
    /// The permanent's effective power: CDA-aware or set base, plus static buffs,
    /// until-EOT deltas, and +1/+1 counters.
    pub fn effective_power(&self, permanent_id: PermanentId) -> i32 {
        self.effective_pt(permanent_id).0
    }

    /// The permanent's effective toughness (same layering as power).
    pub fn effective_toughness(&self, permanent_id: PermanentId) -> i32 {
        self.effective_pt(permanent_id).1
    }

    pub fn effective_pt(&self, permanent_id: PermanentId) -> (i32, i32) {
        let Some(permanent) = self
            .state
            .permanents
            .get(permanent_id.0)
            .and_then(|p| p.as_ref())
        else {
            return (0, 0);
        };
        let card = &self.state.cards[permanent.card];

        // Layer 7a: base P/T (characteristic-defining, or printed); an
        // earthbent permanent is a 0/0 creature.
        let (mut base_power, mut base_toughness) = if permanent.animated {
            (0, 0)
        } else {
            let power = match card.power_cda() {
                Some(cda) => self.cda_power(cda, permanent.controller),
                None => card.power.unwrap_or(0),
            };
            (power, card.toughness.unwrap_or(0))
        };
        // Layer 7b: setters in timestamp order replace the 7a value.
        let mut setters: Vec<(ObjectId, Option<i32>, Option<i32>)> = self
            .effects_on(permanent_id)
            .filter_map(|effect| match effect.modification {
                Modification::SetBasePt { power, toughness } => {
                    Some((effect.timestamp, power, toughness))
                }
                _ => None,
            })
            .collect();
        if let (Some(set), Some(timestamp)) = (
            self.set_characteristics(permanent_id),
            self.set_characteristics_timestamp(permanent_id),
        ) {
            setters.push((timestamp, Some(set.power), Some(set.toughness)));
        }
        setters.sort_by_key(|(timestamp, _, _)| *timestamp);
        for (_, power, toughness) in setters {
            base_power = power.unwrap_or(base_power);
            base_toughness = toughness.unwrap_or(base_toughness);
        }

        // Layer 7c: static continuous effects, records and +1/+1 counters.
        let (static_power, static_toughness) = self.static_pt_bonus(permanent_id);
        let (record_power, record_toughness) = self
            .effects_on(permanent_id)
            .filter_map(|effect| match effect.modification {
                Modification::ModifyPt { power, toughness } => Some((power, toughness)),
                _ => None,
            })
            .fold((0, 0), |(p, t), (dp, dt)| (p + dp, t + dt));

        let plus_one = permanent.counter_count(CounterKind::PlusOnePlusOne);
        (
            base_power + static_power + record_power + plus_one,
            base_toughness + static_toughness + record_toughness + plus_one,
        )
    }

    fn cda_power(&self, cda: &PowerCda, controller: PlayerId) -> i32 {
        match cda {
            PowerCda::CountersOnLandsYouControl => self
                .battlefield_permanents(controller)
                .into_iter()
                .filter(|&id| self.permanent_is_land(id))
                .filter_map(|id| self.state.permanents[id].as_ref())
                .map(|p| p.counter_count(CounterKind::PlusOnePlusOne))
                .sum(),
            PowerCda::CreaturesYouControl => self
                .battlefield_permanents(controller)
                .into_iter()
                .filter(|id| self.permanent_is_creature(*id))
                .count() as i32,
            PowerCda::GraveyardMatching(predicate) => {
                self.count_graveyard_matching(controller, predicate) as i32
            }
        }
    }

    /// Sum of static P/T buffs applying to `target_id` from every
    /// battlefield permanent (including itself for `StaticScope::This`).
    fn static_pt_bonus(&self, target_id: PermanentId) -> (i32, i32) {
        let mut power = 0;
        let mut toughness = 0;
        for (source_id, controller, card) in self.static_sources() {
            for buff in card.pt_buffs() {
                if !self.static_scope_applies(
                    &buff.scope,
                    source_id,
                    controller,
                    target_id,
                    ScopeView::PowerToughness,
                ) || buff
                    .condition
                    .as_ref()
                    .is_some_and(|condition| !self.check_static_condition(condition, controller))
                {
                    continue;
                }
                let per_type = buff.per_creature_type_max.map_or(1, |max| {
                    (self.creature_types(target_id).len() as i32).min(max)
                });
                let per_color = buff
                    .per_color_among
                    .as_ref()
                    .map_or(1, |predicate| self.colors_among(controller, predicate));
                let times = per_type * per_color;
                power += buff.power * times;
                toughness += buff.toughness * times;
            }
        }
        (power, toughness)
    }

    /// Keywords granted to `target_id` by static abilities of battlefield
    /// permanents (layer 6) whose timestamps follow `after`.
    fn static_keyword_bonus(&self, target_id: PermanentId, after: Option<ObjectId>) -> Keywords {
        let mut keywords = Keywords::default();
        for (source_id, controller, card) in self.static_sources() {
            if !self.follows(source_id, after) {
                continue;
            }
            for grant in card.keyword_grants() {
                if self.static_scope_applies(
                    &grant.scope,
                    source_id,
                    controller,
                    target_id,
                    ScopeView::Keywords,
                ) && grant
                    .condition
                    .as_ref()
                    .is_none_or(|condition| self.check_static_condition(condition, controller))
                {
                    keywords = keywords.union(&grant.keywords);
                }
            }
        }
        keywords
    }

    /// A stack spell's keywords: printed, plus static grants to spells its
    /// controller controls (CR 613.1f).
    pub(crate) fn spell_keywords(&self, card_id: CardId, controller: PlayerId) -> Keywords {
        let card = &self.state.cards[card_id];
        let mut keywords = card.keywords;
        for (_, source_controller, source) in self.static_sources() {
            for grant in source.keyword_grants() {
                if let StaticScope::SpellsYouControl(predicate) = &grant.scope {
                    if source_controller == controller
                        && predicate.matches_card(card)
                        && grant.condition.as_ref().is_none_or(|condition| {
                            self.check_static_condition(condition, controller)
                        })
                    {
                        keywords = keywords.union(&grant.keywords);
                    }
                }
            }
        }
        keywords
    }

    /// Battlefield permanents whose cards project static P/T, keyword or
    /// restriction effects, with their controllers.
    pub(crate) fn static_sources(&self) -> impl Iterator<Item = (PermanentId, PlayerId, &Card)> {
        [PlayerId(0), PlayerId(1)]
            .into_iter()
            .flat_map(move |player| {
                self.state
                    .zones
                    .zone_cards(ZoneType::Battlefield, player)
                    .iter()
                    .filter_map(move |&card_id| {
                        let card = &self.state.cards[card_id];
                        if !card.statics.iter().any(|ability| {
                            matches!(
                                ability,
                                StaticAbility::PtBuff(_)
                                    | StaticAbility::KeywordGrant(_)
                                    | StaticAbility::Restriction(_)
                                    | StaticAbility::TypeGrant(_)
                                    | StaticAbility::TriggersAdditionalTime(_)
                                    | StaticAbility::TriggeredGrant(_)
                            )
                        }) {
                            return None;
                        }
                        let permanent_id = self.state.card_to_permanent[card_id]?;
                        let permanent = self.state.permanents[permanent_id].as_ref()?;
                        if self.has_lost_abilities(permanent_id) {
                            return None;
                        }
                        Some((permanent_id, permanent.controller, card))
                    })
            })
    }

    /// Whether the static ability of `source_id` has a timestamp after
    /// `after` (no bound: always).
    fn follows(&self, source_id: PermanentId, after: Option<ObjectId>) -> bool {
        after.is_none_or(|after| {
            self.state.permanents[source_id]
                .as_ref()
                .is_some_and(|source| source.timestamp > after)
        })
    }

    /// The Aura attached to `permanent_id` that sets its characteristics, if
    /// any: the latest by timestamp.
    fn set_characteristics_source(&self, permanent_id: PermanentId) -> Option<PermanentId> {
        self.attachments_on(permanent_id)
            .filter(|(_, card)| card.attached_characteristics().is_some())
            .max_by_key(|(aura, _)| {
                self.state.permanents[*aura]
                    .as_ref()
                    .map(|permanent| permanent.timestamp)
            })
            .map(|(aura, _)| aura)
    }

    /// The timestamp of the Aura that sets the permanent's characteristics,
    /// if any (CR 613.7e).
    pub(crate) fn set_characteristics_timestamp(
        &self,
        permanent_id: PermanentId,
    ) -> Option<ObjectId> {
        let aura = self.set_characteristics_source(permanent_id)?;
        self.state.permanents[aura]
            .as_ref()
            .map(|permanent| permanent.timestamp)
    }

    /// The characteristics an Aura attached to `permanent_id` sets, if any.
    pub(crate) fn set_characteristics(
        &self,
        permanent_id: PermanentId,
    ) -> Option<&SetCharacteristics> {
        let aura = self.set_characteristics_source(permanent_id)?;
        let card = self.state.permanents[aura].as_ref()?.card;
        self.state.cards[card].attached_characteristics()
    }

    /// Whether the permanent has lost all abilities (an effect record, or an
    /// Aura that sets its characteristics). Its printed abilities predate
    /// any loss; `effective_keywords` keeps later grants.
    pub fn has_lost_abilities(&self, permanent_id: PermanentId) -> bool {
        self.ability_loss_timestamp(permanent_id).is_some()
    }

    /// Creature subtypes added by effect records after any characteristic-
    /// setting Aura's timestamp (layer 4).
    pub fn added_subtypes(&self, permanent_id: PermanentId) -> Vec<&str> {
        let aura = self.set_characteristics_timestamp(permanent_id);
        let mut subtypes: Vec<&str> = Vec::new();
        for effect in self.effects_on(permanent_id) {
            if let Modification::AddSubtypes(added) = &effect.modification {
                if effect.follows(aura) {
                    for subtype in added {
                        if !subtypes.contains(&subtype.as_str()) {
                            subtypes.push(subtype);
                        }
                    }
                }
            }
        }
        subtypes
    }

    /// The permanent's name (CR 201.2), as set by a characteristic-setting
    /// Aura or printed.
    pub fn permanent_name(&self, permanent_id: PermanentId) -> &str {
        match self.set_characteristics(permanent_id) {
            Some(set) => &set.named,
            None => self.state.permanents[permanent_id]
                .as_ref()
                .map_or("", |permanent| &self.state.cards[permanent.card].name),
        }
    }

    /// Whether the permanent has `subtype`: printed or temporarily granted.
    /// A characteristic-setting Aura replaces its printed creature types.
    pub fn permanent_has_subtype(&self, permanent_id: PermanentId, subtype: &str) -> bool {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return false;
        };
        let printed = self.state.cards[permanent.card].has_subtype(subtype);
        let printed = match self.set_characteristics(permanent_id) {
            Some(set) => {
                set.creature_types.iter().any(|s| s == subtype)
                    || (printed && NONCREATURE_SUBTYPES.contains(&subtype))
            }
            None => printed,
        };
        printed || self.added_subtypes(permanent_id).contains(&subtype)
    }

    /// Card types static type grants add to `target_id` (layer 4).
    fn granted_card_types(&self, target_id: PermanentId) -> Vec<CardType> {
        let mut types = Vec::new();
        for (source_id, controller, card) in self.static_sources() {
            for grant in card.type_grants() {
                if self.static_scope_applies(
                    &grant.scope,
                    source_id,
                    controller,
                    target_id,
                    ScopeView::Types,
                ) {
                    types.extend(grant.types.iter().copied());
                }
            }
        }
        types
    }

    /// Whether a battlefield permanent is a land, printed or granted.
    pub(crate) fn permanent_is_land(&self, permanent_id: PermanentId) -> bool {
        self.state.permanents[permanent_id]
            .as_ref()
            .is_some_and(|p| self.state.cards[p.card].types.is_land())
            || self
                .granted_card_types(permanent_id)
                .contains(&CardType::Land)
    }

    /// Distinct colors among `controller`'s permanents matching `predicate`
    /// (CR 105.2; Earthen Ally's "each color among Allies you control").
    fn colors_among(&self, controller: PlayerId, predicate: &CardPredicate) -> i32 {
        self.battlefield_permanents(controller)
            .into_iter()
            .filter(|&id| self.permanent_matches_with(id, predicate, ScopeView::Keywords))
            .filter_map(|id| self.state.permanents[id].as_ref())
            .flat_map(|permanent| self.state.cards[permanent.card].colors())
            .collect::<BTreeSet<_>>()
            .len() as i32
    }

    /// Whether a static scope from `source_id` (controlled by `controller`)
    /// covers `target_id`, reading the characteristics `view` allows.
    pub(crate) fn static_scope_applies(
        &self,
        scope: &StaticScope,
        source_id: PermanentId,
        controller: PlayerId,
        target_id: PermanentId,
        view: ScopeView,
    ) -> bool {
        let Some(target) = self.state.permanents[target_id].as_ref() else {
            return false;
        };
        let (predicate, other) = match scope {
            StaticScope::This => return source_id == target_id,
            StaticScope::Attached => return self.attached_creature(source_id) == Some(target_id),
            StaticScope::SpellsYouControl(_) => return false,
            StaticScope::OtherYouControl(predicate) => (predicate, true),
            StaticScope::YouControl(predicate) => (predicate, false),
        };
        if target.controller != controller || (other && source_id == target_id) {
            return false;
        }
        self.permanent_matches_with(target_id, predicate, view)
    }

    /// Whether the permanent is attacking or blocking (CR 506.4).
    pub(crate) fn permanent_in_combat(&self, permanent_id: PermanentId) -> bool {
        self.state.permanents[permanent_id]
            .as_ref()
            .is_some_and(|permanent| permanent.attacking)
            || self.state.combat.as_ref().is_some_and(|combat| {
                combat
                    .attacker_to_blockers
                    .values()
                    .any(|blockers| blockers.contains(&permanent_id))
            })
    }

    /// The permanent an Aura or Equipment is attached to, if that exact
    /// object is still on the battlefield.
    pub(crate) fn attached_permanent(&self, source: PermanentId) -> Option<PermanentId> {
        let attached = self.state.permanents[source].as_ref()?.attached_to?;
        self.lookup_current_permanent(attached).ok()
    }

    /// The attached permanent, if it is still a creature.
    pub(crate) fn attached_creature(&self, source: PermanentId) -> Option<PermanentId> {
        self.attached_permanent(source)
            .filter(|&creature| self.permanent_is_creature(creature))
    }

    /// Auras and Equipment attached to the creature `target`, in battlefield
    /// order, less those that have lost their own abilities.
    pub(crate) fn attachments_on(
        &self,
        target: PermanentId,
    ) -> impl Iterator<Item = (PermanentId, &Card)> {
        [PlayerId(0), PlayerId(1)]
            .into_iter()
            .flat_map(move |player| self.state.zones.zone_cards(ZoneType::Battlefield, player))
            .filter_map(move |&card_id| {
                let source = self.state.card_to_permanent[card_id]?;
                // Only records can remove an attachment's own abilities,
                // so this never recurses.
                (!self
                    .effects_on(source)
                    .any(|effect| effect.modification == Modification::LoseAllAbilities)
                    && self.attached_creature(source) == Some(target))
                .then(|| (source, &self.state.cards[card_id]))
            })
    }

    /// Whether a static restriction from a battlefield permanent applies to
    /// `target_id`.
    pub(crate) fn has_restriction(&self, target_id: PermanentId, kind: Restriction) -> bool {
        self.static_sources().any(|(source_id, controller, card)| {
            card.restrictions().any(|restriction| {
                restriction.kind == kind
                    && restriction
                        .unless
                        .as_ref()
                        .is_none_or(|condition| !self.check_static_condition(condition, controller))
                    && self.static_scope_applies(
                        &restriction.scope,
                        source_id,
                        controller,
                        target_id,
                        ScopeView::Rules,
                    )
            })
        })
    }

    /// Distinct creature types of a battlefield creature (CR 205.3m): printed
    /// subtypes of a printed creature plus temporarily granted subtypes, less
    /// noncreature subtypes. Animated lands keep only granted types.
    pub(crate) fn creature_types(&self, permanent_id: PermanentId) -> Vec<&str> {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return Vec::new();
        };
        let card = &self.state.cards[permanent.card];
        let printed: &[String] = if let Some(set) = self.set_characteristics(permanent_id) {
            &set.creature_types
        } else if card.types.is_creature() {
            &card.subtypes
        } else {
            &[]
        };
        let mut types: Vec<&str> = printed
            .iter()
            .map(String::as_str)
            .chain(self.added_subtypes(permanent_id))
            .filter(|s| !NONCREATURE_SUBTYPES.contains(s))
            .collect();
        types.sort_unstable();
        types.dedup();
        types
    }

    /// The greatest number of creatures `controller` controls that share one
    /// creature type (White Lotus Tile); zero when none has a creature type.
    pub(crate) fn greatest_shared_creature_type(&self, controller: PlayerId) -> usize {
        let mut counts: BTreeMap<&str, usize> = Default::default();
        for id in self.battlefield_permanents(controller) {
            if self.permanent_is_creature(id) {
                for creature_type in self.creature_types(id) {
                    *counts.entry(creature_type).or_default() += 1;
                }
            }
        }
        counts.into_values().max().unwrap_or(0)
    }

    /// Instant timing for `player` casting `card`: printed instant or flash,
    /// or a conditional "as though it had flash" that currently holds.
    pub(crate) fn has_instant_timing(&self, player: PlayerId, card: CardId) -> bool {
        let granted_flash = self
            .exile_grant(player, card)
            .is_some_and(|grant| grant.flash);
        let card = &self.state.cards[card];
        granted_flash
            || card.is_instant_speed()
            || card
                .flash_if()
                .is_some_and(|condition| self.check_static_condition(condition, player))
    }

    /// Answer a game-state condition for `controller` ("you"): statics and
    /// `ActiveIf`. Resolution-only and source questions have no frame or
    /// source here and answer false; definition checks keep them out of
    /// these contexts.
    pub(crate) fn check_static_condition(
        &self,
        condition: &Condition,
        controller: PlayerId,
    ) -> bool {
        debug_assert!(
            !condition.is_resolution_only() && !condition.reads_source(),
            "resolution-only or source condition outside Effect::If: {condition:?}"
        );
        self.condition_holds(condition, controller, None, None, None)
    }

    /// Answer an intervening "if" (CR 603.4) of the ability of the exact
    /// battlefield object `source`, as its event occurs and as it resolves.
    /// `triggering` is the permanent whose event fired it.
    pub(crate) fn check_intervening_if(
        &self,
        condition: &Condition,
        controller: PlayerId,
        source: Option<ObjectRef>,
        triggering: Option<Target>,
    ) -> bool {
        debug_assert!(
            !condition.is_resolution_only(),
            "resolution-only condition outside Effect::If: {condition:?}"
        );
        self.condition_holds(condition, controller, None, source, triggering)
    }

    /// Answer `condition` for `controller`, reading `frame` for the
    /// resolution-only questions (CR 608.2c).
    pub(crate) fn check_condition(
        &self,
        condition: &Condition,
        controller: PlayerId,
        frame: Option<&EffectFrame>,
    ) -> bool {
        self.condition_holds(
            condition,
            controller,
            frame,
            frame.and_then(|frame| frame.source_ref),
            frame.and_then(|frame| frame.context_target),
        )
    }

    /// The source questions read `source`, the ability's exact object, which
    /// answers nothing once it has left the battlefield; the triggering
    /// questions read `triggering`, as it is or last existed.
    fn condition_holds(
        &self,
        condition: &Condition,
        controller: PlayerId,
        frame: Option<&EffectFrame>,
        source: Option<ObjectRef>,
        triggering: Option<Target>,
    ) -> bool {
        let source_permanent = || {
            source
                .and_then(|source| self.lookup_current_permanent(source).ok())
                .and_then(|id| self.state.permanents[id].as_ref())
        };
        match condition {
            Condition::Always => true,
            Condition::YourTurn => self.active_player() == controller,
            Condition::CardsDrawnAtLeast { count } => {
                self.state.turn.history[controller.0].cards_drawn >= *count
            }
            Condition::GraveyardAtLeast { count, predicate } => {
                self.count_graveyard_matching(controller, predicate) >= *count
            }
            Condition::ControlsMatching { predicate } => self
                .battlefield_permanents(controller)
                .into_iter()
                .any(|id| self.permanent_matches_predicate(id, predicate)),
            Condition::UnspentManaAtLeast { count } => {
                u32::from(self.pooled_mana(controller).total()) >= *count
            }
            Condition::AttackedThisTurn => self.state.turn.history[controller.0].attacked,
            Condition::LandEnteredThisTurn => self.state.turn.history[controller.0].land_entered,
            Condition::SacrificedThisTurnAtLeast { count } => {
                self.state.turn.history[controller.0].permanents_sacrificed >= *count
            }
            Condition::AnyOf(conditions) => conditions.iter().any(|condition| {
                self.condition_holds(condition, controller, frame, source, triggering)
            }),
            Condition::Kicked => frame.is_some_and(|frame| frame.kicked),
            Condition::TargetMatches(predicate) => frame
                .and_then(|frame| self.present_primary_target(frame))
                .is_some_and(|target| {
                    matches!(target, Target::Permanent(id)
                        if self.permanent_matches_predicate(id, predicate))
                }),
            Condition::SourceCountersAtLeast { kind, count } => {
                source_permanent().is_some_and(|permanent| permanent.counter_count(*kind) >= *count)
            }
            Condition::SourceTapped => source_permanent().is_some_and(|permanent| permanent.tapped),
            Condition::TriggeringHadCounters => match triggering {
                Some(Target::Permanent(id)) => match self.state.permanents[id].as_ref() {
                    Some(permanent) => permanent.has_counters(),
                    None => self
                        .departed_permanent_lki(id)
                        .is_some_and(|lki| !lki.counters.is_empty()),
                },
                _ => false,
            },
            Condition::TargetCardExiled(predicate) => {
                let Some(Target::GraveyardCard(object)) =
                    frame.and_then(EffectFrame::primary_target)
                else {
                    return false;
                };
                let card = CardId::from(object.entity);
                self.current_object_ref(card) != Some(object)
                    && self.state.zones.zone_of(card) == Some(ZoneType::Exile)
                    && predicate.matches_card(&self.state.cards[card])
            }
            Condition::TargetDealtDamageThisTurn => {
                let Some(Target::Permanent(permanent_id)) =
                    frame.and_then(EffectFrame::primary_target)
                else {
                    return false;
                };
                // A departed target answers from its last known information.
                match self.state.permanents[permanent_id].as_ref() {
                    Some(permanent) => permanent.dealt_damage_this_turn,
                    None => self
                        .departed_permanent_lki(permanent_id)
                        .is_some_and(|lki| lki.dealt_damage_this_turn),
                }
            }
        }
    }

    /// Animation-aware creature check.
    pub(crate) fn permanent_is_creature(&self, permanent_id: PermanentId) -> bool {
        self.state
            .permanents
            .get(permanent_id.0)
            .and_then(|p| p.as_ref())
            .is_some_and(|permanent| self.state.cards[permanent.card].types.is_creature())
            || self.permanent_is_animated(permanent_id)
    }

    /// Whether something other than its printed type makes the permanent a
    /// creature: earthbend, its own counter threshold, or an effect.
    pub(crate) fn permanent_is_animated(&self, permanent_id: PermanentId) -> bool {
        self.state
            .permanents
            .get(permanent_id.0)
            .and_then(|p| p.as_ref())
            .is_some_and(|permanent| permanent.animated_creature(&self.state.cards[permanent.card]))
            || self.becomes_artifact_creature(permanent_id)
    }

    /// Layer 6 keywords: printed keywords, keyword counters and animation
    /// haste, then static grants and records; a loss of all abilities removes
    /// whatever has an earlier timestamp.
    pub fn effective_keywords(&self, permanent_id: PermanentId) -> Keywords {
        let lost = self.ability_loss_timestamp(permanent_id);
        self.recorded_keywords(permanent_id)
            .union(&self.static_keyword_bonus(permanent_id, lost))
    }

    /// Whether the permanent can pay a {T} cost or attack (CR 302.6): it is
    /// untapped, and a creature under its controller's control since the
    /// turn began or with haste, as `effective_keywords` and
    /// `permanent_is_creature` report it. A temporary grant must not erase the
    /// restriction that applies again when the grant expires (CR 702.10).
    pub(crate) fn permanent_can_tap(&self, permanent_id: PermanentId) -> bool {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return false;
        };
        !(permanent.tapped
            || (permanent.summoning_sick
                && self.permanent_is_creature(permanent_id)
                && !self.effective_keywords(permanent_id).haste))
    }

    /// Lethal-damage check (CR 704.5g), animation- and statics-aware.
    pub(crate) fn has_lethal_damage(&self, permanent_id: PermanentId) -> bool {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return false;
        };
        if !self.permanent_is_creature(permanent_id) {
            return false;
        }
        if permanent.deathtouch_damage && permanent.damage > 0 {
            return true;
        }
        permanent.damage >= self.effective_toughness(permanent_id)
    }

    /// Match a battlefield permanent against a predicate using its
    /// effective power, keywords and types.
    pub(crate) fn permanent_matches_predicate(
        &self,
        permanent_id: PermanentId,
        predicate: &CardPredicate,
    ) -> bool {
        self.permanent_matches_with(permanent_id, predicate, ScopeView::Rules)
    }

    /// Keywords before static grants (what layer-4 and layer-6 scopes read):
    /// intrinsic keywords and records, after any loss.
    fn recorded_keywords(&self, permanent_id: PermanentId) -> Keywords {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return Keywords::default();
        };
        let lost = self.ability_loss_timestamp(permanent_id);
        let mut keywords = if lost.is_some() {
            Keywords::default()
        } else {
            permanent.intrinsic_keywords(&self.state.cards[permanent.card])
        };
        for effect in self.effects_on(permanent_id) {
            if let Modification::AddKeywords(granted) = &effect.modification {
                if effect.follows(lost) {
                    keywords = keywords.union(granted);
                }
            }
        }
        keywords
    }

    /// Match a battlefield permanent against a predicate, counting its
    /// temporary subtypes and animation, reading the power, keywords and
    /// types `view` allows without recursing.
    fn permanent_matches_with(
        &self,
        permanent_id: PermanentId,
        predicate: &CardPredicate,
        view: ScopeView,
    ) -> bool {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return false;
        };
        let card = &self.state.cards[permanent.card];
        let flying = || match view {
            ScopeView::Types | ScopeView::Keywords => self.recorded_keywords(permanent_id).flying,
            ScopeView::PowerToughness | ScopeView::Rules => {
                self.effective_keywords(permanent_id).flying
            }
        };
        if (predicate.flying && !flying())
            || (predicate.no_counters && permanent.has_counters())
            || (predicate.in_combat && !self.permanent_in_combat(permanent_id))
        {
            return false;
        }
        if predicate.with_counter && permanent.counter_count(CounterKind::PlusOnePlusOne) <= 0 {
            return false;
        }
        if predicate.attacking
            && !self
                .state
                .combat
                .as_ref()
                .is_some_and(|combat| combat.attackers.contains(&permanent_id))
        {
            return false;
        }
        let has_subtype = |name: &str| self.permanent_has_subtype(permanent_id, name);
        if predicate
            .subtype
            .as_deref()
            .is_some_and(|s| !has_subtype(s))
            || (!predicate.subtypes_any.is_empty()
                && !predicate.subtypes_any.iter().any(|s| has_subtype(s)))
            || predicate.not_subtypes.iter().any(|s| has_subtype(s))
        {
            return false;
        }
        let structural = CardPredicate {
            flying: false,
            with_counter: false,
            attacking: false,
            subtype: None,
            subtypes_any: Vec::new(),
            not_subtypes: Vec::new(),
            ..predicate.clone()
        };
        // Read effective power only for power predicates: a power-free
        // predicate in a static P/T condition ("as long as you control a land
        // creature") must not recompute the P/T it contributes to.
        let power = match view {
            ScopeView::Rules if predicate.reads_power() => self.effective_power(permanent_id),
            _ => card.power.unwrap_or(0),
        };
        let mut types = Vec::new();
        if self.permanent_is_animated(permanent_id) {
            types.push(CardType::Creature);
        }
        let reads_types = structural.card_type.is_some()
            || !structural.card_types_any.is_empty()
            || !structural.not_card_types.is_empty();
        if view != ScopeView::Types && reads_types {
            types.extend(self.granted_card_types(permanent_id));
        }
        structural.matches(card, power, &types)
    }
}
