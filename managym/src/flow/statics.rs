// statics.rs
// A permanent's characteristics after continuous effects, and static-condition
// evaluation. `Game::characteristics` is the one reader of a permanent's
// derived name, types, keywords, abilities and power/toughness; the layer
// stages below are private to it and to static-scope matching. Rules questions
// that depend on characteristics (`permanent_can_tap`, lethal damage,
// predicates) read the view or the stage they need.
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
// read effective power. No computation can recurse: code the stages call
// (static scopes, ability mana outputs) never reads the view. A view computes
// the shared layer inputs (`LayerBasis`) once and hands them to each stage.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    flow::game::Game,
    state::{
        ability::{Effect, StaticCondition},
        card::{
            AbilityGrant, ActivatedAbilityDefinition, Card, CardType, Keywords, PowerCda,
            Restriction, SetCharacteristics, StaticAbility, StaticScope,
        },
        effect::Modification,
        game_object::{CardId, ObjectId, PermanentId, PlayerId},
        mana::{Mana, ManaUse},
        permanent::CounterKind,
        predicate::CardPredicate,
        zone::ZoneType,
    },
};

/// Basic land types and the mana their intrinsic ability adds (CR 305.6).
const BASIC_LAND_TYPES: [(&str, &str); 5] = [
    ("Plains", "W"),
    ("Island", "U"),
    ("Swamp", "B"),
    ("Mountain", "R"),
    ("Forest", "G"),
];

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

/// A battlefield permanent's characteristics with every static ability and
/// continuous-effect record applied (CR 613). A permanent that has left the
/// battlefield reads as the default: nameless, typeless, 0/0, no abilities.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Characteristics<'a> {
    /// Set by a characteristic-setting Aura, or printed (CR 201.2).
    pub name: &'a str,
    /// Whether a characteristic-setting Aura replaced its name, creature
    /// types and base power/toughness.
    pub characteristics_set: bool,
    /// Every subtype it has: printed (creature types replaced by a
    /// characteristic-setting Aura's), then added by records.
    pub subtypes: Vec<&'a str>,
    /// Creature subtypes added by records after any characteristic-setting
    /// Aura (layer 4), in record order.
    pub added_subtypes: Vec<&'a str>,
    /// Distinct creature types (CR 205.3m), sorted.
    pub creature_types: Vec<&'a str>,
    pub is_creature: bool,
    /// Printed or granted land.
    pub is_land: bool,
    /// A creature by something other than its printed type: earthbend, its
    /// own counter threshold, or an effect.
    pub is_animated: bool,
    /// Layer 6 keywords.
    pub keywords: Keywords,
    /// Firebending N granted by records after any ability loss, summed.
    pub firebending: u8,
    /// Whether it has lost all abilities (a record, or a characteristic-
    /// setting Aura); grants with later timestamps still apply.
    pub abilities_lost: bool,
    pub power: i32,
    pub toughness: i32,
    /// Printed abilities, then currently granted abilities in source order,
    /// then the intrinsic mana ability of basic land types gained by an
    /// effect, then abilities granted by attached Equipment and Auras in
    /// battlefield order, each with its cost as it would be paid now. An
    /// ability's index is its position here; offers, activation and payment
    /// read only this list. A permanent that has lost all abilities keeps only
    /// the grants of the Aura that set its characteristics (Humble Merchant's
    /// "{T}: Add {C}").
    pub activated_abilities: Vec<ActivatedAbilityDefinition>,
}

impl Characteristics<'_> {
    pub fn pt(&self) -> (i32, i32) {
        (self.power, self.toughness)
    }

    pub fn has_subtype(&self, subtype: &str) -> bool {
        self.subtypes.contains(&subtype)
    }
}

/// Layer inputs several stages read, computed once per view: the
/// permanent's attachments, the characteristic-setting Aura (latest by
/// timestamp), the ability-loss timestamp and the subtypes records add after
/// that Aura (layer 4).
struct LayerBasis<'a> {
    /// Auras and Equipment attached to it, in battlefield order.
    attachments: Vec<(PermanentId, &'a Card)>,
    set: Option<&'a SetCharacteristics>,
    set_timestamp: Option<ObjectId>,
    lost: Option<ObjectId>,
    added_subtypes: Vec<&'a str>,
}

/// Every subtype: printed or added by records. A characteristic-setting Aura
/// replaces the printed creature types.
fn subtypes_with<'a>(card: &'a Card, basis: &LayerBasis<'a>) -> Vec<&'a str> {
    let printed = card.subtypes.iter().map(String::as_str);
    let mut subtypes: Vec<&str> = match basis.set {
        Some(set) => set
            .creature_types
            .iter()
            .map(String::as_str)
            .chain(printed.filter(|s| NONCREATURE_SUBTYPES.contains(s)))
            .collect(),
        None => printed.collect(),
    };
    for &subtype in &basis.added_subtypes {
        if !subtypes.contains(&subtype) {
            subtypes.push(subtype);
        }
    }
    subtypes
}

/// Distinct creature types (CR 205.3m): printed subtypes of a printed
/// creature (or the Aura-set ones) plus added subtypes, less noncreature
/// subtypes, sorted. Animated lands keep only added types.
fn creature_types_with<'a>(card: &'a Card, basis: &LayerBasis<'a>) -> Vec<&'a str> {
    let printed: &[String] = if let Some(set) = basis.set {
        &set.creature_types
    } else if card.types.is_creature() {
        &card.subtypes
    } else {
        &[]
    };
    let mut types: Vec<&str> = printed
        .iter()
        .map(String::as_str)
        .chain(basis.added_subtypes.iter().copied())
        .filter(|s| !NONCREATURE_SUBTYPES.contains(s))
        .collect();
    types.sort_unstable();
    types.dedup();
    types
}

impl Game {
    /// The permanent's characteristics: static abilities and effect records
    /// applied by layer and then timestamp (see the header).
    pub fn characteristics(&self, permanent_id: PermanentId) -> Characteristics<'_> {
        let Some(permanent) = self
            .state
            .permanents
            .get(permanent_id.0)
            .and_then(|p| p.as_ref())
        else {
            return Characteristics::default();
        };
        let card = &self.state.cards[permanent.card];
        let basis = self.layer_basis(permanent_id);
        let (power, toughness) = self.pt_with(permanent_id, &basis);
        let is_land = self.permanent_is_land(permanent_id);
        Characteristics {
            name: basis
                .set
                .map_or(card.name.as_str(), |set| set.named.as_str()),
            characteristics_set: basis.set.is_some(),
            subtypes: subtypes_with(card, &basis),
            creature_types: creature_types_with(card, &basis),
            is_creature: self.permanent_is_creature(permanent_id),
            is_land,
            is_animated: self.permanent_is_animated(permanent_id),
            keywords: self.keywords_with(permanent_id, basis.lost),
            firebending: self.firebending_with(permanent_id, basis.lost),
            abilities_lost: basis.lost.is_some(),
            power,
            toughness,
            activated_abilities: self.activated_abilities_with(permanent_id, &basis, is_land),
            added_subtypes: basis.added_subtypes,
        }
    }

    /// The layer inputs several stages share, computed once per view.
    fn layer_basis(&self, permanent_id: PermanentId) -> LayerBasis<'_> {
        let attachments: Vec<(PermanentId, &Card)> = self.attachments_on(permanent_id).collect();
        // The latest characteristic-setting Aura wins (CR 613.7e).
        let setter = attachments
            .iter()
            .filter_map(|(aura, card)| {
                let set = card.attached_characteristics()?;
                Some((self.state.permanents[*aura].as_ref()?.timestamp, set))
            })
            .max_by_key(|(timestamp, _)| *timestamp);
        let set_timestamp = setter.map(|(timestamp, _)| timestamp);
        let recorded_loss = self
            .effects_on(permanent_id)
            .filter(|effect| effect.modification == Modification::LoseAllAbilities)
            .map(|effect| effect.timestamp)
            .last();
        let mut added_subtypes: Vec<&str> = Vec::new();
        for effect in self.effects_on(permanent_id) {
            if let Modification::AddSubtypes(added) = &effect.modification {
                if effect.follows(set_timestamp) {
                    for subtype in added {
                        if !added_subtypes.contains(&subtype.as_str()) {
                            added_subtypes.push(subtype);
                        }
                    }
                }
            }
        }
        LayerBasis {
            attachments,
            set: setter.map(|(_, set)| set),
            set_timestamp,
            lost: recorded_loss.max(set_timestamp),
            added_subtypes,
        }
    }

    /// Layer 7: power and toughness (CDA-aware or set base, plus static buffs,
    /// records and +1/+1 counters).
    fn effective_pt(&self, permanent_id: PermanentId) -> (i32, i32) {
        self.pt_with(permanent_id, &self.layer_basis(permanent_id))
    }

    fn pt_with(&self, permanent_id: PermanentId, basis: &LayerBasis) -> (i32, i32) {
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
        if let (Some(set), Some(timestamp)) = (basis.set, basis.set_timestamp) {
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
    fn static_sources(&self) -> impl Iterator<Item = (PermanentId, PlayerId, &Card)> {
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
                            )
                        }) {
                            return None;
                        }
                        let permanent_id = self.state.card_to_permanent[card_id]?;
                        let permanent = self.state.permanents[permanent_id].as_ref()?;
                        if self.ability_loss_timestamp(permanent_id).is_some() {
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

    /// The latest timestamp at which the permanent lost all abilities, if
    /// it has (CR 613.1f); grants with later timestamps still apply.
    fn ability_loss_timestamp(&self, permanent_id: PermanentId) -> Option<ObjectId> {
        self.layer_basis(permanent_id).lost
    }

    /// Whether an effect makes the permanent an artifact creature.
    fn becomes_artifact_creature(&self, permanent_id: PermanentId) -> bool {
        self.permanent_has_modification(permanent_id, &Modification::BecomeArtifactCreature)
    }

    /// Firebending granted by effects later than any ability loss, summed.
    fn firebending_with(&self, permanent_id: PermanentId, lost: Option<ObjectId>) -> u8 {
        self.effects_on(permanent_id)
            .filter(|effect| effect.follows(lost))
            .filter_map(|effect| match effect.modification {
                Modification::Firebending(amount) => Some(amount),
                _ => None,
            })
            .fold(0, u8::saturating_add)
    }

    fn permanent_has_subtype(&self, permanent_id: PermanentId, subtype: &str) -> bool {
        self.state.permanents[permanent_id]
            .as_ref()
            .is_some_and(|permanent| {
                subtypes_with(
                    &self.state.cards[permanent.card],
                    &self.layer_basis(permanent_id),
                )
                .contains(&subtype)
            })
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
    fn permanent_is_land(&self, permanent_id: PermanentId) -> bool {
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
    fn static_scope_applies(
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
                self.state.permanents[source].as_ref()?.attached_to?;
                // Only records can remove an attachment's own abilities,
                // so this never recurses.
                (self.attached_creature(source) == Some(target)
                    && !self
                        .effects_on(source)
                        .any(|effect| effect.modification == Modification::LoseAllAbilities))
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

    /// Distinct creature types of a battlefield creature (CR 205.3m).
    fn creature_types(&self, permanent_id: PermanentId) -> Vec<&str> {
        self.state.permanents[permanent_id]
            .as_ref()
            .map_or_else(Vec::new, |permanent| {
                creature_types_with(
                    &self.state.cards[permanent.card],
                    &self.layer_basis(permanent_id),
                )
            })
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

    /// Evaluate a static condition for `controller` ("you").
    pub(crate) fn check_static_condition(
        &self,
        condition: &StaticCondition,
        controller: PlayerId,
    ) -> bool {
        match condition {
            StaticCondition::Always => true,
            StaticCondition::YourTurn => self.active_player() == controller,
            StaticCondition::CardsDrawnAtLeast { count } => {
                self.state.turn.cards_drawn_this_turn[controller.0] >= *count
            }
            StaticCondition::GraveyardAtLeast { count, predicate } => {
                self.count_graveyard_matching(controller, predicate) >= *count
            }
            StaticCondition::ControlsMatching { predicate } => self
                .battlefield_permanents(controller)
                .into_iter()
                .any(|id| self.permanent_matches_predicate(id, predicate)),
            StaticCondition::UnspentManaAtLeast { count } => {
                u32::from(self.pooled_mana(controller).total()) >= *count
            }
        }
    }

    /// A creature on the battlefield (layer 4). Ability outputs read this
    /// while `characteristics` is computing, so it never reads the view.
    pub(crate) fn permanent_is_battlefield_creature(&self, permanent_id: PermanentId) -> bool {
        let Some(permanent) = self
            .state
            .permanents
            .get(permanent_id.0)
            .and_then(|p| p.as_ref())
        else {
            return false;
        };
        self.state.zones.zone_of(permanent.card) == Some(ZoneType::Battlefield)
            && self.permanent_is_creature(permanent_id)
    }

    /// Animation-aware creature check.
    fn permanent_is_creature(&self, permanent_id: PermanentId) -> bool {
        self.state
            .permanents
            .get(permanent_id.0)
            .and_then(|p| p.as_ref())
            .is_some_and(|permanent| self.state.cards[permanent.card].types.is_creature())
            || self.permanent_is_animated(permanent_id)
    }

    /// Whether something other than its printed type makes the permanent a
    /// creature: earthbend, its own counter threshold, or an effect.
    fn permanent_is_animated(&self, permanent_id: PermanentId) -> bool {
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
    fn effective_keywords(&self, permanent_id: PermanentId) -> Keywords {
        self.keywords_with(permanent_id, self.ability_loss_timestamp(permanent_id))
    }

    fn keywords_with(&self, permanent_id: PermanentId, lost: Option<ObjectId>) -> Keywords {
        self.recorded_keywords_with(permanent_id, lost)
            .union(&self.static_keyword_bonus(permanent_id, lost))
    }

    /// Whether the permanent can pay a {T} cost or attack (CR 302.6): it is
    /// untapped, and a creature under its controller's control since the
    /// turn began or with haste, as its characteristics report it. A
    /// temporary grant must not erase the restriction that applies again when
    /// the grant expires (CR 702.10).
    pub(crate) fn permanent_can_tap(&self, permanent_id: PermanentId) -> bool {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return false;
        };
        if permanent.tapped {
            return false;
        }
        if !permanent.summoning_sick {
            return true;
        }
        let characteristics = self.characteristics(permanent_id);
        !characteristics.is_creature || characteristics.keywords.haste
    }

    /// Lethal-damage check (CR 704.5g) against the permanent's
    /// `characteristics`.
    pub(crate) fn has_lethal_damage(
        &self,
        permanent_id: PermanentId,
        characteristics: &Characteristics,
    ) -> bool {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return false;
        };
        if !characteristics.is_creature {
            return false;
        }
        if permanent.deathtouch_damage && permanent.damage > 0 {
            return true;
        }
        permanent.damage >= characteristics.toughness
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
        self.recorded_keywords_with(permanent_id, self.ability_loss_timestamp(permanent_id))
    }

    fn recorded_keywords_with(
        &self,
        permanent_id: PermanentId,
        lost: Option<ObjectId>,
    ) -> Keywords {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return Keywords::default();
        };
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
        let power = match view {
            ScopeView::Rules => self.effective_pt(permanent_id).0,
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

// -----------------------------------------------------------------------------
// Activated abilities (layer 6)
// -----------------------------------------------------------------------------
impl Game {
    /// `Characteristics::activated_abilities`.
    fn activated_abilities_with(
        &self,
        permanent: PermanentId,
        basis: &LayerBasis,
        is_land: bool,
    ) -> Vec<ActivatedAbilityDefinition> {
        let Some(source) = self.state.permanents[permanent].as_ref() else {
            return Vec::new();
        };
        let card = &self.state.cards[source.card];
        let purpose = ManaUse::ability(
            card.subtypes
                .iter()
                .map(String::as_str)
                .chain(basis.added_subtypes.iter().copied()),
        );
        let lost = basis.lost.is_some();
        let mut abilities = Vec::new();
        if !lost {
            abilities = card.activated_abilities.clone();
            self.extend_own_abilities(permanent, basis, is_land, &mut abilities);
        }
        for &(grantor, card) in &basis.attachments {
            if lost && card.attached_characteristics().is_none() {
                continue;
            }
            let grantor = self
                .permanent_object_ref(grantor)
                .expect("battlefield permanent has an exact identity");
            abilities.extend(card.attached_grants().enumerate().map(|(index, ability)| {
                ActivatedAbilityDefinition {
                    granted_by: Some(AbilityGrant { grantor, index }),
                    ..ability.clone()
                }
            }));
        }
        for ability in &mut abilities {
            self.price_activation(source.controller, purpose, ability);
        }
        abilities
    }

    /// Read state-dependent mana outputs of printed abilities, then add mana
    /// abilities from `mana_grant` permanents and gained basic land types:
    /// everything `permanent` loses with its abilities.
    fn extend_own_abilities(
        &self,
        permanent: PermanentId,
        basis: &LayerBasis,
        is_land: bool,
        abilities: &mut Vec<ActivatedAbilityDefinition>,
    ) {
        let source = self.state.permanents[permanent].as_ref().unwrap();
        for ability in abilities.iter_mut() {
            if let [effect] = ability.effects.as_slice() {
                if let Some(effect) = self.current_mana_output(effect, source.controller) {
                    ability.effects = vec![effect];
                }
            }
        }
        for grant in self.battlefield_permanents(source.controller) {
            let grant = self.state.permanents[grant].as_ref().unwrap();
            if self.state.cards[grant.card].statics.iter().any(|ability| {
                matches!(
                    ability,
                    StaticAbility::GrantsAnyColorManaAbility(predicate)
                        if self.permanent_matches_predicate(permanent, predicate)
                )
            }) {
                abilities.push(ActivatedAbilityDefinition {
                    tap_source: true,
                    effects: vec![Effect::any_color_mana()],
                    ..Default::default()
                });
            }
        }
        if let Some(effect) = self.gained_land_type_mana(permanent, &basis.added_subtypes, is_land)
        {
            abilities.push(ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![effect],
                ..Default::default()
            });
        }
    }

    /// A land with a basic land type has "{T}: Add [its color]" (CR 305.6).
    /// Printed types already carry that ability; types gained until end of
    /// turn (Energybending) add one ability choosing among their colors.
    fn gained_land_type_mana(
        &self,
        permanent: PermanentId,
        added: &[&str],
        is_land: bool,
    ) -> Option<Effect> {
        let source = self.state.permanents[permanent].as_ref()?;
        if added.is_empty() || !is_land {
            return None;
        }
        let card = &self.state.cards[source.card];
        let colors: Vec<Mana> = BASIC_LAND_TYPES
            .iter()
            .filter(|(name, _)| added.contains(name) && !card.has_subtype(name))
            .map(|(_, mana)| Mana::parse(mana))
            .collect();
        match colors.as_slice() {
            [] => None,
            [mana] => Some(Effect::AddMana {
                mana: mana.clone(),
                until_end_of_combat: false,
            }),
            _ => Some(Effect::mana_choice(colors)),
        }
    }
}
