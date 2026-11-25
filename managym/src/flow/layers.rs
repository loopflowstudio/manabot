//! CR 613 evaluation over a single explicit battlefield context. Effects inspect
//! instructions, not final values, when determining dependencies. A context is
//! private derived data; it never calls `Game::characteristics`.

use crate::{
    flow::{game::Game, statics::Characteristics},
    state::{
        ability::{
            Ability, Condition, Effect, EffectValue, EffectiveTrigger, Except, GrantClock,
            ObjectProperty, ObjectSelector, Scope,
        },
        card::{
            AbilityGrant, ActivatedAbilityDefinition, CardType, CardTypes, Keyword, Keywords,
            LayerValue, PowerCda, StaticAbility, StaticPtBuff, StaticScope,
        },
        effect::Modification,
        game_object::{
            BattlefieldFacts, Object, ObjectId, ObjectKind, ObjectLki, PermanentId, PlayerId,
            PublicId,
        },
        mana::{Colors, Mana, SpendingContext, SpendingObject},
        permanent::{CounterKind, Counters, Permanent},
        predicate::CardPredicate,
        subtype::{is_creature_type, subtype_category, subtype_corresponds_to, CreatureTypes},
        zone::ZoneType,
    },
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

const BASIC_LANDS: [(&str, &str); 5] = [
    ("Plains", "W"),
    ("Island", "U"),
    ("Swamp", "B"),
    ("Mountain", "R"),
    ("Forest", "G"),
];

/// Evaluation order, including the four distinct power/toughness sublayers.
/// Copy and control are committed by their event owners before this context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Layer {
    Name,
    Type,
    Color,
    Ability,
    CharacteristicPower, // 7a
    BasePower,           // 7b
    ModifyPower,         // 7c
    SwitchPower,         // 7d
}

impl Layer {
    const ALL: [Self; 8] = [
        Self::Name,
        Self::Type,
        Self::Color,
        Self::Ability,
        Self::CharacteristicPower,
        Self::BasePower,
        Self::ModifyPower,
        Self::SwitchPower,
    ];
}

#[derive(Clone, Debug, PartialEq)]
struct StaticEntry {
    key: Vec<usize>,
    timestamp: PublicId,
    ability: StaticAbility,
}
// Derived fields written by `finish_object`: tracked per object so
// Layer 5-7-only mutations skip recomputing creature types, activation
// purposes and static lists.
const FINISH_TYPES: u8 = 1;
const FINISH_ACTIVATIONS: u8 = 2;
const FINISH_STATICS: u8 = 4;
const FINISH_ALL: u8 = FINISH_TYPES | FINISH_ACTIVATIONS | FINISH_STATICS;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct LayerObject {
    pub view: Arc<Characteristics>,
    statics: Vec<StaticEntry>,
    dirty: u8,
}
#[derive(Clone)]
pub(super) struct Context<'a> {
    game: &'a Game,
    pub objects: BTreeMap<PermanentId, Arc<LayerObject>>,
    /// Land-type overwrites active in layer 4, including when the battlefield
    /// has no matching land yet. Entry consults them before a land exists.
    pub land_type_overwrites: Vec<CardPredicate>,
    /// Includes conditional nodes discovered in any layer, even when an earlier
    /// condition short-circuits or a dependency later removes their source.
    pub reads_mana: bool,
    pub reads_tapped: bool,
    pub reads_life: bool,
}

/// An effect keeps one identity across layers. `targets` is populated on its
/// first application to objects (CR 613.6). Resolved records already own targets.
#[derive(Clone)]
struct Node {
    source: PermanentId,
    key: Vec<usize>,
    timestamp: PublicId,
    targets: Option<Vec<PermanentId>>,
    plan: Arc<SourcePlan>,
}

/// Definition-derived instructions, shared across revisions and search forks.
#[derive(Clone)]
struct SourcePlan {
    cda: bool,
    scope: StaticScope,
    condition: Option<Condition>,
    parts: Vec<(Layer, Operation)>,
    reads: ReadSet,
}

impl std::ops::Deref for Node {
    type Target = SourcePlan;
    fn deref(&self) -> &Self::Target {
        &self.plan
    }
}

/// One live incarnation's copiable object and bound static plans. Holding the
/// card keeps definition identity valid even when a fixture/copy detaches it.
#[derive(Clone)]
pub(super) struct SourceSeed {
    source: Object,
    incarnation: PublicId,
    timestamp: PublicId,
    chosen: Option<Arc<crate::state::permanent::EntryChoice>>,
    object: Arc<LayerObject>,
    nodes: Vec<Node>,
}

/// Conservative reads used for both dependency pruning and cache invalidation.
/// Walk each predicate/condition once, including nested alternatives.
#[derive(Clone, Copy, Default)]
struct ReadSet {
    fields: u8,
    mana: bool,
    tapped: bool,
    life: bool,
}

impl ReadSet {
    fn predicate(&mut self, p: &CardPredicate) {
        if p.chosen_name || p.namesake || p.named.is_some() {
            self.fields |= NAME;
        }
        if p.card_type.is_some() || !p.card_types_any.is_empty() || !p.not_card_types.is_empty() {
            self.fields |= CARD_TYPES;
        }
        if p.chosen_subtype
            || p.subtype.is_some()
            || !p.subtypes_any.is_empty()
            || !p.not_subtypes.is_empty()
        {
            self.fields |= SUBTYPES;
        }
        if p.supertype.is_some() {
            self.fields |= SUPERTYPES;
        }
        if p.chosen_color || !p.colors_any.is_empty() {
            self.fields |= COLORS;
        }
        if p.keyword.is_some() || p.not_flying {
            self.fields |= KEYWORDS;
        }
        if p.min_power.is_some() || p.max_power.is_some() || p.min_toughness.is_some() {
            self.fields |= POWER;
        }
        self.tapped |= p.tapped;
        for alternative in &p.any_of {
            self.predicate(alternative);
        }
    }

    fn selector(&mut self, of: &ObjectSelector) {
        if let ObjectSelector::Each {
            predicate,
            exact_mana_value,
            max_mana_value,
            ..
        } = of
        {
            self.predicate(predicate);
            for bound in [exact_mana_value, max_mana_value].into_iter().flatten() {
                self.value(bound);
            }
        }
    }

    fn value(&mut self, value: &EffectValue) {
        match value {
            EffectValue::Half(inner) | EffectValue::Scaled { value: inner, .. } => {
                self.value(inner)
            }
            EffectValue::GraveyardMatching { predicate, .. } if predicate.reads_power() => {
                self.fields |= CARD_TYPES
            }
            EffectValue::SourcePower => self.fields |= POWER,
            EffectValue::GreatestSharedCreatureType => self.fields |= CARD_TYPES | SUBTYPES,
            EffectValue::CreaturesWithCounters => self.fields |= CARD_TYPES,
            EffectValue::GreatestPowerYouControl { predicate } => {
                self.fields |= CARD_TYPES | POWER;
                self.predicate(predicate);
            }
            EffectValue::BattlefieldMatching { predicate }
            | EffectValue::AttackingMatching { predicate } => self.predicate(predicate),
            EffectValue::Property { of, property, .. } => {
                if matches!(property, ObjectProperty::Power | ObjectProperty::Toughness) {
                    self.fields |= POWER;
                }
                self.selector(of);
            }
            EffectValue::Objects { of, predicate, .. } => {
                if let Some(p) = predicate {
                    self.predicate(p);
                }
                self.selector(of);
            }
            EffectValue::DistinctManaValues { of } => self.selector(of),
            EffectValue::LifeTotal(_)
            | EffectValue::LifeAboveStarting(_)
            | EffectValue::LifeGainedThisTurn(_)
            | EffectValue::LifeLostThisTurn(_) => self.life = true,
            _ => {}
        }
    }

    fn condition(&mut self, condition: &Condition) {
        match condition {
            Condition::Compare { left, right, .. } => {
                self.value(left);
                self.value(right);
            }
            Condition::ControlsAtLeast {
                predicate,
                distinct_names,
                ..
            } => {
                self.predicate(predicate);
                if *distinct_names {
                    self.fields |= NAME;
                }
            }
            Condition::OpponentControlsMoreCreatures => self.fields |= CARD_TYPES,
            Condition::BattlefieldAtLeast { predicate, .. }
            | Condition::SourceMatches(predicate) => {
                self.predicate(predicate);
            }
            Condition::AllOf(conditions) | Condition::AnyOf(conditions) => {
                for condition in conditions {
                    self.condition(condition);
                }
            }
            Condition::Not(condition) => self.condition(condition),
            Condition::SourceTapped => self.tapped = true,
            Condition::UnspentManaAtLeast { .. } => self.mana = true,
            Condition::LifeAtLeast { .. } => self.life = true,
            _ => {}
        }
    }
}

impl SourcePlan {
    fn on_source(parts: Vec<(Layer, Operation)>) -> Self {
        Self {
            cda: false,
            scope: StaticScope::This,
            condition: None,
            parts,
            reads: ReadSet::default(),
        }
    }

    fn summarize(mut self) -> Self {
        let mut reads = ReadSet::default();
        match &self.scope {
            StaticScope::All(p)
            | StaticScope::Other(p)
            | StaticScope::OtherYouControl(p)
            | StaticScope::YouControl(p)
            | StaticScope::SpellsYouControl(p) => reads.predicate(p),
            StaticScope::This | StaticScope::Attached => {}
        }
        if let Some(condition) = &self.condition {
            reads.condition(condition);
        }
        for (_, operation) in &self.parts {
            match operation {
                Operation::Buff(buff) => {
                    if buff.per_creature_type_max.is_some() {
                        reads.fields |= SUBTYPES;
                    }
                    if let Some(predicate) = &buff.per_color_among {
                        reads.fields |= COLORS;
                        reads.predicate(predicate);
                    }
                    if let Some(predicate) = &buff.per_graveyard_matching {
                        reads.predicate(predicate);
                    }
                }
                Operation::Cda { cda, .. } => match cda.as_ref() {
                    PowerCda::CountersOnLandsYouControl | PowerCda::CreaturesYouControl => {
                        reads.fields |= CARD_TYPES;
                    }
                    PowerCda::GraveyardMatching(predicate)
                    | PowerCda::GraveyardMatchingBy { predicate, .. } => reads.predicate(predicate),
                },
                _ => {}
            }
        }
        self.reads = reads;
        self
    }
}

impl Node {
    /// Counters and resolved records already bind one affected object; their
    /// existence and membership no longer depend on a generating static ability.
    fn resolved(
        source: PermanentId,
        key: Vec<usize>,
        timestamp: PublicId,
        part: (Layer, Operation),
    ) -> Self {
        Self {
            source,
            key,
            timestamp,
            targets: Some(vec![source]),
            plan: Arc::new(SourcePlan::on_source(vec![part]).summarize()),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Operation {
    Modify(Modification<LayerValue>),
    LandType(String),
    // Expressions resolve before becoming per-target instructions. Indirection
    // keeps their large predicates out of every small resolved modification.
    /// A characteristic-defining count; `sets_toughness` sets toughness to
    /// the same value ("power and toughness are each equal to").
    Cda {
        cda: Box<PowerCda>,
        sets_toughness: bool,
    },
    Buff(Box<StaticPtBuff>),
    Inherit,
    CatalogTrigger(Box<crate::state::ability::Ability>),
}
// Intermediate fields that instructions can read or change. State-only reads
// (zones, counters, mana, controller, printed mana value) cannot change during
// a layer pass. ABILITIES includes a generating static's continued existence.
const NAME: u8 = 1;
const CARD_TYPES: u8 = 2;
const SUBTYPES: u8 = 4;
const SUPERTYPES: u8 = 8;
const COLORS: u8 = 16;
const KEYWORDS: u8 = 32;
const POWER: u8 = 64;
const ABILITIES: u8 = 128;

impl Operation {
    fn writes(&self) -> u8 {
        match self {
            Self::LandType(_) => SUBTYPES | ABILITIES | KEYWORDS,
            Self::Cda { .. } | Self::Buff(_) => POWER,
            Self::Inherit | Self::CatalogTrigger(_) => 0,
            Self::Modify(m) => match m {
                Modification::SetName(_) => NAME,
                Modification::SetCreatureTypes(_)
                | Modification::AddSubtypes(_)
                | Modification::AllCreatureTypes => SUBTYPES,
                Modification::AddTypes(_) | Modification::BecomeCreature => CARD_TYPES,
                Modification::SetTypes { .. } => CARD_TYPES | SUBTYPES,
                Modification::Nonlegendary => SUPERTYPES,
                Modification::SetColors(_) | Modification::AddColors(_) => COLORS,
                Modification::AddKeywords(_) | Modification::LoseKeywords(_) => KEYWORDS,
                Modification::LoseAllAbilities | Modification::LoseNonManaAbilities => {
                    ABILITIES | KEYWORDS
                }
                Modification::GrantStatic(_) => ABILITIES,
                Modification::SetBasePt { .. }
                | Modification::ModifyPt { .. }
                | Modification::SwitchPt => POWER,
                Modification::Firebending(_)
                | Modification::GrantActivated(_)
                | Modification::GrantTriggered(_)
                | Modification::HasAbilitiesOfSource => 0,
                // Future rules payloads remain conservative until classified.
                _ => u8::MAX,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Instruction {
    target: PermanentId,
    operation: Operation,
}

fn layer<N>(modification: &Modification<N>) -> Option<Layer> {
    Some(match modification {
        Modification::SetName(_) => Layer::Name,
        Modification::SetCreatureTypes(_)
        | Modification::AddSubtypes(_)
        | Modification::AllCreatureTypes
        | Modification::AddTypes(_)
        | Modification::SetTypes { .. }
        | Modification::BecomeCreature
        | Modification::Nonlegendary => Layer::Type,
        Modification::SetColors(_) | Modification::AddColors(_) => Layer::Color,
        Modification::AddKeywords(_)
        | Modification::LoseNonManaAbilities
        | Modification::LoseAllAbilities
        | Modification::LoseKeywords(_)
        | Modification::Firebending(_)
        | Modification::GrantActivated(_)
        | Modification::GrantTriggered(_)
        | Modification::GrantStatic(_)
        | Modification::HasAbilitiesOfSource => Layer::Ability,
        Modification::SetBasePt { .. } => Layer::BasePower,
        Modification::ModifyPt { .. } => Layer::ModifyPower,
        Modification::SwitchPt => Layer::SwitchPower,
        _ => return None,
    })
}

impl<'a> Context<'a> {
    pub fn evaluate(game: &'a Game, seeds: Option<&mut Vec<Option<Arc<SourceSeed>>>>) -> Self {
        #[cfg(feature = "benchmark")]
        let _timer = crate::benchmark_layers::Timer::new(|m, ns| {
            m.boards += 1;
            m.board_ns += ns;
        });
        // The debug oracle rebuilds source plans from definitions. Its fresh
        // objects need no retained seed owner while evaluating modifications.
        let retain_sources = seeds.is_some();
        let mut fresh = Vec::new();
        let seeds = seeds.unwrap_or(&mut fresh);
        let mut context = Self {
            game,
            objects: BTreeMap::new(),
            land_type_overwrites: Vec::new(),
            reads_mana: false,
            reads_tapped: false,
            reads_life: false,
        };
        let mut nodes = Vec::new();
        let mut known = BTreeSet::new();
        seeds.resize(game.state.permanents.len(), None);
        for (slot, permanent) in game.state.permanents.iter().enumerate() {
            let Some(p) = permanent else {
                seeds[slot] = None;
                continue;
            };
            let id = PermanentId(slot);
            let source = &game.state[p.object];
            let reusable = seeds[slot].as_ref().is_some_and(|seed| {
                seed.source.shares_definition_with(source)
                    && seed.incarnation == p.id
                    && seed.timestamp == p.timestamp
                    && match (&seed.chosen, &p.chosen) {
                        (None, None) => true,
                        (Some(a), Some(b)) => Arc::ptr_eq(a, b) || a == b,
                        _ => false,
                    }
            });
            if !reusable {
                seeds[slot] = Some(Arc::new(context.seed(id, p, source)));
            }
            let seed = seeds[slot].as_ref().unwrap();
            context.objects.insert(id, seed.object.clone());
            known.extend(seed.object.statics.iter().map(|entry| entry.key.clone()));
            // A phased-out permanent's static abilities apply to nothing
            // (CR 702.26b); its own view still takes the records on it.
            if !game.is_phased_out(id) {
                for node in &seed.nodes {
                    context.reads_mana |= node.reads.mana;
                    context.reads_tapped |= node.reads.tapped;
                    context.reads_life |= node.reads.life;
                    nodes.push(node.clone());
                }
            }
            for (kind, count) in p.counters.iter() {
                if let Some(keyword) = kind.keyword().filter(|_| count > 0) {
                    nodes.push(Node::resolved(
                        id,
                        vec![slot, usize::MAX - 1, kind as usize],
                        p.keyword_counter_timestamps
                            .get(&kind)
                            .copied()
                            .unwrap_or(p.timestamp),
                        (
                            Layer::Ability,
                            Operation::Modify(Modification::AddKeywords(Keywords::of(&[keyword]))),
                        ),
                    ));
                }
            }
            for (i, record) in game.effects_on(id).enumerate() {
                let Some(layer) = layer(&record.modification) else {
                    continue;
                };
                let operation = if record.modification == Modification::HasAbilitiesOfSource {
                    Operation::Inherit
                } else {
                    Operation::Modify(record.modification.map_amount(|n| LayerValue::Fixed(*n)))
                };
                nodes.push(Node::resolved(
                    id,
                    vec![slot, usize::MAX, i],
                    record.timestamp,
                    (layer, operation),
                ));
            }
        }
        if !retain_sources {
            seeds.clear();
        }
        for layer in Layer::ALL {
            let mut applied = BTreeSet::new();
            loop {
                let pending: Vec<_> = nodes
                    .iter()
                    .enumerate()
                    .filter(|(i, node)| {
                        !applied.contains(i)
                            && node.parts.iter().any(|(part, _)| *part == layer)
                            && context.exists(node)
                    })
                    .map(|(i, _)| i)
                    .collect();
                if pending.is_empty() {
                    break;
                }
                let instructions: Vec<_> = pending
                    .iter()
                    .map(|&i| context.instructions(&nodes[i], layer))
                    .collect();
                for selected in context.next_effects(&nodes, &pending, &instructions, layer) {
                    let index = pending[selected];
                    if nodes[index].targets.is_none()
                        && nodes[index].parts.iter().any(|(part, _)| *part > layer)
                        && !instructions[selected].is_empty()
                    {
                        // Instructions already resolved the affected set. Retain
                        // it only for effects that continue into a later layer.
                        let mut targets: Vec<_> = instructions[selected]
                            .iter()
                            .map(|instruction| instruction.target)
                            .collect();
                        targets.dedup();
                        nodes[index].targets = Some(targets);
                    }
                    // A layer-6 ability loss can remove the printed static
                    // after its type effect has already applied (CR 613.6).
                    let node = &nodes[index];
                    if layer == Layer::Type
                        && matches!(node.parts.as_slice(), [(_, Operation::LandType(_))])
                        && context.condition_met(node)
                    {
                        if let StaticScope::All(predicate) = &node.scope {
                            context.land_type_overwrites.push(predicate.clone());
                        }
                    }
                    context.apply(node, &instructions[selected]);
                    // Only a static grant introduces layer nodes. Removing abilities
                    // changes existence, which the pending filter checks directly.
                    if instructions[selected].iter().any(|i| {
                        matches!(i.operation, Operation::Modify(Modification::GrantStatic(_)))
                    }) {
                        context.discover(&mut nodes, &mut known);
                    }
                    applied.insert(index);
                }
            }
            if layer == Layer::ModifyPower {
                for (&id, object) in &mut context.objects {
                    // CR 613.4c: +1/+1 and -1/-1 counters apply together.
                    let permanent = game.live_permanent(id);
                    let count = permanent.counter_count(CounterKind::PlusOnePlusOne)
                        - permanent.counter_count(CounterKind::MinusOneMinusOne);
                    if count != 0 {
                        let view = Arc::make_mut(&mut Arc::make_mut(object).view);
                        view.power += count;
                        view.toughness += count;
                    }
                }
            }
        }
        for (&id, object) in &mut context.objects {
            if !retain_sources || object.dirty != 0 {
                let object = Arc::make_mut(object);
                if !retain_sources {
                    // The uncached debug oracle recomputes every derived field so
                    // `statics.rs` verifies dirty tracking on every board rebuild.
                    object.dirty = FINISH_ALL;
                }
                let permanent = game.live_permanent(id);
                let record = &game.state[permanent.object];
                Self::finish_object(object, record, permanent.chosen.as_deref());
            }
        }
        context
    }

    /// The copiable object and bound static plans of one live incarnation.
    fn seed(&self, id: PermanentId, permanent: &Permanent, source: &Object) -> SourceSeed {
        #[cfg(feature = "benchmark")]
        crate::benchmark_layers::record(|m| m.source_plans += 1);
        let mut object = LayerObject {
            view: Arc::new(Characteristics {
                name: source.name.clone(),
                subtypes: source.subtypes.clone(),
                creature_types: if source.keywords.has(Keyword::Changeling) {
                    CreatureTypes::All
                } else {
                    CreatureTypes::default()
                },
                types: source.types.clone(),
                colors: source.colors(),
                mana_abilities: source.mana_abilities.clone(),
                triggered_mana_abilities: source.triggered_mana_abilities.clone(),
                is_legendary: source.is_legendary(),
                keywords: source.keywords,
                ward: source.ward().cloned(),
                power: source.power.unwrap_or(0),
                toughness: source.toughness.unwrap_or(0),
                activated_abilities: source.activated_abilities.clone(),
                triggered_abilities: source
                    .abilities
                    .iter()
                    .cloned()
                    .map(|ability| EffectiveTrigger {
                        clock: None,
                        ability,
                        program: false,
                    })
                    .collect(),
                ..Default::default()
            }),
            statics: source
                .statics
                .iter()
                .enumerate()
                .map(|(i, ability)| StaticEntry {
                    key: vec![id.0, i],
                    timestamp: permanent.timestamp,
                    ability: ability.clone(),
                })
                .collect(),
            dirty: FINISH_ALL,
        };
        Self::finish_object(&mut object, source, permanent.chosen.as_deref());
        let nodes = object
            .statics
            .iter()
            .filter_map(|entry| self.node(id, entry))
            .collect();
        SourceSeed {
            source: source.clone(),
            incarnation: permanent.id,
            timestamp: permanent.timestamp,
            chosen: permanent.chosen.clone(),
            object: Arc::new(object),
            nodes,
        }
    }

    fn finish_object(
        object: &mut LayerObject,
        record: &Object,
        chosen: Option<&crate::state::permanent::EntryChoice>,
    ) {
        let dirty = object.dirty;
        if dirty == 0 {
            return;
        }
        object.dirty = 0;
        let view = Arc::make_mut(&mut object.view);
        if dirty & FINISH_TYPES != 0 {
            view.is_animated = view.is_creature() && !record.types.is_creature();
            // During the pass `CreatureTypes::All` marks a record granting all
            // creature types; a listed set is derived from the final subtypes.
            view.creature_types = if !view.is_creature() {
                CreatureTypes::default()
            } else if view.creature_types.is_all() {
                CreatureTypes::All
            } else {
                CreatureTypes::among(&view.subtypes)
            };
        }
        // Activations are paid as abilities of this source (CR 106.6), read
        // from its final subtypes; an unchanged source keeps sharing its
        // earlier snapshot. Fixed entry-choice mana abilities materialize once
        // here so `characteristics()` shares this immutable view without
        // cloning or rebuilding restrictions.
        if dirty & (FINISH_TYPES | FINISH_ACTIVATIONS) != 0 && !view.activated_abilities.is_empty()
        {
            let resolve_entry_mana = dirty & FINISH_ACTIVATIONS != 0;
            let shared = view
                .activated_abilities
                .iter()
                .map(|ability| &ability.mana_cost.purpose)
                .find(|purpose| {
                    matches!(purpose, SpendingContext::Activate {
                        zone: ZoneType::Battlefield,
                        source,
                    } if source.describes(&view.subtypes, view.creature_types.is_all(), view.is_creature()))
                })
                .cloned();
            let purpose = shared.unwrap_or_else(|| SpendingContext::Activate {
                zone: ZoneType::Battlefield,
                source: SpendingObject::new(
                    view.subtypes.iter().map(String::as_str),
                    view.creature_types.is_all(),
                    view.is_creature(),
                ),
            });
            for ability in &mut view.activated_abilities {
                if resolve_entry_mana {
                    ability.resolve_fixed_entry_mana(chosen);
                }
                if ability.mana_cost.purpose != purpose {
                    ability.mana_cost.purpose = purpose.clone();
                }
            }
        }
        if dirty & FINISH_STATICS != 0 {
            view.static_abilities = object
                .statics
                .iter()
                .map(|entry| entry.ability.clone())
                .collect();
        }
    }
    fn discover(&mut self, nodes: &mut Vec<Node>, known: &mut BTreeSet<Vec<usize>>) {
        // Occurrence provenance preserves duplicate grants, including statics
        // discovered during layer 6 that start applying in a later layer.
        for (&id, object) in &self.objects {
            if self.game.is_phased_out(id) {
                continue;
            }
            for entry in &object.statics {
                if known.insert(entry.key.clone()) {
                    if let Some(node) = self.node(id, entry) {
                        self.reads_mana |= node.reads.mana;
                        self.reads_tapped |= node.reads.tapped;
                        self.reads_life |= node.reads.life;
                        nodes.push(node);
                    }
                }
            }
        }
    }

    /// Select a pending effect under CR 613.8: dependencies first, then
    /// characteristic-defining status and timestamps. Independent instructions
    /// can all apply in one pass; dependent sets return one pending index so
    /// discovery, applicability and dependency ordering are reevaluated.
    fn next_effects(
        &self,
        nodes: &[Node],
        pending: &[usize],
        instructions: &[Vec<Instruction>],
        layer: Layer,
    ) -> Vec<usize> {
        if pending.len() == 1 {
            return vec![0];
        }
        let timestamp_order = |&a: &usize| {
            let node = &nodes[pending[a]];
            (!node.cda, node.timestamp, &node.key)
        };
        // A disjoint read/write pair cannot change the other's instructions.
        // Summaries are conservative; true dependencies still use CR 613.8's
        // before/after semantic comparison. Granted statics force rediscovery.
        let writes: Vec<_> = instructions
            .iter()
            .map(|is| is.iter().fold(0, |fields, i| fields | i.operation.writes()))
            .collect();
        let plausible = |a: usize, b: usize| {
            let node = &nodes[pending[a]];
            let reads = node.reads.fields | if node.targets.is_none() { ABILITIES } else { 0 };
            a != b && node.cda == nodes[pending[b]].cda && reads & writes[b] != 0
        };
        let independent = !(0..pending.len()).any(|a| (0..pending.len()).any(|b| plausible(a, b)))
            && !instructions
                .iter()
                .flatten()
                .any(|i| matches!(i.operation, Operation::Modify(Modification::GrantStatic(_))));
        let ordered = || {
            let mut order: Vec<_> = (0..pending.len()).collect();
            order.sort_by_key(timestamp_order);
            order
        };
        if independent && !cfg!(debug_assertions) {
            return ordered();
        }
        let mut dependencies = vec![vec![false; pending.len()]; pending.len()];
        #[cfg(feature = "benchmark")]
        let _timer = crate::benchmark_layers::Timer::new(|m, ns| {
            m.dependency_passes += 1;
            m.dependency_ns += ns;
        });
        for (b, &index) in pending.iter().enumerate() {
            if !cfg!(debug_assertions) && !(0..pending.len()).any(|a| plausible(a, b)) {
                continue;
            }
            let mut after = {
                #[cfg(feature = "benchmark")]
                let _timer = crate::benchmark_layers::Timer::new(|m, ns| {
                    m.contexts_cloned += 1;
                    m.clone_ns += ns;
                });
                self.clone()
            };
            after.apply(&nodes[index], &instructions[b]);
            for (a, &other) in pending.iter().enumerate() {
                if a != b && nodes[other].cda == nodes[index].cda {
                    if !cfg!(debug_assertions) && !plausible(a, b) {
                        continue;
                    }
                    dependencies[a][b] = self.exists(&nodes[other]) != after.exists(&nodes[other])
                        || instructions[a] != after.instructions(&nodes[other], layer);
                    debug_assert!(
                        plausible(a, b) || !dependencies[a][b],
                        "layer read/write summary missed a dependency"
                    );
                    #[cfg(feature = "benchmark")]
                    crate::benchmark_layers::record(|m| {
                        m.comparisons += 1;
                        m.edges += u64::from(dependencies[a][b]);
                    });
                }
            }
        }
        if independent {
            // Keep the complete pairwise oracle in debug, once per independent
            // batch rather than cloning the same board for every suffix.
            debug_assert!(dependencies.iter().flatten().all(|edge| !edge));
            return ordered();
        }
        // Transitive reachability identifies only edges within dependency
        // loops; external prerequisites still precede the loop (CR 613.8b).
        let mut reach = dependencies.clone();
        for k in 0..pending.len() {
            for a in 0..pending.len() {
                for b in 0..pending.len() {
                    reach[a][b] |= reach[a][k] && reach[k][b];
                }
            }
        }
        vec![(0..pending.len())
            .filter(|&a| !(0..pending.len()).any(|b| dependencies[a][b] && !reach[b][a]))
            .min_by_key(timestamp_order)
            .expect("dependency graph has a source component")]
    }

    fn exists(&self, node: &Node) -> bool {
        node.targets.is_some()
            || self
                .objects
                .get(&node.source)
                .is_some_and(|o| o.statics.iter().any(|s| s.key == node.key))
    }
    fn condition_met(&self, node: &Node) -> bool {
        node.condition
            .as_ref()
            .is_none_or(|c| self.condition(c, node.source))
    }

    fn targets(&self, node: &Node) -> Vec<PermanentId> {
        if let Some(targets) = &node.targets {
            return targets.clone();
        }
        if !self.condition_met(node) {
            return Vec::new();
        }
        let (predicate, only_other, only_controlled) = match &node.scope {
            StaticScope::This => {
                return self
                    .objects
                    .contains_key(&node.source)
                    .then_some(node.source)
                    .into_iter()
                    .collect();
            }
            StaticScope::Attached => {
                return self
                    .game
                    .attached_permanent(node.source)
                    .filter(|id| self.objects.contains_key(id))
                    .into_iter()
                    .collect();
            }
            StaticScope::SpellsYouControl(_) => return Vec::new(),
            StaticScope::OtherYouControl(p) => (p, true, true),
            StaticScope::YouControl(p) => (p, false, true),
            StaticScope::Other(p) => (p, true, false),
            StaticScope::All(p) => (p, false, false),
        };
        let source_controller = self.controller(node.source);
        let pred_source = self.predicate_source(predicate, Some(node.source));
        self.objects
            .iter()
            .filter(|&(&id, object)| {
                (!only_other || id != node.source)
                    && (!only_controlled || self.controller(id) == source_controller)
                    && matches_view(self.game, id, predicate, &object.view, pred_source)
            })
            .map(|(&id, _)| id)
            .collect()
    }
    fn instructions(&self, node: &Node, layer: Layer) -> Vec<Instruction> {
        if !self.exists(node) {
            return Vec::new();
        }
        self.targets(node)
            .into_iter()
            .flat_map(|id| {
                node.parts
                    .iter()
                    .filter(move |(part, _)| *part == layer)
                    .map(move |(_, op)| {
                        let operation = match op {
                            Operation::Modify(m) => Operation::Modify(
                                m.map_amount(|n| LayerValue::Fixed(self.value(n, id))),
                            ),
                            Operation::Cda {
                                cda,
                                sets_toughness,
                            } => {
                                let value =
                                    LayerValue::Fixed(self.cda(cda, self.controller(node.source)));
                                Operation::Modify(Modification::SetBasePt {
                                    power: Some(value.clone()),
                                    toughness: sets_toughness.then_some(value),
                                })
                            }
                            Operation::Buff(buff) => self.buff(buff, node.source, id),
                            op => op.clone(),
                        };
                        Instruction {
                            target: id,
                            operation,
                        }
                    })
            })
            .collect()
    }
    fn buff(&self, buff: &StaticPtBuff, source: PermanentId, id: PermanentId) -> Operation {
        let per_type = buff.per_creature_type_max.map_or(1, |max| {
            let view = &self.objects[&id].view;
            if view.has_all_creature_types() {
                max
            } else {
                (view.subtypes.iter().filter(|s| is_creature_type(s)).count() as i32).min(max)
            }
        });
        let controller = self.controller(source);
        let per_color = buff.per_color_among.as_ref().map_or(1, |p| {
            self.objects
                .iter()
                .filter(|(id, _)| {
                    self.controller(**id) == controller && self.matches(**id, p, None)
                })
                .flat_map(|(_, o)| o.view.colors.iter().copied())
                .collect::<BTreeSet<_>>()
                .len() as i32
        });
        let per_card = buff
            .per_graveyard_matching
            .as_ref()
            .map_or(1, |p| self.graveyard_count(controller, p));
        let times = per_type * per_color * per_card;
        Operation::Modify(Modification::ModifyPt {
            power: LayerValue::Fixed(buff.power * times),
            toughness: LayerValue::Fixed(buff.toughness * times),
        })
    }
    fn controller(&self, id: PermanentId) -> PlayerId {
        self.game.live_permanent(id).controller
    }
    fn value(&self, value: &LayerValue, id: PermanentId) -> i32 {
        match value {
            LayerValue::Fixed(n) => *n,
            LayerValue::ManaValue => i32::from(
                self.game
                    .object_mana_value(self.game.live_permanent(id).object),
            ),
            LayerValue::GraveyardCardTypes { offset } => {
                [PlayerId(0), PlayerId(1)]
                    .into_iter()
                    .flat_map(|player| self.game.state.zone_cards(ZoneType::Graveyard, player))
                    .flat_map(|card| {
                        self.game.state[self.game.state.object_of(card)]
                            .types
                            .types
                            .iter()
                    })
                    .collect::<BTreeSet<_>>()
                    .len() as i32
                    + offset
            }
        }
    }
    fn creature_count(&self, controller: PlayerId) -> usize {
        self.objects
            .iter()
            .filter(|(id, object)| self.controller(**id) == controller && object.view.is_creature())
            .count()
    }
    fn cda(&self, cda: &PowerCda, controller: PlayerId) -> i32 {
        match cda {
            PowerCda::CountersOnLandsYouControl => self
                .objects
                .iter()
                .filter(|(id, o)| self.controller(**id) == controller && o.view.is_land())
                .map(|(id, _)| {
                    self.game
                        .live_permanent(*id)
                        .counter_count(CounterKind::PlusOnePlusOne)
                })
                .sum(),
            PowerCda::CreaturesYouControl => self.creature_count(controller) as i32,
            PowerCda::GraveyardMatching(p) => self.graveyard_count(controller, p),
            PowerCda::GraveyardMatchingBy { who, predicate } => (0..self.game.state.players.len())
                .map(PlayerId)
                .filter(|player| who.includes(controller, *player))
                .map(|player| self.graveyard_count(player, predicate))
                .sum(),
        }
    }
    fn graveyard_count(&self, owner: PlayerId, predicate: &CardPredicate) -> i32 {
        self.game
            .state
            .zone_cards(ZoneType::Graveyard, owner)
            .filter(|card| {
                let object = self.game.state.object_of(*card);
                self.game.off_battlefield_matches(object, predicate, || {
                    let card = &self.game.state[object];
                    card.pt_cda().map_or_else(
                        || card.power.unwrap_or(0),
                        |(cda, _)| self.cda(cda, card.owner),
                    )
                })
            })
            .count() as i32
    }

    fn property(&self, id: PermanentId, property: ObjectProperty) -> i32 {
        match property {
            ObjectProperty::Power => self.objects[&id].view.power,
            ObjectProperty::Toughness => self.objects[&id].view.toughness,
            ObjectProperty::ManaValue => self.game.permanent_mana_value(id),
            ObjectProperty::Colors => self.objects[&id].view.colors.len() as i32,
            ObjectProperty::Counters(kind) => self.game.state.permanents[id]
                .as_ref()
                .map_or(0, |permanent| permanent.counter_count(kind)),
        }
    }

    fn select_permanents(
        &self,
        selector: &ObjectSelector,
        source: PermanentId,
    ) -> Vec<PermanentId> {
        let controller = self.controller(source);
        match selector {
            ObjectSelector::This => vec![source],
            ObjectSelector::Each {
                of,
                predicate,
                except,
                exact_mana_value,
                max_mana_value,
            } => {
                let exact = exact_mana_value
                    .as_ref()
                    .map(|n| self.comparison_value(n, source));
                let max = max_mana_value
                    .as_ref()
                    .map(|n| self.comparison_value(n, source));
                self.objects
                    .keys()
                    .copied()
                    .filter(|&id| {
                        !self.game.is_phased_out(id)
                            && match of {
                                Scope::YouControl => self.controller(id) == controller,
                                Scope::OpponentsControl => self.controller(id) != controller,
                                Scope::Battlefield => true,
                                _ => false,
                            }
                            && !(id == source && *except == Some(Except::This))
                            && self.matches(id, predicate, Some(source))
                            && exact.is_none_or(|n| self.game.permanent_mana_value(id) == n)
                            && max.is_none_or(|n| self.game.permanent_mana_value(id) <= n)
                    })
                    .collect()
            }
            _ => Vec::new(),
        }
    }

    /// Static comparisons see the characteristics at this point in the layer system.
    fn comparison_value(&self, value: &EffectValue, source: PermanentId) -> i32 {
        let controller = self.controller(source);
        let controlled = || self.game.iter_battlefield_permanents(controller);
        match value {
            EffectValue::Half(inner) => self.comparison_value(inner, source).div_euclid(2),
            EffectValue::Scaled { factor, value } => {
                factor.saturating_mul(self.comparison_value(value, source))
            }
            EffectValue::SourcePower => self.objects[&source].view.power,
            EffectValue::Property {
                of,
                property,
                aggregate,
            } => aggregate.evaluate(
                self.select_permanents(of, source)
                    .into_iter()
                    .map(|id| self.property(id, *property)),
            ),
            EffectValue::Objects { of, predicate, .. } => self
                .select_permanents(of, source)
                .into_iter()
                .filter(|&id| {
                    predicate
                        .as_ref()
                        .is_none_or(|p| self.matches(id, p, Some(source)))
                })
                .count() as i32,
            EffectValue::DistinctManaValues { of } => self
                .select_permanents(of, source)
                .into_iter()
                .map(|id| self.game.permanent_mana_value(id))
                .collect::<BTreeSet<_>>()
                .len() as i32,
            EffectValue::GreatestPowerYouControl { predicate } => controlled()
                .filter(|&id| {
                    self.objects[&id].view.is_creature() && self.matches(id, predicate, None)
                })
                .map(|id| self.objects[&id].view.power)
                .max()
                .unwrap_or(0)
                .max(0),
            EffectValue::GreatestSharedCreatureType => {
                let mut counts = BTreeMap::new();
                let mut all = 0;
                for id in controlled().filter(|id| self.objects[id].view.is_creature()) {
                    if self.objects[&id].view.has_all_creature_types() {
                        all += 1;
                        continue;
                    }
                    for subtype in self.objects[&id]
                        .view
                        .subtypes
                        .iter()
                        .filter(|s| is_creature_type(s))
                    {
                        *counts.entry(subtype).or_insert(0) += 1;
                    }
                }
                all + counts.into_values().max().unwrap_or(0)
            }
            EffectValue::CreaturesWithCounters => controlled()
                .filter(|&id| {
                    self.objects[&id].view.is_creature()
                        && self
                            .game
                            .live_permanent(id)
                            .counter_count(CounterKind::PlusOnePlusOne)
                            > 0
                })
                .count() as i32,
            EffectValue::AttackingMatching { predicate }
            | EffectValue::BattlefieldMatching { predicate } => controlled()
                .filter(|&id| {
                    self.matches(id, predicate, None)
                        && (!matches!(value, EffectValue::AttackingMatching { .. })
                            || self.game.live_permanent(id).attacking)
                })
                .count() as i32,
            EffectValue::GraveyardMatching { base, predicate } => {
                base.saturating_add(self.graveyard_count(controller, predicate))
            }
            _ => self.game.controller_value(value, controller),
        }
    }

    fn condition(&self, condition: &Condition, source: PermanentId) -> bool {
        match condition {
            Condition::Compare { left, op, right } => op.test(
                self.comparison_value(left, source),
                self.comparison_value(right, source),
            ),
            Condition::OpponentControlsMoreCreatures => {
                let controller = self.controller(source);
                self.creature_count(PlayerId(1 - controller.0)) > self.creature_count(controller)
            }
            Condition::BattlefieldAtLeast { count, predicate } => {
                self.objects
                    .keys()
                    .filter(|&&id| self.matches(id, predicate, Some(source)))
                    .count()
                    >= *count
            }
            Condition::ControlsAtLeast {
                count,
                predicate,
                distinct_names,
            } => {
                let matching = self.objects.keys().filter(|&&id| {
                    self.controller(id) == self.controller(source)
                        && self.matches(id, predicate, Some(source))
                });
                super::statics::controlled_threshold(matching, *count, *distinct_names, |id| {
                    self.objects[id].view.name.as_str()
                })
            }
            Condition::SourceMatches(predicate) => self.matches(source, predicate, Some(source)),
            Condition::AllOf(conditions) => conditions.iter().all(|c| self.condition(c, source)),
            Condition::AnyOf(conditions) => conditions.iter().any(|c| self.condition(c, source)),
            Condition::Not(condition) => !self.condition(condition, source),
            // All other static conditions read state, not characteristics.
            _ => self
                .game
                .layer_state_condition(condition, self.controller(source), source),
        }
    }
    fn predicate_source(
        &self,
        predicate: &CardPredicate,
        source: Option<PermanentId>,
    ) -> Option<PredicateSource<'_>> {
        source
            .filter(|_| predicate.reads_source())
            .map(|s| PredicateSource {
                name: self.objects[&s].view.name.as_str(),
                chosen: self.game.live_permanent(s).chosen.as_deref(),
            })
    }

    fn matches(
        &self,
        id: PermanentId,
        predicate: &CardPredicate,
        source: Option<PermanentId>,
    ) -> bool {
        matches_view(
            self.game,
            id,
            predicate,
            &self.objects[&id].view,
            self.predicate_source(predicate, source),
        )
    }
    fn node(&self, source: PermanentId, entry: &StaticEntry) -> Option<Node> {
        let mut plan = SourcePlan::on_source(Vec::new());
        let mut add = |l, m| plan.parts.push((l, Operation::Modify(m)));
        match &entry.ability {
            StaticAbility::Continuous(effect) => {
                plan.scope = effect.scope.clone();
                plan.condition = effect.condition.clone();
                plan.cda = effect.characteristic_defining;
                for m in &effect.modifications {
                    // A modification outside the layer system (a prohibition
                    // or prevention) has no reader here; a printed static
                    // spelling it would silently do nothing.
                    debug_assert!(
                        layer(m).is_some(),
                        "continuous static modification {m:?} has no layer"
                    );
                    if let Some(l) = layer(m) {
                        let layer = if plan.cda && l == Layer::BasePower {
                            Layer::CharacteristicPower
                        } else {
                            l
                        };
                        add(layer, m.clone());
                    }
                }
            }
            StaticAbility::HasChosenCreatureType => {
                let Some(crate::state::permanent::EntryChoice::CreatureType(name)) =
                    self.game.live_permanent(source).chosen.as_deref()
                else {
                    return None;
                };
                add(Layer::Type, Modification::AddSubtypes(vec![name.clone()]));
            }
            StaticAbility::TypeGrant(g) => {
                plan.scope = g.scope.clone();
                plan.condition = g.condition.clone();
                add(Layer::Type, Modification::AddTypes(g.types.clone()));
                add(Layer::Type, Modification::AddSubtypes(g.subtypes.clone()));
            }
            StaticAbility::KeywordGrant(g) => {
                plan.scope = g.scope.clone();
                plan.condition = g.condition.clone();
                add(Layer::Ability, Modification::AddKeywords(g.keywords));
            }
            StaticAbility::PtBuff(b) => {
                plan.scope = b.scope.clone();
                plan.condition = b.condition.clone();
                plan.parts
                    .push((Layer::ModifyPower, Operation::Buff(Box::new(b.clone()))));
            }
            StaticAbility::CharacteristicPower(cda) | StaticAbility::CharacteristicPt(cda) => {
                plan.cda = true;
                plan.parts.push((
                    Layer::CharacteristicPower,
                    Operation::Cda {
                        cda: Box::new(cda.clone()),
                        sets_toughness: matches!(entry.ability, StaticAbility::CharacteristicPt(_)),
                    },
                ));
            }
            StaticAbility::SetsLandType(set) => {
                plan.scope = StaticScope::All(set.predicate.clone());
                plan.condition = Some(set.condition.clone());
                plan.parts
                    .push((Layer::Type, Operation::LandType(set.subtype.clone())));
            }
            StaticAbility::AttachedSetsCharacteristics(set) => {
                plan.scope = StaticScope::Attached;
                plan.parts = vec![
                    (
                        Layer::Name,
                        Operation::Modify(Modification::SetName(set.named.clone())),
                    ),
                    (
                        Layer::Type,
                        Operation::Modify(Modification::SetCreatureTypes(
                            set.creature_types.clone(),
                        )),
                    ),
                    (
                        Layer::Ability,
                        Operation::Modify(Modification::LoseAllAbilities),
                    ),
                    (
                        Layer::BasePower,
                        Operation::Modify(Modification::SetBasePt {
                            power: Some(LayerValue::Fixed(set.power)),
                            toughness: Some(LayerValue::Fixed(set.toughness)),
                        }),
                    ),
                ];
            }
            StaticAbility::AttachedHas(ability) => {
                plan.scope = StaticScope::Attached;
                let object = &self.game.state[self.game.live_permanent(source).object];
                let index = object
                    .attached_grants()
                    .position(|a| a == ability)
                    .unwrap_or(0);
                let mut granted = ability.clone();
                granted.granted_by = self
                    .game
                    .permanent_object(source)
                    .map(|grantor| AbilityGrant { grantor, index });
                add(
                    Layer::Ability,
                    Modification::GrantActivated(Box::new(granted)),
                );
            }
            StaticAbility::TriggeredGrant(grant) => {
                plan.scope = grant.scope.clone();
                plan.parts.push((
                    Layer::Ability,
                    Operation::CatalogTrigger(grant.ability.clone()),
                ));
            }
            StaticAbility::GrantsAnyColorManaAbility(predicate) => {
                plan.scope = StaticScope::YouControl(predicate.clone());
                add(
                    Layer::Ability,
                    Modification::GrantActivated(Box::new(tap_for_mana(Effect::any_color_mana()))),
                );
            }
            _ => return None,
        }
        Some(Node {
            source,
            key: entry.key.clone(),
            timestamp: entry.timestamp,
            targets: None,
            plan: Arc::new(plan.summarize()),
        })
    }
    fn apply(&mut self, node: &Node, instructions: &[Instruction]) {
        let clock = |ability: &Ability, index| {
            let Ability::Triggered { condition, .. } = ability;
            condition.once_each_turn().then(|| GrantClock {
                timestamp: node.timestamp,
                grantor: node.source,
                static_slot: (node.key[1] != usize::MAX).then_some(node.key[1]),
                ability_index: index,
            })
        };
        for (ordinal, instruction) in instructions.iter().enumerate() {
            let id = instruction.target;
            let object = Arc::make_mut(self.objects.get_mut(&id).unwrap());
            let view = Arc::make_mut(&mut object.view);
            match &instruction.operation {
                Operation::LandType(subtype) => {
                    if !view.is_land() {
                        continue;
                    }
                    view.subtypes
                        .retain(|s| !BASIC_LANDS.iter().any(|(name, _)| s == name));
                    view.subtypes.push(subtype.clone());
                    view.added_subtypes.push(subtype.clone());
                    lose_all_abilities(view, &mut object.statics);
                    view.activated_abilities
                        .extend(intrinsic_mana_ability(|name| name == subtype));
                    object.dirty |= FINISH_TYPES | FINISH_STATICS;
                }
                Operation::Inherit => {
                    if let Some(definition) = self.game.inherited_definition(id) {
                        if !definition.activated_abilities.is_empty() {
                            view.activated_abilities
                                .extend(definition.activated_abilities.clone());
                            object.dirty |= FINISH_ACTIVATIONS;
                        }
                        view.triggered_abilities.extend(
                            definition.abilities.iter().cloned().enumerate().map(
                                |(index, ability)| EffectiveTrigger {
                                    clock: clock(&ability, index),
                                    ability,
                                    program: false,
                                },
                            ),
                        );
                    }
                }
                Operation::CatalogTrigger(ability) => {
                    view.triggered_abilities.push(EffectiveTrigger {
                        clock: clock(ability, 0),
                        ability: (**ability).clone(),
                        program: false,
                    })
                }
                Operation::Modify(modification) => match modification {
                    Modification::SetName(name) => {
                        view.name = name.clone();
                        view.name_set = true;
                    }
                    Modification::AllCreatureTypes => {
                        // A later grant of every type overrides an earlier set list.
                        view.creature_types = CreatureTypes::All;
                        view.creature_types_set = false;
                        object.dirty |= FINISH_TYPES;
                    }
                    Modification::SetCreatureTypes(types) => {
                        view.creature_types = CreatureTypes::default();
                        view.creature_types_set = true;
                        view.subtypes.retain(|s| !is_creature_type(s));
                        view.subtypes.extend(types.iter().cloned());
                        view.added_subtypes.retain(|s| !is_creature_type(s));
                        object.dirty |= FINISH_TYPES;
                    }
                    Modification::AddSubtypes(types) => {
                        if view.is_land() {
                            let gained = intrinsic_mana_ability(|name| {
                                types.iter().any(|subtype| subtype == name)
                                    && !view.subtypes.iter().any(|subtype| subtype == name)
                            });
                            view.activated_abilities.extend(gained);
                        }
                        for subtype in types {
                            if !view.subtypes.contains(subtype) {
                                view.subtypes.push(subtype.clone());
                                view.added_subtypes.push(subtype.clone());
                            }
                        }
                        object.dirty |= FINISH_TYPES;
                    }
                    Modification::AddTypes(types) => {
                        view.types.types.extend(types.iter().copied());
                        object.dirty |= FINISH_TYPES;
                    }
                    Modification::SetTypes { types, subtypes } => {
                        // CR 205.1a, 205.3d: a subtype survives while it
                        // corresponds to a remaining type and the effect sets
                        // no subtype of its kind (a basic land imprisoned as a
                        // land stays a Forest; a Citizen loses other creature
                        // types).
                        let keeps = |subtype: &String| {
                            types.iter().any(|t| subtype_corresponds_to(subtype, *t))
                                && !subtypes
                                    .iter()
                                    .any(|set| subtype_category(set) == subtype_category(subtype))
                        };
                        let sets_creature_types = subtypes.iter().any(|s| is_creature_type(s));
                        if !types.contains(&CardType::Creature) || sets_creature_types {
                            view.creature_types_set |= sets_creature_types
                                || view.creature_types.is_all()
                                || view.subtypes.iter().any(|s| is_creature_type(s));
                            view.creature_types = CreatureTypes::default();
                        }
                        view.types.types = types.iter().copied().collect();
                        view.subtypes.retain(keeps);
                        view.added_subtypes.retain(keeps);
                        for subtype in subtypes {
                            if !view.subtypes.contains(subtype) {
                                view.subtypes.push(subtype.clone());
                                if !is_creature_type(subtype) {
                                    view.added_subtypes.push(subtype.clone());
                                }
                            }
                        }
                        object.dirty |= FINISH_TYPES;
                    }
                    Modification::BecomeCreature => {
                        view.types.types.insert(CardType::Creature);
                        object.dirty |= FINISH_TYPES;
                    }
                    Modification::Nonlegendary => view.is_legendary = false,
                    Modification::SetColors(colors) => view.colors = colors.clone(),
                    Modification::AddColors(colors) => view.colors.extend(colors),
                    Modification::AddKeywords(keywords) => {
                        view.keywords = view.keywords.union(*keywords)
                    }
                    Modification::LoseKeywords(keywords) => {
                        view.keywords = view.keywords.without(*keywords)
                    }
                    Modification::LoseNonManaAbilities => {
                        lose_nonmana_abilities(view, &mut object.statics);
                        object.dirty |= FINISH_STATICS;
                    }
                    Modification::LoseAllAbilities => {
                        lose_all_abilities(view, &mut object.statics);
                        object.dirty |= FINISH_STATICS;
                    }
                    Modification::Firebending(n) => {
                        view.firebending = view.firebending.saturating_add(*n);
                        view.triggered_abilities.push(EffectiveTrigger {
                            clock: None,
                            ability: Ability::firebending(*n),
                            program: true,
                        });
                    }
                    Modification::GrantActivated(ability) => {
                        view.activated_abilities.push((**ability).clone());
                        object.dirty |= FINISH_ACTIVATIONS;
                    }
                    Modification::GrantTriggered(ability) => {
                        view.triggered_abilities.push(EffectiveTrigger {
                            clock: clock(ability, 0),
                            ability: (**ability).clone(),
                            program: true,
                        })
                    }
                    Modification::GrantStatic(ability) => {
                        if let StaticAbility::Ward(cost) = ability.as_ref() {
                            view.ward = Some(cost.clone());
                            view.triggered_abilities.push(EffectiveTrigger {
                                clock: None,
                                ability: Ability::ward(cost.clone()),
                                program: true,
                            });
                        }
                        let mut key = node.key.clone();
                        key.extend([id.0, ordinal]);
                        object.statics.push(StaticEntry {
                            key,
                            timestamp: node.timestamp.max(self.game.live_permanent(id).timestamp),
                            ability: (**ability).clone(),
                        });
                        object.dirty |= FINISH_STATICS;
                    }
                    Modification::SetBasePt { power, toughness } => {
                        if let Some(LayerValue::Fixed(p)) = power {
                            view.power = *p;
                        }
                        if let Some(LayerValue::Fixed(t)) = toughness {
                            view.toughness = *t;
                        }
                    }
                    Modification::ModifyPt {
                        power: LayerValue::Fixed(p),
                        toughness: LayerValue::Fixed(t),
                    } => {
                        view.power += p;
                        view.toughness += t;
                    }
                    Modification::SwitchPt => std::mem::swap(&mut view.power, &mut view.toughness),
                    _ => unreachable!("layer instruction is resolved"),
                },
                _ => unreachable!("layer instruction is resolved"),
            }
        }
    }
}

impl Game {
    /// Layer evaluation only visits battlefield slots, which are always live.
    pub(super) fn live_permanent(&self, id: PermanentId) -> &Permanent {
        self.state.permanents[id].as_ref().unwrap()
    }
}

fn lose_nonmana_abilities(view: &mut Characteristics, statics: &mut Vec<StaticEntry>) {
    view.abilities_lost = true;
    view.keywords = Keywords::default();
    view.ward = None;
    view.firebending = 0;
    view.activated_abilities
        .retain(ActivatedAbilityDefinition::is_mana_ability);
    view.triggered_abilities.clear();
    statics.clear();
}

fn lose_all_abilities(view: &mut Characteristics, statics: &mut Vec<StaticEntry>) {
    view.mana_abilities.clear();
    view.triggered_mana_abilities.clear();
    view.activated_abilities.clear();
    lose_nonmana_abilities(view, statics);
}

fn tap_for_mana(effect: Effect) -> ActivatedAbilityDefinition {
    ActivatedAbilityDefinition {
        tap_source: true,
        effects: vec![effect],
        ..Default::default()
    }
}

/// CR 305.6 gives a land each basic type's intrinsic mana ability. Types gained
/// by one effect share one modal ability, the engine's form for a tap that
/// produces one chosen color; modes follow WUBRG order.
fn intrinsic_mana_ability(gained: impl Fn(&str) -> bool) -> Option<ActivatedAbilityDefinition> {
    let mut outputs: Vec<_> = BASIC_LANDS
        .iter()
        .filter(|(name, _)| gained(name))
        .map(|(_, mana)| Mana::parse(mana))
        .collect();
    match outputs.len() {
        0 => None,
        1 => Some(tap_for_mana(Effect::add_mana(outputs.remove(0)))),
        _ => Some(tap_for_mana(Effect::mana_choice(outputs))),
    }
}

pub(super) fn matches_static_scope(
    scope: &StaticScope,
    source: PermanentId,
    source_controller: PlayerId,
    attached: Option<PermanentId>,
    target: PermanentId,
    target_controller: PlayerId,
    matches: impl FnOnce(&CardPredicate) -> bool,
) -> bool {
    let predicate = match scope {
        StaticScope::This => return source == target,
        StaticScope::Attached => return attached == Some(target),
        StaticScope::SpellsYouControl(_) => return false,
        StaticScope::OtherYouControl(p)
            if source != target && source_controller == target_controller =>
        {
            p
        }
        StaticScope::YouControl(p) if source_controller == target_controller => p,
        StaticScope::Other(p) if source != target => p,
        StaticScope::All(p) => p,
        _ => return false,
    };
    matches(predicate)
}

/// Source-permanent facts read by source-relative predicates (`namesake`, `chosen_*`).
#[derive(Clone, Copy)]
pub(super) struct PredicateSource<'a> {
    pub name: &'a str,
    pub chosen: Option<&'a crate::state::permanent::EntryChoice>,
}

/// A live permanent under `view` (an intermediate layer context or its
/// final view).
pub(super) fn matches_view(
    game: &Game,
    id: PermanentId,
    p: &CardPredicate,
    view: &Characteristics,
    source: Option<PredicateSource<'_>>,
) -> bool {
    let Some(permanent) = game.state.permanents[id].as_ref() else {
        return false;
    };
    let object = &game.state[permanent.object];
    CharacteristicView {
        types: &view.types,
        subtypes: &view.subtypes,
        all_creature_types: view.has_all_creature_types(),
        supertypes: &object.supertypes,
        colors: &view.colors,
        name: &view.name,
        keywords: view.keywords,
        power: view.power,
        toughness: view.toughness,
        is_legendary: view.is_legendary,
        token: object.kind() == ObjectKind::Token,
        attacking: permanent.attacking,
        tapped: permanent.tapped,
        derived: DerivedStatus::Live {
            game,
            id,
            object: permanent.object,
        },
        counters: &permanent.counters,
    }
    .matches(p, source)
}

/// The characteristics and permanent status a battlefield predicate reads,
/// borrowed from a live permanent or a departed object's last known facts.
struct CharacteristicView<'a> {
    types: &'a CardTypes,
    subtypes: &'a [String],
    all_creature_types: bool,
    supertypes: &'a [String],
    colors: &'a Colors,
    name: &'a str,
    keywords: Keywords,
    power: i32,
    toughness: i32,
    is_legendary: bool,
    token: bool,
    attacking: bool,
    tapped: bool,
    derived: DerivedStatus<'a>,
    counters: &'a Counters,
}

/// Mana value and combat status, which few predicates read: a live
/// permanent computes them only when asked, a departure recorded them.
enum DerivedStatus<'a> {
    Live {
        game: &'a Game,
        id: PermanentId,
        object: ObjectId,
    },
    Departed(&'a BattlefieldFacts),
}

impl DerivedStatus<'_> {
    fn mana_value(&self) -> u8 {
        match self {
            Self::Live { game, object, .. } => game.object_mana_value(*object),
            Self::Departed(facts) => facts.mana_value,
        }
    }

    fn in_combat(&self) -> bool {
        match self {
            Self::Live { game, id, .. } => game.permanent_in_combat(*id),
            Self::Departed(facts) => facts.in_combat,
        }
    }
}

/// A departed object's predicate match against its last battlefield
/// characteristics (CR 603.10a), never the card it became. Namesake
/// predicates name a live source, so a departure has none.
pub(super) fn matches_departure(lki: &ObjectLki, p: &CardPredicate) -> bool {
    let facts = &lki.battlefield;
    CharacteristicView {
        types: &facts.types,
        subtypes: &facts.subtypes,
        all_creature_types: lki.creature_types.is_all(),
        supertypes: &facts.supertypes,
        colors: &facts.colors,
        name: &facts.name,
        keywords: lki.keywords,
        power: lki.power,
        toughness: lki.toughness,
        is_legendary: facts.is_legendary,
        token: facts.token,
        attacking: facts.attacking,
        tapped: facts.tapped,
        derived: DerivedStatus::Departed(facts),
        counters: &lki.counters,
    }
    .matches(p, None)
}

impl CharacteristicView<'_> {
    fn has_subtype(&self, subtype: &str) -> bool {
        self.subtypes.iter().any(|s| s == subtype)
            || (self.all_creature_types && is_creature_type(subtype))
    }

    fn matches(&self, p: &CardPredicate, source: Option<PredicateSource<'_>>) -> bool {
        // A permanent is on the battlefield, never a graveyard card.
        !p.from_battlefield_this_turn
            && !p.put_into_graveyard_this_turn
            && p.accepts_name(self.name)
            && (!p.namesake || source.map(|s| s.name) == Some(self.name))
            && p.matches_choice(
                source.and_then(|s| s.chosen),
                self.name,
                |c| self.colors.contains(&c),
                |s| self.has_subtype(s),
            )
            && p.keyword.is_none_or(|k| self.keywords.has(k))
            && (!p.not_flying || !self.keywords.has(Keyword::Flying))
            && (!p.no_counters || self.counters.is_empty())
            && (!p.in_combat || self.derived.in_combat())
            && (!p.with_counter || self.counters.get(CounterKind::PlusOnePlusOne) > 0)
            && (!p.attacking || self.attacking)
            && (!p.tapped || self.tapped)
            && p.token.is_none_or(|token| token == self.token)
            && (p.any_of.is_empty()
                || p.any_of
                    .iter()
                    .any(|alternative| self.matches(alternative, source)))
            && p.supertype.as_ref().is_none_or(|s| {
                if s.eq_ignore_ascii_case("legendary") {
                    self.is_legendary
                } else {
                    self.supertypes.iter().any(|t| t.eq_ignore_ascii_case(s))
                }
            })
            && (!p.nonlegendary || !self.is_legendary)
            && (!p.nonbasic
                || !self
                    .supertypes
                    .iter()
                    .any(|s| s.eq_ignore_ascii_case("basic")))
            && p.card_type.is_none_or(|t| self.types.types.contains(&t))
            && (p.card_types_any.is_empty()
                || p.card_types_any
                    .iter()
                    .any(|t| self.types.types.contains(t)))
            && p.not_card_types
                .iter()
                .all(|t| !self.types.types.contains(t))
            && p.subtype.as_ref().is_none_or(|s| self.has_subtype(s))
            && (p.subtypes_any.is_empty() || p.subtypes_any.iter().any(|s| self.has_subtype(s)))
            && (p.colors_any.is_empty() || p.colors_any.iter().any(|c| self.colors.contains(c)))
            && p.not_subtypes.iter().all(|s| !self.has_subtype(s))
            && p.min_power.is_none_or(|min| self.power >= min)
            && p.min_toughness
                .is_none_or(|min| self.toughness >= i32::from(min))
            && p.max_power.is_none_or(|max| self.power <= max)
            && (p.min_mana_value.is_none() && p.max_mana_value.is_none() || {
                let mana_value = self.derived.mana_value();
                p.min_mana_value.is_none_or(|min| mana_value >= min)
                    && p.max_mana_value.is_none_or(|max| mana_value <= max)
            })
    }
}
