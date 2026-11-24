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
            AbilityGrant, ActivatedAbilityDefinition, CardType, Keyword, Keywords, LayerValue,
            PowerCda, StaticAbility, StaticPtBuff, StaticScope,
        },
        effect::Modification,
        game_object::{Object, ObjectKind, PermanentId, PlayerId, PublicId},
        mana::{Mana, SpendingContext, SpendingObject},
        permanent::{CounterKind, Permanent},
        predicate::CardPredicate,
        subtype::is_creature_type,
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
#[derive(Clone, Debug, PartialEq)]
pub(super) struct LayerObject {
    pub view: Arc<Characteristics>,
    statics: Vec<StaticEntry>,
}
#[derive(Clone)]
pub(super) struct Context<'a> {
    game: &'a Game,
    pub objects: BTreeMap<PermanentId, Arc<LayerObject>>,
    /// Includes conditional nodes discovered in any layer, even when an earlier
    /// condition short-circuits or a dependency later removes their source.
    pub reads_mana: bool,
    pub reads_tapped: bool,
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
}

impl ReadSet {
    fn predicate(&mut self, p: &CardPredicate) {
        if p.namesake {
            self.fields |= NAME;
        }
        if p.card_type.is_some() || !p.card_types_any.is_empty() || !p.not_card_types.is_empty() {
            self.fields |= CARD_TYPES;
        }
        if p.subtype.is_some() || !p.subtypes_any.is_empty() || !p.not_subtypes.is_empty() {
            self.fields |= SUBTYPES;
        }
        if p.supertype.is_some() {
            self.fields |= SUPERTYPES;
        }
        if !p.colors_any.is_empty() {
            self.fields |= COLORS;
        }
        if p.flying || p.deathtouch {
            self.fields |= KEYWORDS;
        }
        if p.min_power.is_some() || p.max_power.is_some() {
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
            EffectValue::Half(inner) => self.value(inner),
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
            EffectValue::Objects { of, predicate } => {
                if let Some(p) = predicate {
                    self.predicate(p);
                }
                self.selector(of);
            }
            EffectValue::DistinctManaValues { of } => self.selector(of),
            _ => self.mana |= value.reads_mana_pools(),
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
            Condition::BattlefieldAtLeast { predicate, .. }
            | Condition::SourceMatches(predicate) => {
                self.predicate(predicate);
            }
            Condition::AllOf(conditions) | Condition::AnyOf(conditions) => {
                for condition in conditions {
                    self.condition(condition);
                }
            }
            Condition::SourceTapped => self.tapped = true,
            _ => self.mana |= condition.reads_mana_pools(),
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
                Operation::Power(cda) => match cda.as_ref() {
                    PowerCda::CountersOnLandsYouControl | PowerCda::CreaturesYouControl => {
                        reads.fields |= CARD_TYPES;
                    }
                    PowerCda::GraveyardMatching(predicate) => reads.predicate(predicate),
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
    Name(String),
    // Expressions resolve before becoming per-target instructions. Indirection
    // keeps their large predicates out of every small resolved modification.
    Power(Box<PowerCda>),
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
            Self::Name(_) => NAME,
            Self::LandType(_) => SUBTYPES | ABILITIES | KEYWORDS,
            Self::Power(_) | Self::Buff(_) => POWER,
            Self::Inherit | Self::CatalogTrigger(_) => 0,
            Self::Modify(m) => match m {
                Modification::SetCreatureTypes(_) | Modification::AddSubtypes(_) => SUBTYPES,
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
        Modification::SetCreatureTypes(_)
        | Modification::AddSubtypes(_)
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
            reads_mana: false,
            reads_tapped: false,
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
                    context.apply(&nodes[index], &instructions[selected]);
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
            if !retain_sources || !Arc::ptr_eq(object, &seeds[id.0].as_ref().unwrap().object) {
                let record = &game.state[game.live_permanent(id).object];
                Self::finish_object(Arc::make_mut(object), record);
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
        };
        Self::finish_object(&mut object, source);
        let nodes = object
            .statics
            .iter()
            .filter_map(|entry| self.node(id, entry))
            .collect();
        SourceSeed {
            source: source.clone(),
            incarnation: permanent.id,
            timestamp: permanent.timestamp,
            object: Arc::new(object),
            nodes,
        }
    }

    fn finish_object(object: &mut LayerObject, record: &Object) {
        let view = Arc::make_mut(&mut object.view);
        view.is_animated = view.is_creature() && !record.types.is_creature();
        view.creature_types = if view.is_creature() {
            view.subtypes
                .iter()
                .filter(|s| is_creature_type(s))
                .cloned()
                .collect()
        } else {
            Vec::new()
        };
        view.creature_types.sort();
        view.creature_types.dedup();
        // Activations are paid as abilities of this source (CR 106.6), read
        // from its final subtypes; an unchanged source keeps sharing its
        // earlier snapshot.
        if !view.activated_abilities.is_empty() {
            let shared = view
                .activated_abilities
                .iter()
                .map(|ability| &ability.mana_cost.purpose)
                .find(|purpose| {
                    matches!(purpose, SpendingContext::Activate {
                        zone: ZoneType::Battlefield,
                        source,
                    } if source.describes(&view.subtypes))
                })
                .cloned();
            let purpose = shared.unwrap_or_else(|| SpendingContext::Activate {
                zone: ZoneType::Battlefield,
                source: SpendingObject::new(view.subtypes.iter().map(String::as_str)),
            });
            for ability in &mut view.activated_abilities {
                if ability.mana_cost.purpose != purpose {
                    ability.mana_cost.purpose = purpose.clone();
                }
            }
        }
        view.static_abilities = object
            .statics
            .iter()
            .map(|entry| entry.ability.clone())
            .collect();
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
    fn targets(&self, node: &Node) -> Vec<PermanentId> {
        if let Some(targets) = &node.targets {
            return targets.clone();
        }
        if node
            .condition
            .as_ref()
            .is_some_and(|c| !self.condition(c, node.source))
        {
            return Vec::new();
        }
        self.objects
            .keys()
            .copied()
            .filter(|&id| self.scope(&node.scope, node.source, id))
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
                            Operation::Power(cda) => Operation::Modify(Modification::SetBasePt {
                                power: Some(LayerValue::Fixed(
                                    self.cda(cda, self.controller(node.source)),
                                )),
                                toughness: None,
                            }),
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
            (self.objects[&id]
                .view
                .subtypes
                .iter()
                .filter(|s| is_creature_type(s))
                .count() as i32)
                .min(max)
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
            LayerValue::ManaValue => self.game.permanent_mana_value(id),
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
            PowerCda::CreaturesYouControl => self
                .objects
                .iter()
                .filter(|(id, o)| self.controller(**id) == controller && o.view.is_creature())
                .count() as i32,
            PowerCda::GraveyardMatching(p) => self.graveyard_count(controller, p),
        }
    }
    fn graveyard_count(&self, controller: PlayerId, predicate: &CardPredicate) -> i32 {
        self.game
            .state
            .zone_cards(ZoneType::Graveyard, controller)
            .filter(|card| {
                let object = self.game.state.object_of(*card);
                self.game.off_battlefield_matches(object, predicate, || {
                    let card = &self.game.state[object];
                    card.power_cda()
                        .map_or_else(|| card.power.unwrap_or(0), |cda| self.cda(cda, card.owner))
                })
            })
            .count() as i32
    }

    fn property(&self, id: PermanentId, property: ObjectProperty) -> i32 {
        match property {
            ObjectProperty::Power => self.objects[&id].view.power,
            ObjectProperty::Toughness => self.objects[&id].view.toughness,
            ObjectProperty::ManaValue => self.game.permanent_mana_value(id),
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
            EffectValue::Objects { of, predicate } => self
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
                for id in controlled().filter(|id| self.objects[id].view.is_creature()) {
                    for subtype in self.objects[&id]
                        .view
                        .subtypes
                        .iter()
                        .filter(|s| is_creature_type(s))
                    {
                        *counts.entry(subtype).or_insert(0) += 1;
                    }
                }
                counts.into_values().max().unwrap_or(0)
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
            // All other static conditions read state, not characteristics.
            _ => self
                .game
                .layer_state_condition(condition, self.controller(source), source),
        }
    }
    fn scope(&self, scope: &StaticScope, source: PermanentId, target: PermanentId) -> bool {
        let predicate = match scope {
            StaticScope::This => return source == target,
            StaticScope::Attached => return self.game.attached_permanent(source) == Some(target),
            StaticScope::SpellsYouControl(_) => return false,
            StaticScope::OtherYouControl(p)
                if source != target && self.controller(source) == self.controller(target) =>
            {
                p
            }
            StaticScope::YouControl(p) if self.controller(source) == self.controller(target) => p,
            StaticScope::Other(p) if source != target => p,
            StaticScope::All(p) => p,
            _ => return false,
        };
        self.matches(target, predicate, Some(source))
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
            source.map(|s| self.objects[&s].view.name.as_str()),
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
            StaticAbility::CharacteristicPower(cda) => {
                plan.cda = true;
                plan.parts.push((
                    Layer::CharacteristicPower,
                    Operation::Power(Box::new(cda.clone())),
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
                    (Layer::Name, Operation::Name(set.named.clone())),
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
                Operation::Name(name) => {
                    view.name = name.clone();
                    view.characteristics_set = true;
                }
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
                }
                Operation::Inherit => {
                    if let Some(definition) = self.game.inherited_definition(id) {
                        view.activated_abilities
                            .extend(definition.activated_abilities.clone());
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
                    Modification::SetCreatureTypes(types) => {
                        view.subtypes.retain(|s| !is_creature_type(s));
                        view.subtypes.extend(types.iter().cloned());
                        view.added_subtypes.retain(|s| !is_creature_type(s));
                        view.added_subtypes.extend(types.iter().cloned());
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
                    }
                    Modification::AddTypes(types) => view.types.types.extend(types.iter().copied()),
                    Modification::SetTypes { types, subtypes } => {
                        view.types.types = types.iter().copied().collect();
                        view.subtypes = subtypes.clone();
                        view.added_subtypes = subtypes.clone();
                    }
                    Modification::BecomeCreature => {
                        view.types.types.insert(CardType::Creature);
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
                        lose_nonmana_abilities(view, &mut object.statics)
                    }
                    Modification::LoseAllAbilities => lose_all_abilities(view, &mut object.statics),
                    Modification::Firebending(n) => {
                        view.firebending = view.firebending.saturating_add(*n);
                        view.triggered_abilities.push(EffectiveTrigger {
                            clock: None,
                            ability: Ability::firebending(*n),
                            program: true,
                        });
                    }
                    Modification::GrantActivated(ability) => {
                        view.activated_abilities.push((**ability).clone())
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
    fn live_permanent(&self, id: PermanentId) -> &Permanent {
        self.state.permanents[id].as_ref().unwrap()
    }
}

fn lose_nonmana_abilities(view: &mut Characteristics, statics: &mut Vec<StaticEntry>) {
    view.abilities_lost = true;
    view.keywords = Keywords::default();
    view.ward = None;
    view.firebending = 0;
    view.activated_abilities
        .retain(|ability| ability.mana_choices().is_some());
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

/// What a battlefield predicate reads of one permanent: its layered view,
/// the printed facts layers leave alone (mana value, token status, other
/// supertypes) and its battlefield status. A live permanent and a departed
/// one's last known information (CR 603.10a) supply the same facts, so one
/// matcher answers both.
pub(crate) struct PermanentFacts<'a> {
    pub name: &'a str,
    pub keywords: Keywords,
    pub types: &'a crate::state::card::CardTypes,
    pub subtypes: &'a [String],
    pub colors: &'a crate::state::mana::Colors,
    pub is_legendary: bool,
    pub power: i32,
    pub supertypes: &'a [String],
    pub mana_value: u8,
    pub token: bool,
    pub counters: &'a crate::state::permanent::Counters,
    pub attacking: bool,
    pub tapped: bool,
    pub in_combat: bool,
}

impl<'a> PermanentFacts<'a> {
    /// A live permanent's facts under `view`, its current layered view.
    fn live(
        game: &Game,
        id: PermanentId,
        permanent: &'a Permanent,
        object: &'a Object,
        view: &'a Characteristics,
    ) -> Self {
        Self {
            name: &view.name,
            keywords: view.keywords,
            types: &view.types,
            subtypes: &view.subtypes,
            colors: &view.colors,
            is_legendary: view.is_legendary,
            power: view.power,
            supertypes: &object.supertypes,
            mana_value: object.mana_value_cost().map_or(0, |m| m.mana_value),
            token: object.kind() == ObjectKind::Token,
            counters: &permanent.counters,
            attacking: permanent.attacking,
            tapped: permanent.tapped,
            in_combat: game.permanent_in_combat(id),
        }
    }

    /// A departed permanent's facts: its last known view and status, with
    /// mana value and supertypes from its retained printing.
    pub(crate) fn departed(
        lki: &'a crate::state::game_object::ObjectLki,
        printing: &'a crate::state::card::PrintedCharacteristics,
        kind: ObjectKind,
    ) -> Self {
        Self {
            name: &lki.name,
            keywords: lki.keywords,
            types: &lki.types,
            subtypes: &lki.subtypes,
            colors: &lki.colors,
            is_legendary: lki.is_legendary,
            power: lki.power,
            supertypes: &printing.supertypes,
            mana_value: printing
                .mana_cost
                .as_ref()
                .map_or(0, |cost| cost.mana_value),
            token: kind == ObjectKind::Token,
            counters: &lki.counters,
            attacking: lki.was_attacking,
            tapped: lki.was_tapped,
            in_combat: lki.was_in_combat,
        }
    }
}

/// Shared predicate semantics for live permanents, intermediate layer
/// contexts and departed permanents' last known information.
pub(crate) fn matches_permanent_facts(
    p: &CardPredicate,
    facts: &PermanentFacts,
    source_name: Option<&str>,
) -> bool {
    // A permanent is on the battlefield, never a graveyard card.
    !p.from_battlefield_this_turn
        && (!p.namesake || source_name == Some(facts.name))
        && (!p.deathtouch || facts.keywords.has(Keyword::Deathtouch))
        && (!p.flying || facts.keywords.has(Keyword::Flying))
        && (!p.no_counters || facts.counters.is_empty())
        && (!p.in_combat || facts.in_combat)
        && (!p.with_counter || facts.counters.get(CounterKind::PlusOnePlusOne) > 0)
        && (!p.attacking || facts.attacking)
        && (!p.tapped || facts.tapped)
        && p.token.is_none_or(|token| token == facts.token)
        && (p.any_of.is_empty()
            || p.any_of
                .iter()
                .any(|alternative| matches_permanent_facts(alternative, facts, source_name)))
        && p.supertype.as_ref().is_none_or(|s| {
            if s.eq_ignore_ascii_case("legendary") {
                facts.is_legendary
            } else {
                facts.supertypes.iter().any(|t| t.eq_ignore_ascii_case(s))
            }
        })
        && (!p.nonbasic
            || !facts
                .supertypes
                .iter()
                .any(|s| s.eq_ignore_ascii_case("basic")))
        && p.card_type.is_none_or(|t| facts.types.types.contains(&t))
        && (p.card_types_any.is_empty()
            || p.card_types_any
                .iter()
                .any(|t| facts.types.types.contains(t)))
        && p.not_card_types
            .iter()
            .all(|t| !facts.types.types.contains(t))
        && p.subtype
            .as_ref()
            .is_none_or(|s| facts.subtypes.contains(s))
        && (p.subtypes_any.is_empty() || p.subtypes_any.iter().any(|s| facts.subtypes.contains(s)))
        && (p.colors_any.is_empty() || p.colors_any.iter().any(|c| facts.colors.contains(c)))
        && p.not_subtypes.iter().all(|s| !facts.subtypes.contains(s))
        && p.min_power.is_none_or(|min| facts.power >= min)
        && p.max_power.is_none_or(|max| facts.power <= max)
        && p.min_mana_value.is_none_or(|min| facts.mana_value >= min)
        && p.max_mana_value.is_none_or(|max| facts.mana_value <= max)
}

/// A live permanent under `view` (an intermediate layer context or its
/// final view).
pub(super) fn matches_view(
    game: &Game,
    id: PermanentId,
    p: &CardPredicate,
    view: &Characteristics,
    source_name: Option<&str>,
) -> bool {
    let Some(permanent) = game.state.permanents[id].as_ref() else {
        return false;
    };
    let object = &game.state[permanent.object];
    matches_permanent_facts(
        p,
        &PermanentFacts::live(game, id, permanent, object, view),
        source_name,
    )
}
