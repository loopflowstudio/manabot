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
            LayerCount, LayerValue, StaticAbility, StaticScope,
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
/// Copiable values are committed at entry, before this context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Layer {
    Name,
    Control,
    Type,
    Color,
    Ability,
    CharacteristicPower, // 7a
    BasePower,           // 7b
    ModifyPower,         // 7c
    SwitchPower,         // 7d
}

impl Layer {
    const ALL: [Self; 9] = [
        Self::Name,
        Self::Control,
        Self::Type,
        Self::Color,
        Self::Ability,
        Self::CharacteristicPower,
        Self::BasePower,
        Self::ModifyPower,
        Self::SwitchPower,
    ];

    #[inline]
    const fn bit(self) -> u16 {
        1u16 << (self as u8)
    }

    #[inline]
    const fn after_mask(self) -> u16 {
        !0u16 << (self as u8 + 1)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct StaticEntry {
    key: Vec<usize>,
    timestamp: PublicId,
    ability: StaticAbility,
}
const DIRTY_TYPES: u8 = 1 << 0;
const DIRTY_ACTIVATIONS: u8 = 1 << 1;
const DIRTY_STATICS: u8 = 1 << 2;
const DIRTY_ALL: u8 = DIRTY_TYPES | DIRTY_ACTIVATIONS | DIRTY_STATICS;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct LayerObject {
    pub view: Arc<Characteristics>,
    pub controller: PlayerId,
    statics: Vec<StaticEntry>,
    dirty: u8,
}
#[derive(Clone)]
pub(super) struct Context<'a> {
    game: &'a Game,
    pub objects: BTreeMap<PermanentId, Arc<LayerObject>>,
    /// Includes conditional nodes discovered in any layer, even when an earlier
    /// condition short-circuits or a dependency later removes their source.
    pub reads_mana: bool,
    pub reads_tapped: bool,
    pub reads_life: bool,
}

#[derive(Clone)]
enum NodeKind {
    Static {
        plan: Arc<SourcePlan>,
        /// Populated on first application for multi-layer static effects (CR 613.6).
        targets: Option<Vec<PermanentId>>,
    },
    Resolved {
        layer: Layer,
        operation: Operation<i32>,
    },
}

/// An effect keeps one identity across layers. Resolved records already bind
/// `source` and carry resolved `i32` amounts.
#[derive(Clone)]
struct Node {
    source: PermanentId,
    key: Vec<usize>,
    timestamp: PublicId,
    kind: NodeKind,
}

/// Definition-derived instructions, shared across revisions and search forks.
#[derive(Clone)]
struct SourcePlan {
    cda: bool,
    scope: StaticScope,
    condition: Option<Condition>,
    parts: Vec<(Layer, Operation)>,
    layers: u16,
    reads: ReadSet,
}

impl Node {
    fn cda(&self) -> bool {
        match &self.kind {
            NodeKind::Static { plan, .. } => plan.cda,
            NodeKind::Resolved { .. } => false,
        }
    }

    fn reads(&self) -> ReadSet {
        match &self.kind {
            NodeKind::Static { plan, .. } => plan.reads,
            NodeKind::Resolved { .. } => ReadSet::default(),
        }
    }

    #[inline]
    fn layers(&self) -> u16 {
        match &self.kind {
            NodeKind::Static { plan, .. } => plan.layers,
            NodeKind::Resolved { layer, .. } => layer.bit(),
        }
    }

    #[inline]
    fn has_layer(&self, layer: Layer) -> bool {
        self.layers() & layer.bit() != 0
    }

    fn is_static_unbound(&self) -> bool {
        matches!(self.kind, NodeKind::Static { targets: None, .. })
    }

    fn is_unconditional_attached_or_self(&self) -> bool {
        matches!(
            &self.kind,
            NodeKind::Static { plan, .. }
                if plan.condition.is_none()
                    && matches!(plan.scope, StaticScope::This | StaticScope::Attached)
        )
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
    fields: u16,
    mana: bool,
    tapped: bool,
    life: bool,
}

impl ReadSet {
    fn predicate(&mut self, p: &CardPredicate) {
        if p.namesake || p.named.is_some() {
            self.fields |= NAME;
        }
        if p.card_type.is_some() || !p.card_types_any.is_empty() || !p.not_card_types.is_empty() {
            self.fields |= CARD_TYPES;
        }
        if p.chosen_type_of_source
            || p.subtype.is_some()
            || !p.subtypes_any.is_empty()
            || !p.not_subtypes.is_empty()
        {
            self.fields |= SUBTYPES;
        }
        if p.supertype.is_some() {
            self.fields |= SUPERTYPES;
        }
        if !p.colors_any.is_empty() {
            self.fields |= COLORS;
        }
        if p.flying || p.not_flying || p.deathtouch {
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

    fn layer_value(&mut self, value: &LayerValue) {
        if let LayerValue::Count(count) = value {
            match count.as_ref() {
                LayerCount::CreatureTypes { .. } => self.fields |= SUBTYPES,
                LayerCount::ColorsAmong(p) => {
                    self.fields |= COLORS;
                    self.predicate(p);
                }
                LayerCount::Graveyard { predicate, .. }
                | LayerCount::YouControl(predicate)
                | LayerCount::CountersOn { predicate, .. } => self.predicate(predicate),
                LayerCount::SourceCounters(_) => {}
            }
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
    fn new() -> Self {
        Self {
            cda: false,
            scope: StaticScope::This,
            condition: None,
            parts: Vec::new(),
            layers: 0,
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
        let mut layers = 0u16;
        for (part_layer, operation) in &self.parts {
            layers |= part_layer.bit();
            if let Operation::Modify(modification) = operation {
                match modification {
                    Modification::Control => reads.fields |= CONTROL,
                    Modification::SetBasePt { power, toughness } => {
                        for value in power.iter().chain(toughness.iter()) {
                            reads.layer_value(value);
                        }
                    }
                    Modification::ModifyPt { power, toughness } => {
                        reads.layer_value(power);
                        reads.layer_value(toughness);
                    }
                    _ => {}
                }
            }
        }
        self.layers = layers;
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
        layer: Layer,
        operation: Operation<i32>,
    ) -> Self {
        Self {
            source,
            key,
            timestamp,
            kind: NodeKind::Resolved { layer, operation },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Operation<N = LayerValue> {
    Control(PlayerId),
    Modify(Modification<N>),
    LandType(String),
    Inherit,
    CatalogTrigger(Box<crate::state::ability::Ability>),
}
// Intermediate fields that instructions can read or change. State-only reads
// (zones, counters, mana, printed mana value) cannot change during
// a layer pass. ABILITIES includes a generating static's continued existence.
const CONTROL: u16 = 256;
const NAME: u16 = 1;
const CARD_TYPES: u16 = 2;
const SUBTYPES: u16 = 4;
const SUPERTYPES: u16 = 8;
const COLORS: u16 = 16;
const KEYWORDS: u16 = 32;
const POWER: u16 = 64;
const ABILITIES: u16 = 128;

impl<N> Operation<N> {
    fn writes(&self) -> u16 {
        match self {
            Self::Control(_) => CONTROL,
            Self::LandType(_) => SUBTYPES | ABILITIES | KEYWORDS,
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
                Modification::LoseKeywords(keywords) if keywords.has(Keyword::Hexproof) => {
                    ABILITIES | KEYWORDS
                }
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
                _ => u16::MAX,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Instruction {
    target: PermanentId,
    operation: Operation<i32>,
}

fn layer<N>(modification: &Modification<N>) -> Option<Layer> {
    Some(match modification {
        Modification::Control => Layer::Control,
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
            reads_mana: false,
            reads_tapped: false,
            reads_life: false,
        };
        let mut nodes = Vec::new();
        let mut active_layers = 0u16;
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
                    && seed.object.controller == p.entry_controller()
            });
            if !reusable {
                seeds[slot] = Some(Arc::new(context.seed(id, p, source)));
            }
            let seed = seeds[slot].as_ref().unwrap();
            context.objects.insert(id, seed.object.clone());
            // A phased-out permanent's static abilities apply to nothing
            // (CR 702.26b); its own view still takes the records on it.
            if !game.is_phased_out(id) {
                for node in &seed.nodes {
                    context.push_node(&mut nodes, &mut active_layers, node.clone());
                }
            }
            for (kind, count) in p.counters.iter() {
                if let Some(keyword) = kind.keyword().filter(|_| count > 0) {
                    context.push_node(
                        &mut nodes,
                        &mut active_layers,
                        Node::resolved(
                            id,
                            vec![slot, usize::MAX - 1, kind as usize],
                            p.keyword_counter_timestamps
                                .get(&kind)
                                .copied()
                                .unwrap_or(p.timestamp),
                            Layer::Ability,
                            Operation::Modify(Modification::AddKeywords(Keywords::of(&[keyword]))),
                        ),
                    );
                }
            }
            for (i, record) in game.effects_on(id).enumerate() {
                let Some(layer) = layer(&record.modification) else {
                    continue;
                };
                let operation = if record.modification == Modification::HasAbilitiesOfSource {
                    Operation::Inherit
                } else if record.modification == Modification::Control {
                    Operation::Control(record.controller)
                } else {
                    Operation::Modify(record.modification.clone())
                };
                context.push_node(
                    &mut nodes,
                    &mut active_layers,
                    Node::resolved(
                        id,
                        vec![slot, usize::MAX, i],
                        record.timestamp,
                        layer,
                        operation,
                    ),
                );
            }
        }
        if !retain_sources {
            seeds.clear();
        }
        let mut pending = Vec::new();
        let mut applied = Vec::new();
        let mut granted = Vec::new();
        for layer in Layer::ALL {
            if active_layers & layer.bit() != 0 {
                applied.clear();
                loop {
                    pending.clear();
                    for (i, node) in nodes.iter().enumerate() {
                        if !applied.get(i).copied().unwrap_or(false)
                            && node.has_layer(layer)
                            && context.exists(node)
                        {
                            pending.push(i);
                        }
                    }
                    if pending.is_empty() {
                        break;
                    }
                    let instructions: Vec<_> = pending
                        .iter()
                        .map(|&i| context.instructions(&nodes[i], layer))
                        .collect();
                    let selected = context.next_effects(&nodes, &pending, &instructions, layer);
                    let all_applied = selected.len() == pending.len();
                    let mut discovered_this_layer = false;
                    for &slot in &selected {
                        let index = pending[slot];
                        if let NodeKind::Static { plan, targets } = &mut nodes[index].kind {
                            if targets.is_none()
                                && plan.layers & layer.after_mask() != 0
                                && !instructions[slot].is_empty()
                            {
                                // Instructions already resolved the affected set. Retain
                                // it only for effects that continue into a later layer.
                                let mut bound: Vec<_> = instructions[slot]
                                    .iter()
                                    .map(|instruction| instruction.target)
                                    .collect();
                                bound.dedup();
                                *targets = Some(bound);
                            }
                        }
                        context.apply(&nodes[index], &instructions[slot], Some(&mut granted));
                        for granted_node in granted.drain(..) {
                            let added =
                                context.push_node(&mut nodes, &mut active_layers, granted_node);
                            discovered_this_layer |= added & layer.bit() != 0;
                        }
                    }
                    if all_applied && !discovered_this_layer {
                        debug_assert!(nodes.iter().enumerate().all(|(i, node)| {
                            applied.get(i).copied().unwrap_or(false)
                                || pending.contains(&i)
                                || !node.has_layer(layer)
                                || !context.exists(node)
                        }));
                        break;
                    }
                    if applied.len() < nodes.len() {
                        applied.resize(nodes.len(), false);
                    }
                    for slot in selected {
                        applied[pending[slot]] = true;
                    }
                }
            } else {
                debug_assert!(!nodes.iter().any(|node| node.has_layer(layer)));
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
            if object.dirty != 0 {
                let record = &game.state[game.live_permanent(id).object];
                Self::finish_object(Arc::make_mut(object), record);
            }
            #[cfg(debug_assertions)]
            {
                let mut expected = (**object).clone();
                expected.dirty = DIRTY_ALL;
                let record = &game.state[game.live_permanent(id).object];
                Self::finish_object(&mut expected, record);
                debug_assert_eq!(
                    expected.view, object.view,
                    "incremental finish_object diverged from full recomputation"
                );
            }
        }
        context
    }

    /// The copiable object and bound static plans of one live incarnation.
    fn seed(&self, id: PermanentId, permanent: &Permanent, source: &Object) -> SourceSeed {
        #[cfg(feature = "benchmark")]
        crate::benchmark_layers::record(|m| m.source_plans += 1);
        let mut object = LayerObject {
            controller: permanent.entry_controller(),
            view: Arc::new(Characteristics {
                name: source.name.clone(),
                subtypes: source.subtypes.clone(),
                creature_types: source.creature_types(),
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
            dirty: DIRTY_ACTIVATIONS | DIRTY_STATICS,
        };
        Self::finish_object(&mut object, source);
        let nodes = object
            .statics
            .iter()
            .filter_map(|entry| Self::static_node(self.game, id, entry))
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
        let dirty = object.dirty;
        object.dirty = 0;
        let view = Arc::make_mut(&mut object.view);
        if dirty & DIRTY_TYPES != 0 {
            view.is_animated = view.is_creature() && !record.types.is_creature();
            // During the pass `CreatureTypes::All` marks a record granting all
            // creature types; a listed set is derived from the final subtypes.
            view.creature_types = if !view.types.has_creature_subtypes() {
                CreatureTypes::default()
            } else if view.creature_types.is_all() {
                CreatureTypes::All
            } else {
                CreatureTypes::among(&view.subtypes)
            };
        }
        if dirty & (DIRTY_TYPES | DIRTY_ACTIVATIONS) != 0 {
            Self::finish_activation_purposes(view);
        }
        if dirty & DIRTY_STATICS != 0 {
            view.static_abilities = object
                .statics
                .iter()
                .map(|entry| entry.ability.clone())
                .collect();
        }
    }

    fn finish_activation_purposes(view: &mut Characteristics) {
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
                    } if source.describes(&view.subtypes, view.creature_types.is_all()))
                })
                .cloned();
            let purpose = shared.unwrap_or_else(|| SpendingContext::Activate {
                zone: ZoneType::Battlefield,
                source: SpendingObject::new(
                    view.subtypes.iter().map(String::as_str),
                    view.creature_types.is_all(),
                ),
            });
            for ability in &mut view.activated_abilities {
                if ability.mana_cost.purpose != purpose {
                    ability.mana_cost.purpose = purpose.clone();
                }
            }
        }
    }
    fn push_node(&mut self, nodes: &mut Vec<Node>, active_layers: &mut u16, node: Node) -> u16 {
        let reads = node.reads();
        self.reads_mana |= reads.mana;
        self.reads_tapped |= reads.tapped;
        self.reads_life |= reads.life;
        let layers = node.layers();
        *active_layers |= layers;
        nodes.push(node);
        layers
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
            (!node.cda(), node.timestamp, &node.key)
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
            let reads = node.reads().fields
                | if node.is_static_unbound() {
                    ABILITIES
                } else {
                    0
                };
            // An unconditional attachment reads only its own controller
            // in layer 2. Independent Auras can apply in one timestamp pass.
            if layer == Layer::Control
                && node.is_unconditional_attached_or_self()
                && !instructions[b].iter().any(|i| i.target == node.source)
            {
                return false;
            }
            a != b && node.cda() == nodes[pending[b]].cda() && reads & writes[b] != 0
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
            after.apply(&nodes[index], &instructions[b], None);
            for (a, &other) in pending.iter().enumerate() {
                if a != b && nodes[other].cda() == nodes[index].cda() {
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
        match &node.kind {
            NodeKind::Resolved { .. }
            | NodeKind::Static {
                targets: Some(_), ..
            } => true,
            NodeKind::Static { targets: None, .. } => self
                .objects
                .get(&node.source)
                .is_some_and(|o| o.statics.iter().any(|s| s.key == node.key)),
        }
    }
    fn targets(
        &self,
        source: PermanentId,
        plan: &SourcePlan,
        bound: Option<&[PermanentId]>,
    ) -> Vec<PermanentId> {
        if let Some(targets) = bound {
            return targets.to_vec();
        }
        if plan
            .condition
            .as_ref()
            .is_some_and(|c| !self.condition(c, source))
        {
            return Vec::new();
        }
        match plan.scope {
            StaticScope::Attached => {
                return self.game.attached_permanent(source).into_iter().collect();
            }
            StaticScope::This => return vec![source],
            _ => {}
        }
        self.objects
            .keys()
            .copied()
            .filter(|&id| self.scope(&plan.scope, source, id))
            .collect()
    }
    fn instructions(&self, node: &Node, layer: Layer) -> Vec<Instruction> {
        let (plan, bound_targets) = match &node.kind {
            NodeKind::Resolved {
                layer: part_layer,
                operation,
            } => {
                return (*part_layer == layer)
                    .then(|| {
                        vec![Instruction {
                            target: node.source,
                            operation: operation.clone(),
                        }]
                    })
                    .unwrap_or_default();
            }
            NodeKind::Static { plan, targets } => (plan, targets.as_deref()),
        };
        if !self.exists(node) {
            return Vec::new();
        }
        let targets = self.targets(node.source, plan, bound_targets);
        if targets.is_empty() {
            return Vec::new();
        }
        let controller = self.controller(node.source);
        let mut out = Vec::with_capacity(targets.len());
        for id in targets {
            for (part, op) in &plan.parts {
                if *part != layer {
                    continue;
                }
                let operation = match op {
                    Operation::Modify(Modification::Control) => Operation::Control(controller),
                    Operation::Modify(m) => Operation::Modify(m.resolve(|n| {
                        self.layer_value(
                            n,
                            || self.game.live_permanent(id).object,
                            controller,
                            Some(node.source),
                            Some(id),
                        )
                    })),
                    Operation::Control(player) => Operation::Control(*player),
                    Operation::LandType(subtype) => Operation::LandType(subtype.clone()),
                    Operation::Inherit => Operation::Inherit,
                    Operation::CatalogTrigger(ability) => {
                        Operation::CatalogTrigger(ability.clone())
                    }
                };
                out.push(Instruction {
                    target: id,
                    operation,
                });
            }
        }
        out
    }
    fn controlled(
        &self,
        controller: PlayerId,
    ) -> impl Iterator<Item = (PermanentId, &LayerObject)> {
        self.objects
            .iter()
            .filter(move |(&id, object)| {
                object.controller == controller && !self.game.is_phased_out(id)
            })
            .map(|(&id, object)| (id, object.as_ref()))
    }
    fn controller(&self, id: PermanentId) -> PlayerId {
        self.objects[&id].controller
    }
    fn layer_value(
        &self,
        value: &LayerValue,
        object: impl FnOnce() -> ObjectId,
        controller: PlayerId,
        source: Option<PermanentId>,
        affected: Option<PermanentId>,
    ) -> i32 {
        match value {
            LayerValue::Fixed(n) => *n,
            LayerValue::ManaValue => i32::from(self.game.object_mana_value(object())),
            LayerValue::GraveyardCardTypes { offset } => self.game.graveyard_card_types() + offset,
            LayerValue::Count(count) => self.count(count, controller, source, affected),
        }
    }
    fn count(
        &self,
        count: &LayerCount,
        controller: PlayerId,
        source: Option<PermanentId>,
        affected: Option<PermanentId>,
    ) -> i32 {
        let controlled = |p| {
            self.controlled(controller)
                .filter(move |(id, _)| self.matches(*id, p, source))
        };
        match count {
            LayerCount::CreatureTypes { max } => {
                let Some(id) = affected else {
                    return 0;
                };
                let view = &self.objects[&id].view;
                if view.has_all_creature_types() {
                    *max
                } else {
                    (view.subtypes.iter().filter(|s| is_creature_type(s)).count() as i32).min(*max)
                }
            }
            LayerCount::ColorsAmong(p) => controlled(p)
                .flat_map(|(_, o)| o.view.colors.iter().copied())
                .collect::<BTreeSet<_>>()
                .len() as i32,
            LayerCount::Graveyard { who, predicate } => (0..self.game.state.players.len())
                .map(PlayerId)
                .filter(|player| who.includes(controller, *player))
                .map(|player| self.graveyard_count(player, predicate))
                .sum(),
            LayerCount::YouControl(p) => controlled(p).count() as i32,
            LayerCount::SourceCounters(kind) => source.map_or(0, |source| {
                self.game.live_permanent(source).counter_count(*kind)
            }),
            LayerCount::CountersOn { kind, predicate } => controlled(predicate)
                .map(|(id, _)| self.game.live_permanent(id).counter_count(*kind))
                .sum(),
        }
    }
    fn creature_count(&self, controller: PlayerId) -> usize {
        self.objects
            .iter()
            .filter(|(id, object)| self.controller(**id) == controller && object.view.is_creature())
            .count()
    }
    fn graveyard_count(&self, owner: PlayerId, predicate: &CardPredicate) -> i32 {
        self.game
            .state
            .zone_cards(ZoneType::Graveyard, owner)
            .filter(|card| {
                let object = self.game.state.object_of(*card);
                self.game.off_battlefield_matches(object, predicate, || {
                    let card = &self.game.state[object];
                    card.pt_cda().and_then(|(power, _)| power).map_or_else(
                        || card.power.unwrap_or(0),
                        |value| self.layer_value(value, || object, card.owner, None, None),
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
        let controlled = || self.controlled(controller).map(|(id, _)| id);
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
    fn scope(&self, scope: &StaticScope, source: PermanentId, target: PermanentId) -> bool {
        matches_static_scope(
            scope,
            source,
            self.controller(source),
            self.game.attached_permanent(source),
            target,
            self.controller(target),
            |predicate| self.matches(target, predicate, Some(source)),
        )
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
            self.game.chosen_type_for(predicate, source),
        )
    }
    fn static_node(game: &Game, source: PermanentId, entry: &StaticEntry) -> Option<Node> {
        let mut plan = SourcePlan::new();
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
                let object = &game.state[game.live_permanent(source).object];
                let index = object
                    .attached_grants()
                    .position(|a| a == ability)
                    .unwrap_or(0);
                let mut granted = ability.clone();
                granted.granted_by = game
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
            kind: NodeKind::Static {
                plan: Arc::new(plan.summarize()),
                targets: None,
            },
        })
    }
    fn apply(
        &mut self,
        node: &Node,
        instructions: &[Instruction],
        mut granted: Option<&mut Vec<Node>>,
    ) {
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
            if let Operation::Control(controller) = instruction.operation {
                object.controller = controller;
                continue;
            }
            let view = Arc::make_mut(&mut object.view);
            match &instruction.operation {
                Operation::LandType(subtype) => {
                    if !view.is_land() {
                        continue;
                    }
                    object.dirty |= DIRTY_ALL;
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
                        if !definition.activated_abilities.is_empty() {
                            object.dirty |= DIRTY_ACTIVATIONS;
                        }
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
                    Modification::SetName(name) => {
                        view.name = name.clone();
                        view.name_set = true;
                    }
                    Modification::AllCreatureTypes => {
                        // A later grant of every type overrides an earlier set list.
                        object.dirty |= DIRTY_TYPES;
                        view.creature_types = CreatureTypes::All;
                        view.creature_types_set = false;
                    }
                    Modification::SetCreatureTypes(types) => {
                        object.dirty |= DIRTY_TYPES;
                        view.creature_types = CreatureTypes::default();
                        view.creature_types_set = true;
                        view.subtypes.retain(|s| !is_creature_type(s));
                        view.subtypes.extend(types.iter().cloned());
                        view.added_subtypes.retain(|s| !is_creature_type(s));
                    }
                    Modification::AddSubtypes(types) => {
                        object.dirty |= DIRTY_TYPES;
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
                    Modification::AddTypes(types) => {
                        object.dirty |= DIRTY_TYPES;
                        view.types.types.extend(types.iter().copied());
                    }
                    Modification::SetTypes { types, subtypes } => {
                        object.dirty |= DIRTY_TYPES;
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
                        if !(types.contains(&CardType::Creature)
                            || types.contains(&CardType::Kindred))
                            || sets_creature_types
                        {
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
                    }
                    Modification::BecomeCreature => {
                        object.dirty |= DIRTY_TYPES;
                        view.types.types.insert(CardType::Creature);
                    }
                    Modification::Nonlegendary => view.is_legendary = false,
                    Modification::SetColors(colors) => view.colors = colors.clone(),
                    Modification::AddColors(colors) => view.colors.extend(colors),
                    Modification::AddKeywords(keywords) => {
                        view.keywords = view.keywords.union(*keywords)
                    }
                    Modification::LoseKeywords(keywords) => {
                        view.keywords = view.keywords.without(*keywords);
                        if keywords.has(Keyword::Hexproof) {
                            let before = object.statics.len();
                            object.statics.retain(|entry| {
                                !matches!(entry.ability, StaticAbility::HexproofFrom(_))
                            });
                            if object.statics.len() != before {
                                object.dirty |= DIRTY_STATICS;
                            }
                        }
                    }
                    Modification::LoseNonManaAbilities => {
                        if !object.statics.is_empty() {
                            object.dirty |= DIRTY_STATICS;
                        }
                        lose_nonmana_abilities(view, &mut object.statics)
                    }
                    Modification::LoseAllAbilities => {
                        if !object.statics.is_empty() {
                            object.dirty |= DIRTY_STATICS;
                        }
                        lose_all_abilities(view, &mut object.statics)
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
                        object.dirty |= DIRTY_ACTIVATIONS;
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
                        object.dirty |= DIRTY_STATICS;
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
                        let entry = StaticEntry {
                            key,
                            timestamp: node.timestamp.max(self.game.live_permanent(id).timestamp),
                            ability: (**ability).clone(),
                        };
                        if let Some(granted) = granted.as_mut() {
                            if !self.game.is_phased_out(id) {
                                if let Some(granted_node) = Self::static_node(self.game, id, &entry)
                                {
                                    granted.push(granted_node);
                                }
                            }
                        }
                        object.statics.push(entry);
                    }
                    Modification::SetBasePt { power, toughness } => {
                        if let Some(p) = power {
                            view.power = *p;
                        }
                        if let Some(t) = toughness {
                            view.toughness = *t;
                        }
                    }
                    Modification::ModifyPt { power, toughness } => {
                        view.power += power;
                        view.toughness += toughness;
                    }
                    Modification::SwitchPt => std::mem::swap(&mut view.power, &mut view.toughness),
                    _ => unreachable!("non-layer modification in layer instruction"),
                },
                Operation::Control(_) => unreachable!("control handled above"),
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

/// A live permanent under `view` (an intermediate layer context or its
/// final view).
pub(super) fn matches_view(
    game: &Game,
    id: PermanentId,
    p: &CardPredicate,
    view: &Characteristics,
    source_name: Option<&str>,
    chosen_type: Option<&str>,
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
    .matches(p, source_name, chosen_type)
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
    .matches(p, None, None)
}

impl CharacteristicView<'_> {
    fn has_subtype(&self, subtype: &str) -> bool {
        self.subtypes.iter().any(|s| s == subtype)
            || (self.all_creature_types && is_creature_type(subtype))
    }

    fn matches(
        &self,
        p: &CardPredicate,
        source_name: Option<&str>,
        chosen_type: Option<&str>,
    ) -> bool {
        // A permanent is on the battlefield, never a graveyard card.
        !p.from_battlefield_this_turn
            && p.accepts_name(self.name)
            && (!p.namesake || source_name == Some(self.name))
            && (!p.chosen_type_of_source || chosen_type.is_some_and(|name| self.has_subtype(name)))
            && (!p.deathtouch || self.keywords.has(Keyword::Deathtouch))
            && (!p.flying || self.keywords.has(Keyword::Flying))
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
                    .any(|alternative| self.matches(alternative, source_name, chosen_type)))
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
