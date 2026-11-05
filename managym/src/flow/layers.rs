//! CR 613 evaluation over a single explicit battlefield context. Effects inspect
//! instructions, not final values, when determining dependencies. A context is
//! private derived data; it never calls `Game::characteristics`.

use crate::{
    flow::{game::Game, statics::Characteristics},
    state::{
        ability::{Condition, Effect, EffectiveTrigger},
        card::{
            AbilityGrant, ActivatedAbilityDefinition, CardType, Keyword, Keywords, LayerValue,
            PowerCda, StaticAbility, StaticScope,
        },
        effect::Modification,
        game_object::{ObjectId, PermanentId, PlayerId},
        mana::Mana,
        permanent::CounterKind,
        predicate::CardPredicate,
        zone::ZoneType,
    },
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) const BASIC_LANDS: [(&str, &str); 5] = [
    ("Plains", "W"),
    ("Island", "U"),
    ("Swamp", "B"),
    ("Mountain", "R"),
    ("Forest", "G"),
];
const NONCREATURE: &[&str] = &[
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
    "Saga",
    "Aura",
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
    timestamp: ObjectId,
    ability: StaticAbility,
}
#[derive(Clone, Debug, PartialEq)]
pub(super) struct LayerObject {
    pub view: Characteristics,
    statics: Vec<StaticEntry>,
}
#[derive(Clone)]
pub(super) struct Context<'a> {
    game: &'a Game,
    pub objects: BTreeMap<PermanentId, LayerObject>,
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
    timestamp: ObjectId,
    cda: bool,
    scope: StaticScope,
    condition: Option<Condition>,
    parts: Vec<(Layer, Operation)>,
    targets: Option<Vec<PermanentId>>,
}
impl Node {
    /// Dependencies inspect the same scope/condition/value predicates used to
    /// resolve instructions. Recurse through alternatives once at this owner.
    fn reads_predicate(&self, reads: fn(&CardPredicate) -> bool) -> bool {
        fn predicate_reads(p: &CardPredicate, reads: fn(&CardPredicate) -> bool) -> bool {
            reads(p) || p.any_of.iter().any(|p| predicate_reads(p, reads))
        }
        fn condition_reads(c: &Condition, reads: fn(&CardPredicate) -> bool) -> bool {
            match c {
                Condition::ControlsAtLeast { predicate, .. }
                | Condition::SourceMatches(predicate) => predicate_reads(predicate, reads),
                Condition::AnyOf(conditions) => {
                    conditions.iter().any(|c| condition_reads(c, reads))
                }
                _ => false,
            }
        }
        let scope_reads = match &self.scope {
            StaticScope::All(p)
            | StaticScope::Other(p)
            | StaticScope::OtherYouControl(p)
            | StaticScope::YouControl(p)
            | StaticScope::SpellsYouControl(p) => predicate_reads(p, reads),
            StaticScope::This | StaticScope::Attached => false,
        };
        scope_reads
            || self
                .condition
                .as_ref()
                .is_some_and(|c| condition_reads(c, reads))
            || self.parts.iter().any(|(_, op)| match op {
                Operation::Buff(buff) => {
                    buff.per_color_among
                        .as_ref()
                        .is_some_and(|p| predicate_reads(p, reads))
                        || buff
                            .per_graveyard_matching
                            .as_ref()
                            .is_some_and(|p| predicate_reads(p, reads))
                }
                Operation::Power(cda) => match cda.as_ref() {
                    PowerCda::GraveyardMatching(p) => predicate_reads(p, reads),
                    _ => false,
                },
                _ => false,
            })
    }

    fn reads_tapped(&self) -> bool {
        fn condition_reads(c: &Condition) -> bool {
            match c {
                Condition::SourceTapped => true,
                Condition::AnyOf(conditions) => conditions.iter().any(condition_reads),
                _ => false,
            }
        }
        self.reads_predicate(|p| p.tapped) || self.condition.as_ref().is_some_and(condition_reads)
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
    Buff(Box<crate::state::card::StaticPtBuff>),
    Inherit,
    CatalogTrigger(Box<crate::state::ability::Ability>),
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
    pub fn evaluate(game: &'a Game) -> Self {
        #[cfg(feature = "benchmark")]
        let _timer = crate::benchmark_layers::Timer::new(|m, ns| {
            m.boards += 1;
            m.board_ns += ns;
        });
        let mut context = Self {
            game,
            objects: BTreeMap::new(),
            reads_mana: false,
            reads_tapped: false,
        };
        let mut nodes = Vec::new();
        for (slot, permanent) in game.state.permanents.iter().enumerate() {
            let Some(p) = permanent else { continue };
            let id = PermanentId(slot);
            let card = &game.state.cards[p.card];
            context.objects.insert(
                id,
                LayerObject {
                    view: Characteristics {
                        name: card.name.clone(),
                        subtypes: card.subtypes.clone(),
                        types: card.types.clone(),
                        colors: card.colors(),
                        mana_abilities: card.mana_abilities.clone(),
                        triggered_mana_abilities: card.triggered_mana_abilities.clone(),
                        is_legendary: card.is_legendary(),
                        keywords: card.keywords,
                        ward: card.ward().cloned(),
                        power: card.power.unwrap_or(0),
                        toughness: card.toughness.unwrap_or(0),
                        activated_abilities: card.activated_abilities.clone(),
                        triggered_abilities: card
                            .abilities
                            .iter()
                            .cloned()
                            .map(|ability| EffectiveTrigger {
                                ability,
                                program: false,
                            })
                            .collect(),
                        ..Default::default()
                    },
                    statics: card
                        .statics
                        .iter()
                        .enumerate()
                        .map(|(i, ability)| StaticEntry {
                            key: vec![slot, i],
                            timestamp: p.timestamp,
                            ability: ability.clone(),
                        })
                        .collect(),
                },
            );
            for (kind, count) in p.counters.iter() {
                if let Some(keyword) = kind.keyword().filter(|_| count > 0) {
                    nodes.push(Node {
                        source: id,
                        key: vec![slot, usize::MAX - 1, kind as usize],
                        timestamp: p
                            .keyword_counter_timestamps
                            .get(&kind)
                            .copied()
                            .unwrap_or(p.timestamp),
                        cda: false,
                        scope: StaticScope::This,
                        condition: None,
                        targets: Some(vec![id]),
                        parts: vec![(
                            Layer::Ability,
                            Operation::Modify(Modification::AddKeywords(Keywords::of(&[keyword]))),
                        )],
                    });
                }
            }
            for (i, record) in game.effects_on(id).enumerate() {
                let Some(l) = layer(&record.modification) else {
                    continue;
                };
                nodes.push(Node {
                    source: id,
                    key: vec![slot, usize::MAX, i],
                    timestamp: record.timestamp,
                    cda: false,
                    scope: StaticScope::This,
                    condition: None,
                    targets: Some(vec![id]),
                    parts: vec![(
                        l,
                        if record.modification == Modification::HasAbilitiesOfSource {
                            Operation::Inherit
                        } else {
                            Operation::Modify(
                                record.modification.map_amount(|n| LayerValue::Fixed(*n)),
                            )
                        },
                    )],
                });
            }
        }
        let mut known = BTreeSet::new();
        context.discover(&mut nodes, &mut known);
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
                    let count = game.state.permanents[id]
                        .as_ref()
                        .unwrap()
                        .counter_count(CounterKind::PlusOnePlusOne);
                    object.view.power += count;
                    object.view.toughness += count;
                }
            }
        }
        for (&id, object) in &mut context.objects {
            let card = &game.state.cards[game.state.permanents[id].as_ref().unwrap().card];
            object.view.is_animated = object.view.is_creature() && !card.types.is_creature();
            object.view.creature_types = if object.view.is_creature() {
                object
                    .view
                    .subtypes
                    .iter()
                    .filter(|s| !NONCREATURE.contains(&s.as_str()))
                    .cloned()
                    .collect()
            } else {
                Vec::new()
            };
            object.view.creature_types.sort();
            object.view.creature_types.dedup();
            object.view.static_abilities = object
                .statics
                .iter()
                .map(|entry| entry.ability.clone())
                .collect();
        }
        context
    }
    fn discover(&mut self, nodes: &mut Vec<Node>, known: &mut BTreeSet<Vec<usize>>) {
        // Occurrence provenance preserves duplicate grants, including statics
        // discovered during layer 6 that start applying in a later layer.
        for (&id, object) in &self.objects {
            for entry in &object.statics {
                if known.insert(entry.key.clone()) {
                    if let Some(node) = self.node(id, entry) {
                        self.reads_mana |= node
                            .condition
                            .as_ref()
                            .is_some_and(Condition::reads_mana_pools);
                        self.reads_tapped |= node.reads_tapped();
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
        let timestamp_order = |&a: &usize| {
            let node = &nodes[pending[a]];
            (!node.cda, node.timestamp, &node.key)
        };
        let pt_independent = layer >= Layer::CharacteristicPower
            && pending
                .iter()
                .all(|&i| !nodes[i].reads_predicate(CardPredicate::reads_power));
        // These grants cannot remove generating statics or change instructions.
        // Flying is the only keyword that a current predicate can inspect.
        // Loss and static grants retain full dependency evaluation.
        let ability_independent = layer == Layer::Ability
            && pending
                .iter()
                .all(|&i| !nodes[i].reads_predicate(|p| p.flying))
            && instructions.iter().flatten().all(|i| {
                matches!(
                    i.operation,
                    Operation::CatalogTrigger(_)
                        | Operation::Inherit
                        | Operation::Modify(
                            Modification::AddKeywords(_)
                                | Modification::LoseKeywords(_)
                                | Modification::Firebending(_)
                                | Modification::GrantActivated(_)
                                | Modification::GrantTriggered(_)
                        )
                )
            });
        let independent = pt_independent || ability_independent;
        let ordered = || {
            let mut order: Vec<_> = (0..pending.len()).collect();
            order.sort_by_key(timestamp_order);
            order
        };
        if pending.len() == 1 || (independent && !cfg!(debug_assertions)) {
            return ordered();
        }
        let mut dependencies = vec![vec![false; pending.len()]; pending.len()];
        #[cfg(feature = "benchmark")]
        let _timer = crate::benchmark_layers::Timer::new(|m, ns| {
            m.dependency_passes += 1;
            m.dependency_ns += ns;
        });
        for (b, &index) in pending.iter().enumerate() {
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
                    dependencies[a][b] = self.exists(&nodes[other]) != after.exists(&nodes[other])
                        || instructions[a] != after.instructions(&nodes[other], layer);
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
                                power: Some(LayerValue::Fixed(self.cda(cda, node.source))),
                                toughness: None,
                            }),
                            Operation::Buff(buff) => {
                                let object = &self.objects[&id].view;
                                let per_type = buff.per_creature_type_max.map_or(1, |max| {
                                    (object
                                        .subtypes
                                        .iter()
                                        .filter(|s| !NONCREATURE.contains(&s.as_str()))
                                        .count() as i32)
                                        .min(max)
                                });
                                let controller = self.game.state.permanents[node.source]
                                    .as_ref()
                                    .unwrap()
                                    .controller;
                                let per_color = buff.per_color_among.as_ref().map_or(1, |p| {
                                    self.objects
                                        .iter()
                                        .filter(|(id, _)| {
                                            self.controller(**id) == controller
                                                && self.matches(**id, p, None)
                                        })
                                        .flat_map(|(_, o)| o.view.colors.iter().copied())
                                        .collect::<BTreeSet<_>>()
                                        .len() as i32
                                });
                                let per_card =
                                    buff.per_graveyard_matching.as_ref().map_or(1, |p| {
                                        self.game.count_graveyard_matching(controller, p) as i32
                                    });
                                let times = per_type * per_color * per_card;
                                Operation::Modify(Modification::ModifyPt {
                                    power: LayerValue::Fixed(buff.power * times),
                                    toughness: LayerValue::Fixed(buff.toughness * times),
                                })
                            }
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
    fn controller(&self, id: PermanentId) -> PlayerId {
        self.game.state.permanents[id].as_ref().unwrap().controller
    }
    fn value(&self, value: &LayerValue, id: PermanentId) -> i32 {
        match value {
            LayerValue::Fixed(n) => *n,
            LayerValue::ManaValue => self.game.state.cards
                [self.game.state.permanents[id].as_ref().unwrap().card]
                .mana_value_cost()
                .map_or(0, |m| i32::from(m.mana_value)),
            LayerValue::GraveyardCardTypes { offset } => {
                [PlayerId(0), PlayerId(1)]
                    .into_iter()
                    .flat_map(|p| self.game.state.zones.zone_cards(ZoneType::Graveyard, p))
                    .flat_map(|id| self.game.state.cards[*id].types.types.iter())
                    .collect::<BTreeSet<_>>()
                    .len() as i32
                    + offset
            }
        }
    }
    fn cda(&self, cda: &PowerCda, source: PermanentId) -> i32 {
        let controller = self.controller(source);
        match cda {
            PowerCda::CountersOnLandsYouControl => self
                .objects
                .iter()
                .filter(|(id, o)| self.controller(**id) == controller && o.view.is_land())
                .map(|(id, _)| {
                    self.game.state.permanents[*id]
                        .as_ref()
                        .unwrap()
                        .counter_count(CounterKind::PlusOnePlusOne)
                })
                .sum(),
            PowerCda::CreaturesYouControl => self
                .objects
                .iter()
                .filter(|(id, o)| self.controller(**id) == controller && o.view.is_creature())
                .count() as i32,
            PowerCda::GraveyardMatching(p) => {
                self.game.count_graveyard_matching(controller, p) as i32
            }
        }
    }
    fn condition(&self, condition: &Condition, source: PermanentId) -> bool {
        match condition {
            Condition::ControlsAtLeast { count, predicate } => {
                self.objects
                    .keys()
                    .filter(|&&id| {
                        self.controller(id) == self.controller(source)
                            && self.matches(id, predicate, Some(source))
                    })
                    .count()
                    >= *count
            }
            Condition::SourceMatches(predicate) => self.matches(source, predicate, Some(source)),
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
        let mut node = Node {
            source,
            key: entry.key.clone(),
            timestamp: entry.timestamp,
            cda: false,
            scope: StaticScope::This,
            condition: None,
            parts: Vec::new(),
            targets: None,
        };
        let mut add = |l, m| node.parts.push((l, Operation::Modify(m)));
        match &entry.ability {
            StaticAbility::Continuous(effect) => {
                node.scope = effect.scope.clone();
                node.condition = effect.condition.clone();
                node.cda = effect.characteristic_defining;
                for m in &effect.modifications {
                    if let Some(l) = layer(m) {
                        let layer = if node.cda && l == Layer::BasePower {
                            Layer::CharacteristicPower
                        } else {
                            l
                        };
                        add(layer, m.clone());
                    }
                }
            }
            StaticAbility::TypeGrant(g) => {
                node.scope = g.scope.clone();
                node.condition = g.condition.clone();
                add(Layer::Type, Modification::AddTypes(g.types.clone()));
                add(Layer::Type, Modification::AddSubtypes(g.subtypes.clone()));
            }
            StaticAbility::KeywordGrant(g) => {
                node.scope = g.scope.clone();
                node.condition = g.condition.clone();
                add(Layer::Ability, Modification::AddKeywords(g.keywords));
            }
            StaticAbility::PtBuff(b) => {
                node.scope = b.scope.clone();
                node.condition = b.condition.clone();
                node.parts
                    .push((Layer::ModifyPower, Operation::Buff(Box::new(b.clone()))));
            }
            StaticAbility::CharacteristicPower(cda) => {
                node.cda = true;
                node.parts.push((
                    Layer::CharacteristicPower,
                    Operation::Power(Box::new(cda.clone())),
                ));
            }
            StaticAbility::SetsLandType(set) => {
                node.scope = StaticScope::All(set.predicate.clone());
                node.condition = Some(set.condition.clone());
                node.parts
                    .push((Layer::Type, Operation::LandType(set.subtype.clone())));
            }
            StaticAbility::AttachedSetsCharacteristics(set) => {
                node.scope = StaticScope::Attached;
                node.parts = vec![
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
                node.scope = StaticScope::Attached;
                let card = &self.game.state.cards
                    [self.game.state.permanents[source].as_ref().unwrap().card];
                let index = card
                    .attached_grants()
                    .position(|a| a == ability)
                    .unwrap_or(0);
                let mut granted = ability.clone();
                granted.granted_by = self
                    .game
                    .permanent_object_ref(source)
                    .map(|grantor| AbilityGrant { grantor, index });
                add(
                    Layer::Ability,
                    Modification::GrantActivated(Box::new(granted)),
                );
            }
            StaticAbility::TriggeredGrant(grant) => {
                node.scope = grant.scope.clone();
                node.parts.push((
                    Layer::Ability,
                    Operation::CatalogTrigger(grant.ability.clone()),
                ));
            }
            StaticAbility::GrantsAnyColorManaAbility(predicate) => {
                node.scope = StaticScope::YouControl(predicate.clone());
                add(
                    Layer::Ability,
                    Modification::GrantActivated(Box::new(ActivatedAbilityDefinition {
                        tap_source: true,
                        effects: vec![Effect::any_color_mana()],
                        ..Default::default()
                    })),
                );
            }
            _ => return None,
        }
        Some(node)
    }
    fn apply(&mut self, node: &Node, instructions: &[Instruction]) {
        for (ordinal, instruction) in instructions.iter().enumerate() {
            let id = instruction.target;
            let object = self.objects.get_mut(&id).unwrap();
            let view = &mut object.view;
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
                    view.abilities_lost = true;
                    view.keywords = Keywords::default();
                    view.ward = None;
                    view.mana_abilities.clear();
                    view.triggered_mana_abilities.clear();
                    view.activated_abilities.clear();
                    view.triggered_abilities.clear();
                    object.statics.clear();
                    if let Some((_, mana)) = BASIC_LANDS.iter().find(|(name, _)| *name == subtype) {
                        view.activated_abilities.push(mana_ability(mana));
                    }
                }
                Operation::Inherit => {
                    if let Some(definition) = self.game.inherited_definition(id) {
                        view.activated_abilities
                            .extend(definition.activated_abilities.clone());
                        view.triggered_abilities
                            .extend(definition.abilities.iter().cloned().map(|ability| {
                                EffectiveTrigger {
                                    ability,
                                    program: false,
                                }
                            }));
                    }
                }
                Operation::CatalogTrigger(ability) => {
                    view.triggered_abilities.push(EffectiveTrigger {
                        ability: (**ability).clone(),
                        program: false,
                    })
                }
                Operation::Modify(modification) => match modification {
                    Modification::SetCreatureTypes(types) => {
                        view.subtypes.retain(|s| NONCREATURE.contains(&s.as_str()));
                        view.subtypes.extend(types.iter().cloned());
                        view.added_subtypes
                            .retain(|s| NONCREATURE.contains(&s.as_str()));
                        view.added_subtypes.extend(types.iter().cloned());
                    }
                    Modification::AddSubtypes(types) => {
                        for subtype in types {
                            if !view.subtypes.contains(subtype) {
                                view.subtypes.push(subtype.clone());
                                view.added_subtypes.push(subtype.clone());
                                if view.is_land() {
                                    if let Some((_, mana)) =
                                        BASIC_LANDS.iter().find(|(name, _)| *name == subtype)
                                    {
                                        view.activated_abilities.push(mana_ability(mana));
                                    }
                                }
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
                        view.abilities_lost = true;
                        view.keywords = Keywords::default();
                        view.ward = None;
                        view.firebending = 0;
                        view.activated_abilities
                            .retain(|ability| ability.mana_choices().is_some());
                        view.triggered_abilities.clear();
                        object.statics.clear();
                    }
                    Modification::LoseAllAbilities => {
                        view.abilities_lost = true;
                        view.keywords = Keywords::default();
                        view.ward = None;
                        view.firebending = 0;
                        view.mana_abilities.clear();
                        view.triggered_mana_abilities.clear();
                        view.activated_abilities.clear();
                        view.triggered_abilities.clear();
                        object.statics.clear();
                    }
                    Modification::Firebending(n) => {
                        view.firebending = view.firebending.saturating_add(*n)
                    }
                    Modification::GrantActivated(ability) => {
                        view.activated_abilities.push((**ability).clone())
                    }
                    Modification::GrantTriggered(ability) => {
                        view.triggered_abilities.push(EffectiveTrigger {
                            ability: (**ability).clone(),
                            program: true,
                        })
                    }
                    Modification::GrantStatic(ability) => {
                        if let StaticAbility::Ward(cost) = ability.as_ref() {
                            view.ward = Some(cost.clone());
                            view.triggered_abilities.push(EffectiveTrigger {
                                ability: crate::state::ability::Ability::ward(cost.clone()),
                                program: true,
                            });
                        }
                        let mut key = node.key.clone();
                        key.extend([id.0, ordinal]);
                        object.statics.push(StaticEntry {
                            key,
                            timestamp: node
                                .timestamp
                                .max(self.game.state.permanents[id].as_ref().unwrap().timestamp),
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
fn mana_ability(mana: &str) -> ActivatedAbilityDefinition {
    ActivatedAbilityDefinition {
        tap_source: true,
        effects: vec![Effect::add_mana(Mana::parse(mana))],
        ..Default::default()
    }
}

/// Shared predicate semantics for intermediate contexts and the final view.
/// Printed identity contributes only mana value, token status and supertypes.
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
    let card = &game.state.cards[permanent.card];
    let mana_value = card.mana_value_cost().map_or(0, |m| m.mana_value);
    (!p.namesake || source_name == Some(view.name.as_str()))
        && (!p.flying || view.keywords.has(Keyword::Flying))
        && (!p.no_counters || !permanent.has_counters())
        && (!p.in_combat || game.permanent_in_combat(id))
        && (!p.with_counter || permanent.counter_count(CounterKind::PlusOnePlusOne) > 0)
        && (!p.attacking || permanent.attacking)
        && (!p.tapped || permanent.tapped)
        && p.token.is_none_or(|token| token == card.is_token)
        && (p.any_of.is_empty()
            || p.any_of
                .iter()
                .any(|alternative| matches_view(game, id, alternative, view, source_name)))
        && p.supertype.as_ref().is_none_or(|s| {
            if s.eq_ignore_ascii_case("legendary") {
                view.is_legendary
            } else {
                card.supertypes.iter().any(|t| t.eq_ignore_ascii_case(s))
            }
        })
        && (!p.nonbasic
            || !card
                .supertypes
                .iter()
                .any(|s| s.eq_ignore_ascii_case("basic")))
        && p.card_type.is_none_or(|t| view.types.types.contains(&t))
        && (p.card_types_any.is_empty()
            || p.card_types_any
                .iter()
                .any(|t| view.types.types.contains(t)))
        && p.not_card_types
            .iter()
            .all(|t| !view.types.types.contains(t))
        && p.subtype.as_ref().is_none_or(|s| view.subtypes.contains(s))
        && (p.subtypes_any.is_empty() || p.subtypes_any.iter().any(|s| view.subtypes.contains(s)))
        && p.not_subtypes.iter().all(|s| !view.subtypes.contains(s))
        && p.min_power.is_none_or(|min| view.power >= min)
        && p.max_power.is_none_or(|max| view.power <= max)
        && p.min_mana_value.is_none_or(|min| mana_value >= min)
        && p.max_mana_value.is_none_or(|max| mana_value <= max)
}
