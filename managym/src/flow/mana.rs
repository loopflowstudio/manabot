// mana.rs
// One payment transition model for offers, automatic production and nested
// mana activations. PaymentModel answers a question two ways with one result:
// directly, by arithmetic over the pool and the fixed outputs, when every
// source in play for the question taps itself for one fixed output and the
// obligations are one mana cost with an optional split that taps may pay
// (`PaymentModel::direct`); otherwise by searching identity-preserving
// resource steps. The witness search prunes with optimistic whole-payment and
// color-set bounds and memoizes feasibility across interchangeable sources
// (PaymentSearch), always returning the first concrete path in source order.
// The search defines a legal payment; debug builds run it beside every direct
// answer and assert equal affordability and an equal witness.
// Game owns their events and explicit Decision continuations, and shares one
// player's sources across the questions of a `PaymentScan`.
// Source lists are immutable shared values; reservations detach on writes.
// Pool and spend-purpose restrictions belong to state/mana.rs. Game also retains
// bounded exact queries (PaymentCache); debug builds recompute every hit.

mod autotapper;
mod help;
mod interchangeable;
mod preview;
pub use preview::PaymentPreview;

use crate::state::player::{ManaFloats, ManaPayment};
use std::{cell::RefCell, collections::VecDeque, num::NonZeroUsize, sync::Arc};

use crate::{
    agent::action::{Action, ActionSpace, AgentError},
    flow::{
        decision::{
            Decision, EffectFrame, ManaProduction, PendingManaBonus, SpendCompletion,
            SuspendedResolution, TappedForMana,
        },
        game::{ActivationSource, Game, PendingChoice},
        payment::{CostComponent, CostPermissionContext, Help, Operation, Payment},
        proposed_event::{CounterProvenance, EventCause, ProposedEvent},
        statics::Characteristics,
        turn::StepKind,
    },
    state::{
        ability::{Effect, EffectValue},
        card::{
            ActivatedAbilityDefinition, ActivationZone, ReplacementEffect, TriggeredManaAbility,
            TriggeredManaOutput,
        },
        game_object::{ObjectId, PermanentId, PlayerId},
        mana::{
            Color, EmptyDisposition, Mana, ManaClass, ManaCost, ManaExpiry, ManaPool,
            ManaRestriction, ManaSpendChoice, ManaTypeSet, SpellSnapshot, SpendOrder,
            SpendingContext,
        },
        payment::{How, ManaOrigin, PaymentReceipt, SpentMana},
        zone::ZoneType,
    },
};

/// Fast deterministic non-cryptographic hasher for internal per-plan search
/// tables (`PaymentSearch::visited`, `PurposeSupply::allows`,
/// `PurposeProjection`), avoiding SipHash-1-3 compression overhead.
#[derive(Default)]
struct FastHasher(u64);

impl FastHasher {
    const K: u64 = 0x517c_c1b7_2722_0a95;

    #[inline]
    fn add_word(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(Self::K);
    }
}

impl std::hash::Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        while bytes.len() >= 8 {
            let word = u64::from_ne_bytes(bytes[..8].try_into().unwrap());
            self.add_word(word);
            bytes = &bytes[8..];
        }
        if bytes.len() >= 4 {
            let word = u64::from(u32::from_ne_bytes(bytes[..4].try_into().unwrap()));
            self.add_word(word);
            bytes = &bytes[4..];
        }
        if bytes.len() >= 2 {
            let word = u64::from(u16::from_ne_bytes(bytes[..2].try_into().unwrap()));
            self.add_word(word);
            bytes = &bytes[2..];
        }
        if let Some(&byte) = bytes.first() {
            self.add_word(u64::from(byte));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add_word(u64::from(i));
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add_word(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add_word(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add_word(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add_word(i as u64);
    }
}

type FastBuildHasher = std::hash::BuildHasherDefault<FastHasher>;
type FastHashSet<T> = std::collections::HashSet<T, FastBuildHasher>;
type FastHashMap<K, V> = std::collections::HashMap<K, V, FastBuildHasher>;

/// Bitset of which reachable `SpendingContext`s allow a `ManaRestriction`.
/// Inline `u64` covers up to 64 distinct reachable purposes without heap work;
/// `Wide` handles larger sets with no upper cap.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum PurposeMask {
    Inline(u64),
    Wide(Vec<u64>),
}

impl PurposeMask {
    fn of(restriction: &ManaRestriction, purposes: &[SpendingContext]) -> Self {
        if purposes.len() <= 64 {
            let mut bits = 0_u64;
            for (index, purpose) in purposes.iter().enumerate() {
                if restriction.allows(purpose) {
                    bits |= 1_u64 << index;
                }
            }
            Self::Inline(bits)
        } else {
            let mut words = vec![0_u64; purposes.len().div_ceil(64)];
            for (index, purpose) in purposes.iter().enumerate() {
                if restriction.allows(purpose) {
                    words[index / 64] |= 1_u64 << (index % 64);
                }
            }
            Self::Wide(words)
        }
    }

    fn is_all(&self, count: usize) -> bool {
        match self {
            Self::Inline(bits) => {
                let all = if count >= 64 {
                    u64::MAX
                } else {
                    (1_u64 << count) - 1
                };
                *bits == all
            }
            Self::Wide(words) => {
                let full = count / 64;
                let rem = count % 64;
                words[..full].iter().all(|&word| word == u64::MAX)
                    && (rem == 0 || words[full] == (1_u64 << rem) - 1)
            }
        }
    }
}

/// Non-ordinary restricted classes on a feasibility-only `ProjectedMana`.
/// `ManaExpiry` and `SpendEffect` are stripped in feasibility search, so
/// unrestricted classes merge directly into `ProjectedMana.mana`. Inline
/// single-class storage avoids per-mode `Box` allocations for uniform
/// restricted sources such as `Secluded Courtyard` and `Hermitic Herbalist`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
enum ProjectedClasses {
    #[default]
    None,
    One(PurposeMask, [u8; 6]),
    Many(Box<[(PurposeMask, [u8; 6])]>),
}

/// Canonical `Mana` projected onto reachable spending purposes for pure
/// feasibility search (`PaymentSearch`). Unrestricted non-ordinary classes
/// (differing only in `ManaExpiry` or `SpendEffect`) and restrictions that
/// admit every reachable purpose merge into `mana`; remaining restrictions
/// merge by `PurposeMask`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct ProjectedMana {
    mana: [u8; 6],
    classes: ProjectedClasses,
}

/// Immutable choice list with a precomputed hash digest so search states,
/// obligations and projected outputs compare and hash in O(1).
#[derive(Debug)]
struct Choices<T>(Arc<ChoicesInner<T>>);

#[derive(Debug)]
struct ChoicesInner<T> {
    modes: Vec<T>,
    digest: u64,
}

impl<T> Clone for Choices<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: std::hash::Hash> From<Vec<T>> for Choices<T> {
    fn from(modes: Vec<T>) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = FastHasher::default();
        modes.hash(&mut hasher);
        Self(Arc::new(ChoicesInner {
            modes,
            digest: hasher.finish(),
        }))
    }
}

impl<T> std::ops::Deref for Choices<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        &self.0.modes
    }
}

impl<T: PartialEq> PartialEq for Choices<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || (self.0.digest == other.0.digest && self.0.modes == other.0.modes)
    }
}

impl<T: Eq> Eq for Choices<T> {}

impl<T> std::hash::Hash for Choices<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.0.digest);
    }
}

type ManaChoices = Choices<Mana>;
type ProjectedChoices = Choices<ProjectedMana>;

/// Projects `ManaRestriction`s onto the finite set of `SpendingContext`s
/// reachable in a payment query (pending obligations plus nonzero-cost mana
/// abilities), interning projected output choice lists so identically-usable
/// restricted sources compare in O(1).
struct PurposeProjection {
    purposes: Vec<SpendingContext>,
    masks: FastHashMap<ManaRestriction, PurposeMask>,
    choices: FastHashMap<ManaChoices, ProjectedChoices>,
    interned: FastHashSet<ProjectedChoices>,
}

impl PurposeProjection {
    fn for_model(
        model: &PaymentModel,
        obligations: &[Obligation],
        extra: &[SpendingContext],
    ) -> Self {
        let mut purposes = Vec::new();
        let mut push = |purpose: &SpendingContext| {
            if !purposes.contains(purpose) {
                purposes.push(purpose.clone());
            }
        };
        for purpose in extra {
            push(purpose);
        }
        for obligation in obligations {
            if let Obligation::Pay { cost, .. } = obligation {
                if cost.mana_value > 0 {
                    push(&cost.purpose);
                }
            }
        }
        for source in model.sources.iter() {
            for ability in &source.abilities {
                if ability.cost.mana_value > 0 {
                    push(&ability.cost.purpose);
                }
            }
        }
        Self {
            purposes,
            masks: FastHashMap::default(),
            choices: FastHashMap::default(),
            interned: FastHashSet::default(),
        }
    }

    fn mask(&mut self, restriction: &ManaRestriction) -> PurposeMask {
        if let Some(mask) = self.masks.get(restriction) {
            return mask.clone();
        }
        let mask = PurposeMask::of(restriction, &self.purposes);
        self.masks.insert(restriction.clone(), mask.clone());
        mask
    }

    /// The restriction's bitset when it actually excludes at least one
    /// reachable spending purpose; `None` when `class` can spend everywhere
    /// in this query and therefore merges into ordinary `ProjectedMana.mana`.
    fn effective_mask(&mut self, class: &ManaClass) -> Option<PurposeMask> {
        let restriction = class.restriction.as_ref()?;
        let mask = self.mask(restriction);
        (!mask.is_all(self.purposes.len())).then_some(mask)
    }

    /// The single shared `ManaClass` when every mode carries exactly one class
    /// and all modes share it (e.g. restricted five-color lands).
    fn uniform_class(modes: &[Mana]) -> Option<&ManaClass> {
        let first = modes.first()?;
        let [(class, _)] = first.classes() else {
            return None;
        };
        modes[1..]
            .iter()
            .all(|m| matches!(m.classes(), [(other, _)] if other == class))
            .then_some(class)
    }

    fn project_mana(&mut self, mana: &Mana) -> ProjectedMana {
        let mut plain = mana.mana;
        let raw_classes = mana.classes();
        if raw_classes.is_empty() {
            return ProjectedMana {
                mana: plain,
                classes: ProjectedClasses::None,
            };
        }
        if let [(class, amounts)] = raw_classes {
            let classes = match self.effective_mask(class) {
                Some(mask) => ProjectedClasses::One(mask, *amounts),
                None => {
                    add_amounts(&mut plain, amounts);
                    ProjectedClasses::None
                }
            };
            return ProjectedMana {
                mana: plain,
                classes,
            };
        }
        let mut classes: Vec<(PurposeMask, [u8; 6])> = Vec::new();
        for (class, amounts) in raw_classes {
            let Some(mask) = self.effective_mask(class) else {
                add_amounts(&mut plain, amounts);
                continue;
            };
            match classes.binary_search_by(|(existing, _)| existing.cmp(&mask)) {
                Ok(index) => add_amounts(&mut classes[index].1, amounts),
                Err(index) => classes.insert(index, (mask, *amounts)),
            }
        }
        let classes = match classes.len() {
            0 => ProjectedClasses::None,
            1 => {
                let (mask, amounts) = classes.pop().unwrap();
                ProjectedClasses::One(mask, amounts)
            }
            _ => ProjectedClasses::Many(classes.into_boxed_slice()),
        };
        ProjectedMana {
            mana: plain,
            classes,
        }
    }

    fn project_choices(&mut self, raw: &ManaChoices) -> ProjectedChoices {
        if let Some(cached) = self.choices.get(raw) {
            return cached.clone();
        }
        let modes: Vec<ProjectedMana> = if let Some(uniform) = Self::uniform_class(raw) {
            match self.effective_mask(uniform) {
                Some(mask) => raw
                    .iter()
                    .map(|m| ProjectedMana {
                        mana: m.mana,
                        classes: ProjectedClasses::One(mask.clone(), m.classes()[0].1),
                    })
                    .collect(),
                None => raw
                    .iter()
                    .map(|m| {
                        let mut plain = m.mana;
                        add_amounts(&mut plain, &m.classes()[0].1);
                        ProjectedMana {
                            mana: plain,
                            classes: ProjectedClasses::None,
                        }
                    })
                    .collect(),
            }
        } else {
            raw.iter().map(|m| self.project_mana(m)).collect()
        };
        let candidate = ProjectedChoices::from(modes);
        let interned = if let Some(existing) = self.interned.get(&candidate) {
            existing.clone()
        } else {
            self.interned.insert(candidate.clone());
            candidate
        };
        self.choices.insert(raw.clone(), interned.clone());
        interned
    }

    /// Whether two sources have identical feasibility behavior, lazily
    /// constructing or borrowing a `PurposeProjection` only when non-ordinary
    /// outputs differ in raw representation (excluding `untapped`, which
    /// search canonicalizes in `PaymentResources`). `ManaExpiry` and
    /// `SpendEffect` never affect payment feasibility, so both `PaymentSearch`
    /// and `SourceClassCache` project through this single check.
    fn sources_equivalent<'a>(
        projection: impl FnOnce() -> Option<&'a mut PurposeProjection>,
        a: &ManaSource,
        b: &ManaSource,
        candidates: &[&Arc<Vec<PermanentId>>],
    ) -> bool {
        if a.waterbend != b.waterbend
            || a.keyword != b.keyword
            || a.abilities.len() != b.abilities.len()
            || !candidates
                .iter()
                .all(|set| set.contains(&a.permanent) == set.contains(&b.permanent))
        {
            return false;
        }
        let mut fixed_pair = None;
        if a.fixed != b.fixed {
            let (Some(af), Some(bf)) = (&a.fixed, &b.fixed) else {
                return false;
            };
            if !mana_has_classes(af) && !mana_has_classes(bf) {
                return false;
            }
            fixed_pair = Some((af, bf));
        }
        let mut need_ability_proj = false;
        for (x, y) in a.abilities.iter().zip(&b.abilities) {
            if x.tap_source != y.tap_source
                || x.tap_count != y.tap_count
                || (x.tap_count != 0 && x.tap_candidates != y.tap_candidates)
                || x.limited != y.limited
                || x.sacrifice_source != y.sacrifice_source
                || x.life_cost != y.life_cost
                || !((x.cost.mana_value == 0 && y.cost.mana_value == 0) || x.cost == y.cost)
                || x.outputs.len() != y.outputs.len()
            {
                return false;
            }
            if x.outputs != y.outputs {
                if !x.outputs.iter().any(mana_has_classes)
                    && !y.outputs.iter().any(mana_has_classes)
                {
                    return false;
                }
                need_ability_proj = true;
            }
        }
        if fixed_pair.is_none() && !need_ability_proj {
            return true;
        }
        let Some(proj) = projection() else {
            return false;
        };
        if let Some((af, bf)) = fixed_pair {
            if proj.project_mana(af) != proj.project_mana(bf) {
                return false;
            }
        }
        if need_ability_proj {
            for (x, y) in a.abilities.iter().zip(&b.abilities) {
                if x.outputs != y.outputs
                    && proj.project_choices(&x.outputs) != proj.project_choices(&y.outputs)
                {
                    return false;
                }
            }
        }
        true
    }
}

/// Memoizes a per-source feasibility answer across purpose-equivalent sources,
/// lazily building a `PurposeProjection` only when non-ordinary outputs differ
/// in raw representation. Callers still emit each source's own action.
struct SourceClassCache<'a, V, K = u8> {
    model: &'a PaymentModel<'a>,
    obligations: &'a [Obligation],
    extra: &'a [SpendingContext],
    candidates: Vec<&'a Arc<Vec<PermanentId>>>,
    projection: Option<PurposeProjection>,
    classes: Vec<(usize, K, V)>,
}

impl<'a, V: Copy, K: Copy + Eq> SourceClassCache<'a, V, K> {
    fn new(
        model: &'a PaymentModel<'a>,
        obligations: &'a [Obligation],
        extra: &'a [SpendingContext],
    ) -> Self {
        Self {
            candidates: model
                .tap_candidate_sets(obligations.iter().filter_map(Obligation::tap_candidates)),
            model,
            obligations,
            extra,
            projection: None,
            classes: Vec::new(),
        }
    }

    /// Cached value for an equivalent representative, or `None` if
    /// `source_index` starts a new equivalence class.
    fn get(&mut self, source_index: usize, key: K) -> Option<V> {
        let source = &self.model.sources[source_index];
        self.classes
            .iter()
            .find(|(rep_index, rep_key, _)| {
                let rep = &self.model.sources[*rep_index];
                rep.untapped == source.untapped
                    && *rep_key == key
                    && PurposeProjection::sources_equivalent(
                        || {
                            Some(self.projection.get_or_insert_with(|| {
                                PurposeProjection::for_model(
                                    self.model,
                                    self.obligations,
                                    self.extra,
                                )
                            }))
                        },
                        rep,
                        source,
                        &self.candidates,
                    )
            })
            .map(|(_, _, value)| *value)
    }

    fn insert(&mut self, source_index: usize, key: K, value: V) {
        self.classes.push((source_index, key, value));
    }
}

fn add_amounts(dst: &mut [u8; 6], src: &[u8; 6]) {
    for (slot, &more) in dst.iter_mut().zip(src) {
        *slot = slot.saturating_add(more);
    }
}

fn mana_has_classes(mana: &Mana) -> bool {
    !mana.classes().is_empty()
}

fn obligation_has_classes(obligation: &Obligation) -> bool {
    matches!(obligation, Obligation::Output { choices, .. } if choices.iter().any(mana_has_classes))
}

fn state_has_classes(state: &PaymentSearchState) -> bool {
    mana_has_classes(&state.pool) || state.obligations.iter().any(obligation_has_classes)
}

fn model_has_classes(model: &PaymentModel) -> bool {
    model.sources.iter().any(|source| {
        source.fixed.as_ref().is_some_and(mana_has_classes)
            || source
                .abilities
                .iter()
                .any(|ability| ability.outputs.iter().any(mana_has_classes))
    })
}

/// One finite resource transition model. A source is reserved before its
/// upfront obligation, so filters cannot finance themselves or form seedless
/// cycles. Witnesses retain identities and concrete typed pool remainders.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Obligation<C = ManaChoices> {
    Pay {
        cost: ManaCost,
        taps: Taps,
    },
    Output {
        // Alternatives and candidate sets are immutable query payloads. Branches
        // and memo keys share them while keeping their own progress and pool.
        choices: C,
        life_cost: u32,
    },
    Tap {
        candidates: Arc<Vec<PermanentId>>,
        remaining: usize,
    },
}

impl Obligation {
    /// `cost`, paid only with mana.
    fn mana(cost: ManaCost) -> Self {
        Obligation::Pay {
            cost,
            taps: Taps::None,
        }
    }
}

impl<C> Obligation<C> {
    fn tap_candidates(&self) -> Option<&Arc<Vec<PermanentId>>> {
        match self {
            Self::Tap { candidates, .. } => Some(candidates),
            _ => None,
        }
    }

    fn map_choices<D>(&self, f: impl FnOnce(&C) -> D) -> Obligation<D> {
        match self {
            Self::Pay { cost, taps } => Obligation::Pay {
                cost: cost.clone(),
                taps: *taps,
            },
            Self::Output { choices, life_cost } => Obligation::Output {
                choices: f(choices),
                life_cost: *life_cost,
            },
            Self::Tap {
                candidates,
                remaining,
            } => Obligation::Tap {
                candidates: candidates.clone(),
                remaining: *remaining,
            },
        }
    }
}

/// Which untapped artifacts and creatures may pay a `Pay` obligation's
/// generic mana by tapping, {1} each, instead of with mana.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Taps {
    /// None: the cost is paid with mana.
    None,
    /// Those the cost's convoke or improvise names (`ManaSource::keyword`).
    Keyword,
    /// Any of them: a waterbend cost (`ManaSource::waterbend`).
    Any,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ManaSource {
    permanent: PermanentId,
    object: ObjectId,
    untapped: bool,
    fixed: Option<Mana>,
    abilities: Vec<ManaActivation>,
    /// An untapped artifact or creature, whose tap can pay {1} of a
    /// waterbend cost.
    waterbend: bool,
    /// Whether the cost being paid lets this tap pay any of its generic
    /// mana (convoke, improvise; `Game::mark_help_taps`).
    keyword: bool,
}

impl ManaSource {
    fn same_payment_profile(&self, other: &Self) -> bool {
        self.fixed == other.fixed
            && self.abilities == other.abilities
            && self.waterbend == other.waterbend
            && self.keyword == other.keyword
    }

    /// Whether tapping this source may pay {1} of an obligation that `taps`
    /// may pay.
    fn pays_by_tap(&self, taps: Taps) -> bool {
        match taps {
            Taps::None => false,
            Taps::Keyword => self.keyword,
            Taps::Any => self.waterbend,
        }
    }
}

/// One representative output of a useful tap. A determined payment keeps
/// the first members of each interchangeable class in battlefield order so
/// two residual costs can compare counts and spend effects directly.
#[derive(Clone, Copy)]
struct UsefulTap<'a> {
    source: &'a ManaSource,
    output: &'a Mana,
}

enum PaymentTaps<'a> {
    Determined(Vec<UsefulTap<'a>>),
    Choose(Vec<Action>),
}

#[derive(Clone, Copy, Default)]
struct TapOptions {
    fixed_payable: bool,
    explicit: bool,
    allow_plain_pool: bool,
    other_choice: bool,
    need_taps: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ManaActivation {
    /// An unrestricted-use tap with no other costs, effects or triggered output choice.
    automatic: bool,
    tap_source: bool,
    tap_count: usize,
    tap_candidates: Arc<Vec<PermanentId>>,
    limited: bool,
    sacrifice_source: bool,
    ability_index: usize,
    cost: ManaCost,
    outputs: ManaChoices,
    life_cost: u32,
}

impl ManaActivation {
    fn max_net(&self) -> u64 {
        self.outputs
            .iter()
            .map(|output| u64::from(output.total()).saturating_sub(u64::from(self.cost.mana_value)))
            .max()
            .unwrap_or(0)
    }
}

/// Optimistic color upper bound and maximum single-mode usable total over
/// `outputs`, projected onto one spending purpose. Maxima are taken within
/// each allowed class *before* combining classes so two modes with different
/// classes both contribute to the bound.
fn optimistic_supply(
    outputs: &[Mana],
    mut allows: impl FnMut(&ManaClass) -> bool,
) -> ([u8; 6], u64) {
    let mut upper = [0_u8; 6];
    let mut max_usable = 0_u64;
    let mut has_classes = false;
    let single = outputs.len() == 1;
    for output in outputs {
        let mut mode_usable = 0_u64;
        for (amount, more) in upper.iter_mut().zip(output.mana) {
            *amount = (*amount).max(more);
            mode_usable += u64::from(more);
        }
        for (class, amounts) in output.classes() {
            has_classes = true;
            if allows(class) {
                for (amount, &more) in upper.iter_mut().zip(amounts) {
                    mode_usable += u64::from(more);
                    if single {
                        *amount = amount.saturating_add(more);
                    }
                }
            }
        }
        max_usable = max_usable.max(mode_usable);
    }
    if single || !has_classes {
        return (upper, max_usable);
    }
    for (index, output) in outputs.iter().enumerate() {
        for (class, _) in output.classes() {
            if !allows(class)
                || outputs[..index]
                    .iter()
                    .any(|previous| previous.classes().iter().any(|(seen, _)| seen == class))
            {
                continue;
            }
            let mut amounts = [0_u8; 6];
            for mode in &outputs[index..] {
                for (amount, more) in amounts.iter_mut().zip(mode.class_of(class)) {
                    *amount = (*amount).max(more);
                }
            }
            for (amount, more) in upper.iter_mut().zip(amounts) {
                *amount = amount.saturating_add(more);
            }
        }
    }
    (upper, max_usable)
}

/// Reserve incompatible uses, not the whole object. A tapped mana creature
/// can still be sacrificed later, but an announced tap cannot be spent twice.
#[derive(Clone, Copy)]
struct SourceReservation {
    permanent: PermanentId,
    tap: bool,
    sacrifice: bool,
}

impl SourceReservation {
    /// `permanent` is tapped for something other than mana.
    fn tap(permanent: PermanentId) -> Self {
        Self {
            permanent,
            tap: true,
            sacrifice: false,
        }
    }

    /// `permanent` is owed to a cost that taps it or makes it leave. Either
    /// way it cannot also leave to pay another cost.
    fn owed(permanent: PermanentId, tap: bool, leaves: bool) -> Self {
        Self {
            permanent,
            tap,
            sacrifice: tap || leaves,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "benchmark", derive(serde::Serialize))]
enum PaymentStep {
    Fixed(PermanentId),
    Activate(PermanentId, usize),
    Spend,
    Output(usize),
    Waterbend(PermanentId),
    Tap(ObjectId),
}

impl PaymentStep {
    fn action(&self, player: PlayerId) -> Option<Action> {
        match self {
            Self::Activate(permanent, ability_index) => Some(Action::ActivateAbility {
                player,
                permanent: *permanent,
                ability_index: *ability_index,
            }),
            Self::Tap(object) => Some(Action::SelectCard {
                player,
                object: *object,
            }),
            Self::Output(mode) => Some(Action::ChooseMode {
                player,
                mode: *mode,
            }),
            Self::Waterbend(permanent) => Some(Action::WaterbendTap {
                player,
                permanent: *permanent,
            }),
            Self::Fixed(_) | Self::Spend => None,
        }
    }
}

/// A source's presence, untapped resource and per-ability limited uses are
/// independent. A tapped source can still pay a sacrifice cost; a repeatable
/// ability can consume other objects without consuming its own source.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PaymentResources {
    present: Vec<bool>,
    untapped: Vec<bool>,
    /// Per-source bitmasks for limited abilities (`ability < 64`). Empty until
    /// the first limited use, so ordinary transitions allocate nothing.
    used_masks: Vec<u64>,
}

impl std::hash::Hash for PaymentResources {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        fn hash_bits<H: std::hash::Hasher>(bits: &[bool], state: &mut H) {
            state.write_usize(bits.len());
            for chunk in bits.chunks(64) {
                let mut word = 0_u64;
                for (bit, &flag) in chunk.iter().enumerate() {
                    word |= u64::from(flag) << bit;
                }
                state.write_u64(word);
            }
        }
        hash_bits(&self.present, state);
        hash_bits(&self.untapped, state);
        state.write_usize(self.used_masks.len());
        for &mask in &self.used_masks {
            state.write_u64(mask);
        }
    }
}

impl PaymentResources {
    #[inline]
    fn used_mask(&self, source: usize) -> u64 {
        self.used_masks.get(source).copied().unwrap_or(0)
    }

    #[inline]
    fn is_used(&self, source: usize, ability: usize) -> bool {
        self.used_masks.get(source).is_some_and(|&mask| {
            assert!(ability < 64, "source has >= 64 activated mana abilities");
            mask & (1_u64 << ability) != 0
        })
    }

    #[inline]
    fn source_state(&self, index: usize) -> (bool, bool, u64) {
        (
            self.present[index],
            self.untapped[index],
            self.used_mask(index),
        )
    }

    /// Allocation-free O(1) total order over one source's resource state for
    /// canonicalizing interchangeable sources.
    #[inline]
    fn cmp_source(&self, a: usize, b: usize) -> std::cmp::Ordering {
        self.source_state(a).cmp(&self.source_state(b))
    }

    fn use_ability(&mut self, source: usize, ability: usize) {
        assert!(ability < 64, "source has >= 64 activated mana abilities");
        if self.used_masks.is_empty() {
            self.used_masks.resize(self.present.len(), 0);
        }
        self.used_masks[source] |= 1_u64 << ability;
    }
}

struct AbilitySupply {
    upper: [u8; 6],
    max_usable: u64,
    max_net: u64,
}

struct SourceSupply {
    fixed_upper: [u8; 6],
    fixed_usable: u64,
    fixed_net: u64,
    abilities: std::ops::Range<usize>,
}

/// Per-`SpendingContext` supply memo for one `PaymentSearch`: caches
/// `ManaRestriction::allows` and each immutable source/ability's optimistic
/// color and usable-total bounds.
struct PurposeSupply {
    purpose: SpendingContext,
    allows: FastHashMap<ManaRestriction, bool>,
    sources: Vec<SourceSupply>,
    abilities: Vec<AbilitySupply>,
}

impl PurposeSupply {
    fn allows(&mut self, class: &ManaClass) -> bool {
        let Some(restriction) = &class.restriction else {
            return true;
        };
        if let Some(&allowed) = self.allows.get(restriction) {
            return allowed;
        }
        let allowed = restriction.allows(&self.purpose);
        self.allows.insert(restriction.clone(), allowed);
        allowed
    }

    fn allows_cached(&self, class: &ManaClass) -> bool {
        class.restriction.as_ref().is_none_or(|restriction| {
            self.allows
                .get(restriction)
                .copied()
                .unwrap_or_else(|| restriction.allows(&self.purpose))
        })
    }

    fn abilities_of(&self, source: usize) -> &[AbilitySupply] {
        &self.abilities[self.sources[source].abilities.clone()]
    }

    fn build(model: &PaymentModel, purpose: &SpendingContext) -> Self {
        let mut supply = Self {
            purpose: purpose.clone(),
            allows: FastHashMap::default(),
            sources: Vec::new(),
            abilities: Vec::new(),
        };
        let mut sources = Vec::with_capacity(model.sources.len());
        let mut abilities = Vec::new();
        for source in model.sources.iter() {
            let (fixed_upper, fixed_usable, fixed_net) = match &source.fixed {
                Some(output) => {
                    let (upper, usable) =
                        optimistic_supply(std::slice::from_ref(output), |class| {
                            supply.allows(class)
                        });
                    (upper, usable, u64::from(output.total()))
                }
                None => ([0; 6], 0, 0),
            };
            let start = abilities.len();
            for ability in &source.abilities {
                let (upper, max_usable) =
                    optimistic_supply(&ability.outputs, |class| supply.allows(class));
                abilities.push(AbilitySupply {
                    upper,
                    max_usable,
                    max_net: ability.max_net(),
                });
            }
            sources.push(SourceSupply {
                fixed_upper,
                fixed_usable,
                fixed_net,
                abilities: start..abilities.len(),
            });
        }
        supply.sources = sources;
        supply.abilities = abilities;
        supply
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct SearchState<M = Mana, C = ManaChoices> {
    resources: PaymentResources,
    pool: M,
    life: u32,
    obligations: Vec<Obligation<C>>,
    /// Fixed-source budget until the first Spend; later obligations are unrestricted.
    prefix: Option<usize>,
}

type PaymentSearchState = SearchState;
type ProjectedSearchState = SearchState<ProjectedMana, ProjectedChoices>;

impl PaymentSearchState {
    fn with_resources(&self, resources: PaymentResources) -> Self {
        Self {
            resources,
            pool: self.pool.clone(),
            life: self.life,
            obligations: self.obligations.clone(),
            prefix: self.prefix,
        }
    }
}

/// Equivalence and purpose-projection state allocated only once a search
/// grows past `PaymentSearch::GROWTH`, keeping ordinary small searches small
/// on the stack and free of extra table setup/teardown.
///
/// While `projection` is `Some` (any non-ordinary `ManaClass` present), every
/// visited state routes into `projected_visited` so unrestricted classed mana
/// collapses with ordinary mana; otherwise states stay in `PaymentSearch::visited`.
struct SearchEquivalence {
    groups: Vec<Vec<usize>>,
    /// Equivalence group index per source (`u32::MAX` for singletons).
    source_group: Vec<u32>,
    projection: Option<PurposeProjection>,
    projected_visited: FastHashSet<ProjectedSearchState>,
    order_buf: Vec<usize>,
}

impl SearchEquivalence {
    /// Whether an earlier sibling in the same equivalence group with identical
    /// resource state was already tried and failed in the current ascending
    /// source loop. Preserves the first concrete witness in source order
    /// because `prev < index` is always tried first.
    #[inline]
    fn has_tried_sibling(&self, resources: &PaymentResources, index: usize) -> bool {
        let Some(&group_id) = self.source_group.get(index) else {
            return false;
        };
        if group_id == u32::MAX {
            return false;
        }
        let target = resources.source_state(index);
        self.groups[group_id as usize]
            .iter()
            .take_while(|&&prev| prev < index)
            .any(|&prev| resources.source_state(prev) == target)
    }

    /// Sort interchangeable source groups in `resources`. Returns `None` when
    /// every group is already in non-decreasing order so the caller avoids
    /// cloning the state.
    fn canonical_resources(&mut self, resources: &PaymentResources) -> Option<PaymentResources> {
        let mut permuted: Option<PaymentResources> = None;
        let has_masks = !resources.used_masks.is_empty();
        for group in &self.groups {
            // Source-order success usually consumes an already sorted prefix.
            // Permute only for branches that need one.
            if group
                .windows(2)
                .all(|pair| resources.cmp_source(pair[0], pair[1]) != std::cmp::Ordering::Greater)
            {
                continue;
            }
            self.order_buf.clear();
            self.order_buf.extend_from_slice(group);
            self.order_buf
                .sort_unstable_by(|&a, &b| resources.cmp_source(a, b));
            let dst = permuted.get_or_insert_with(|| resources.clone());
            for (&to, &from) in group.iter().zip(self.order_buf.iter()) {
                dst.present[to] = resources.present[from];
                dst.untapped[to] = resources.untapped[from];
                if has_masks {
                    dst.used_masks[to] = resources.used_masks[from];
                }
            }
        }
        permuted
    }

    /// Insert `state` into `projected_visited` when feasibility projection is
    /// active (any non-ordinary `ManaClass` present in the query). Routes every
    /// state while active so unrestricted classed mana collapses with ordinary
    /// mana into one table; returns `None` when projection is inactive so
    /// ordinary-only searches stay in exact `visited`.
    fn insert_projected(&mut self, state: &PaymentSearchState) -> Option<bool> {
        if self.projection.is_none() {
            return None;
        }
        let resources = self
            .canonical_resources(&state.resources)
            .unwrap_or_else(|| state.resources.clone());
        let projection = self.projection.as_mut().unwrap();
        let pool = projection.project_mana(&state.pool);
        let obligations = state
            .obligations
            .iter()
            .map(|obligation| obligation.map_choices(|c| projection.project_choices(c)))
            .collect();
        Some(self.projected_visited.insert(ProjectedSearchState {
            resources,
            pool,
            life: state.life,
            obligations,
            prefix: state.prefix,
        }))
    }
}

/// Live branch state held during `witness`: shared in `Arc` when inserted
/// unmodified into `PaymentSearch::visited`, and kept unboxed on the stack when
/// resources are permuted or projected into `SearchEquivalence::projected_visited`.
enum LiveState {
    Owned(PaymentSearchState),
    Shared(Arc<PaymentSearchState>),
}

impl std::ops::Deref for LiveState {
    type Target = PaymentSearchState;
    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(state) => state,
            Self::Shared(state) => state,
        }
    }
}

/// Which transitions may be automatic at the caller's current boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourcePolicy {
    All,
    FixedOnly,
    AutomaticPrefix,
}

/// Memoize feasibility modulo interchangeable resources, but never rewrite the
/// live state or a returned witness. The first concrete DFS path still wins.
///
/// Ordinary plans expand a handful of states, so a search starts with exact
/// identity keys and only the root's color-set check. Once it has expanded
/// `GROWTH` states it classifies the interchangeable sources once (projecting
/// `ManaRestriction`s onto the reachable `SpendingContext`s), re-keys what it
/// visited and from then on canonicalizes every key and applies the color-set
/// bound to every state. Both protections only reject states with no witness,
/// so when they start does not change the path returned.
#[derive(Default)]
struct PaymentSearch {
    /// Counted-tap sets of the root obligations; with the abilities' own sets
    /// they decide which sources are interchangeable.
    root_taps: Vec<Arc<Vec<PermanentId>>>,
    /// Exact visited states (all states before `GROWTH`, and restriction-free
    /// canonical states after `GROWTH`).
    visited: FastHashSet<Arc<PaymentSearchState>>,
    /// Allocated once the search reaches `GROWTH`.
    equivalence: Option<Box<SearchEquivalence>>,
    /// Per-purpose supply and restriction-allowance memos for this search.
    /// Populated on the first branching state before any nonzero `Pay` pops,
    /// so it already holds every root nonzero spending context at `GROWTH`.
    purposes: Vec<PurposeSupply>,
    #[cfg(test)]
    max_states: Option<usize>,
    /// Reference search for tests: skip the color-set bound.
    #[cfg(test)]
    total_bound_only: bool,
    #[cfg(test)]
    gross_bound_only: bool,
}

impl PaymentSearch {
    /// Visited states after which a search pays for source classes and the
    /// per-state color-set bound.
    const GROWTH: usize = 16;

    fn new(obligations: &[Obligation]) -> Self {
        Self {
            root_taps: obligations
                .iter()
                .filter_map(Obligation::tap_candidates)
                .cloned()
                .collect(),
            ..Default::default()
        }
    }

    #[cfg(test)]
    fn identity_only(gross_bound_only: bool, max_states: usize) -> Self {
        Self {
            equivalence: Some(Box::new(SearchEquivalence {
                groups: Vec::new(),
                source_group: Vec::new(),
                projection: None,
                projected_visited: FastHashSet::default(),
                order_buf: Vec::new(),
            })),
            gross_bound_only,
            max_states: Some(max_states),
            ..Default::default()
        }
    }

    #[cfg(test)]
    fn visited_len(&self) -> usize {
        self.visited.len()
            + self
                .equivalence
                .as_deref()
                .map_or(0, |eq| eq.projected_visited.len())
    }

    fn visited_is_empty(&self) -> bool {
        self.visited.is_empty()
            && self
                .equivalence
                .as_deref()
                .is_none_or(|eq| eq.projected_visited.is_empty())
    }

    fn clear_visited(&mut self) {
        self.visited.clear();
        if let Some(eq) = self.equivalence.as_deref_mut() {
            eq.projected_visited.clear();
        }
    }

    /// Whether the color-set bound applies to the state about to be checked:
    /// at the root, and everywhere once the search has grown.
    fn bounds_colors(&self) -> bool {
        self.visited_is_empty() || self.equivalence.is_some()
    }

    #[cfg(test)]
    fn groups(&mut self, model: &PaymentModel) -> Vec<Vec<usize>> {
        self.groups_for(model, &[])
    }

    #[cfg(test)]
    fn groups_for(&mut self, model: &PaymentModel, obligations: &[Obligation]) -> Vec<Vec<usize>> {
        let project = model_has_classes(model) || obligations.iter().any(obligation_has_classes);
        let eq = self.classify_sources(model, obligations, project);
        let groups = eq.groups.clone();
        self.equivalence = Some(Box::new(eq));
        groups
    }

    #[inline]
    fn has_tried_sibling(&self, resources: &PaymentResources, index: usize) -> bool {
        self.equivalence
            .as_deref()
            .is_some_and(|eq| eq.has_tried_sibling(resources, index))
    }

    fn classify_sources(
        &self,
        model: &PaymentModel,
        obligations: &[Obligation],
        project: bool,
    ) -> SearchEquivalence {
        // Membership in every possible counted-tap set is part of identity:
        // equal mana producers need not be equal candidates for a saved cost.
        let candidates = model.tap_candidate_sets(&self.root_taps);
        let mut projection = project.then(|| {
            let extra: Vec<SpendingContext> = self
                .purposes
                .iter()
                .map(|entry| entry.purpose.clone())
                .collect();
            PurposeProjection::for_model(model, obligations, &extra)
        });
        // Most boards have few duplicates. Do not allocate a member vector
        // for every singleton just to discard it after classification.
        let mut representatives: Vec<(usize, Option<usize>)> = Vec::new();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for (index, source) in model.sources.iter().enumerate() {
            let equivalent = representatives.iter_mut().find(|(representative, _)| {
                let other = &model.sources[*representative];
                PurposeProjection::sources_equivalent(
                    || projection.as_mut(),
                    source,
                    other,
                    &candidates,
                )
            });
            if let Some((representative, group)) = equivalent {
                let group = group.get_or_insert_with(|| {
                    groups.push(vec![*representative]);
                    groups.len() - 1
                });
                groups[*group].push(index);
            } else {
                representatives.push((index, None));
            }
        }
        let mut source_group = vec![u32::MAX; model.sources.len()];
        for (gid, group) in groups.iter().enumerate() {
            for &index in group {
                source_group[index] = gid as u32;
            }
        }
        SearchEquivalence {
            groups,
            source_group,
            projection,
            projected_visited: FastHashSet::default(),
            order_buf: Vec::new(),
        }
    }

    fn purpose_index(&mut self, model: &PaymentModel, purpose: &SpendingContext) -> usize {
        if let Some(index) = self
            .purposes
            .iter()
            .position(|entry| entry.purpose == *purpose)
        {
            return index;
        }
        self.purposes.push(PurposeSupply::build(model, purpose));
        self.purposes.len() - 1
    }

    /// Distinct nonzero-cost `Pay` purpose indices for `obligations`, cached on `self`.
    fn pay_purposes(&mut self, model: &PaymentModel, obligations: &[Obligation]) -> Vec<usize> {
        let mut indices = Vec::new();
        for obligation in obligations {
            if let Obligation::Pay { cost, .. } = obligation {
                if cost.mana_value == 0 {
                    continue;
                }
                let index = self.purpose_index(model, &cost.purpose);
                if !indices.contains(&index) {
                    indices.push(index);
                }
            }
        }
        indices
    }

    /// Mana by kind that any pending cost in `purpose_indices` could spend,
    /// reading cached restriction allowances.
    fn usable_for(&mut self, purpose_indices: &[usize], mana: &Mana) -> [u64; 6] {
        let mut usable = mana.mana.map(u64::from);
        for (class, amounts) in mana.classes() {
            if purpose_indices
                .iter()
                .any(|&pi| self.purposes[pi].allows(class))
            {
                for (amount, more) in usable.iter_mut().zip(amounts) {
                    *amount += u64::from(*more);
                }
            }
        }
        usable
    }

    fn usable_total(&mut self, purpose_indices: &[usize], mana: &Mana) -> u64 {
        self.usable_for(purpose_indices, mana).iter().sum()
    }

    fn usable_total_cached(&self, purpose_indices: &[usize], mana: &Mana) -> u64 {
        let mut total: u64 = mana.mana.iter().map(|&n| u64::from(n)).sum();
        for (class, amounts) in mana.classes() {
            if purpose_indices
                .iter()
                .any(|&pi| self.purposes[pi].allows_cached(class))
            {
                total += amounts.iter().map(|&n| u64::from(n)).sum::<u64>();
            }
        }
        total
    }

    fn rekey_visited(&mut self) {
        let Some(eq) = self.equivalence.as_deref_mut() else {
            return;
        };
        if eq.groups.is_empty() && eq.projection.is_none() {
            return;
        }
        let exact = std::mem::take(&mut self.visited);
        self.visited.reserve(exact.len() + 1);
        for key in exact {
            if eq.insert_projected(&key).is_some() {
                continue;
            }
            let key = match eq.canonical_resources(&key.resources) {
                Some(resources) => Arc::new(key.with_resources(resources)),
                None => key,
            };
            self.visited.insert(key);
        }
    }

    fn insert(&mut self, model: &PaymentModel, state: PaymentSearchState) -> Option<LiveState> {
        if self.equivalence.is_none() && self.visited.len() >= Self::GROWTH {
            // A visited identity state failed or is being expanded, and so is
            // every state interchangeable with it: its canonical key is one
            // the grouped search would hold too.
            let has_classes = model_has_classes(model)
                || state_has_classes(&state)
                || self.visited.iter().any(|s| state_has_classes(s));
            let eq = self.classify_sources(model, &state.obligations, has_classes);
            self.equivalence = Some(Box::new(eq));
            self.rekey_visited();
        }
        let live = 'insert: {
            if let Some(eq) = self.equivalence.as_deref_mut() {
                if let Some(fresh) = eq.insert_projected(&state) {
                    if !fresh {
                        return None;
                    }
                    break 'insert LiveState::Owned(state);
                }
                if let Some(resources) = eq.canonical_resources(&state.resources) {
                    if !self
                        .visited
                        .insert(Arc::new(state.with_resources(resources)))
                    {
                        return None;
                    }
                    break 'insert LiveState::Owned(state);
                }
            }
            let shared = Arc::new(state);
            if !self.visited.insert(shared.clone()) {
                return None;
            }
            LiveState::Shared(shared)
        };
        #[cfg(test)]
        if let Some(limit) = self.max_states {
            assert!(
                self.visited_len() <= limit,
                "payment exceeded {limit} states"
            );
        }
        Some(live)
    }
}

/// Exact derived queries, copied by value with a fork and excluded from rules
/// state. Each fork owns its FIFO; immutable entries can share their payload.
/// FIFO bounds retention; equality includes ordered sources and callers,
/// not just board revisions. A selected tap is part of the query identity.
#[derive(Clone, Debug, Default)]
pub(crate) struct PaymentCache {
    entries: VecDeque<Arc<CachedPayment>>,
}

#[cfg(feature = "benchmark")]
impl PaymentCache {
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }
}

#[derive(Debug)]
struct CachedPayment {
    sources: Arc<Vec<ManaSource>>,
    life: u32,
    pool: Mana,
    obligations: Vec<Obligation>,
    policy: SourcePolicy,
    tap: Option<PermanentId>,
    path: Option<Vec<PaymentStep>>,
}

struct PaymentModel<'a> {
    cache: Option<&'a RefCell<PaymentCache>>,
    sources: Arc<Vec<ManaSource>>,
    life: u32,
}

/// A question the direct answer covers: `cost` paid with mana, then `split`,
/// generic that mana or the taps `taps` names may pay.
struct DirectPayment<'a> {
    cost: &'a ManaCost,
    split: Option<&'a ManaCost>,
    taps: Taps,
}

impl DirectPayment<'_> {
    fn split_generic(&self) -> u32 {
        self.split.map_or(0, |split| u32::from(split.generic()))
    }
}

/// What tapping one source offers a direct payment.
#[derive(Clone, Copy)]
struct DirectTap<'a> {
    permanent: PermanentId,
    /// The fixed output of an untapped source, tapped in source order even
    /// when the cost can spend none of it.
    output: Option<&'a Mana>,
    /// How much of `output` the cost's purpose may spend.
    mana: u32,
    /// Whether the tap may instead pay {1} of the split.
    waterbend: bool,
}

impl DirectTap<'_> {
    /// A waterbend tap that gives up no mana. A source worth mana is never
    /// better tapped for waterbend, which pays exactly {1}.
    fn spare(self) -> bool {
        self.waterbend && self.mana == 0
    }

    /// The most this tap adds towards the split.
    fn worth(self) -> u32 {
        self.mana.max(self.waterbend.into())
    }
}

/// One player's unreserved sources, kept while a `PaymentScan` is live, plus
/// one-shot staged actions bridging `advance_payment` into the immediately
/// following `payment_action_space` without a state-revision key.
#[derive(Debug, Default)]
pub(crate) struct SharedSources {
    scans: u32,
    sources: [Option<Arc<Vec<ManaSource>>>; 2],
    staged_announced: Option<StagedAnnouncedActions>,
}

#[derive(Debug)]
struct StagedAnnouncedActions {
    player: PlayerId,
    floats: ManaFloats,
    automatic: bool,
    actions: Vec<Action>,
}

/// One legal-action computation. The sources a player could tap are a
/// function of the game alone, and the guard's shared borrow excludes every
/// write to it, so the payment questions asked while it lives build those
/// sources once per player. Reservations of the pending payment, a sacrifice
/// under consideration and payable life are still applied per question.
/// Debug builds rebuild the sources on every reuse and assert they match.
pub(crate) struct PaymentScan<'a> {
    game: &'a Game,
}

impl Drop for PaymentScan<'_> {
    fn drop(&mut self) {
        let mut shared = self.game.payment_sources.borrow_mut();
        shared.scans -= 1;
        if shared.scans == 0 {
            shared.sources = Default::default();
        }
    }
}

impl PaymentModel<'_> {
    fn reserve(&mut self, reservation: SourceReservation) {
        let Some(index) = self.sources.iter().position(|source| {
            source.permanent == reservation.permanent
                && (reservation.tap
                    && (source.untapped || source.fixed.is_some() || source.waterbend)
                    || source.abilities.iter().any(|ability| {
                        reservation.tap && ability.tap_source
                            || reservation.sacrifice && ability.sacrifice_source
                    }))
        }) else {
            return;
        };
        let source = &mut Arc::make_mut(&mut self.sources)[index];
        if reservation.tap {
            source.fixed = None;
            source.waterbend = false;
            source.keyword = false;
            source.untapped = false;
        }
        source.abilities.retain(|ability| {
            !(reservation.tap && ability.tap_source
                || reservation.sacrifice && ability.sacrifice_source)
        });
    }

    /// Gross usable output and net new units. Each tap can choose one mode.
    fn source_bounds(
        &self,
        resources: &PaymentResources,
        life: u32,
        mut fixed_supply: impl FnMut(usize, &Mana) -> (u64, u64),
        mut ability_supply: impl FnMut(usize, usize, &ManaActivation) -> (u64, u64),
        pays_by_tap: impl Fn(&ManaSource) -> bool,
    ) -> (u64, u64) {
        let mut net_total = 0;
        let mut total = 0;
        for (index, source) in self.sources.iter().enumerate() {
            if !resources.present[index] {
                continue;
            }
            // Fixed mana, tapping abilities and waterbending compete for the
            // same tap. Non-tapping activations can still use this object later.
            let mut tap_total = 0;
            let mut net_tap_total = 0;
            if resources.untapped[index] {
                if let Some(output) = &source.fixed {
                    (tap_total, net_tap_total) = fixed_supply(index, output);
                }
                if pays_by_tap(source) {
                    tap_total = tap_total.max(1);
                    net_tap_total = net_tap_total.max(1);
                }
            }
            for (ability_index, ability) in source.abilities.iter().enumerate() {
                let uses = self.uses(resources, index, ability_index) as u64;
                if uses == 0 || ability.life_cost > life {
                    continue;
                }
                let (usable, net) = ability_supply(index, ability_index, ability);
                let output = uses * usable;
                let net_output = uses * net;
                if ability.tap_source {
                    net_tap_total = net_tap_total.max(net_output);
                    tap_total = tap_total.max(output);
                } else {
                    total += output;
                    net_total += net_output;
                }
            }
            total += tap_total;
            net_total += net_tap_total;
        }
        (total, net_total)
    }

    fn resources(&self) -> PaymentResources {
        PaymentResources {
            present: vec![true; self.sources.len()],
            untapped: self.sources.iter().map(|source| source.untapped).collect(),
            used_masks: Vec::new(),
        }
    }

    /// Counted-tap candidate sets from every ability and any extra tap
    /// obligations; membership in these sets is part of source identity.
    fn tap_candidate_sets<'a>(
        &'a self,
        extra: impl IntoIterator<Item = &'a Arc<Vec<PermanentId>>>,
    ) -> Vec<&'a Arc<Vec<PermanentId>>> {
        self.sources
            .iter()
            .flat_map(|source| &source.abilities)
            .filter(|ability| ability.tap_count > 0)
            .map(|ability| &ability.tap_candidates)
            .chain(extra)
            .collect()
    }

    /// An optimistic count, ignoring competition between different abilities.
    /// Every activation consumes presence, a tap, a limited use, or counted taps.
    /// Summing these upper bounds also covers multiple uses of one source.
    fn uses(&self, resources: &PaymentResources, index: usize, ability_index: usize) -> usize {
        if !resources.present[index] {
            return 0;
        }
        let ability = &self.sources[index].abilities[ability_index];
        if resources.is_used(index, ability_index)
            || (ability.tap_source && !resources.untapped[index])
        {
            return 0;
        }
        // A tapped, sacrificed or limited source activates at most once.
        let cap = if ability.tap_source || ability.sacrifice_source || ability.limited {
            1
        } else {
            usize::MAX
        };
        // Only a counted tap cost reads the board, and only as far as `cap`
        // activations need: a repeatable cost counts every candidate for its
        // optimistic production bound.
        let uses = if let Some(tap_count) = NonZeroUsize::new(ability.tap_count) {
            self.tap_candidates(resources, &ability.tap_candidates)
                .take(cap.saturating_mul(tap_count.get()))
                .count()
                / tap_count.get()
        } else {
            cap
        };
        assert_ne!(
            uses,
            usize::MAX,
            "mana activation must consume a finite resource"
        );
        uses
    }

    fn tap_candidates<'a>(
        &'a self,
        resources: &'a PaymentResources,
        candidates: &'a [PermanentId],
    ) -> impl Iterator<Item = usize> + 'a {
        self.sources
            .iter()
            .enumerate()
            .filter_map(|(index, source)| {
                (resources.present[index]
                    && resources.untapped[index]
                    && candidates.contains(&source.permanent))
                .then_some(index)
            })
    }

    fn witness(
        &self,
        state: PaymentSearchState,
        policy: SourcePolicy,
        visited: &mut PaymentSearch,
    ) -> Option<Vec<PaymentStep>> {
        #[cfg(feature = "benchmark")]
        crate::benchmark_payments::call();
        if state.obligations.is_empty() {
            return Some(Vec::new());
        }
        // No saved caller reads the final remainder. Exact affordability is
        // enough here; only a payment with later obligations needs to search
        // unit assignments. Execution still offers the native spending choices.
        if let [Obligation::Pay { cost, .. }] = state.obligations.as_slice() {
            if state.pool.can_pay(cost) {
                return Some(vec![PaymentStep::Spend]);
            }
        }
        // These transitions have no branch to memoize or bound. Advance the
        // same ordered obligations before scanning every source; the next
        // branching state still proves affordability and retains every step.
        match &state.obligations[0] {
            Obligation::Pay { cost, .. } if cost.mana_value == 0 => {
                let mut next = state;
                next.obligations.remove(0);
                next.prefix = None;
                return self.witness(next, policy, visited).map(|mut path| {
                    path.insert(0, PaymentStep::Spend);
                    path
                });
            }
            Obligation::Tap { remaining: 0, .. } => {
                let mut next = state;
                next.obligations.remove(0);
                return self.witness(next, policy, visited);
            }
            Obligation::Output { choices, life_cost } if choices.len() == 1 => {
                let life = state.life.checked_sub(*life_cost)?;
                let mut next = state;
                let Obligation::Output { choices, .. } = next.obligations.remove(0) else {
                    unreachable!()
                };
                next.life = life;
                next.pool.add(&choices[0]);
                return self.witness(next, policy, visited).map(|mut path| {
                    path.insert(0, PaymentStep::Output(0));
                    path
                });
            }
            _ => {}
        }
        // Presence and untapped resources only decrease during payment. Once
        // any saved counted tap lacks candidates, no production or spending
        // order can restore it. Reject before exploring the earlier costs;
        // checking only when that tap becomes first repeats doomed payments.
        if state.obligations.iter().any(|obligation| {
            matches!(obligation, Obligation::Tap { candidates, remaining }
                if self.tap_candidates(&state.resources, candidates).take(*remaining).count() < *remaining)
        }) {
            return None;
        }
        // Bound the whole pending payment, not just its first obligation:
        // otherwise an impossible later cost retries every way to fund an
        // earlier one.
        let purpose_indices = visited.pay_purposes(self, &state.obligations);
        // Whether some pending cost lets tapping `source` pay {1} of it.
        let pays_by_tap = |source: &ManaSource| {
            state.obligations.iter().any(|obligation| {
                matches!(obligation, Obligation::Pay { taps, .. } if source.pays_by_tap(*taps))
            })
        };
        // A second conservation bound counts *all* pool classes, even those
        // unusable by pending costs, and only the net gain of future
        // activations. Restricted seeds can fund filters without being lost
        // from this accounting. Already saved outputs remain gross supply;
        // their upfront fees are explicit pending Pay obligations.
        let (gross, net) = if let [only] = purpose_indices.as_slice() {
            let supply = &visited.purposes[*only];
            self.source_bounds(
                &state.resources,
                u32::MAX,
                |i, _| (supply.sources[i].fixed_usable, supply.sources[i].fixed_net),
                |i, ai, _| {
                    let ab = &supply.abilities_of(i)[ai];
                    (ab.max_usable, ab.max_net)
                },
                pays_by_tap,
            )
        } else {
            self.source_bounds(
                &state.resources,
                u32::MAX,
                |_, output| {
                    (
                        visited.usable_total_cached(&purpose_indices, output),
                        u64::from(output.total()),
                    )
                },
                |_, _, ability| {
                    let max_usable = ability
                        .outputs
                        .iter()
                        .map(|output| visited.usable_total_cached(&purpose_indices, output))
                        .max()
                        .unwrap_or(0);
                    (max_usable, ability.max_net())
                },
                pays_by_tap,
            )
        };
        let mut total = visited.usable_total(&purpose_indices, &state.pool) + gross;
        let mut net_total = u64::from(state.pool.total()) + net;
        for obligation in &state.obligations {
            if let Obligation::Output { choices, .. } = obligation {
                total += choices
                    .iter()
                    .map(|mana| visited.usable_total(&purpose_indices, mana))
                    .max()
                    .unwrap_or(0);
                net_total += choices
                    .iter()
                    .map(|output| u64::from(output.total()))
                    .max()
                    .unwrap_or(0);
            }
        }
        let owed = state
            .obligations
            .iter()
            .filter_map(|obligation| match obligation {
                Obligation::Pay { cost, .. } => Some(u64::from(cost.mana_value)),
                _ => None,
            })
            .sum::<u64>();
        #[cfg(test)]
        if visited.gross_bound_only {
            net_total = u64::MAX;
        }
        // Only states that need exploration need an owned visited key.
        if total < owed
            || net_total < owed
            || (visited.bounds_colors()
                && !self.colors_coverable(&state, &purpose_indices, visited))
        {
            return None;
        }
        let state = visited.insert(self, state)?;
        #[cfg(feature = "benchmark")]
        crate::benchmark_payments::state();
        match &state.obligations[0] {
            Obligation::Tap {
                candidates,
                remaining,
            } => {
                for index in self.tap_candidates(&state.resources, candidates) {
                    if visited.has_tried_sibling(&state.resources, index) {
                        continue;
                    }
                    let mut next = (*state).clone();
                    next.resources.untapped[index] = false;
                    next.obligations[0] = Obligation::Tap {
                        candidates: candidates.clone(),
                        remaining: *remaining - 1,
                    };
                    if let Some(mut path) = self.witness(next, policy, visited) {
                        path.insert(0, PaymentStep::Tap(self.sources[index].object));
                        return Some(path);
                    }
                }
            }
            Obligation::Output {
                choices, life_cost, ..
            } => {
                let life = state.life.checked_sub(*life_cost)?;
                for (mode, output) in choices.iter().enumerate() {
                    let mut pool = state.pool.clone();
                    pool.add(output);
                    if let Some(mut path) = self.witness(
                        PaymentSearchState {
                            pool,
                            life,
                            resources: state.resources.clone(),
                            obligations: state.obligations[1..].to_vec(),
                            prefix: state.prefix,
                        },
                        policy,
                        visited,
                    ) {
                        path.insert(0, PaymentStep::Output(mode));
                        return Some(path);
                    }
                }
            }
            Obligation::Pay { cost, taps } => {
                let pooled = state.pool.can_pay(cost);
                if !pooled
                    && !self.cost_coverable(
                        &state,
                        cost,
                        *taps,
                        &mut visited.purposes[purpose_indices[0]],
                    )
                {
                    return None;
                }
                // Pool the whole current cost before spending starts;
                // production cannot interleave with its unit choices. If the
                // pool cannot cover it, no first unit can leave a payable rest.
                // In pure feasibility search, classes sharing a `ManaRestriction`
                // (differing only in `expiry`/`SpendEffect`) leave identically
                // feasible pools, so each `(restriction, color)` is tried once.
                if pooled {
                    if let Some((allowed, remaining)) = cost.pop_next_pip() {
                        let mut seen_unrestricted = ManaTypeSet::EMPTY;
                        let mut seen_restricted: Vec<(&ManaRestriction, ManaTypeSet)> = Vec::new();
                        for (class_index, restriction, amounts) in
                            std::iter::once((None, None, &state.pool.mana)).chain(
                                state
                                    .pool
                                    .classes()
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, (class, _))| class.allows(&cost.purpose))
                                    .map(|(idx, (class, amounts))| {
                                        (Some(idx), class.restriction.as_ref(), amounts)
                                    }),
                            )
                        {
                            for color in allowed.iter() {
                                if amounts[color] == 0 {
                                    continue;
                                }
                                let seen = match restriction {
                                    None => &mut seen_unrestricted,
                                    Some(r) => match seen_restricted
                                        .iter()
                                        .position(|(prev, _)| *prev == r)
                                    {
                                        Some(i) => &mut seen_restricted[i].1,
                                        None => {
                                            seen_restricted.push((r, ManaTypeSet::EMPTY));
                                            &mut seen_restricted.last_mut().unwrap().1
                                        }
                                    },
                                };
                                if seen.contains(color) {
                                    continue;
                                }
                                seen.insert(ManaTypeSet::bit(color));
                                let mut pool = state.pool.clone();
                                pool.spend_one(class_index, color);
                                if !pool.can_pay(&remaining) {
                                    continue;
                                }
                                let mut next = PaymentSearchState {
                                    pool,
                                    resources: state.resources.clone(),
                                    obligations: state.obligations.clone(),
                                    life: state.life,
                                    prefix: state.prefix,
                                };
                                next.obligations[0] = Obligation::Pay {
                                    cost: remaining.clone(),
                                    taps: *taps,
                                };
                                // The zero-cost arm owns completion and the Spend boundary,
                                // including when this was the last unit of a nonzero cost.
                                if let Some(path) = self.witness(next, policy, visited) {
                                    return Some(path);
                                }
                            }
                        }
                    }
                }
                if state.prefix == Some(0) {
                    return None;
                }
                for (index, source) in self.sources.iter().enumerate() {
                    if !state.resources.present[index]
                        || visited.has_tried_sibling(&state.resources, index)
                    {
                        continue;
                    }
                    if state.resources.untapped[index] {
                        if let Some(output) = &source.fixed {
                            let mut next = (*state).clone();
                            next.resources.untapped[index] = false;
                            next.pool.add(output);
                            next.prefix = next.prefix.map(|n| n - 1);
                            if let Some(mut path) = self.witness(next, policy, visited) {
                                path.insert(0, PaymentStep::Fixed(source.permanent));
                                return Some(path);
                            }
                        }
                    }
                    // Chosen activations cannot nest before the automatic
                    // prefix's first Spend, so its budget needs no depth.
                    if policy == SourcePolicy::FixedOnly
                        || (policy == SourcePolicy::AutomaticPrefix && state.prefix.is_some())
                    {
                        continue;
                    }
                    for (ability_index, ability) in source.abilities.iter().enumerate() {
                        if self.uses(&state.resources, index, ability_index) == 0 {
                            continue;
                        }
                        let mut obligations = Self::activation_obligations(ability);
                        obligations.extend_from_slice(&state.obligations);
                        let mut next = PaymentSearchState {
                            resources: state.resources.clone(),
                            pool: state.pool.clone(),
                            life: state.life,
                            obligations,
                            prefix: None,
                        };
                        if ability.tap_source {
                            next.resources.untapped[index] = false;
                        }
                        if ability.sacrifice_source {
                            next.resources.present[index] = false;
                        }
                        if ability.limited {
                            next.resources.use_ability(index, ability_index);
                        }
                        if let Some(mut path) = self.witness(next, policy, visited) {
                            path.insert(
                                0,
                                PaymentStep::Activate(source.permanent, ability.ability_index),
                            );
                            return Some(path);
                        }
                    }
                    if source.pays_by_tap(*taps)
                        && state.resources.untapped[index]
                        && cost.generic() > 0
                    {
                        let mut next = (*state).clone();
                        next.resources.untapped[index] = false;
                        next.obligations[0] = Obligation::Pay {
                            cost: cost.reduced_generic(1),
                            taps: *taps,
                        };
                        next.prefix = next.prefix.map(|n| n - 1);
                        if let Some(mut path) = self.witness(next, policy, visited) {
                            path.insert(0, PaymentStep::Waterbend(source.permanent));
                            return Some(path);
                        }
                    }
                }
            }
        }
        None
    }

    /// Optimistic color/restriction supply for the current cost. Shared
    /// resources are relaxed and every output mode contributes its colors;
    /// the whole-payment and color-set bounds handle competing obligations.
    fn cost_coverable(
        &self,
        state: &PaymentSearchState,
        cost: &ManaCost,
        taps: Taps,
        supply: &mut PurposeSupply,
    ) -> bool {
        let (mut upper, mut usable) =
            optimistic_supply(std::slice::from_ref(&state.pool), |class| {
                supply.allows(class)
            });
        for (index, source) in self.sources.iter().enumerate() {
            if !state.resources.present[index] {
                continue;
            }
            let src = &supply.sources[index];
            if state.resources.untapped[index] {
                if source.fixed.is_some() {
                    for (slot, &more) in upper.iter_mut().zip(&src.fixed_upper) {
                        *slot = slot.saturating_add(more);
                    }
                    usable += src.fixed_usable;
                }
                if source.pays_by_tap(taps) {
                    upper[5] = upper[5].saturating_add(1);
                    usable += 1;
                }
            }
            for (ability_index, ab) in supply.abilities_of(index).iter().enumerate() {
                let uses = self.uses(&state.resources, index, ability_index);
                if uses == 0 {
                    continue;
                }
                let mul = uses.min(u8::MAX as usize) as u8;
                for (slot, &more) in upper.iter_mut().zip(&ab.upper) {
                    *slot = slot.saturating_add(more.saturating_mul(mul));
                }
                usable += uses as u64 * ab.max_usable;
            }
        }
        let optimistic = Mana {
            mana: upper,
            classes: None,
        };
        usable >= u64::from(cost.mana_value) && optimistic.can_pay(cost)
    }

    /// Hall's condition over the colors owed: for every set S of the mana
    /// kinds named by pending `Pay` pips (W/U/B/R/G and {C}), the pips payable
    /// only from S need that many units of mana in S. A pip is confined to S
    /// when it is a colored or {C} pip of S or a hybrid pip with both halves
    /// in S; generic pips and waterbend taps are left to the total bound.
    ///
    /// Supply is optimistic in the same way as the total bound: mana counts
    /// when the payment could spend it (`PaymentSearch::usable_for`), one tap yields the best
    /// of the fixed output and the tapping abilities, and non-tapping
    /// abilities and pending outputs add their best choice. Unlike the
    /// per-cost check, each use contributes one output choice, not the union
    /// of its modes, so one "U or R" source cannot cover {U}{R}. Mana is never
    /// spent as another color during payment (`Mana::spending_choices`), so
    /// a failed set has no witness and pruning it keeps the first witness.
    fn colors_coverable(
        &self,
        state: &PaymentSearchState,
        purpose_indices: &[usize],
        search: &mut PaymentSearch,
    ) -> bool {
        #[cfg(test)]
        if search.total_bound_only {
            return true;
        }
        const KINDS: usize = 6;
        let mut pips = [0_u64; KINDS];
        let mut hybrid: Vec<ManaTypeSet> = Vec::new();
        let mut owed_kinds = ManaTypeSet::EMPTY;
        let mut pays = 0;
        for obligation in &state.obligations {
            let Obligation::Pay { cost, .. } = obligation else {
                continue;
            };
            pays += 1;
            for (kind, owed) in pips.iter_mut().enumerate() {
                *owed += u64::from(cost.cost[kind]);
                if cost.cost[kind] > 0 {
                    owed_kinds.insert(ManaTypeSet::bit(kind));
                }
            }
            for pair in &cost.hybrid {
                let pip = ManaTypeSet::hybrid(*pair);
                owed_kinds.insert(pip);
                hybrid.push(pip);
            }
        }
        // One color owed by one cost is already bounded by its own check.
        if owed_kinds.is_empty() || (owed_kinds.len() < 2 && pays < 2) {
            return true;
        }
        // Each group is one exclusive choice: the supply of S is the pool
        // plus, per group, its best member counted inside S.
        let mut choices: Vec<[u64; KINDS]> = Vec::new();
        let mut groups: Vec<usize> = Vec::new();
        for (index, source) in self.sources.iter().enumerate() {
            if !state.resources.present[index] {
                continue;
            }
            // Fixed mana and tapping abilities compete for the same tap.
            let tap_group = choices.len();
            if state.resources.untapped[index] {
                if let Some(output) = &source.fixed {
                    choices.push(search.usable_for(purpose_indices, output));
                }
            }
            let mut separate: Vec<(u64, &[Mana])> = Vec::new();
            for (ability_index, ability) in source.abilities.iter().enumerate() {
                let uses = self.uses(&state.resources, index, ability_index) as u64;
                if uses == 0 {
                    continue;
                }
                if ability.tap_source {
                    choices.extend(
                        ability
                            .outputs
                            .iter()
                            .map(|output| search.usable_for(purpose_indices, output)),
                    );
                } else {
                    separate.push((uses, &ability.outputs));
                }
            }
            if choices.len() > tap_group {
                groups.push(choices.len());
            }
            for (uses, outputs) in separate {
                choices.extend(outputs.iter().map(|output| {
                    search
                        .usable_for(purpose_indices, output)
                        .map(|amount| amount.saturating_mul(uses))
                }));
                groups.push(choices.len());
            }
        }
        for obligation in &state.obligations {
            if let Obligation::Output { choices: modes, .. } = obligation {
                choices.extend(
                    modes
                        .iter()
                        .map(|output| search.usable_for(purpose_indices, output)),
                );
                groups.push(choices.len());
            }
        }
        let pool = search.usable_for(purpose_indices, &state.pool);
        let within = |amounts: &[u64; KINDS], set: ManaTypeSet| -> u64 {
            set.iter()
                .fold(0_u64, |sum, kind| sum.saturating_add(amounts[kind]))
        };
        // Every non-empty subset of the owed kinds, largest first.
        for set in owed_kinds.nonempty_subsets() {
            let owed = within(&pips, set)
                + hybrid.iter().filter(|pip| pip.is_subset_of(set)).count() as u64;
            let mut supply = within(&pool, set);
            let mut start = 0;
            for &end in &groups {
                if supply >= owed {
                    break;
                }
                supply = supply.saturating_add(
                    choices[start..end]
                        .iter()
                        .map(|choice| within(choice, set))
                        .max()
                        .unwrap_or(0),
                );
                start = end;
            }
            if supply < owed {
                return false;
            }
        }
        true
    }

    fn activation_obligations(ability: &ManaActivation) -> Vec<Obligation> {
        let mut obligations = Vec::new();
        if ability.tap_count > 0 {
            obligations.push(Obligation::Tap {
                candidates: ability.tap_candidates.clone(),
                remaining: ability.tap_count,
            });
        }
        obligations.push(Obligation::mana(ability.cost.clone()));
        obligations.push(Obligation::Output {
            choices: ability.outputs.clone(),
            life_cost: ability.life_cost,
        });
        obligations
    }

    /// Consume one selected tap before searching the remaining obligations.
    /// Ordinary and nested activations use the same reserved resources and
    /// unpaid count, so neither can offer an object promised to another cost.
    fn plan_after_tap(
        &self,
        pool: Mana,
        obligations: Vec<Obligation>,
        permanent: PermanentId,
    ) -> Option<Vec<PaymentStep>> {
        #[cfg(feature = "benchmark")]
        crate::benchmark_payments::begin_plan(self.sources.len());
        let path = self.cached_plan(pool, obligations, SourcePolicy::All, Some(permanent));
        #[cfg(feature = "benchmark")]
        crate::benchmark_payments::end_plan(&path);
        path
    }

    fn search_after_tap(
        &self,
        pool: Mana,
        mut obligations: Vec<Obligation>,
        permanent: PermanentId,
    ) -> Option<Vec<PaymentStep>> {
        let Obligation::Tap {
            candidates,
            remaining,
        } = obligations.first_mut()?
        else {
            return None;
        };
        let mut resources = self.resources();
        let index = self
            .tap_candidates(&resources, candidates)
            .find(|index| self.sources[*index].permanent == permanent)?;
        *remaining = remaining.checked_sub(1)?;
        resources.untapped[index] = false;
        let mut search = PaymentSearch::new(&obligations);
        self.witness(
            PaymentSearchState {
                resources,
                pool,
                life: self.life,
                obligations,
                prefix: None,
            },
            SourcePolicy::All,
            &mut search,
        )
    }

    /// The first witness for `obligations`, or `None` when they cannot be paid.
    fn plan(
        &self,
        pool: Mana,
        obligations: Vec<Obligation>,
        policy: SourcePolicy,
    ) -> Option<Vec<PaymentStep>> {
        #[cfg(feature = "benchmark")]
        crate::benchmark_payments::begin_plan(self.sources.len());
        let path = self.cached_plan(pool, obligations, policy, None);
        #[cfg(feature = "benchmark")]
        crate::benchmark_payments::end_plan(&path);
        path
    }

    /// The over-tap criterion, asked of the board after the payment's taps:
    /// without one source's complete `output` (triggered bonuses included),
    /// `pool` still holds `cost` and the saved callers in `tail` stay payable.
    /// Callers first check that the pool exceeds the cost at all.
    fn is_surplus(&self, pool: &Mana, output: &Mana, cost: &ManaCost, tail: &[Obligation]) -> bool {
        let mut remaining = pool.clone();
        remaining.spend(output);
        if !remaining.can_pay(cost) {
            return false;
        }
        if tail.is_empty() {
            return true;
        }
        let mut obligations = vec![Obligation::mana(cost.clone())];
        obligations.extend_from_slice(tail);
        self.payable(remaining, obligations, SourcePolicy::All)
    }

    fn cached_plan(
        &self,
        pool: Mana,
        obligations: Vec<Obligation>,
        policy: SourcePolicy,
        tap: Option<PermanentId>,
    ) -> Option<Vec<PaymentStep>> {
        // Arithmetic plans are cheaper than retaining and comparing a full
        // board key. Keep the cache for identity-preserving searches.
        if tap.is_none() {
            if let Some(direct) = self.direct(&obligations, policy) {
                let path = self.direct_witness(pool.clone(), &direct);
                debug_assert_eq!(path, self.search(pool, obligations.clone(), policy));
                return path;
            }
        }
        let uncached = || match tap {
            Some(permanent) => self.search_after_tap(pool.clone(), obligations.clone(), permanent),
            None => {
                #[cfg(test)]
                tests::SEARCHED.with(|searched| searched.set(searched.get() + 1));
                self.search(pool.clone(), obligations.clone(), policy)
            }
        };
        let Some(cache) = self.cache else {
            return uncached();
        };
        let cached = cache
            .borrow()
            .entries
            .iter()
            .rev()
            .find(|entry| {
                entry.life == self.life
                    && entry.policy == policy
                    && entry.tap == tap
                    && entry.pool == pool
                    && entry.obligations == obligations
                    && (Arc::ptr_eq(&entry.sources, &self.sources) || entry.sources == self.sources)
            })
            .map(|entry| entry.path.clone());
        #[cfg(feature = "benchmark")]
        crate::benchmark_payments::cache(cached.is_some());
        if let Some(path) = cached {
            debug_assert_eq!(
                path,
                uncached(),
                "cached payment witness differs from exact query"
            );
            return path;
        }
        let path = uncached();
        let mut cache = cache.borrow_mut();
        if cache.entries.len() == 64 {
            cache.entries.pop_front();
        }
        cache.entries.push_back(Arc::new(CachedPayment {
            sources: self.sources.clone(),
            life: self.life,
            pool,
            obligations,
            policy,
            tap,
            path: path.clone(),
        }));
        path
    }

    /// The pool already includes the tap's output. Reserve its untapped
    /// resource before asking whether all remaining costs can be paid.
    fn payable_after_tap(
        &self,
        permanent: PermanentId,
        pool: Mana,
        obligations: Vec<Obligation>,
    ) -> bool {
        self.after_tap(permanent)
            .payable(pool, obligations, SourcePolicy::All)
    }

    /// This model once `permanent` is tapped for something else. Retains the
    /// shared `PaymentCache` because cache keys include the reserved `sources`.
    fn after_tap(&self, permanent: PermanentId) -> Self {
        let mut after = PaymentModel {
            cache: self.cache,
            sources: self.sources.clone(),
            life: self.life,
        };
        after.reserve(SourceReservation::tap(permanent));
        after
    }

    /// `pool` plus every untapped fixed source's output.
    fn fixed_total(&self, pool: &Mana) -> Mana {
        let mut total = pool.clone();
        for source in self.sources.iter().filter(|s| s.untapped) {
            if let Some(output) = &source.fixed {
                total.add(output);
            }
        }
        total
    }

    /// Whether `obligations` can be paid: `plan(..).is_some()` without
    /// building the witness when the direct answer applies, or when fixed
    /// sources alone already suffice under `SourcePolicy::All`.
    fn payable(&self, pool: Mana, obligations: Vec<Obligation>, policy: SourcePolicy) -> bool {
        if let Some(direct) = self.direct(&obligations, policy) {
            let payable = self.direct_payable(&pool, &direct);
            debug_assert_eq!(
                payable,
                self.search(pool, obligations.clone(), policy).is_some(),
                "direct payment answer differs from the search"
            );
            return payable;
        }
        if policy == SourcePolicy::All && self.fixed_sufficient(&pool, self.life, &obligations) {
            debug_assert!(
                self.search(pool, obligations.clone(), policy).is_some(),
                "fixed-sufficient payment fast-path differed from the search"
            );
            return true;
        }
        self.plan(pool, obligations, policy).is_some()
    }

    /// Direct-class obligation shape, independent of whether `self.sources`
    /// has activated mana abilities.
    fn direct_shape(obligations: &[Obligation], allow_split: bool) -> Option<DirectPayment<'_>> {
        match obligations {
            [Obligation::Pay {
                cost,
                taps: Taps::None,
            }] => Some(DirectPayment {
                cost,
                split: None,
                taps: Taps::None,
            }),
            [Obligation::Pay {
                cost,
                taps: Taps::None,
            }, Obligation::Pay { cost: split, taps }]
                if allow_split
                    && *taps != Taps::None
                    && split.purpose == cost.purpose
                    && split.mana_value == split.generic() =>
            {
                Some(DirectPayment {
                    cost,
                    split: Some(split),
                    taps: *taps,
                })
            }
            _ => None,
        }
    }

    /// O(N) sufficient condition for `SourcePolicy::All`: returns `true` only
    /// when `pool` and untapped fixed sources (plus waterbend/keyword taps on
    /// a generic split) already pay `obligations` without activating any mana
    /// ability. Conservatively returns `false` whenever activated abilities
    /// may be needed.
    fn fixed_sufficient(&self, pool: &Mana, life: u32, mut obligations: &[Obligation]) -> bool {
        while let Some(first) = obligations.first() {
            match first {
                Obligation::Pay { cost, .. } if cost.mana_value == 0 => {
                    obligations = &obligations[1..];
                }
                Obligation::Tap { remaining: 0, .. } => {
                    obligations = &obligations[1..];
                }
                _ => break,
            }
        }
        if obligations.is_empty() {
            return true;
        }
        if let Some(direct) = Self::direct_shape(obligations, true) {
            return self.direct_payable(pool, &direct);
        }
        if let Some((Obligation::Output { choices, life_cost }, rest)) = obligations.split_first() {
            if *life_cost > life || choices.is_empty() {
                return false;
            }
            let remaining_life = life - *life_cost;
            if rest.is_empty() {
                return true;
            }
            if let Some(direct) = Self::direct_shape(rest, true) {
                return self.direct_covers(pool, choices, &direct, |_| false);
            }
            return choices.iter().any(|choice| {
                let mut next = pool.clone();
                next.add(choice);
                self.fixed_sufficient(&next, remaining_life, rest)
            });
        }
        if let [Obligation::Pay {
            cost: fee,
            taps: Taps::None,
        }, Obligation::Output { choices, life_cost }, rest @ ..] = obligations
        {
            if *life_cost > life || choices.is_empty() {
                return false;
            }
            if rest.is_empty() {
                return self.direct_payable(
                    pool,
                    &DirectPayment {
                        cost: fee,
                        split: None,
                        taps: Taps::None,
                    },
                );
            }
            if let Some(rest_direct) = Self::direct_shape(rest, true) {
                return self.fixed_filter_payable(pool, fee, choices, &rest_direct);
            }
        }
        false
    }

    /// Whether `pool`, the non-skipped untapped fixed sources (plus split taps),
    /// and optionally one mode from `choices` cover `payment`.
    fn direct_covers(
        &self,
        pool: &Mana,
        choices: &[Mana],
        payment: &DirectPayment,
        skip: impl Fn(usize) -> bool,
    ) -> bool {
        let has_split = payment.split.is_some();
        if !has_split && pool.can_pay(payment.cost) {
            return true;
        }
        let mut base = pool.clone();
        let mut spare_taps = 0_u32;
        let mut added = false;
        for (index, source) in self.sources.iter().enumerate() {
            if !source.untapped || skip(index) {
                continue;
            }
            if let Some(output) = &source.fixed {
                base.add(output);
                added = true;
            }
            if has_split
                && source.pays_by_tap(payment.taps)
                && source
                    .fixed
                    .as_ref()
                    .is_none_or(|output| output.usable_total(payment.cost) == 0)
            {
                spare_taps += 1;
            }
        }
        let covers = |total: &Mana| -> bool {
            total.can_pay(payment.cost)
                && (!has_split
                    || u32::from(total.usable_total(payment.cost))
                        .checked_sub(payment.cost.mana_value.into())
                        .is_some_and(|slack| slack + spare_taps >= payment.split_generic()))
        };
        if (has_split || added) && covers(&base) {
            return true;
        }
        choices.iter().any(|choice| {
            let mut total = base.clone();
            total.add(choice);
            covers(&total)
        })
    }

    /// Constructive O(N) check for `[Pay(fee), Output(choices), rest]` using
    /// only fixed sources (plus waterbend/keyword taps on `rest`'s split):
    /// pay `fee` from `pool` plus fixed taps, add one choice from `choices`,
    /// and check whether the remaining fixed taps cover `rest`.
    fn fixed_filter_payable(
        &self,
        pool: &Mana,
        fee: &ManaCost,
        choices: &[Mana],
        rest: &DirectPayment,
    ) -> bool {
        let mut after_fee = pool.clone();
        let mut tapped_bits = 0_u64;
        let mut tapped_wide = Vec::new();
        for (index, source) in self.sources.iter().enumerate() {
            if after_fee.can_pay(fee) {
                break;
            }
            if !source.untapped {
                continue;
            }
            let Some(output) = &source.fixed else {
                continue;
            };
            if output.usable_total(fee) == 0 {
                continue;
            }
            after_fee.add(output);
            if index < 64 {
                tapped_bits |= 1_u64 << index;
            } else {
                tapped_wide.push(index);
            }
        }
        if !after_fee.can_pay(fee) {
            return false;
        }
        let mut rem = fee.clone();
        while rem.mana_value > 0 {
            let Some(choice) = after_fee.spending_choices(&rem).into_iter().find(|c| {
                let mut next = after_fee.clone();
                next.spend(&c.unit);
                next.can_pay(&c.remaining)
            }) else {
                return false;
            };
            after_fee.spend(&choice.unit);
            rem = choice.remaining;
        }
        self.direct_covers(&after_fee, choices, rest, |index| {
            if index < 64 {
                tapped_bits & (1_u64 << index) != 0
            } else {
                tapped_wide.contains(&index)
            }
        })
    }

    /// The question as the direct answer reads it, or `None` when only the
    /// search is exact.
    ///
    /// Direct class: the obligations are one mana cost, optionally followed
    /// by one generic-only split of the same purpose that taps may pay
    /// (`help_obligations`), and no source offers a searched ability.
    /// Tapping a fixed source then only adds mana, and mana of one purpose
    /// only adds ways to pay, so affordability is one sum and the search's
    /// first witness is a single pass in source order. `FixedOnly` never
    /// searches abilities, so a lone cost is direct whatever the sources
    /// offer; it never taps for waterbend, so a split is not. The automatic
    /// prefix minimizes taps instead of taking them in order and stays with
    /// the search, as do counted taps, pending outputs and life.
    fn direct<'a>(
        &self,
        obligations: &'a [Obligation],
        policy: SourcePolicy,
    ) -> Option<DirectPayment<'a>> {
        let direct = Self::direct_shape(obligations, policy == SourcePolicy::All)?;
        let searched = match policy {
            SourcePolicy::FixedOnly => false,
            SourcePolicy::All => self
                .sources
                .iter()
                .any(|source| !source.abilities.is_empty()),
            SourcePolicy::AutomaticPrefix => true,
        };
        (!searched).then_some(direct)
    }

    fn direct_taps<'a>(
        &'a self,
        payment: &'a DirectPayment,
    ) -> impl Iterator<Item = DirectTap<'a>> + Clone {
        self.sources.iter().map(|source| {
            let output = source.fixed.as_ref().filter(|_| source.untapped);
            DirectTap {
                permanent: source.permanent,
                output,
                mana: output.map_or(0, |output| output.usable_total(payment.cost).into()),
                waterbend: source.untapped && source.pays_by_tap(payment.taps),
            }
        })
    }

    /// Affordability by arithmetic, as `(slack, spare taps)`: every fixed
    /// source tapped for mana pays the cost with `slack` usable mana left
    /// over, and that slack plus the spare taps covers the split. `None` when
    /// the payment cannot be made.
    fn direct_margin<'a>(
        pool: &Mana,
        taps: impl Iterator<Item = DirectTap<'a>> + Clone,
        payment: &DirectPayment,
    ) -> Option<(u32, u32)> {
        let mut total = pool.clone();
        for output in taps.clone().filter_map(|tap| tap.output) {
            total.add(output);
        }
        if !total.can_pay(payment.cost) {
            return None;
        }
        let slack = u32::from(total.usable_total(payment.cost))
            .checked_sub(payment.cost.mana_value.into())?;
        let spare_taps = taps.filter(|tap| tap.spare()).count() as u32;
        (slack + spare_taps >= payment.split_generic()).then_some((slack, spare_taps))
    }

    fn direct_payable(&self, pool: &Mana, payment: &DirectPayment) -> bool {
        self.direct_covers(pool, &[], payment, |_| false)
    }

    /// The witness the search returns first, built without searching.
    ///
    /// The search spends as soon as the pool pays the cost and the split can
    /// still be covered afterwards, and until then taps the first fixed source
    /// whose tap leaves the whole payment possible. That skips only a
    /// worthless mana tap of a waterbend source the split needs. After the
    /// spend it covers the split the same way, preferring each source's mana
    /// tap to its waterbend tap. A feasible state stays feasible along this
    /// path, so the search never backtracks out of it and one pass in source
    /// order visits every decision it makes.
    fn direct_witness(&self, mut pool: Mana, payment: &DirectPayment) -> Option<Vec<PaymentStep>> {
        let cost = payment.cost;
        let owed = u32::from(cost.mana_value);
        let mut need = payment.split_generic();
        let taps = self.direct_taps(payment).collect::<Vec<_>>();
        // Tapping a source for mana moves its output into the pool and
        // leaves the slack alone.
        let (slack, mut spare_taps) = Self::direct_margin(&pool, taps.iter().copied(), payment)?;
        // What the untapped sources can still add towards the split.
        let mut ahead = taps.iter().map(|tap| tap.worth()).sum::<u32>();
        let mut tapped = vec![false; taps.len()];
        let mut path = Vec::new();
        let mut index = 0;
        while !(pool.can_pay(cost) && u32::from(pool.usable_total(cost)) - owed + ahead >= need) {
            let tap = *taps
                .get(index)
                .expect("a payable cost is paid before its fixed sources run out");
            let needed_for_split = tap.spare() && slack + spare_taps <= need;
            if let Some(output) = tap.output.filter(|_| !needed_for_split) {
                pool.add(output);
                ahead -= tap.worth();
                spare_taps -= u32::from(tap.spare());
                tapped[index] = true;
                path.push(PaymentStep::Fixed(tap.permanent));
            }
            index += 1;
        }
        path.push(PaymentStep::Spend);
        let Some(split) = payment.split else {
            return Some(path);
        };
        // Both costs have the same purpose and the split is generic, so every
        // legal spending choice leaves the same usable total. The witness
        // never chooses units or commits a hypothetical pool remainder.
        let mut have = u32::from(pool.usable_total(split)) - owed;
        for (tap, tapped) in taps.iter().zip(tapped) {
            if have >= need {
                break;
            }
            if tapped {
                continue;
            }
            let rest = ahead - tap.worth();
            match tap.output {
                Some(_) if have + tap.mana + rest >= need => {
                    have += tap.mana;
                    path.push(PaymentStep::Fixed(tap.permanent));
                }
                _ if tap.waterbend => {
                    need -= 1;
                    path.push(PaymentStep::Waterbend(tap.permanent));
                }
                _ => continue,
            }
            ahead = rest;
        }
        path.push(PaymentStep::Spend);
        Some(path)
    }

    /// Before AutomaticPrefix's first Spend, only fixed production and
    /// waterbend can add payment capacity. Even ignoring colors and counted
    /// tap competition, the largest outputs must cover the pool's shortfall.
    /// Budgets below this lower bound have no witness. Other leading
    /// obligations retain the ordinary search's zero start.
    fn prefix_lower_bound(&self, pool: &Mana, obligations: &[Obligation]) -> usize {
        let Some(Obligation::Pay { cost, taps }) = obligations.first() else {
            return 0;
        };
        let mut need =
            u64::from(cost.mana_value).saturating_sub(u64::from(pool.usable_total(cost)));
        if need == 0 {
            return 0;
        }
        let mut outputs: Vec<u64> = self
            .sources
            .iter()
            .filter(|source| source.untapped)
            .map(|source| {
                let fixed = source
                    .fixed
                    .as_ref()
                    .map_or(0, |output| u64::from(output.usable_total(cost)));
                fixed.max(u64::from(source.pays_by_tap(*taps) && cost.generic() > 0))
            })
            .collect();
        outputs.sort_unstable_by(|left, right| right.cmp(left));
        for (count, output) in outputs.into_iter().enumerate() {
            need = need.saturating_sub(output);
            if need == 0 {
                return count + 1;
            }
        }
        // No automatic prefix can fill the shortfall.
        self.sources.len() + 1
    }

    fn search(
        &self,
        pool: Mana,
        obligations: Vec<Obligation>,
        policy: SourcePolicy,
    ) -> Option<Vec<PaymentStep>> {
        let mut search = PaymentSearch::new(&obligations);
        // Offers need one executable witness, not a proof of the fewest taps.
        // Only a nested automatic prefix minimizes early commitments.
        if policy != SourcePolicy::AutomaticPrefix {
            return self.witness(
                PaymentSearchState {
                    resources: self.resources(),
                    pool,
                    life: self.life,
                    obligations,
                    prefix: None,
                },
                policy,
                &mut search,
            );
        }
        // Minimize taps before the first Spend without starving a saved caller.
        for budget in self.prefix_lower_bound(&pool, &obligations)..=self.sources.len() {
            search.clear_visited();
            if let Some(path) = self.witness(
                PaymentSearchState {
                    resources: self.resources(),
                    pool: pool.clone(),
                    life: self.life,
                    obligations: obligations.clone(),
                    prefix: Some(budget),
                },
                policy,
                &mut search,
            ) {
                return Some(path);
            }
        }
        None
    }
}

/// A pending payment's hold on its own source, from what it still owes.
fn owed_reservation(payment: &Payment) -> Option<SourceReservation> {
    let Operation::Activation {
        source: ActivationSource::Permanent(permanent),
        ..
    } = payment.operation
    else {
        return None;
    };
    let owes = |how| payment.owes(&CostComponent::Source(how));
    Some(SourceReservation::owed(
        permanent,
        owes(How::Tapped),
        owes(How::Sacrificed) || owes(How::Exiled) || owes(How::Returned),
    ))
}

impl Game {
    /// Mana abilities resolve at once (CR 605.3a), so an output that depends
    /// on the game is fixed when the ability is offered: a branch between two
    /// immediate outputs, or X mana of one chosen color. Activation costs do
    /// not change these readings for current content.
    pub(crate) fn current_mana_output(
        &self,
        effect: &Effect,
        controller: PlayerId,
        chosen: Option<&crate::state::permanent::EntryChoice>,
    ) -> Option<Effect> {
        match effect {
            Effect::If {
                condition,
                then,
                otherwise,
            } if [then, otherwise].iter().all(|branch| {
                matches!(
                    branch.as_slice(),
                    [Effect::AddMana {
                        until_end_of_combat: false,
                        ..
                    }]
                )
            }) =>
            {
                let branch = if self.check_static_condition(condition, controller) {
                    then
                } else {
                    otherwise
                };
                Some(
                    self.current_mana_output(&branch[0], controller, chosen)
                        .unwrap_or_else(|| branch[0].clone()),
                )
            }
            Effect::AddMana {
                mana,
                amount,
                until_end_of_combat: false,
            } if !matches!(amount, EffectValue::Fixed(1)) => {
                let mut output = Mana::default();
                for _ in 0..self.controller_value(amount, controller) {
                    output.add(mana);
                }
                Some(Effect::add_mana(output))
            }
            Effect::AddOneColorManaValue {
                amount,
                chosen_color,
                creature_purposes,
            } => Some(crate::state::card::materialize_one_color_mana(
                self.controller_value(amount, controller),
                *chosen_color,
                creature_purposes,
                chosen,
            )),
            _ => None,
        }
    }

    /// The activation's mana cost as paid now: per-permanent reductions apply
    /// to its generic part (CR 601.2f, 602.2b). Its purpose already names the
    /// source (CR 106.6).
    pub(crate) fn price_activation(
        &self,
        controller: PlayerId,
        ability: &mut ActivatedAbilityDefinition,
    ) {
        if let Some(predicate) = &ability.cost_reduction_per {
            let count = self.count_battlefield_matching(controller, predicate);
            ability.mana_cost = ability
                .mana_cost
                .reduced_generic(u8::try_from(count).unwrap_or(u8::MAX));
        }
    }

    pub(crate) fn capture_priority_mana(&mut self, player: PlayerId) {
        if self
            .card_capture
            .as_ref()
            .and_then(|cards| cards.exposure.as_ref())
            .is_none()
            || player != self.active_player()
        {
            return;
        }
        let model = self.payment_model(player, None);
        let (_, net) = model.source_bounds(
            &model.resources(),
            model.life,
            |_, output| (0, u64::from(output.total())),
            |_, _, ability| (0, ability.max_net()),
            |_| false,
        );
        let supply = net + u64::from(self.pooled_mana(player).total());
        if let Some(turn) = self
            .exposure_mut()
            .and_then(|capture| capture.mana_turns[player.0].last_mut())
        {
            turn.available = turn.available.max(turn.paid + supply);
        }
    }

    fn payment_model(&self, player: PlayerId, sacrifice: Option<PermanentId>) -> PaymentModel<'_> {
        let _scan = self.board_scan();
        let mut model = PaymentModel {
            cache: Some(&self.payment_cache),
            sources: self.shared_payment_sources(player),
            life: self.payable_life(player).max(0) as u32,
        };
        if let Some(source) = sacrifice {
            model.reserve(SourceReservation::owed(source, false, true));
        }
        // An announcement survives nested mana decisions in pending_choice;
        // its resources are owed in every phase, not only mana spending.
        if let Some(PendingChoice::Pay { payment, .. }) = &self.pending_choice {
            self.reserve_announced_payment(payment, &mut model);
            self.mark_help_taps(&mut model, payment.help());
        }
        fn reserve(decision: &Decision, game: &Game, model: &mut PaymentModel) {
            match decision {
                Decision::ManaSpend { completion, .. } => match completion.as_ref() {
                    SpendCompletion::Upfront(next) => reserve(next, game, model),
                    SpendCompletion::Announced(payment) => {
                        game.reserve_announced_payment(payment, model);
                    }
                    _ => {}
                },
                Decision::ManaUpfront {
                    permanent,
                    ability_index,
                    resume,
                    ..
                } => {
                    let ability =
                        &game.characteristics(*permanent).activated_abilities[*ability_index];
                    model.reserve(SourceReservation::owed(
                        *permanent,
                        ability.tap_source,
                        ability.source_leaves(),
                    ));
                    if let Some(next) = resume {
                        reserve(next, game, model);
                    }
                }
                Decision::ManaChoice {
                    resume: Some(next), ..
                } => reserve(next, game, model),
                _ => {}
            }
        }
        if let Some(suspended) = &self.state.suspended_decision {
            reserve(&suspended.decision, self, &mut model);
        }
        model
    }

    /// Begin one legal-action computation; see `PaymentScan`.
    pub(crate) fn payment_scan(&self) -> PaymentScan<'_> {
        self.payment_sources.borrow_mut().scans += 1;
        PaymentScan { game: self }
    }

    /// `player`'s unreserved sources, reused inside a `PaymentScan`.
    fn shared_payment_sources(&self, player: PlayerId) -> Arc<Vec<ManaSource>> {
        if let Some(sources) = &self.payment_sources.borrow().sources[player.0] {
            debug_assert_eq!(
                **sources,
                self.payment_sources(player),
                "stale shared payment sources"
            );
            return sources.clone();
        }
        let sources = Arc::new(self.payment_sources(player));
        let mut shared = self.payment_sources.borrow_mut();
        if shared.scans > 0 {
            shared.sources[player.0] = Some(sources.clone());
        }
        sources
    }

    /// Every permanent `player` controls as a mana source, before any
    /// reservation: a function of the game alone.
    fn payment_sources(&self, player: PlayerId) -> Vec<ManaSource> {
        let _scan = self.board_scan();
        let triggers = self.mana_triggers();
        self.characteristics_many(self.iter_battlefield_permanents(player))
            .map(|(permanent, characteristics)| {
                let source = self.state.permanents[permanent].as_ref().unwrap();
                // A source that lost its abilities offers no zero-mana tap.
                let can_tap = self.permanent_can_tap(permanent);
                let prohibited = self.activations_prohibited(permanent);
                let fixed = (can_tap && !prohibited && !characteristics.mana_abilities.is_empty())
                    .then(|| self.fixed_source_mana(player, permanent, &characteristics, &triggers))
                    .flatten();
                let abilities = characteristics
                    .activated_abilities
                    .iter()
                    .enumerate()
                    .filter(|(index, ability)| {
                        !prohibited
                            && (!ability.tap_source || can_tap)
                            && self.check_activation_terms(player, ability).is_ok()
                            && !source.activation_used(*index, ability)
                            && source.can_pay_counters(ability)
                    })
                    .filter_map(|(index, ability)| {
                        let mut outputs = ability.mana_choices()?;
                        let mut automatic = ability.is_simple_mana_tap();
                        if ability.tap_source {
                            let bonuses = self.tap_bonuses(&triggers, player, permanent);
                            automatic &= bonuses.is_empty();
                            outputs = production_alternatives(outputs, &bonuses);
                        }
                        Some(ManaActivation {
                            automatic,
                            tap_source: ability.tap_source,
                            tap_count: ability.tap_cost.as_ref().map_or(0, |cost| cost.count),
                            tap_candidates: self.tap_cost_candidates(player, ability).into(),
                            limited: ability.is_use_limited() || ability.remove_counters.is_some(),
                            sacrifice_source: ability.sacrifice_source,
                            ability_index: index,
                            cost: ability.mana_cost.clone(),
                            outputs: outputs.into(),
                            life_cost: ability.life_cost,
                        })
                    })
                    .collect();
                ManaSource {
                    permanent,
                    object: source.object,
                    untapped: !source.tapped,
                    fixed,
                    abilities,
                    keyword: false,
                    waterbend: !source.tapped
                        && (characteristics.types.is_artifact() || characteristics.is_creature()),
                }
            })
            .collect()
    }

    /// Reserve costs still owed by an announced payment: its source's own
    /// tap or sacrifice, its life, and the permanents already chosen to be
    /// sacrificed as it completes. ManaUpfront's life fee instead lives on
    /// its Output obligation and must not be debited here.
    fn reserve_announced_payment(&self, payment: &Payment, model: &mut PaymentModel) {
        if let Some(reservation) = owed_reservation(payment) {
            model.reserve(reservation);
        }
        model.life = model.life.saturating_sub(payment.life());
        let grantors = payment.owed.iter().filter_map(|component| match component {
            CostComponent::ExileGrantor(object) => Some((*object, false)),
            CostComponent::TapGrantor(object) => Some((*object, true)),
            _ => None,
        });
        let sacrificed = payment
            .receipt
            .paid_as(How::Sacrificed)
            .map(|object| (object, false));
        for (object, tap) in sacrificed.chain(grantors) {
            if let Some(permanent) = self.state.permanent_of(object) {
                model.reserve(SourceReservation::owed(permanent, tap, true));
            }
        }
    }

    /// The model for paying `payment`, or `None` when its life cost alone
    /// cannot be paid. The pending payment is reserved by every model; one
    /// only being considered at priority is reserved here.
    fn announced_model(&self, player: PlayerId, operation: &Operation) -> Option<PaymentModel<'_>> {
        let mut model = self.payment_model(player, None);
        if self.paying().is_some() {
            return Some(model);
        }
        if let Operation::Activation {
            source,
            ability_index,
            ..
        } = operation
        {
            let ability = self.activation_ability(*source, *ability_index)?;
            // Hold the source's own tap/leave cost while considering activation;
            // once announced, `owed_reservation` reads its cost.
            if let ActivationSource::Permanent(permanent) = source {
                model.reserve(SourceReservation::owed(
                    *permanent,
                    ability.tap_source,
                    ability.source_leaves(),
                ));
            }
            if ability.has_grantor_cost() {
                model.reserve(SourceReservation::owed(
                    self.state.permanent_of(ability.granted_by?.grantor)?,
                    ability.tap_grantor,
                    ability.exile_grantor,
                ));
            }
            model.life = model.life.checked_sub(ability.life_cost)?;
        }
        Some(model)
    }

    fn payment_obligations(&self, player: PlayerId) -> Vec<Obligation> {
        self.payment_obligations_owing(player, None)
    }

    /// The obligations with the payment in progress owing `owed` mana
    /// instead of what it still records.
    fn payment_obligations_owing(
        &self,
        player: PlayerId,
        owed: Option<&ManaCost>,
    ) -> Vec<Obligation> {
        fn append(decision: &Decision, game: &Game, result: &mut Vec<Obligation>) {
            match decision {
                Decision::ManaSpend {
                    cost, completion, ..
                } => {
                    result.push(Obligation::mana(cost.clone()));
                    if let SpendCompletion::Upfront(next) = completion.as_ref() {
                        let mut saved = Vec::new();
                        append(next, game, &mut saved);
                        result.extend(saved.into_iter().skip(1));
                    }
                }
                Decision::ManaUpfront {
                    permanent,
                    ability_index,
                    tap_remaining,
                    resume,
                    ..
                } => {
                    let ability = game.characteristics(*permanent).activated_abilities
                        [*ability_index]
                        .clone();
                    let mut outputs = ability.mana_choices().unwrap();
                    if ability.tap_source {
                        let bonuses =
                            game.tap_bonuses(&game.mana_triggers(), decision.player(), *permanent);
                        outputs = production_alternatives(outputs, &bonuses);
                    }
                    if *tap_remaining > 0 {
                        result.push(Obligation::Tap {
                            candidates: game
                                .tap_cost_candidates(decision.player(), &ability)
                                .into(),
                            remaining: *tap_remaining,
                        });
                    }
                    result.push(Obligation::mana(ability.mana_cost));
                    result.push(Obligation::Output {
                        choices: outputs.into(),
                        life_cost: ability.life_cost,
                    });
                    if let Some(next) = resume {
                        append(next, game, result);
                    }
                }
                Decision::ManaChoice {
                    resume: Some(next), ..
                } => append(next, game, result),
                // A life cost is paid or declined in one action: it never
                // commits, so it never reaches the solver as an obligation.
                Decision::Payment(progress) if !progress.can_decline() => {
                    if let Some(CostComponent::Mana { cost, .. }) = progress.owed.first() {
                        result.push(Obligation::Pay {
                            cost: cost.clone(),
                            taps: if progress.waterbends() {
                                Taps::Any
                            } else {
                                Taps::None
                            },
                        });
                    }
                }
                _ => {}
            }
        }
        let mut result = Vec::new();
        if let Some(suspended) = &self.state.suspended_decision {
            append(&suspended.decision, self, &mut result);
        }
        match self.paying() {
            Some((payer, payment, CostComponent::Mana { .. })) if payer == player => {
                let (recorded, _) = payment.mana();
                result.extend(self.help_obligations(
                    player,
                    owed.unwrap_or(recorded),
                    payment.mana_permissions(),
                ));
            }
            _ => {}
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn activation_payable_from(
        &self,
        player: PlayerId,
        base: &PaymentModel<'_>,
        source_index: usize,
        ability_pos: usize,
        pool: &Mana,
        convoke: bool,
        pending_obligations: &[Obligation],
    ) -> bool {
        let source = &base.sources[source_index];
        let permanent = source.permanent;
        let ability = &source.abilities[ability_pos];
        let ability_index = ability.ability_index;
        let mut model = PaymentModel {
            cache: base.cache,
            sources: base.sources.clone(),
            life: base.life,
        };
        model.reserve(SourceReservation::owed(
            permanent,
            ability.tap_source,
            ability.sacrifice_source,
        ));
        if ability.limited {
            Arc::make_mut(&mut model.sources)[source_index]
                .abilities
                .retain(|a| a.ability_index != ability_index);
        }
        let obligations = PaymentModel::activation_obligations(ability);
        if convoke {
            self.while_paying(player, &mut model, &mut |model, owed| {
                let mut obligations = obligations.clone();
                obligations.extend(owed);
                model
                    .payable(pool.clone(), obligations, SourcePolicy::All)
                    .then_some(())
            })
            .is_some()
        } else {
            let mut obligations = obligations;
            obligations.extend_from_slice(pending_obligations);
            model.payable(pool.clone(), obligations, SourcePolicy::All)
        }
    }

    fn paying_convoke(&self, player: PlayerId) -> bool {
        matches!(
            self.paying(),
            Some((payer, payment, CostComponent::Mana { .. }))
                if payer == player && payment.help().convoke
        )
    }

    /// Under convoke, colored creatures can pay colored pips of their colors
    /// in `while_paying`, so convoker colors are part of source identity.
    /// Bit 7 distinguishes a convoking creature (even colorless) from non-convokers.
    fn convoker_key(&self, convoke: bool, source: &ManaSource) -> u8 {
        if !convoke || !source.keyword {
            return 0;
        }
        let view = self.characteristics(source.permanent);
        if !view.is_creature() {
            return 0;
        }
        view.colors
            .iter()
            .fold(0x80_u8, |mask, &color| mask | (1_u8 << (color as u8)))
    }

    pub(crate) fn can_activate_mana_ability(
        &self,
        player: PlayerId,
        permanent: PermanentId,
        ability_index: usize,
    ) -> bool {
        let base = self.payment_model(player, None);
        let Some(source_index) = base
            .sources
            .iter()
            .position(|source| source.permanent == permanent)
        else {
            return false;
        };
        let Some(ability_pos) = base.sources[source_index]
            .abilities
            .iter()
            .position(|ability| ability.ability_index == ability_index)
        else {
            return false;
        };
        let convoke = self.paying_convoke(player);
        let pending_obligations = if convoke {
            Vec::new()
        } else {
            self.payment_obligations(player)
        };
        let pool = self.pooled_mana(player);
        self.activation_payable_from(
            player,
            &base,
            source_index,
            ability_pos,
            &pool,
            convoke,
            &pending_obligations,
        )
    }

    /// Legal activated mana abilities in source and ability order. Sources in
    /// the same purpose-projected equivalence class share one feasibility check
    /// while preserving exact action identities and order; debug builds verify
    /// every deduplicated source against an independent query.
    pub(crate) fn mana_ability_actions(
        &self,
        player: PlayerId,
        include_simple_taps: bool,
    ) -> Vec<Action> {
        let model = self.payment_model(player, None);
        self.mana_ability_actions_with(&model, player, include_simple_taps)
    }

    fn mana_ability_actions_with(
        &self,
        base: &PaymentModel<'_>,
        player: PlayerId,
        include_simple_taps: bool,
    ) -> Vec<Action> {
        let mut active_sources = base.sources.iter().filter(|s| {
            s.abilities
                .iter()
                .any(|a| include_simple_taps || !a.automatic)
        });
        if active_sources.next().is_none() {
            return Vec::new();
        }
        let dedup = active_sources.next().is_some();
        let convoke = self.paying_convoke(player);
        let pending_obligations = if convoke && !dedup {
            Vec::new()
        } else {
            self.payment_obligations(player)
        };
        let pool = self.pooled_mana(player);
        let check_ability = |source_index, pos| {
            self.activation_payable_from(
                player,
                base,
                source_index,
                pos,
                &pool,
                convoke,
                &pending_obligations,
            )
        };
        let mut cache =
            dedup.then(|| SourceClassCache::<u64, (u8, u64)>::new(base, &pending_obligations, &[]));
        let mut actions = Vec::new();
        for (source_index, source) in base.sources.iter().enumerate() {
            if !source
                .abilities
                .iter()
                .any(|a| include_simple_taps || !a.automatic)
            {
                continue;
            }
            let Some(cache) = cache.as_mut() else {
                for (pos, ability) in source.abilities.iter().enumerate() {
                    if (include_simple_taps || !ability.automatic)
                        && check_ability(source_index, pos)
                    {
                        actions.push(Action::ActivateAbility {
                            player,
                            permanent: source.permanent,
                            ability_index: ability.ability_index,
                        });
                    }
                }
                break;
            };
            let auto_mask = if include_simple_taps {
                0_u64
            } else {
                source
                    .abilities
                    .iter()
                    .enumerate()
                    .fold(0_u64, |mask, (pos, a)| {
                        assert!(pos < 64, "source has >= 64 activated mana abilities");
                        if a.automatic {
                            mask | (1_u64 << pos)
                        } else {
                            mask
                        }
                    })
            };
            let key = (self.convoker_key(convoke, source), auto_mask);
            let feasible_mask = if let Some(cached) = cache.get(source_index, key) {
                #[cfg(debug_assertions)]
                for (pos, ability) in source.abilities.iter().enumerate() {
                    if !include_simple_taps && ability.automatic {
                        continue;
                    }
                    assert!(pos < 64, "source has >= 64 activated mana abilities");
                    debug_assert_eq!(
                        cached & (1_u64 << pos) != 0,
                        self.can_activate_mana_ability(
                            player,
                            source.permanent,
                            ability.ability_index
                        ),
                        "deduplicated mana ability feasibility differed from exact witness"
                    );
                }
                cached
            } else {
                let mut mask = 0_u64;
                for (pos, ability) in source.abilities.iter().enumerate() {
                    assert!(pos < 64, "source has >= 64 activated mana abilities");
                    if (include_simple_taps || !ability.automatic)
                        && check_ability(source_index, pos)
                    {
                        mask |= 1_u64 << pos;
                    }
                }
                cache.insert(source_index, key, mask);
                mask
            };
            for (pos, ability) in source.abilities.iter().enumerate() {
                if (include_simple_taps || !ability.automatic)
                    && feasible_mask & (1_u64 << pos) != 0
                {
                    actions.push(Action::ActivateAbility {
                        player,
                        permanent: source.permanent,
                        ability_index: ability.ability_index,
                    });
                }
            }
        }
        actions
    }

    /// Append fixed sources after the ordinary offers. A committed payment
    /// must still have a witness after the tap, including any saved callers.
    pub(crate) fn append_mana_floats(&self, player: PlayerId, actions: &mut Vec<Action>) {
        if self.mana_float_mode(player).is_off()
            || (self.automatic_mana_payment(player)
                && (self.pending_choice.is_some() || self.state.suspended_decision.is_some()))
        {
            return;
        }
        let obligations = self.payment_obligations(player);
        let model = self.payment_model(player, None);
        let ordinary_len = actions.len();
        for source in model.sources.iter().filter(|source| source.untapped) {
            let Some(output) = &source.fixed else {
                continue;
            };
            let tap = Action::TapForMana {
                player,
                permanent: source.permanent,
            };
            if actions[..ordinary_len].contains(&tap) {
                continue;
            }
            if !obligations.is_empty() {
                let mut pool = self.pooled_mana(player);
                pool.add(output);
                let mut after = model.after_tap(source.permanent);
                let payable = self.while_paying(player, &mut after, &mut |model, owed| {
                    model
                        .payable(pool.clone(), owed, SourcePolicy::All)
                        .then_some(())
                });
                if payable.is_none() {
                    continue;
                }
            }
            actions.push(tap);
        }
    }

    /// Whether the seat deciding for `player` was set up with mana tap offers.
    pub(crate) fn offers_extra_payment_taps(&self, player: PlayerId) -> bool {
        !self.mana_float_mode(player).is_off() && !self.automatic_mana_payment(player)
    }

    pub(crate) fn automatic_mana_payment(&self, player: PlayerId) -> bool {
        self.mana_payment != [ManaPayment::Manual; 2]
            && self.mana_payment[self.decision_controller(player).0] == ManaPayment::Auto
    }

    /// Follow one step of the live payment witness. The automatic payer pays
    /// mana with mana and never picks an object: units of one mana class are
    /// interchangeable, creatures and cards are not. Every other component is
    /// the player's, and so is a convoke or improvise tap or a delve exile
    /// when mana alone cannot pay; the payer resumes once it can. Waterbend
    /// taps are the one exception and are still made for the player. Optional
    /// costs still offer Pay/Decline, and a deliberate priority float still
    /// chooses its output.
    pub(crate) fn automatic_payment_action(&self, space: &ActionSpace) -> Option<Action> {
        let player = space.player?;
        if !self.automatic_mana_payment(player) {
            return None;
        }
        // Whether convoke, improvise or delve may pay part of this mana step.
        let helped = match self.state.suspended_decision.as_ref().map(|s| &s.decision) {
            Some(Decision::Payment(progress)) if !progress.can_decline() => false,
            Some(Decision::ManaUpfront { .. }) => false,
            // The most specific unit that can pay goes first, so mana only
            // this kind of cost can use is not left floating beside
            // ordinary mana; a rider this cost would waste goes last. The
            // offers keep their order: this is the automatic payer's choice.
            Some(Decision::ManaSpend { cost, .. }) => {
                let order = |choice: &ManaSpendChoice| {
                    choice
                        .unit
                        .classes()
                        .first()
                        .map_or(SpendOrder::Ordinary, |(class, _)| {
                            class.spend_order(&cost.purpose)
                        })
                };
                let choices = self.mana_spend_choices(player, cost);
                let mode = (0..choices.len()).min_by_key(|mode| order(&choices[*mode]))?;
                return space.actions.get(mode).cloned();
            }
            Some(Decision::ManaChoice {
                choices, resume, ..
            }) if resume.is_some() || self.pending_choice.is_some() => {
                if space.actions.len() == 1 {
                    return space.actions.first().cloned();
                }
                let model = self.payment_model(player, None);
                return space
                    .actions
                    .iter()
                    .min_by_key(|action| {
                        let Action::ChooseMode { mode, .. } = action else {
                            unreachable!()
                        };
                        let (pool, obligations) = self.after_mana_choice(player, &choices[*mode]);
                        model
                            .plan(pool, obligations, SourcePolicy::All)
                            .map_or(usize::MAX, |remaining| remaining.len())
                    })
                    .cloned();
            }
            None => match self.paying() {
                Some((_, _, CostComponent::Mana { help, .. })) => help.keyworded(),
                _ => return None,
            },
            _ => return None,
        };
        if let Some(pay) = space
            .actions
            .iter()
            .find(|a| matches!(a, Action::PayCost { .. }))
        {
            return Some(pay.clone());
        }
        // Mana alone cannot pay: which creature or artifact to tap, or card
        // to exile, is the player's pick.
        if helped {
            return None;
        }
        // The witness models the ordinary and waterbend portions as separate
        // spends, but the game charges their combined remainder only once.
        // Every offered waterbend tap preserves a witness for that remainder.
        if let Some(tap) = space
            .actions
            .iter()
            .find(|a| matches!(a, Action::WaterbendTap { .. }))
        {
            return Some(tap.clone());
        }
        let path = self
            .payment_model(player, None)
            .plan(
                self.pooled_mana(player),
                self.payment_obligations(player),
                SourcePolicy::All,
            )
            .expect("committed cost has a payment witness");
        let next = if self
            .paying()
            .is_some_and(|(_, payment, _)| payment.waterbends())
            && self.state.suspended_decision.is_none()
        {
            path.iter().find(|step| !matches!(step, PaymentStep::Spend))
        } else {
            path.first()
        };
        match next.expect("committed payment has a next step") {
            PaymentStep::Spend => Some(Action::PayCost { player }),
            PaymentStep::Fixed(permanent) => Some(Action::TapForMana {
                player,
                permanent: *permanent,
            }),
            step => step.action(player),
        }
    }

    pub(crate) fn mana_float_mode(&self, player: PlayerId) -> ManaFloats {
        if self.mana_floats == [ManaFloats::Off; 2] {
            return ManaFloats::Off;
        }
        self.mana_floats[self.decision_controller(player).0]
    }

    fn has_mana_floats(&self, player: PlayerId) -> bool {
        let mut taps = Vec::new();
        self.append_mana_floats(player, &mut taps);
        !taps.is_empty()
    }

    fn payment_reads_spent_colors(&self, payment: &Payment) -> bool {
        use crate::cards::content_pack::{spent_mana_read, SpentManaRead};
        self.payment_source_object(&payment.operation)
            .and_then(|object| self.observed_source_definition(object))
            .and_then(spent_mana_read)
            == Some(SpentManaRead::Colors)
    }

    pub(crate) fn clear_staged_announced_actions(&self) {
        self.payment_sources.borrow_mut().staged_announced = None;
    }

    pub(crate) fn stage_announced_mana_actions(&self, player: PlayerId, actions: Vec<Action>) {
        if self.state.suspended_decision.is_some() {
            return;
        }
        self.payment_sources.borrow_mut().staged_announced = Some(StagedAnnouncedActions {
            player,
            floats: self.mana_float_mode(player),
            automatic: self.automatic_mana_payment(player),
            actions,
        });
    }

    /// Consume the actions just computed by `advance_payment` when publishing
    /// the pending `Pay` space; recompute fresh whenever no staged entry is
    /// present or player payment settings changed.
    pub(crate) fn take_announced_mana_actions(
        &self,
        player: PlayerId,
        payment: &Payment,
    ) -> Vec<Action> {
        let staged = self.payment_sources.borrow_mut().staged_announced.take();
        if let Some(entry) = staged.filter(|entry| {
            self.state.suspended_decision.is_none()
                && entry.player == player
                && entry.floats == self.mana_float_mode(player)
                && entry.automatic == self.automatic_mana_payment(player)
        }) {
            debug_assert_eq!(
                entry.actions,
                self.compute_announced_mana_actions(player, payment),
                "stale staged announced_mana_actions"
            );
            return entry.actions;
        }
        self.compute_announced_mana_actions(player, payment)
    }

    pub(crate) fn compute_announced_mana_actions(
        &self,
        player: PlayerId,
        payment: &Payment,
    ) -> Vec<Action> {
        let _scan = self.payment_scan();
        let cost = payment.mana().0;
        let extra = self.offers_extra_payment_taps(player);
        let explicit = extra || payment.waterbends();
        let model = self.payment_model(player, None);
        let pool = self.pooled_mana(player);
        let mut actions = self.mana_ability_actions_with(&model, player, explicit);
        let help = self.help_actions(player);
        let (fixed_payable, can_pay) = self.automatic_payability(&model, &pool, player, cost);
        if !payment.waterbends() {
            if let PaymentTaps::Choose(taps) = self.useful_payment_taps(
                &model,
                &pool,
                player,
                cost,
                payment.help(),
                TapOptions {
                    fixed_payable,
                    explicit,
                    allow_plain_pool: false,
                    other_choice: !actions.is_empty() || !help.is_empty() || !can_pay,
                    need_taps: false,
                },
            ) {
                actions.extend(taps);
            }
        }
        actions.extend(help);
        if can_pay {
            actions.push(Action::PayCost { player });
        }
        self.append_mana_floats(player, &mut actions);
        actions
    }

    /// Resolution payments (`TurnFaceUp`, `Ward`, etc.) under manual payment
    /// share the same forced/interchangeable skip and per-source tap rules for
    /// both basics and simple choice sources.
    pub(crate) fn resolution_mana_actions(
        &self,
        player: PlayerId,
        payment: &crate::flow::payment::PaymentProgress,
        cost: &ManaCost,
        permissions: crate::flow::payment::CostPermissionContext,
        tail_payable: bool,
    ) -> Vec<Action> {
        let _scan = self.payment_scan();
        let model = self.payment_model(player, None);
        let pool = self.pooled_mana(player);
        let (fixed_payable, auto_can_pay) = self.automatic_payability(&model, &pool, player, cost);
        let can_pay = tail_payable && auto_can_pay;
        let extra = self.offers_extra_payment_taps(player);
        let explicit = !can_pay || payment.waterbends() || extra;
        let mut actions = self.mana_ability_actions_with(&model, player, explicit);
        let help_actions: Vec<Action> =
            if tail_payable && (permissions.help != Help::default() || permissions.waterbend > 0) {
                self.help_taps(
                    player,
                    cost,
                    permissions,
                    &self.waterbend_candidates(player),
                )
                .into_iter()
                .map(|(permanent, _)| Action::WaterbendTap { player, permanent })
                .collect()
            } else {
                Vec::new()
            };
        if tail_payable && !payment.waterbends() && !self.automatic_mana_payment(player) {
            if let PaymentTaps::Choose(taps) = self.useful_payment_taps(
                &model,
                &pool,
                player,
                cost,
                permissions.help,
                TapOptions {
                    fixed_payable,
                    explicit,
                    allow_plain_pool: true,
                    other_choice: !actions.is_empty() || !help_actions.is_empty() || !can_pay,
                    need_taps: false,
                },
            ) {
                actions.extend(taps);
            }
        }
        actions.extend(help_actions);
        if can_pay {
            actions.push(Action::PayCost { player });
        }
        if payment.can_decline() {
            actions.push(Action::Decline { player });
        }
        self.append_mana_floats(player, &mut actions);
        actions
    }

    /// Count a permanent's best useful output once, even if it has several
    /// abilities. If all units are required there is no source decision,
    /// unless the choice of output can change the receipt or floating pool.
    fn useful_payment_taps<'a>(
        &self,
        model: &'a PaymentModel<'a>,
        pool: &Mana,
        player: PlayerId,
        cost: &ManaCost,
        help: Help,
        options: TapOptions,
    ) -> PaymentTaps<'a> {
        let spendable = pool.spendable_units(cost);
        let missing = u16::from(cost.mana_value).saturating_sub(spendable);
        if missing == 0 {
            return PaymentTaps::Determined(Vec::new());
        }
        let helped = help.keyworded();
        let color_matters = pool.usable_total(cost) > spendable
            || self
                .paying()
                .is_some_and(|(_, payment, _)| self.payment_reads_spent_colors(payment));
        // When no caller needs the concrete tap list and total untapped
        // automatic units cannot exceed the shortfall, every useful tap is
        // mandatory unless a source can vary spend effects.
        if !options.need_taps
            && !options.other_choice
            && !helped
            && !color_matters
            && automatic_units_within(&model.sources, missing)
        {
            return PaymentTaps::Determined(Vec::new());
        }
        let mut obligations = vec![Obligation::mana(cost.clone())];
        obligations.extend(self.saved_obligations(player));
        let extra = self.offers_extra_payment_taps(player);
        // Moving an already affordable fixed output into the pool preserves
        // aggregate affordability. A choice output can forfeit another color.
        let single_mana_obligation = !helped && obligations.len() == 1;
        let fixed_only = single_mana_obligation && options.fixed_payable;
        let mut fixed_total: Option<Mana> = None;
        let has_counted_taps = obligations
            .iter()
            .any(|obligation| matches!(obligation, Obligation::Tap { .. }))
            || model
                .sources
                .iter()
                .any(|s| s.abilities.iter().any(|a| a.tap_count > 0));
        let dedup_profiles = !helped && !has_counted_taps;
        let dedup_fixed = !fixed_only
            && dedup_profiles
            && model
                .sources
                .iter()
                .filter(|source| source.untapped && source.fixed.is_some())
                .nth(1)
                .is_some();
        let mut fixed_cache = dedup_fixed.then(|| {
            SourceClassCache::<bool, ()>::new(
                model,
                &obligations,
                std::slice::from_ref(&cost.purpose),
            )
        });
        struct ProfileTap<'a> {
            source: &'a ManaSource,
            fixed_useful: bool,
            offered_abilities: Vec<usize>,
            units: u16,
            first: Option<&'a Mana>,
            output_choice: bool,
        }
        let mut seen_profiles: Vec<ProfileTap<'a>> = Vec::new();
        let mut needed_colors = [false; 6];
        for (needed, &count) in needed_colors.iter_mut().zip(&cost.cost[..6]) {
            *needed = count > 0;
        }
        for &[left, right] in &cost.hybrid {
            needed_colors[left as usize] = true;
            needed_colors[right as usize] = true;
        }
        let is_plain_generic_only = |output: &Mana| -> bool {
            output.classes().is_empty()
                && output.total() == 1
                && output
                    .mana
                    .iter()
                    .position(|&n| n == 1)
                    .is_some_and(|color| !needed_colors[color])
        };
        let mut offered_units = 0_u16;
        let mut useful_taps = Vec::new();
        let mut single_units = true;
        let mut output_choice = false;
        let mut abilities = Vec::new();
        let mut fixed = Vec::new();
        for (source_index, source) in model.sources.iter().enumerate() {
            if !source.untapped {
                continue;
            }
            let cached = dedup_profiles
                .then(|| {
                    seen_profiles
                        .iter()
                        .position(|entry| entry.source.same_payment_profile(source))
                })
                .flatten();
            let fresh = cached.is_none().then(|| {
                let mut after: Option<PaymentModel> = None;
                let mut generic_failed = false;
                let mut check_useful = |output: &Mana, is_fixed: bool| -> bool {
                    let mut tapped = pool.clone();
                    tapped.add(output);
                    if tapped.spendable_units(cost) <= spendable {
                        return false;
                    }
                    if fixed_only {
                        let Some(fixed_output) = source.fixed.as_ref().filter(|_| !is_fixed) else {
                            return true;
                        };
                        let mut without = fixed_total
                            .get_or_insert_with(|| model.fixed_total(pool))
                            .clone();
                        without.spend(fixed_output);
                        without.add(output);
                        if without.can_pay(cost) {
                            return true;
                        }
                    }
                    let plain_generic =
                        !is_fixed && single_mana_obligation && is_plain_generic_only(output);
                    if plain_generic && generic_failed {
                        return false;
                    }
                    let ok = if helped {
                        let mut after_help = model.after_tap(source.permanent);
                        self.while_paying(player, &mut after_help, &mut |model, owed| {
                            model
                                .payable(tapped.clone(), owed, SourcePolicy::All)
                                .then_some(())
                        })
                        .is_some()
                    } else if is_fixed {
                        if let Some(cache) = fixed_cache.as_mut() {
                            if let Some(cached) = cache.get(source_index, ()) {
                                debug_assert_eq!(
                                    cached,
                                    model.payable_after_tap(
                                        source.permanent,
                                        tapped.clone(),
                                        obligations.clone(),
                                    ),
                                    "deduplicated useful payment tap differed from exact check"
                                );
                                cached
                            } else {
                                let after =
                                    after.get_or_insert_with(|| model.after_tap(source.permanent));
                                let result =
                                    after.payable(tapped, obligations.clone(), SourcePolicy::All);
                                cache.insert(source_index, (), result);
                                result
                            }
                        } else {
                            let after =
                                after.get_or_insert_with(|| model.after_tap(source.permanent));
                            after.payable(tapped, obligations.clone(), SourcePolicy::All)
                        }
                    } else {
                        let after = after.get_or_insert_with(|| model.after_tap(source.permanent));
                        after.payable(tapped, obligations.clone(), SourcePolicy::All)
                    };
                    if plain_generic && !ok {
                        generic_failed = true;
                    }
                    ok
                };
                let mut units = 0;
                let mut first = None;
                let mut fixed_useful = false;
                let mut offered_abilities = Vec::new();
                let mut source_output_choice = false;
                if let Some(output) = source
                    .fixed
                    .as_ref()
                    .filter(|output| check_useful(output, true))
                {
                    fixed_useful = true;
                    units = output.total();
                    first = Some(output);
                }
                for ability in source.abilities.iter().filter(|ability| ability.automatic) {
                    let mut offered = false;
                    for output in ability.outputs.iter() {
                        if offered
                            && output.total() <= units
                            && (output_choice
                                || source_output_choice
                                || first.is_some_and(|prev| {
                                    prev == output
                                        || (!color_matters && same_spend_effects(prev, output))
                                }))
                        {
                            continue;
                        }
                        if !check_useful(output, false) {
                            continue;
                        }
                        offered = true;
                        units = units.max(output.total());
                        let previous = *first.get_or_insert(output);
                        if previous != output {
                            source_output_choice |=
                                color_matters || !same_spend_effects(previous, output);
                        }
                    }
                    if offered && !options.explicit {
                        offered_abilities.push(ability.ability_index);
                    }
                }
                ProfileTap {
                    source,
                    fixed_useful,
                    offered_abilities,
                    units,
                    first,
                    output_choice: source_output_choice,
                }
            });
            let entry = cached.map_or_else(|| fresh.as_ref().unwrap(), |i| &seen_profiles[i]);
            if entry.fixed_useful {
                fixed.push(Action::TapForMana {
                    player,
                    permanent: source.permanent,
                });
            }
            if !options.explicit {
                for &ability_index in &entry.offered_abilities {
                    abilities.push(Action::ActivateAbility {
                        player,
                        permanent: source.permanent,
                        ability_index,
                    });
                }
            }
            output_choice |= entry.output_choice;
            if let Some(output) = entry.first {
                useful_taps.push(UsefulTap { source, output });
                single_units &= entry.units == 1;
                offered_units = offered_units.saturating_add(entry.units);
            }
            if dedup_profiles {
                if let Some(entry) = fresh {
                    seen_profiles.push(entry);
                }
            }
        }
        if !options.other_choice && !output_choice {
            if offered_units <= missing {
                return PaymentTaps::Determined(useful_taps);
            }
            let plain_committed_pool = spendable == 0
                || (options.allow_plain_pool
                    && pool.classes().is_empty()
                    && pool.usable_total(cost) == spendable);
            if single_units
                && plain_committed_pool
                && !self.automatic_mana_payment(player)
                && obligations.len() == 1
                && !extra
            {
                if let Some(taps) = self.determined_source_counts(
                    model,
                    pool,
                    cost,
                    usize::from(missing),
                    useful_taps,
                    options.need_taps,
                ) {
                    return PaymentTaps::Determined(taps);
                }
            }
        }
        abilities.extend(fixed);
        PaymentTaps::Choose(abilities)
    }

    /// Tap one offered fixed source; an announced payment then advances
    /// (`advance_payment`), finishing once no mana is owed.
    pub(crate) fn tap_for_mana_action(
        &mut self,
        player: PlayerId,
        permanent: PermanentId,
    ) -> Result<(), AgentError> {
        let before = (self.pending_choice.is_none()
            && self.state.suspended_decision.is_none()
            && self.mana_capture(player).is_some())
        .then(|| self.state.players[player.0].mana_pool.total());
        let start = self.state.players[player.0].mana_pool.unit_count();
        self.tap_fixed_mana_source(player, permanent);
        if let Some(before) = before {
            let produced = self.state.players[player.0].mana_pool.total() - before;
            self.mana_capture(player).unwrap().float(produced);
        }
        if self.state.suspended_decision.is_some() {
            return self.advance_mana_upfront();
        }
        self.commit_produced_mana(player, start);
        match self.paying() {
            Some((_, _, CostComponent::Mana { .. })) => {
                let payment = self.pending_payment(player)?.clone();
                return self.advance_payment(player, payment);
            }
            None => self.state.priority_mut().on_non_pass_action(player),
            _ => {}
        }
        Ok(())
    }

    pub(crate) fn can_pay_with_mana_abilities(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        sacrifice: Option<PermanentId>,
    ) -> bool {
        self.payment_model(player, sacrifice).payable(
            self.pooled_mana(player),
            vec![Obligation::mana(cost.clone())],
            SourcePolicy::All,
        )
    }

    /// Feasibility of `payment`'s chosen-object tap cost and its mana
    /// together. The owed taps include the hypothetical `selected` one; the
    /// shared transition checks and consumes it before searching the rest.
    pub(crate) fn tap_activation_payment_possible(
        &self,
        player: PlayerId,
        payment: &Payment,
        selected: Option<PermanentId>,
    ) -> bool {
        let Some((tap, toward)) = payment.owed.iter().find_map(|component| match component {
            CostComponent::Tap(tap, toward) => Some((tap, *toward)),
            _ => None,
        }) else {
            return false;
        };
        let Some(model) = self.announced_model(player, &payment.operation) else {
            return false;
        };
        let (owed, _) = payment.mana();
        let mut obligations = vec![Obligation::Tap {
            candidates: self
                .tap_component_candidates(player, &payment.operation, tap, toward)
                .into(),
            remaining: tap.count,
        }];
        obligations.extend(self.help_obligations(player, owed, payment.mana_permissions()));
        let pool = self.pooled_mana(player);
        match selected {
            Some(permanent) => model.plan_after_tap(pool, obligations, permanent).is_some(),
            None => model.payable(pool, obligations, SourcePolicy::All),
        }
    }

    /// Price the announced X with the same mana component used to pay the activation.
    pub(crate) fn activation_x_payable(
        &self,
        player: PlayerId,
        permanent: PermanentId,
        ability_index: usize,
        x: u8,
    ) -> bool {
        let operation = Operation::Activation {
            source: ActivationSource::Permanent(permanent),
            ability_index,
            targets: Vec::new(),
            x,
        };
        let Some(model) = self.announced_model(player, &operation) else {
            return false;
        };
        let payment = Payment {
            owed: self.cost_components(player, &operation),
            operation,
            receipt: Default::default(),
        };
        let (cost, _) = payment.mana();
        model.payable(
            self.pooled_mana(player),
            self.help_obligations(player, cost, payment.mana_permissions()),
            SourcePolicy::All,
        )
    }

    /// Whether `operation`, considered at priority, can pay `cost` beside
    /// its source's own costs and a `sacrifice` chosen for it.
    pub(crate) fn can_pay_announced_activation(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        operation: &Operation,
        sacrifice: Option<PermanentId>,
    ) -> bool {
        let Some(mut model) = self.announced_model(player, operation) else {
            return false;
        };
        if let Some(permanent) = sacrifice {
            model.reserve(SourceReservation::owed(permanent, false, true));
        }
        model.payable(
            self.pooled_mana(player),
            vec![Obligation::mana(cost.clone())],
            SourcePolicy::All,
        )
    }

    /// Whether choosing `mana` at the pending `ManaChoice` leaves a complete
    /// witness, counting the triggered mana abilities that still resolve
    /// after it from the same production.
    pub(crate) fn mana_choice_completes_payment(&self, player: PlayerId, mana: &Mana) -> bool {
        let (pool, obligations) = self.after_mana_choice(player, mana);
        self.payment_model(player, None)
            .payable(pool, obligations, SourcePolicy::All)
    }

    /// Feasible `Decision::ManaChoice` modes; when every offered mode leads to
    /// an indistinguishable payment state under manual payment, collapse to the
    /// single deterministic first mode (`len == 1`) so games with and without
    /// `skip_trivial` stay in lockstep.
    pub(crate) fn mana_choice_actions(&self, player: PlayerId, choices: &[Mana]) -> Vec<Action> {
        let _scan = self.payment_scan();
        let model = self.payment_model(player, None);
        let mut modes: Vec<usize> = (0..choices.len())
            .filter(|&mode| {
                let (pool, obligations) = self.after_mana_choice(player, &choices[mode]);
                model.payable(pool, obligations, SourcePolicy::All)
            })
            .collect();
        if modes.len() > 1
            && !self.automatic_mana_payment(player)
            && self.equivalent_mana_choices(player, &model, &modes)
        {
            modes.truncate(1);
        }
        modes
            .into_iter()
            .map(|mode| Action::ChooseMode { player, mode })
            .collect()
    }

    /// Compare the real commitment of every offered output before collapsing
    /// `Decision::ManaChoice` for a manual seat. Surplus mana and receipts read
    /// by converge keep the color choice open. A priority float has no
    /// announced cost.
    fn equivalent_mana_choices(
        &self,
        player: PlayerId,
        model: &PaymentModel,
        modes: &[usize],
    ) -> bool {
        let Some(Decision::ManaChoice {
            choices,
            resume,
            production:
                ManaProduction::Output {
                    source: Some(source),
                    tapped: Some(tapped),
                },
            ..
        }) = self.state.suspended_decision.as_ref().map(|s| &s.decision)
        else {
            return false;
        };
        let Some((&first_mode, rest_modes)) = modes.split_first() else {
            return false;
        };
        let first_output = &choices[first_mode];
        if rest_modes
            .iter()
            .any(|&mode| !same_spend_effects(first_output, &choices[mode]))
        {
            return false;
        }
        enum Enclosing<'a> {
            Announced(&'a Payment, &'a ManaCost),
            Resolution(&'a ManaCost),
        }
        let enclosing = match resume.as_deref() {
            None => {
                let Some((payer, payment, CostComponent::Mana { cost, .. })) = self.paying() else {
                    return false;
                };
                if payer != player || self.payment_reads_spent_colors(payment) {
                    return false;
                }
                Enclosing::Announced(payment, cost)
            }
            Some(Decision::Payment(progress)) => {
                let Some(CostComponent::Mana { cost, .. }) = progress.owed.first() else {
                    return false;
                };
                if progress.player != player
                    || progress.waterbends()
                    || self.offers_extra_payment_taps(player)
                {
                    return false;
                }
                Enclosing::Resolution(cost)
            }
            _ => return false,
        };
        if !self.production_bonuses(player, tapped).is_empty() {
            return false;
        }
        let Some(permanent) = self.state.permanent_of(*source) else {
            return false;
        };
        let view = self.characteristics(permanent);
        // A ManaChoice retains the output, not the ability index. If the
        // source could have paid an additional cost for these modes, ask.
        let mut candidates = view
            .activated_abilities
            .iter()
            .filter(|ability| ability.mana_choices().as_ref() == Some(choices));
        if !candidates
            .next()
            .is_some_and(|ability| ability.is_simple_mana_tap())
            || candidates.any(|ability| !ability.is_simple_mana_tap())
        {
            return false;
        }
        let pool = self.pooled_mana(player);
        match enclosing {
            Enclosing::Announced(payment, cost) => {
                let mut remainders = Vec::new();
                for &mode in modes {
                    let output = &choices[mode];
                    let units = ManaPool::from(output.clone());
                    let mut left = pool.clone();
                    left.add(output);
                    let mut rest = cost.clone();
                    for i in 0..units.unit_count() {
                        let unit = units.unit(i);
                        let color = units.unit_color(i);
                        left.spend(&unit);
                        let Some(next) = self
                            .cost_after_produced_unit(player, &rest, payment, &left, &unit, color)
                        else {
                            return false;
                        };
                        rest = next;
                    }
                    if !remainders.contains(&rest) {
                        remainders.push(rest);
                    }
                }
                if remainders.len() == 1 {
                    return true;
                }
                if payment.help() != Help::default()
                    || self.offers_extra_payment_taps(player)
                    || !remainders
                        .iter()
                        .all(|rest| pool.spendable_units(rest) == 0)
                {
                    return false;
                }
                self.same_determined_taps(
                    model,
                    player,
                    remainders.iter().map(|rest| (&pool, rest)),
                    false,
                )
            }
            Enclosing::Resolution(cost) => {
                // Resolution payments never run `commit_produced_mana`: every
                // previously tapped unit stays in `pool` against the unchanged
                // `cost`. Compare each mode's `(next_pool, cost)` directly.
                let mut next_pools = Vec::with_capacity(modes.len());
                for &mode in modes {
                    let mut next_pool = pool.clone();
                    next_pool.add(&choices[mode]);
                    let spendable = next_pool.spendable_units(cost);
                    if spendable != next_pool.total() || next_pool.usable_total(cost) != spendable {
                        return false;
                    }
                    next_pools.push(next_pool);
                }
                // 1. Exact completion: every mode pays `cost` with zero surplus.
                if next_pools.iter().all(|next_pool| next_pool.can_pay(cost)) {
                    return true;
                }
                // 2. Identical unambiguous remaining cost when no remaining
                //    source can overshoot into floating mana.
                let single_unit_remaining = model.sources.iter().all(|s| {
                    (!s.untapped || s.fixed.as_ref().is_none_or(|m| m.total() <= 1))
                        && s.abilities.iter().all(|a| {
                            a.cost.mana_value == 0 && a.outputs.iter().all(|m| m.total() <= 1)
                        })
                });
                if single_unit_remaining
                    && resolution_remaining_pips(&next_pools[0], cost).is_some_and(|first| {
                        next_pools[1..].iter().all(|next_pool| {
                            resolution_remaining_pips(next_pool, cost) == Some(first)
                        })
                    })
                {
                    return true;
                }
                // 3. Every mode leads to the exact same determined completion.
                self.same_determined_taps(
                    model,
                    player,
                    next_pools.iter().map(|next_pool| (next_pool, cost)),
                    true,
                )
            }
        }
    }

    fn same_determined_taps<'a>(
        &self,
        model: &PaymentModel,
        player: PlayerId,
        states: impl IntoIterator<Item = (&'a Mana, &'a ManaCost)>,
        allow_plain_pool: bool,
    ) -> bool {
        if model
            .sources
            .iter()
            .any(|s| s.abilities.iter().any(|a| !a.automatic))
        {
            return false;
        }
        let mut first: Option<Vec<UsefulTap>> = None;
        for (state_pool, state_cost) in states {
            let (fixed_payable, can_pay) =
                self.automatic_payability(model, state_pool, player, state_cost);
            if !can_pay {
                return false;
            }
            let PaymentTaps::Determined(taps) = self.useful_payment_taps(
                model,
                state_pool,
                player,
                state_cost,
                Help::default(),
                TapOptions {
                    fixed_payable,
                    allow_plain_pool,
                    need_taps: true,
                    ..Default::default()
                },
            ) else {
                return false;
            };
            if let Some(previous) = &first {
                if previous.len() != taps.len()
                    || !previous.iter().zip(taps).all(|(a, b)| {
                        a.source.permanent == b.source.permanent
                            && same_spend_effects(a.output, b.output)
                    })
                {
                    return false;
                }
            } else {
                first = Some(taps);
            }
        }
        true
    }

    fn after_mana_choice(&self, player: PlayerId, mana: &Mana) -> (Mana, Vec<Obligation>) {
        let mut pool = self.pooled_mana(player);
        pool.add(mana);
        let mut obligations = self.payment_obligations(player);
        if let Some(SuspendedResolution {
            decision: Decision::ManaChoice { production, .. },
            ..
        }) = &self.state.suspended_decision
        {
            let (produced, bonuses) = match production {
                ManaProduction::Output {
                    tapped: Some(tapped),
                    ..
                } => (mana, self.production_bonuses(player, tapped)),
                ManaProduction::Output { .. } => (mana, Vec::new()),
                ManaProduction::Bonus {
                    produced, pending, ..
                } => (produced, pending.clone()),
            };
            let bonuses: Vec<_> = bonuses.into_iter().map(|bonus| bonus.output).collect();
            let after = bonus_alternatives(produced, &bonuses);
            if after != [Mana::default()] {
                obligations.insert(
                    0,
                    Obligation::Output {
                        choices: after.into(),
                        life_cost: 0,
                    },
                );
            }
        }
        (pool, obligations)
    }

    pub(crate) fn activate_mana_ability(
        &mut self,
        player: PlayerId,
        permanent: PermanentId,
        ability_index: usize,
    ) -> Result<(), AgentError> {
        if !self.can_activate_mana_ability(player, permanent, ability_index) {
            return Err(AgentError("illegal mana activation".into()));
        }
        let ability = self.characteristics(permanent).activated_abilities[ability_index].clone();
        self.state
            .record_activation(permanent, ability_index, &ability);
        let (frame, resume) = match self.state.suspended_decision_mut().take() {
            Some(caller) => (caller.frame, Some(Box::new(caller.decision))),
            None => (EffectFrame::new(player, Vec::new()), None),
        };
        *self.state.suspended_decision_mut() = Some(SuspendedResolution {
            frame,
            decision: Decision::ManaUpfront {
                player,
                permanent,
                ability_index,
                tap_remaining: ability.tap_cost.as_ref().map_or(0, |cost| cost.count),
                resume,
            },
        });
        self.advance_mana_upfront()
    }

    /// Keep only selections that leave a complete witness for every caller.
    pub(crate) fn mana_tap_cost_actions(&self, player: PlayerId) -> Vec<Action> {
        let obligations = self.payment_obligations(player);
        let Some(Obligation::Tap { candidates, .. }) = obligations.first() else {
            return Vec::new();
        };
        let model = self.payment_model(player, None);
        let pool = self.pooled_mana(player);
        let mut actions = Vec::new();
        for permanent in candidates.iter() {
            if model
                .plan_after_tap(pool.clone(), obligations.clone(), *permanent)
                .is_some()
            {
                actions.push(Action::SelectCard {
                    player,
                    object: self.state.permanents[*permanent].as_ref().unwrap().object,
                });
            }
        }
        actions
    }

    pub(crate) fn pay_mana_tap_cost(
        &mut self,
        player: PlayerId,
        object: ObjectId,
    ) -> Result<(), AgentError> {
        if !self
            .mana_tap_cost_actions(player)
            .contains(&Action::SelectCard { player, object })
        {
            return Err(AgentError("illegal mana tap cost selection".into()));
        }
        let permanent = self.state.permanent_of(object).unwrap();
        self.tap_permanent(permanent, false);
        if let Some(SuspendedResolution {
            decision: Decision::ManaUpfront { tap_remaining, .. },
            ..
        }) = self.state.suspended_decision_mut()
        {
            *tap_remaining -= 1;
        }
        self.advance_mana_upfront()
    }

    /// Commit the automatic fixed-source prefix, then enter native spending.
    /// Neither spending nor chosen output is selected by a search witness.
    pub(crate) fn advance_mana_upfront(&mut self) -> Result<(), AgentError> {
        let Some(suspended) = &self.state.suspended_decision else {
            return Ok(());
        };
        let Decision::ManaUpfront {
            player,
            permanent,
            ability_index,
            tap_remaining,
            ..
        } = suspended.decision
        else {
            return Ok(());
        };
        if tap_remaining > 0 {
            return Ok(());
        }
        // With taps on, a mana-costed activation waits for the payer's taps
        // or PayCost instead of committing the automatic prefix.
        if self.characteristics(permanent).activated_abilities[ability_index]
            .mana_cost
            .mana_value
            > 0
            && self.has_mana_floats(player)
        {
            return Ok(());
        }
        self.pay_mana_upfront(player)
    }

    fn automatic_upfront_path(&self, player: PlayerId) -> Option<Vec<PaymentStep>> {
        let path = self.payment_model(player, None).plan(
            self.pooled_mana(player),
            self.payment_obligations(player),
            SourcePolicy::AutomaticPrefix,
        )?;
        path.iter()
            .take_while(|step| !matches!(step, PaymentStep::Spend))
            .all(|step| step.action(player).is_none())
            .then_some(path)
    }

    pub(crate) fn can_pay_mana_upfront(&self, player: PlayerId) -> bool {
        self.automatic_upfront_path(player).is_some()
    }

    pub(crate) fn pay_mana_upfront(&mut self, player: PlayerId) -> Result<(), AgentError> {
        let Some(path) = self.automatic_upfront_path(player) else {
            return Ok(());
        };
        let Decision::ManaUpfront {
            permanent,
            ability_index,
            ..
        } = self.state.suspended_decision.as_ref().unwrap().decision
        else {
            unreachable!()
        };
        for step in path {
            match step {
                PaymentStep::Fixed(source) => self.tap_fixed_mana_source(player, source),
                PaymentStep::Spend => break,
                _ => unreachable!("validated automatic prefix"),
            }
        }
        let cost = self.characteristics(permanent).activated_abilities[ability_index]
            .mana_cost
            .clone();
        let caller = self.state.suspended_decision_mut().take().unwrap();
        self.begin_mana_spend(
            player,
            cost,
            SpendCompletion::Upfront(Box::new(caller.decision)),
            caller.frame,
        )
    }

    /// Finish the paid activation using its owned caller. Costs commit once;
    /// only the output decision is restored after the source's effects run.
    fn complete_mana_upfront(&mut self, mut caller: SuspendedResolution) {
        let Decision::ManaUpfront {
            player,
            permanent,
            ability_index,
            resume,
            ..
        } = caller.decision
        else {
            unreachable!()
        };
        let ability = self.characteristics(permanent).activated_abilities[ability_index].clone();
        // Tapping the source for mana (CR 106.12) records what the source
        // was, so a sacrifice cost leaves its last known information; the
        // triggered mana abilities it triggers are read from the board once
        // the output is produced (CR 106.12a).
        let mut tapped = None;
        if ability.tap_source {
            self.tap_permanent(permanent, true);
            tapped = Some(self.tapped_for_mana(permanent));
        }
        let counter_costs = [
            ability
                .remove_counters
                .map(|(kind, count)| (kind, -i32::from(count))),
            ability
                .add_counters
                .map(|(kind, count)| (kind, i32::from(count))),
        ];
        for (kind, delta) in counter_costs.into_iter().flatten() {
            let Some(target) = self.permanent_object(permanent) else {
                break;
            };
            self.apply_immediate_event(ProposedEvent::CounterChange {
                provenance: CounterProvenance::new(EventCause::Cost),
                target,
                kind,
                delta,
                by: Some(player),
            });
        }
        if ability.life_cost > 0 {
            self.change_life(player, -(ability.life_cost as i32));
        }
        // Retain the output and original payment frame before any cost
        // effect can suspend. Resumption must not spend mana, tap or pay life
        // again, nor expose this output before sacrifice/zone work finishes.
        let source_ref = self.permanent_object(permanent);
        caller.decision = Decision::ManaChoice {
            player,
            choices: ability.mana_choices().unwrap(),
            production: ManaProduction::Output {
                source: source_ref,
                tapped,
            },
            resume,
        };
        // The sacrifice cost, then the ability's non-mana instructions, run
        // once as the source's own child frame (CR 605.3b): damage it deals
        // names the source and can be prevented. Resuming the choice never
        // reaches this frame again.
        let mut effects = Vec::new();
        if ability.sacrifice_source {
            effects.push(Effect::sacrifice_this());
        }
        effects.extend_from_slice(ability.mana_suffix());
        if !effects.is_empty() {
            let mut child = EffectFrame::new(player, effects);
            child.bind_source(
                self.state.permanents[permanent].as_ref().map(|p| p.object),
                source_ref,
            );
            self.run_before_decision(child, caller);
        } else {
            *self.state.suspended_decision_mut() = Some(caller);
        }
    }

    /// Tap a source of fixed mana for mana and resolve its triggered mana
    /// abilities. The planner offers only sources whose bonuses need no
    /// choice (`fixed_source_mana`), so this never suspends.
    fn tap_fixed_mana_source(&mut self, player: PlayerId, permanent: PermanentId) {
        self.perform_automatic_tap(player, AutomaticTap::fixed(permanent));
    }

    fn perform_automatic_tap(&mut self, player: PlayerId, tap: AutomaticTap) {
        let permanent = tap.permanent;
        let produced = tap
            .output
            .unwrap_or_else(|| fixed_output(&self.characteristics(permanent)));
        let source = self
            .permanent_object(permanent)
            .expect("an automatic mana source is on the battlefield");
        self.tap_permanent(permanent, true);
        let origin = self.mana_origin(source);
        self.state
            .mana_pool_mut(player)
            .add(&produced, Some(origin));
        let tapped = self.tapped_for_mana(permanent);
        let bonuses = self.production_bonuses(player, &tapped);
        let resolved = self.resolve_mana_bonuses(player, source, &produced, bonuses, None);
        assert!(
            resolved.is_none(),
            "an automatic tap's bonuses need no choice"
        );
    }

    /// Fixed tap output plus its triggered mana abilities' mana, or `None`
    /// when a bonus would need a choice among several produced types (no
    /// current fixed source produces more than one type). Callers check that
    /// the source has mana abilities, can tap and has not lost its abilities.
    fn fixed_source_mana(
        &self,
        player: PlayerId,
        permanent: PermanentId,
        view: &Characteristics,
        triggers: &[ManaTrigger],
    ) -> Option<Mana> {
        let bonuses = self.tap_bonuses(triggers, player, permanent);
        let output = fixed_output(view);
        if bonuses.is_empty() {
            return Some(output);
        }
        let mut outputs = production_alternatives(vec![output], &bonuses);
        (outputs.len() == 1).then(|| outputs.pop().unwrap())
    }

    /// Every triggered mana ability on the battlefield, with its source and
    /// controller, read from one board.
    fn mana_triggers(&self) -> Vec<ManaTrigger> {
        let mut triggers = Vec::new();
        for controller in [PlayerId(0), PlayerId(1)] {
            for (source, view) in self.layer_views(self.iter_battlefield_permanents(controller)) {
                for ability in &view.triggered_mana_abilities {
                    triggers.push(ManaTrigger {
                        source,
                        controller,
                        ability: ability.clone(),
                    });
                }
            }
        }
        triggers
    }

    /// The planner's estimate of the outputs of the triggered mana abilities
    /// that `producer` tapping `tapped` for mana would trigger, from the
    /// current board. Play reads them at production (`production_bonuses`).
    fn tap_bonuses(
        &self,
        triggers: &[ManaTrigger],
        producer: PlayerId,
        tapped: PermanentId,
    ) -> Vec<TriggeredManaOutput> {
        triggers
            .iter()
            .filter(|trigger| {
                trigger.ability.who.includes(trigger.controller, producer)
                    && self.permanent_matches_predicate(tapped, &trigger.ability.predicate)
            })
            .map(|trigger| trigger.ability.output.clone())
            .collect()
    }

    /// Record `permanent`, just tapped for mana, against every triggered
    /// mana ability predicate on the board.
    fn tapped_for_mana(&self, permanent: PermanentId) -> TappedForMana {
        let mut matched = Vec::new();
        for trigger in self.mana_triggers() {
            let predicate = trigger.ability.predicate;
            if !matched.contains(&predicate)
                && self.permanent_matches_predicate(permanent, &predicate)
            {
                matched.push(predicate);
            }
        }
        TappedForMana { matched }
    }

    /// The triggered mana abilities that `producer` tapping a permanent
    /// for mana triggers, read from the board as the mana is produced
    /// (CR 106.12a): a trigger source removed while the ability's costs were
    /// paid triggers nothing, while the tapped permanent answers from
    /// `tapped` even after it left. In board order.
    pub(crate) fn production_bonuses(
        &self,
        producer: PlayerId,
        tapped: &TappedForMana,
    ) -> Vec<PendingManaBonus> {
        if tapped.matched.is_empty() {
            return Vec::new();
        }
        self.mana_triggers()
            .into_iter()
            .filter(|trigger| {
                trigger.ability.who.includes(trigger.controller, producer)
                    && tapped.matched.contains(&trigger.ability.predicate)
            })
            .filter_map(|trigger| {
                Some(PendingManaBonus {
                    source: self.permanent_object(trigger.source)?,
                    output: trigger.ability.output,
                })
            })
            .collect()
    }

    /// Resolve `bonuses` after `tapped`'s production of `produced` into
    /// `player`'s pool (CR 605.4a): a fixed or single-type bonus adds at
    /// once; a choice among several produced types suspends as a
    /// `ManaChoice` naming its source and holding the rest. Returns that
    /// choice, or the caller's `resume` when every bonus resolved.
    pub(crate) fn resolve_mana_bonuses(
        &mut self,
        player: PlayerId,
        tapped: ObjectId,
        produced: &Mana,
        bonuses: Vec<PendingManaBonus>,
        resume: Option<Box<Decision>>,
    ) -> Option<Box<Decision>> {
        let mut pending = bonuses.into_iter();
        while let Some(bonus) = pending.next() {
            let mut choices = bonus.output.alternatives(produced);
            if choices.len() > 1 {
                return Some(Box::new(Decision::ManaChoice {
                    player,
                    choices,
                    production: ManaProduction::Bonus {
                        source: bonus.source,
                        tapped,
                        produced: produced.clone(),
                        pending: pending.collect(),
                    },
                    resume,
                }));
            }
            if let Some(mana) = choices.pop() {
                let origin = self.mana_origin(bonus.source);
                self.state.mana_pool_mut(player).add(&mana, Some(origin));
            }
        }
        resume
    }

    /// The same witness used by automatic production determines PayCost offers.
    pub(crate) fn can_pay_automatic_mana(&self, player: PlayerId, cost: &ManaCost) -> bool {
        let model = self.payment_model(player, None);
        let pool = self.pooled_mana(player);
        self.automatic_payability(&model, &pool, player, cost).1
    }

    /// `(fixed_payable, can_pay)` for `cost` from `pool` using `model`.
    fn automatic_payability(
        &self,
        model: &PaymentModel,
        pool: &Mana,
        player: PlayerId,
        cost: &ManaCost,
    ) -> (bool, bool) {
        let fixed_payable = model.fixed_total(pool).can_pay(cost);
        debug_assert_eq!(
            fixed_payable,
            model.payable(
                pool.clone(),
                vec![Obligation::mana(cost.clone())],
                SourcePolicy::FixedOnly,
            ),
            "direct FixedOnly payability differs from model"
        );
        let can_pay = fixed_payable || self.choice_tap_witness(player, cost, model, pool).is_some();
        (fixed_payable, can_pay)
    }

    /// Pooled mana with every class and expiry intact.
    pub(crate) fn pooled_mana(&self, player: PlayerId) -> Mana {
        self.state.players[player.0].mana_pool.mana().clone()
    }

    /// `object` as the producer of mana it is adding to a pool.
    pub(crate) fn mana_origin(&self, object: ObjectId) -> ManaOrigin {
        ManaOrigin {
            object,
            definition: self
                .observed_source_definition(object)
                .unwrap_or(crate::state::card::CardDefId::ANONYMOUS),
        }
    }

    pub(crate) fn produce_mana(
        &mut self,
        player: PlayerId,
        cost: &ManaCost,
    ) -> Result<(), AgentError> {
        for tap in self.automatic_taps(player, cost)? {
            self.perform_automatic_tap(player, tap);
        }
        Ok(())
    }

    /// The exact production prefix shared by execution and inspection.
    fn automatic_taps(
        &self,
        player: PlayerId,
        cost: &ManaCost,
    ) -> Result<Vec<AutomaticTap>, AgentError> {
        let model = self.payment_model(player, None);
        let Some(path) = model.plan(
            self.pooled_mana(player),
            vec![Obligation::mana(cost.clone())],
            SourcePolicy::FixedOnly,
        ) else {
            return self
                .automatic_choice_taps(player, cost, model)
                .ok_or_else(|| AgentError("not enough automatic mana".into()));
        };
        // The witness proved the cost payable; the choice selects
        // which fixed sources pay it. Only the payer's own hand and
        // permanents are read.
        let choice = model.choose_taps(
            &self.pooled_mana(player),
            cost,
            &self.saved_obligations(player),
            &path,
            || self.source_reluctance(&model),
            || self.payment_followups(player, cost, &model),
        );
        let sources = model.sources;
        Ok(choice
            .taps
            .into_iter()
            .map(|index| AutomaticTap::fixed(sources[index].permanent))
            .collect())
    }

    /// If fixed sources cannot fund a payment, let the existing witness
    /// select colors from simple tap abilities. Costs such as sacrificing a
    /// Treasure and non-mana instructions still require an explicit activation.
    fn automatic_choice_taps(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        model: PaymentModel<'_>,
    ) -> Option<Vec<AutomaticTap>> {
        let pool = self.pooled_mana(player);
        let path = self.choice_tap_witness(player, cost, &model, &pool)?;
        let mut taps = Vec::new();
        let mut steps = path.into_iter();
        while let Some(step) = steps.next() {
            let permanent = match step {
                PaymentStep::Fixed(source) | PaymentStep::Activate(source, _) => source,
                PaymentStep::Spend => break,
                _ => unreachable!("automatic witness contains only tap abilities"),
            };
            let index = model
                .sources
                .iter()
                .position(|s| s.permanent == permanent)?;
            let source = &model.sources[index];
            let output = match step {
                PaymentStep::Activate(_, ability) => {
                    let (Some(PaymentStep::Spend), Some(PaymentStep::Output(mode))) =
                        (steps.next(), steps.next())
                    else {
                        unreachable!("simple tap witness")
                    };
                    let activation = source
                        .abilities
                        .iter()
                        .find(|a| a.ability_index == ability)?;
                    Some(activation.outputs[mode].clone())
                }
                _ => None,
            };
            taps.push((index, AutomaticTap { permanent, output }));
        }

        // A search witness may tap a basic or restricted source before finding
        // the dual needed for a colored pip. Prune its concrete outputs in
        // preference order, retaining the witness's execution order.
        fn tap_output<'a>(
            sources: &'a [ManaSource],
            index: usize,
            tap: &'a AutomaticTap,
        ) -> &'a Mana {
            tap.output
                .as_ref()
                .unwrap_or_else(|| sources[index].fixed.as_ref().unwrap())
        }
        let mut produced = pool;
        for (index, tap) in &taps {
            produced.add(tap_output(&model.sources, *index, tap));
        }
        let owed = u16::from(cost.mana_value);
        if produced.total() > owed {
            let tail = self.saved_obligations(player);
            let reluctance = self.source_reluctance(&model);
            let mut order: Vec<_> = (0..taps.len()).collect();
            order.sort_unstable_by_key(|&i| std::cmp::Reverse((reluctance[taps[i].0], i)));
            let mut keep = vec![true; taps.len()];
            for i in order {
                if produced.total() <= owed {
                    break;
                }
                let mana = tap_output(&model.sources, taps[i].0, &taps[i].1);
                let surplus = if tail.is_empty() {
                    model.is_surplus(&produced, mana, cost, &[])
                } else {
                    model
                        .tapped(
                            taps.iter()
                                .enumerate()
                                .filter(|(j, _)| *j != i && keep[*j])
                                .map(|(_, (index, _))| *index),
                        )
                        .is_surplus(&produced, mana, cost, &tail)
                };
                if surplus {
                    keep[i] = false;
                    produced.spend(mana);
                }
            }
            let mut keep = keep.into_iter();
            taps.retain(|_| keep.next().unwrap());
        }
        Some(taps.into_iter().map(|(_, tap)| tap).collect())
    }

    /// Restrict the model to automatic taps and find their concrete modes.
    /// Execution and preview prune the returned witness.
    fn choice_tap_witness(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        model: &PaymentModel,
        pool: &Mana,
    ) -> Option<Vec<PaymentStep>> {
        let mut has_automatic = false;
        let mut has_manual = false;
        for ability in model.sources.iter().flat_map(|s| &s.abilities) {
            if ability.automatic {
                has_automatic = true;
            } else {
                has_manual = true;
            }
        }
        if !has_automatic {
            return None;
        }
        let mut obligations = vec![Obligation::mana(cost.clone())];
        obligations.extend(self.saved_obligations(player));
        if !has_manual {
            return model.plan(pool.clone(), obligations, SourcePolicy::All);
        }
        let mut choice_model = PaymentModel {
            cache: model.cache,
            sources: model.sources.clone(),
            life: model.life,
        };
        for source in Arc::make_mut(&mut choice_model.sources) {
            source.abilities.retain(|activation| activation.automatic);
        }
        choice_model.plan(pool.clone(), obligations, SourcePolicy::All)
    }

    /// Costs of the callers suspended behind the payment now being made.
    fn saved_obligations(&self, player: PlayerId) -> Vec<Obligation> {
        let Some(caller) = &self.state.suspended_decision else {
            // The colored, convoke and waterbend obligations can all belong
            // to this one announced cost. Paying it with mana replaces them all.
            return Vec::new();
        };
        let obligations = self.payment_obligations(player);
        if matches!(&caller.decision, Decision::Payment(payment) if payment.can_decline()) {
            // An optional cost only enters the witness when it is accepted.
            return obligations;
        }
        obligations.into_iter().skip(1).collect()
    }

    /// How reluctantly the smart payer taps each of `model`'s sources, read
    /// from the layered view: a creature (a land that is currently one
    /// included), then any other nonland permanent, then a land with an
    /// activated ability on the battlefield, mana abilities with a choice or
    /// a cost among them, then a land without the basic supertype, then a
    /// basic land. Static and triggered abilities do not count: tapping the
    /// land for mana gives none of them up.
    fn source_reluctance(&self, model: &PaymentModel) -> Vec<autotapper::Reluctance> {
        use autotapper::Reluctance;
        let _scan = self.board_scan();
        self.characteristics_many(model.sources.iter().map(|source| source.permanent))
            .zip(model.sources.iter())
            .map(|((id, view), source)| {
                debug_assert_eq!(id, source.permanent);
                let basic = || {
                    self.state[source.object]
                        .supertypes
                        .iter()
                        .any(|supertype| supertype.eq_ignore_ascii_case("basic"))
                };
                if view.is_creature() {
                    Reluctance::Creature
                } else if !view.is_land() {
                    Reluctance::Permanent
                } else if view
                    .activated_abilities
                    .iter()
                    .any(|ability| ability.zone == ActivationZone::Battlefield)
                {
                    Reluctance::UtilityLand
                } else if basic() {
                    Reluctance::BasicLand
                } else {
                    Reluctance::Land
                }
            })
            .collect()
    }

    /// Bounded mana-cost options for automatic source preference. Hand costs
    /// are deduplicated; this is not a count of playable cards. Each other
    /// castable card in the payer's hand at its cost with default cast
    /// choices, less one copy of the spell being paid for, which is still in
    /// hand at this boundary. Then each non-mana battlefield ability of a
    /// permanent the payer controls with a positive mana cost as charged now.
    ///
    /// An ability is listed when its source's activations are not prohibited,
    /// its use limit is not spent, its turn restriction and life cost hold
    /// and, for a {T} cost, its source can tap; `tapping` then names that
    /// source, which neither pays for the ability nor survives a payment
    /// that taps it. Timing within the turn, targets and the other costs
    /// (sacrifice, counted taps, crew, discard, waterbend taps) are not
    /// checked, as for hand cards. Single-use abilities matching the paid
    /// cost are excluded by cost equality, not activation identity; this
    /// can also exclude another source with the same cost.
    fn payment_followups(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        model: &PaymentModel,
    ) -> autotapper::PaymentProbe {
        let mut paying = match &cost.purpose {
            SpendingContext::Cast { spell, .. } => Some(spell.clone()),
            _ => None,
        };
        let mut probe = autotapper::PaymentProbe::default();
        let mut instant = Vec::new();
        for card in self.state.zone_cards(ZoneType::Hand, player) {
            let object_id = self.state.object_of(card);
            let object = &self.state[object_id];
            if !object.types.is_castable() {
                continue;
            }
            let Some(followup) =
                self.effective_spell_cost(player, object_id, Default::default(), None)
            else {
                continue;
            };
            if paying.as_ref() == Some(&SpellSnapshot(object.copiable_definition())) {
                paying = None;
                continue;
            }
            if followup.mana_value == 0 {
                continue;
            }
            let timing = self.has_instant_timing(player, object_id, Default::default());
            match probe.followups.iter().position(|known| *known == followup) {
                Some(index) => instant[index] |= timing,
                None => {
                    probe.followups.push(followup);
                    instant.push(timing);
                }
            }
        }
        probe.followups.truncate(autotapper::FOLLOWUPS);
        probe.instant = instant
            .iter()
            .take(autotapper::FOLLOWUPS)
            .enumerate()
            .fold(0, |set, (i, timing)| set | u64::from(*timing) << i);
        probe.hand = match probe.followups.len() {
            autotapper::FOLLOWUPS => u64::MAX,
            cards => (1 << cards) - 1,
        };
        probe.tapping = vec![None; probe.followups.len()];
        // One validated board for every view; a permanent with no activated
        // ability (a basic land's mana ability is not one) costs nothing more.
        let _scan = self.board_scan();
        let views = self.characteristics_many(model.sources.iter().map(|source| source.permanent));
        for (index, (id, view)) in views.enumerate() {
            if view.activated_abilities.is_empty() {
                continue;
            }
            let source = &model.sources[index];
            debug_assert_eq!(id, source.permanent);
            let Some(permanent) = self.state.permanents[source.permanent].as_ref() else {
                continue;
            };
            if self.activations_prohibited(source.permanent)
                || self.nonmana_activation_prohibited(permanent.object)
            {
                continue;
            }
            for (ability_index, ability) in view.activated_abilities.iter().enumerate() {
                if probe.followups.len() == autotapper::FOLLOWUPS {
                    return probe;
                }
                let single_use =
                    ability.tap_source || ability.sacrifice_source || ability.is_use_limited();
                if ability.zone != ActivationZone::Battlefield
                    || ability.mana_cost.mana_value == 0
                    || ability.mana_choices().is_some()
                    || permanent.activation_used(ability_index, ability)
                    || self.check_activation_terms(player, ability).is_err()
                    || (ability.tap_source && !self.permanent_can_tap(source.permanent))
                    || (single_use && ability.mana_cost == *cost)
                {
                    continue;
                }
                let tapping = ability.tap_source.then_some(index);
                let known = probe
                    .followups
                    .iter()
                    .zip(&probe.tapping)
                    .skip(probe.hand.count_ones() as usize)
                    .any(|(known, tap)| *known == ability.mana_cost && *tap == tapping);
                if !known {
                    probe.followups.push(ability.mana_cost.clone());
                    probe.tapping.push(tapping);
                }
            }
        }
        probe
    }

    /// Commit precisely the selected unit, retaining its expiry identity.
    fn spend_mana_unit(
        &mut self,
        player: PlayerId,
        unit: &Mana,
        cost: &ManaCost,
    ) -> Vec<SpentMana> {
        let combat_unit = unit
            .classes()
            .first()
            .is_some_and(|(class, _)| class.expiry == ManaExpiry::EndOfCombat);
        let pool = self.state.mana_pool_mut(player);
        let spent = pool.spend(unit, &cost.purpose);
        if !combat_unit {
            let remaining_non_combat = pool.total() - pool.expiry_total(ManaExpiry::EndOfCombat);
            if let Some(capture) = self.mana_capture(player) {
                capture.remove(unit.total(), remaining_non_combat, false);
            }
        }
        spent
    }

    /// Feasible unit selections preserve every saved caller, including the
    /// output of an in-progress filter before the outer cast's remaining cost.
    pub(crate) fn mana_spend_choices(
        &self,
        player: PlayerId,
        cost: &ManaCost,
    ) -> Vec<ManaSpendChoice> {
        let pool = self.pooled_mana(player);
        let mut tail = self.payment_obligations(player);
        if !tail.is_empty() {
            tail.remove(0);
        }
        let model = (!tail.is_empty()).then(|| self.payment_model(player, None));
        // When every usable unit is needed, only the first feasible ordering
        // matters: all orders leave the same receipt and remaining pool.
        let limit = if pool.usable_total(cost) == u16::from(cost.mana_value) {
            1
        } else {
            usize::MAX
        };
        pool.spending_choices(cost)
            .into_iter()
            .filter(|choice| {
                let mut remaining = pool.clone();
                remaining.spend(&choice.unit);
                if !remaining.can_pay(&choice.remaining) {
                    return false;
                }
                let Some(model) = &model else {
                    return true;
                };
                let mut obligations = vec![Obligation::mana(choice.remaining.clone())];
                obligations.extend(tail.clone());
                model.payable(remaining, obligations, SourcePolicy::All)
            })
            .take(limit)
            .collect()
    }

    pub(crate) fn begin_mana_spend(
        &mut self,
        player: PlayerId,
        cost: ManaCost,
        completion: SpendCompletion,
        frame: EffectFrame,
    ) -> Result<(), AgentError> {
        *self.state.suspended_decision_mut() = Some(SuspendedResolution {
            frame,
            decision: Decision::ManaSpend {
                player,
                cost,
                spent: PaymentReceipt::default(),
                completion: Box::new(completion),
            },
        });
        self.advance_mana_spend(None)
    }

    pub(crate) fn advance_mana_spend(
        &mut self,
        mut selected: Option<usize>,
    ) -> Result<(), AgentError> {
        loop {
            let Some(suspended) = self.state.suspended_decision.as_ref() else {
                return Ok(());
            };
            let Decision::ManaSpend {
                player,
                cost,
                completion,
                ..
            } = &suspended.decision
            else {
                return Ok(());
            };
            let player = *player;
            let cost = cost.clone();
            if cost.mana_value > 0 {
                let choices = self.mana_spend_choices(player, &cost);
                let index = match selected.take() {
                    Some(index) => index,
                    None if choices.len() == 1 => 0,
                    None if !choices.is_empty() => return Ok(()),
                    None => return Err(AgentError("no feasible mana spending choice".into())),
                };
                let choice = choices
                    .get(index)
                    .ok_or_else(|| AgentError("illegal mana spending choice".into()))?;
                let count_paid = !matches!(**completion, SpendCompletion::Upfront(_));
                let receipt = self.spend_mana_unit(player, &choice.unit, &cost);
                if count_paid {
                    if let Some(exposure) = self.exposure_mut() {
                        exposure.paid(player, u64::from(choice.unit.total()));
                    }
                }
                let suspended = self.state.suspended_decision_mut().as_mut().unwrap();
                let Decision::ManaSpend { cost, spent, .. } = &mut suspended.decision else {
                    unreachable!()
                };
                *cost = choice.remaining.clone();
                spent.record_mana(receipt);
                continue;
            }
            let suspended = self.state.suspended_decision_mut().take().unwrap();
            let Decision::ManaSpend {
                spent, completion, ..
            } = suspended.decision
            else {
                unreachable!()
            };
            match *completion {
                SpendCompletion::Announced(mut payment) => {
                    payment.receipt.record_mana(spent.mana().iter().cloned());
                    self.complete_announced_payment(player, payment)?;
                }
                SpendCompletion::Upfront(decision) => {
                    self.complete_mana_upfront(SuspendedResolution {
                        frame: suspended.frame,
                        decision: *decision,
                    });
                }
                SpendCompletion::Resolution(mut payment) => {
                    payment.receipt.record_mana(spent.mana().iter().cloned());
                    self.advance_resolution_payment(payment, suspended.frame);
                }
                SpendCompletion::Foretell(card) => self.complete_foretell(player, card),
            }
            return Ok(());
        }
    }

    /// The mana `player` could produce now by tapping each untapped
    /// permanent's mana abilities once, read from the layered board.
    pub fn producible_mana(&self, player: PlayerId) -> Mana {
        let mut total = Mana::default();
        let triggers = self.mana_triggers();
        for (permanent_id, view) in self.layer_views(self.iter_battlefield_permanents(player)) {
            if view.mana_abilities.is_empty()
                || !self.permanent_can_tap(permanent_id)
                || self.activations_prohibited(permanent_id)
            {
                continue;
            }
            if let Some(mana) = self.fixed_source_mana(player, permanent_id, &view, &triggers) {
                total.add(&mana);
            }
        }
        total
    }

    /// CR 106.4 — pools empty as each step/phase ends, applying a single
    /// per-unit disposition (`Drop | Keep | Recolor(Color)`) across classes
    /// whose expiry ends here while leaving later-expiring classes intact.
    pub(crate) fn clear_mana_pools(&mut self, step: StepKind) {
        let combat_expires = matches!(step, StepKind::EndOfCombat | StepKind::Cleanup);
        let expires = |expiry: ManaExpiry| match expiry {
            ManaExpiry::Step => true,
            ManaExpiry::EndOfCombat => combat_expires,
            ManaExpiry::EndOfTurn => step == StepKind::Cleanup,
        };
        for player in [PlayerId(0), PlayerId(1)] {
            if self.state.players[player.0].mana_pool.total() == 0 {
                continue;
            }
            let becomes_red = self.unspent_mana_becomes_red(player);
            let keeps_red = !becomes_red && self.keeps_unspent_mana(player, Color::Red);
            let summary = self
                .state
                .mana_pool_mut(player)
                .empty_expiring(expires, |color| {
                    if becomes_red {
                        EmptyDisposition::Recolor(Color::Red)
                    } else if keeps_red && color == Color::Red {
                        EmptyDisposition::Keep
                    } else {
                        EmptyDisposition::Drop
                    }
                });
            if let Some(capture) = self.mana_capture(player) {
                if summary.regular_expired > 0 {
                    capture.remove(summary.regular_lost, summary.remaining_non_combat, true);
                }
                capture.combat_expired += u64::from(summary.combat_lost);
            }
        }
    }

    /// Whether `player` controls a permanent replacing mana loss (CR 614.1a).
    fn unspent_mana_becomes_red(&self, player: PlayerId) -> bool {
        self.iter_battlefield_permanents(player).any(|id| {
            self.state.permanents[id].as_ref().is_some_and(|permanent| {
                self.state[permanent.object]
                    .replacement_effects
                    .contains(&ReplacementEffect::UnspentManaBecomesRed)
            }) && !self.characteristics(id).abilities_lost
        })
    }
}

/// A triggered mana ability on the battlefield and its controller, the
/// "you" its `who` is relative to.
struct ManaTrigger {
    source: PermanentId,
    controller: PlayerId,
    ability: TriggeredManaAbility,
}

/// Each distinct total the triggered mana abilities `bonuses` can add after a
/// production of `produced`: one per combination of their alternatives.
fn bonus_alternatives(produced: &Mana, bonuses: &[TriggeredManaOutput]) -> Vec<Mana> {
    let mut totals = vec![Mana::default()];
    for bonus in bonuses {
        let alternatives = bonus.alternatives(produced);
        if alternatives.is_empty() {
            continue;
        }
        let mut next = Vec::new();
        for total in &totals {
            for alternative in &alternatives {
                let mut sum = total.clone();
                sum.add(alternative);
                if !next.contains(&sum) {
                    next.push(sum);
                }
            }
        }
        totals = next;
    }
    totals
}

/// The mana a fixed source adds when tapped: every fixed production at once.
pub(crate) fn fixed_output(view: &Characteristics) -> Mana {
    let mut output = Mana::default();
    for ability in &view.mana_abilities {
        output.add(&ability.mana);
    }
    output
}

/// Whether untapped automatic taps produce at most `limit` total units and no
/// source has two outputs with different spend effects.
fn automatic_units_within(sources: &[ManaSource], limit: u16) -> bool {
    let mut total = 0_u16;
    for source in sources.iter().filter(|s| s.untapped) {
        let mut source_units = 0_u16;
        let mut first_output: Option<&Mana> = None;
        for output in source.fixed.iter().chain(
            source
                .abilities
                .iter()
                .filter(|a| a.automatic)
                .flat_map(|a| a.outputs.iter()),
        ) {
            source_units = source_units.max(output.total());
            if let Some(first) = first_output {
                if !same_spend_effects(first, output) {
                    return false;
                }
            } else {
                first_output = Some(output);
            }
        }
        total = total.saturating_add(source_units);
        if total > limit {
            return false;
        }
    }
    true
}

/// Remaining `[W, U, B, R, G, C, generic]` pips after spending every unit of
/// `pool` against a non-hybrid `cost` (`pool.spendable_units(cost) == pool.total()`).
fn resolution_remaining_pips(pool: &Mana, cost: &ManaCost) -> Option<[u8; 7]> {
    if !cost.hybrid.is_empty() {
        return None;
    }
    let mut avail = pool.mana;
    for (_, amounts) in pool
        .classes()
        .iter()
        .filter(|(class, _)| class.allows(&cost.purpose))
    {
        for (total, amount) in avail.iter_mut().zip(amounts) {
            *total = total.saturating_add(*amount);
        }
    }
    let mut rest = cost.cost;
    let mut excess: u8 = 0;
    for (have, required) in avail.into_iter().zip(rest[..6].iter_mut()) {
        let used = have.min(*required);
        *required -= used;
        excess = excess.saturating_add(have - used);
    }
    rest[Color::Generic as usize] = rest[Color::Generic as usize].checked_sub(excess)?;
    Some(rest)
}

/// A fully consumed output leaves only applicable riders on its receipt.
fn same_spend_effects(a: &Mana, b: &Mana) -> bool {
    fn effects(mana: &Mana) -> impl Iterator<Item = (&crate::state::mana::SpendEffect, u16)> {
        mana.classes().iter().filter_map(|(class, amounts)| {
            Some((
                class.effect.as_ref()?,
                amounts.iter().copied().map(u16::from).sum(),
            ))
        })
    }
    effects(a).eq(effects(b))
}

/// Fixed sources derive their output from their characteristics. A simple
/// choice-output activation carries the specific printed mode the payer chose.
struct AutomaticTap {
    permanent: PermanentId,
    output: Option<Mana>,
}

impl AutomaticTap {
    fn fixed(permanent: PermanentId) -> Self {
        Self {
            permanent,
            output: None,
        }
    }
}

/// Each distinct pool addition of a mana ability with `outputs` followed by
/// the triggered mana abilities it triggers. The planner reads these as one
/// output choice; play resolves the bonuses' own choices afterward.
fn production_alternatives(mut outputs: Vec<Mana>, bonuses: &[TriggeredManaOutput]) -> Vec<Mana> {
    if bonuses.is_empty() {
        // Keep the owned modes and their class metadata. Deduplicate in place,
        // preserving the first occurrence and therefore the first witness.
        let mut index = 0;
        while index < outputs.len() {
            if outputs[..index].contains(&outputs[index]) {
                outputs.remove(index);
            } else {
                index += 1;
            }
        }
        return outputs;
    }
    let mut totals = Vec::new();
    for mut output in outputs {
        let mut alternatives = bonus_alternatives(&output, bonuses);
        // A deterministic bonus needs no copy of the original production.
        if alternatives.len() == 1 {
            output.add(&alternatives.pop().unwrap());
            if !totals.contains(&output) {
                totals.push(output);
            }
        } else {
            for bonus in alternatives {
                let mut sum = output.clone();
                sum.add(&bonus);
                if !totals.contains(&sum) {
                    totals.push(sum);
                }
            }
        }
    }
    totals
}

/// `cost` as obligations: the rest paid with mana, then `keyword` of its
/// generic that the taps convoke or improvise names may pay, then
/// `waterbend` that any artifact's or creature's tap may pay. Every
/// keyword tapper may also waterbend, so a tap pays the waterbend part only
/// when nothing else could.
fn tap_obligations(cost: &ManaCost, keyword: u8, waterbend: u8) -> Vec<Obligation> {
    let generic = |amount: u8| {
        ManaCost {
            purpose: cost.purpose.clone(),
            ..ManaCost::default()
        }
        .with_generic(amount)
    };
    let mut obligations = vec![Obligation::mana(cost.reduced_generic(keyword + waterbend))];
    for (amount, taps) in [(keyword, Taps::Keyword), (waterbend, Taps::Any)] {
        if amount > 0 {
            obligations.push(Obligation::Pay {
                cost: generic(amount),
                taps,
            });
        }
    }
    obligations
}

#[cfg(test)]
mod tests;
