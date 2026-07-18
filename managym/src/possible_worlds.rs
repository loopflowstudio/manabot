// possible_worlds.rs
// First viewer-relative possible-world space and typed WorldQuery grammar.
//
// managym owns the hidden hand-count hypothesis domain, the exact
// compatible-deal measure, and deterministic materialization into exact Game
// branches. manabot beliefs, learned inference, and normalized distributions
// over this domain are out of scope; weights here are exact combinatorial
// counts, never probabilities.

use std::collections::BTreeMap;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::{
    agent::observation::Observation,
    flow::game::Game,
    state::{
        game_object::{CardId, PlayerId},
        zone::ZoneType,
    },
};

/// Stable, viewer-meaningful identity for a canonical query. Equivalent query
/// constructions share one canonical form and therefore one digest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorldQueryDigest(blake3::Hash);

impl WorldQueryDigest {
    pub fn as_bytes(&self) -> [u8; 32] {
        *self.0.as_bytes()
    }

    pub fn hex(&self) -> String {
        self.0.to_hex().to_string()
    }
}

impl std::fmt::Display for WorldQueryDigest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0.to_hex(), f)
    }
}

fn digest_of(value: &impl serde::Serialize) -> WorldQueryDigest {
    let bytes = serde_json::to_vec(value).expect("query serialization is infallible");
    WorldQueryDigest(blake3::hash(&bytes))
}

/// Exact binomial coefficient for the compatible-deal measure. The two-deck
/// pool keeps `n <= 40`, so `u128` is ample and the incremental formula is
/// exact at every step.
fn binom(n: u32, k: u32) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut acc: u128 = 1;
    for i in 0..k {
        acc = acc * (n - i) as u128 / (i + 1) as u128;
    }
    acc
}

/// One opponent-hand card-**name** multiset of the public hand size, drawn from
/// the opponent's unseen pool. The weight is the number of physical `CardId`
/// deals that yield this name-multiset, not a probability.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PossibleWorld {
    /// Name multiset of the hypothesized hand (only names with count > 0).
    pub hand: BTreeMap<String, u32>,
    /// Exact multivariate-hypergeometric count `Π C(n_i, k_i)`.
    pub weight: u128,
}

/// Exact-count query over a world's per-name hand count. `Not` applies only to
/// this form (per "Not(Q)"), never to `Has`/`Lacks`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum CountQuery {
    Exactly { card: String, count: u32 },
}

/// The minimal hand-count hypothesis grammar: `True`, `Has`, `Lacks`, `Q`, and
/// `Not(Q)`. This is deliberately not a general boolean algebra; it covers the
/// complete set of single-card count comparisons (`>=`, `<`, `=`, `!=`) plus
/// the tautology that a hand-count hypothesis needs.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum WorldQuery {
    True,
    Has { card: String, at_least: u32 },
    Lacks { card: String, fewer_than: u32 },
    Q(CountQuery),
    Not(CountQuery),
}

/// Normalized query form. Tautologies and impossibilities are folded against
/// the space's pool, so semantically-equivalent queries share one canonical
/// form, one digest, and one conditioned support.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum CanonicalWorldQuery {
    True,
    /// No world can satisfy the query.
    Empty,
    Has {
        card: String,
        at_least: u32,
    },
    Lacks {
        card: String,
        fewer_than: u32,
    },
    Q {
        card: String,
        count: u32,
    },
    NotQ {
        card: String,
        count: u32,
    },
}

impl CanonicalWorldQuery {
    pub fn digest(&self) -> WorldQueryDigest {
        digest_of(self)
    }

    /// Whether a world with per-name hand counts `hand` satisfies this
    /// canonical query. `Empty` never matches; `True` always matches.
    pub fn satisfies(&self, hand: &BTreeMap<String, u32>) -> bool {
        let count = |name: &str| hand.get(name).copied().unwrap_or(0);
        match self {
            CanonicalWorldQuery::True => true,
            CanonicalWorldQuery::Empty => false,
            CanonicalWorldQuery::Has { card, at_least } => count(card) >= *at_least,
            CanonicalWorldQuery::Lacks { card, fewer_than } => count(card) < *fewer_than,
            CanonicalWorldQuery::Q { card, count: c } => count(card) == *c,
            CanonicalWorldQuery::NotQ { card, count: c } => count(card) != *c,
        }
    }
}

impl WorldQuery {
    /// Syntactic digest of the input query form (distinct for
    /// `Has(x, 0)` vs `True`); useful for provenance alongside the canonical
    /// digest.
    pub fn digest(&self) -> WorldQueryDigest {
        digest_of(self)
    }

    /// Fold tautologies and impossibilities against `space`'s pool.
    pub fn canonicalize(&self, space: &PossibleWorldSpace) -> CanonicalWorldQuery {
        let pool = |name: &str| space.pool.get(name).copied().unwrap_or(0);
        match self {
            WorldQuery::True => CanonicalWorldQuery::True,
            WorldQuery::Has { card, at_least } => {
                if *at_least == 0 {
                    CanonicalWorldQuery::True
                } else if *at_least > pool(card) {
                    CanonicalWorldQuery::Empty
                } else {
                    CanonicalWorldQuery::Has {
                        card: card.clone(),
                        at_least: *at_least,
                    }
                }
            }
            WorldQuery::Lacks { card, fewer_than } => {
                if *fewer_than == 0 {
                    CanonicalWorldQuery::Empty
                } else if *fewer_than > pool(card) {
                    CanonicalWorldQuery::True
                } else {
                    CanonicalWorldQuery::Lacks {
                        card: card.clone(),
                        fewer_than: *fewer_than,
                    }
                }
            }
            WorldQuery::Q(CountQuery::Exactly { card, count }) => {
                if *count > pool(card) {
                    CanonicalWorldQuery::Empty
                } else if *count == 0 {
                    CanonicalWorldQuery::Lacks {
                        card: card.clone(),
                        fewer_than: 1,
                    }
                } else {
                    CanonicalWorldQuery::Q {
                        card: card.clone(),
                        count: *count,
                    }
                }
            }
            WorldQuery::Not(CountQuery::Exactly { card, count }) => {
                if *count > pool(card) {
                    CanonicalWorldQuery::True
                } else if *count == 0 {
                    CanonicalWorldQuery::Has {
                        card: card.clone(),
                        at_least: 1,
                    }
                } else {
                    CanonicalWorldQuery::NotQ {
                        card: card.clone(),
                        count: *count,
                    }
                }
            }
        }
    }
}

/// Always-succeeds summary of a query's support over the space. Zero support
/// is reported, not an error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SupportReceipt {
    pub query_digest: WorldQueryDigest,
    pub canonical_digest: WorldQueryDigest,
    pub support_size: usize,
    pub total_weight: u128,
}

/// Conditioned support: the worlds matching a query, with their exact weights,
/// plus the support receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConditioningReceipt {
    pub worlds: Vec<PossibleWorld>,
    pub receipt: SupportReceipt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConditioningError {
    /// No world matches the query. Never falls back to the unconditioned
    /// space; checking the query must not reveal whether the actual hidden
    /// world satisfies it.
    EmptySupport {
        query_digest: WorldQueryDigest,
        canonical_digest: WorldQueryDigest,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterializeError {
    /// The world is not realizable from the opponent's unseen pool, or the
    /// source state does not match the space's domain.
    InconsistentWorld,
}

/// A viewer-relative, revision-bound space over the opponent's hand-count
/// domain. Built from any live `Game` state; the decklist is implicit (the
/// unseen pool is the names of opponent cards in Hand ∪ Library, so public
/// cards are excluded by construction).
#[derive(Clone, Debug)]
pub struct PossibleWorldSpace {
    opponent: PlayerId,
    hand_size: u32,
    pool: BTreeMap<String, u32>,
    worlds: Vec<PossibleWorld>,
    total_weight: u128,
}

impl PossibleWorldSpace {
    /// The opponent whose hidden hand this space hypothesizes about.
    pub fn opponent(&self) -> PlayerId {
        self.opponent
    }

    /// The public hand size `H` every world must sum to.
    pub fn hand_size(&self) -> u32 {
        self.hand_size
    }

    /// The opponent unseen pool as a name multiset (`{name -> n_i}`).
    pub fn pool(&self) -> &BTreeMap<String, u32> {
        &self.pool
    }

    /// `C(N, H)`: the total number of physical deals over the pool.
    pub fn total_weight(&self) -> u128 {
        self.total_weight
    }

    /// Every compatible name-multiset world with its exact weight, in
    /// deterministic lexicographic order.
    pub fn worlds(&self) -> &[PossibleWorld] {
        &self.worlds
    }

    /// Build the space for `viewer` from a live `Game`. The opponent is the
    /// other player; the unseen pool is the opponent's Hand ∪ Library name
    /// multiset; `H` is the opponent's current hand size.
    pub fn for_viewer(game: &Game, viewer: PlayerId) -> Self {
        assert!(
            viewer.0 < game.state.players.len(),
            "viewer must name a player"
        );
        let opponent = PlayerId((viewer.0 + 1) % 2);
        let mut pool: BTreeMap<String, u32> = BTreeMap::new();
        for zone in [ZoneType::Hand, ZoneType::Library] {
            for &card in game.state.zones.zone_cards(zone, opponent) {
                *pool.entry(game.state.cards[card].name.clone()).or_insert(0) += 1;
            }
        }
        let hand_size = game.state.zones.size(ZoneType::Hand, opponent) as u32;
        Self::from_pool(opponent, hand_size, pool)
    }

    /// Build the space directly from a pool. Used by weight/canonicalization
    /// tests that do not need a live `Game`; `materialize` still works when
    /// given a source `Game` whose opponent pool matches.
    pub(crate) fn from_pool(
        opponent: PlayerId,
        hand_size: u32,
        pool: BTreeMap<String, u32>,
    ) -> Self {
        let names: Vec<(String, u32)> = pool.iter().map(|(k, &v)| (k.clone(), v)).collect();
        let total_weight = binom(pool.values().copied().sum(), hand_size);
        let mut worlds = Vec::new();
        let mut current: BTreeMap<String, u32> = BTreeMap::new();
        enumerate(&names, 0, hand_size, &mut current, &mut worlds, &pool);
        Self {
            opponent,
            hand_size,
            pool,
            worlds,
            total_weight,
        }
    }

    /// Always-succeeds support summary for `query`.
    pub fn support_receipt(&self, query: &WorldQuery) -> SupportReceipt {
        let canonical = query.canonicalize(self);
        let canonical_digest = canonical.digest();
        let query_digest = query.digest();
        let mut support_size = 0usize;
        let mut total_weight: u128 = 0;
        for world in &self.worlds {
            if canonical.satisfies(&world.hand) {
                support_size += 1;
                total_weight += world.weight;
            }
        }
        SupportReceipt {
            query_digest,
            canonical_digest,
            support_size,
            total_weight,
        }
    }

    /// Condition the space on `query`. Returns the matching worlds (with
    /// weights) and the receipt, or `Err(EmptySupport)` when no world matches.
    pub fn condition(&self, query: &WorldQuery) -> Result<ConditioningReceipt, ConditioningError> {
        let canonical = query.canonicalize(self);
        let canonical_digest = canonical.digest();
        let query_digest = query.digest();
        let mut support = Vec::new();
        let mut total_weight: u128 = 0;
        for world in &self.worlds {
            if canonical.satisfies(&world.hand) {
                support.push(world.clone());
                total_weight += world.weight;
            }
        }
        if support.is_empty() {
            return Err(ConditioningError::EmptySupport {
                query_digest,
                canonical_digest,
            });
        }
        let receipt = SupportReceipt {
            query_digest,
            canonical_digest,
            support_size: support.len(),
            total_weight,
        };
        Ok(ConditioningReceipt {
            worlds: support,
            receipt,
        })
    }

    /// Materialize `world` into an exact `Game` branch of `source`.
    ///
    /// Clones the source, reassigns only the opponent's hand+library split to
    /// match the world (lowest `CardId`s of each name to hand; seeded library
    /// shuffle), reusing the `resample_hidden` zone-update pattern. The
    /// viewer's zones, all public zones, and any viewer suspended-decision
    /// revealed cards are untouched. Deterministic per `(source, world, seed)`.
    /// No `journal_zones` call: the branch is an independent clone; undo is the
    /// driver's job.
    pub fn materialize(
        &self,
        source: &Game,
        world: &PossibleWorld,
        seed: u64,
    ) -> Result<Game, MaterializeError> {
        let sum_k: u32 = world.hand.values().copied().sum();
        if sum_k != self.hand_size {
            return Err(MaterializeError::InconsistentWorld);
        }
        for (name, &k) in &world.hand {
            if k > self.pool.get(name).copied().unwrap_or(0) {
                return Err(MaterializeError::InconsistentWorld);
            }
        }

        let mut branch = source.clone();
        let opponent = self.opponent;

        // Group the opponent's Hand ∪ Library CardIds by name, sorted ascending
        // so the lowest CardIds of each name go to the hand deterministically.
        let mut by_name: BTreeMap<String, Vec<CardId>> = BTreeMap::new();
        for zone in [ZoneType::Hand, ZoneType::Library] {
            for &card in branch.state.zones.zone_cards(zone, opponent) {
                by_name
                    .entry(branch.state.cards[card].name.clone())
                    .or_default()
                    .push(card);
            }
        }
        for cards in by_name.values_mut() {
            cards.sort_unstable_by_key(|c| c.0);
        }

        let mut new_hand: Vec<CardId> = Vec::with_capacity(self.hand_size as usize);
        let mut new_library: Vec<CardId> = Vec::new();
        for (name, cards) in &by_name {
            let k = world.hand.get(name).copied().unwrap_or(0) as usize;
            if k > cards.len() {
                return Err(MaterializeError::InconsistentWorld);
            }
            new_hand.extend_from_slice(&cards[..k]);
            new_library.extend_from_slice(&cards[k..]);
        }
        if new_hand.len() as u32 != self.hand_size {
            return Err(MaterializeError::InconsistentWorld);
        }

        new_hand.sort_unstable_by_key(|c| c.0);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        new_library.shuffle(&mut rng);

        branch
            .state
            .zones
            .reassign_hidden(opponent, new_hand, new_library);

        // Fail closed: the realized hand name-multiset must equal the world.
        let mut realized: BTreeMap<String, u32> = BTreeMap::new();
        for &card in branch.state.zones.zone_cards(ZoneType::Hand, opponent) {
            *realized
                .entry(branch.state.cards[card].name.clone())
                .or_insert(0) += 1;
        }
        if realized != world.hand {
            return Err(MaterializeError::InconsistentWorld);
        }

        Ok(branch)
    }
}

fn world_weight(hand: &BTreeMap<String, u32>, pool: &BTreeMap<String, u32>) -> u128 {
    hand.iter()
        .map(|(name, &k)| binom(pool.get(name).copied().unwrap_or(0), k))
        .product()
}

/// Recursively enumerate every name-multiset of size `remaining` drawn from
/// `names[idx..]` with per-name caps `n_i`, in deterministic lexicographic
/// order.
fn enumerate(
    names: &[(String, u32)],
    idx: usize,
    remaining: u32,
    current: &mut BTreeMap<String, u32>,
    out: &mut Vec<PossibleWorld>,
    pool: &BTreeMap<String, u32>,
) {
    if remaining == 0 {
        out.push(PossibleWorld {
            hand: current.clone(),
            weight: world_weight(current, pool),
        });
        return;
    }
    if idx >= names.len() {
        return;
    }
    let (name, n_i) = &names[idx];
    let max_k = (*n_i).min(remaining);
    for k in 0..=max_k {
        if k == 0 {
            current.remove(name);
        } else {
            current.insert(name.clone(), k);
        }
        enumerate(names, idx + 1, remaining - k, current, out, pool);
    }
    current.remove(name);
}

/// Narrow adapter over the current (pre-semantic) viewer projection.
///
/// RUL-7 owns the semantic `Observation` contracts (no published parent PR
/// yet). This slice depends on the viewer projection through exactly this one
/// call site, so later RUL-7 integration is mechanical: swap the projection
/// here, leave the space, query grammar, and materializer untouched.
pub fn viewer_observation(game: &Game, viewer: PlayerId) -> Observation {
    Observation::for_player(game, viewer)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(p: &[(&str, u32)]) -> BTreeMap<String, u32> {
        p.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn binom_basic() {
        assert_eq!(binom(0, 0), 1);
        assert_eq!(binom(4, 2), 6);
        assert_eq!(binom(40, 7), 18_643_560);
        assert_eq!(binom(3, 5), 0);
    }

    #[test]
    fn weights_match_brute_force_physical_deals() {
        let space =
            PossibleWorldSpace::from_pool(PlayerId(0), 2, pool(&[("A", 2), ("B", 1), ("C", 1)]));
        // Independent brute force over physical CardId deals: enumerate every
        // unordered 2-card deal from [A, A, B, C] and count name-multisets.
        let physical = ["A", "A", "B", "C"];
        let mut brute: BTreeMap<BTreeMap<String, u32>, u128> = BTreeMap::new();
        for i in 0..physical.len() {
            for j in (i + 1)..physical.len() {
                let mut m = BTreeMap::new();
                *m.entry(physical[i].to_string()).or_insert(0) += 1;
                *m.entry(physical[j].to_string()).or_insert(0) += 1;
                *brute.entry(m).or_insert(0) += 1;
            }
        }
        let mut space_map: BTreeMap<BTreeMap<String, u32>, u128> = BTreeMap::new();
        for w in &space.worlds {
            assert_eq!(
                space_map.insert(w.hand.clone(), w.weight),
                None,
                "duplicate world"
            );
        }
        assert_eq!(space_map, brute);
        assert_eq!(space.total_weight, binom(4, 2));
        let sum: u128 = space.worlds.iter().map(|w| w.weight).sum();
        assert_eq!(sum, space.total_weight);
    }

    #[test]
    fn empty_hand_size_is_single_empty_world() {
        let space = PossibleWorldSpace::from_pool(PlayerId(0), 0, pool(&[("A", 2)]));
        assert_eq!(space.worlds.len(), 1);
        assert!(space.worlds[0].hand.is_empty());
        assert_eq!(space.worlds[0].weight, 1);
        assert_eq!(space.total_weight, 1);
    }

    #[test]
    fn canonicalization_folds_tautologies_and_impossibilities() {
        let space =
            PossibleWorldSpace::from_pool(PlayerId(0), 7, pool(&[("Firebending Lesson", 3)]));
        let card = "Firebending Lesson";

        assert_eq!(
            WorldQuery::Has {
                card: card.into(),
                at_least: 0
            }
            .canonicalize(&space),
            CanonicalWorldQuery::True,
        );
        assert_eq!(
            WorldQuery::Has {
                card: card.into(),
                at_least: 99
            }
            .canonicalize(&space),
            CanonicalWorldQuery::Empty,
        );
        assert_eq!(
            WorldQuery::Lacks {
                card: card.into(),
                fewer_than: 0
            }
            .canonicalize(&space),
            CanonicalWorldQuery::Empty,
        );
        assert_eq!(
            WorldQuery::Lacks {
                card: card.into(),
                fewer_than: 99
            }
            .canonicalize(&space),
            CanonicalWorldQuery::True,
        );
        assert_eq!(
            WorldQuery::Q(CountQuery::Exactly {
                card: card.into(),
                count: 0
            })
            .canonicalize(&space),
            CanonicalWorldQuery::Lacks {
                card: card.into(),
                fewer_than: 1
            },
        );
        assert_eq!(
            WorldQuery::Q(CountQuery::Exactly {
                card: card.into(),
                count: 99
            })
            .canonicalize(&space),
            CanonicalWorldQuery::Empty,
        );
        assert_eq!(
            WorldQuery::Not(CountQuery::Exactly {
                card: card.into(),
                count: 0
            })
            .canonicalize(&space),
            CanonicalWorldQuery::Has {
                card: card.into(),
                at_least: 1
            },
        );
        assert_eq!(
            WorldQuery::Not(CountQuery::Exactly {
                card: card.into(),
                count: 99
            })
            .canonicalize(&space),
            CanonicalWorldQuery::True,
        );
        // A card absent from the pool folds against pool count 0.
        assert_eq!(
            WorldQuery::Has {
                card: "Absent".into(),
                at_least: 1
            }
            .canonicalize(&space),
            CanonicalWorldQuery::Empty,
        );
        assert_eq!(
            WorldQuery::Lacks {
                card: "Absent".into(),
                fewer_than: 1
            }
            .canonicalize(&space),
            CanonicalWorldQuery::True,
        );
    }

    #[test]
    fn equivalent_queries_share_canonical_digest_and_support() {
        let space =
            PossibleWorldSpace::from_pool(PlayerId(0), 3, pool(&[("A", 2), ("B", 2), ("C", 1)]));
        let card = "A";
        let true_worlds = space.condition(&WorldQuery::True).unwrap();

        let true_cases = [
            WorldQuery::Has {
                card: card.into(),
                at_least: 0,
            },
            WorldQuery::Lacks {
                card: card.into(),
                fewer_than: 99,
            },
            WorldQuery::Not(CountQuery::Exactly {
                card: card.into(),
                count: 99,
            }),
        ];
        for q in true_cases {
            let canon = q.canonicalize(&space);
            assert_eq!(
                canon,
                CanonicalWorldQuery::True,
                "{q:?} should canonicalize to True"
            );
            assert_eq!(canon.digest(), CanonicalWorldQuery::True.digest());
            let got = space.condition(&q).unwrap();
            assert_eq!(
                got.worlds, true_worlds.worlds,
                "{q:?} support must equal True support"
            );
            assert_eq!(
                got.receipt.canonical_digest,
                true_worlds.receipt.canonical_digest
            );
            assert_eq!(got.receipt.support_size, true_worlds.receipt.support_size);
            assert_eq!(got.receipt.total_weight, true_worlds.receipt.total_weight);
            // The syntactic query_digest is a provenance fingerprint and
            // deliberately differs across non-identical input forms.
            assert_ne!(got.receipt.query_digest, true_worlds.receipt.query_digest);
        }

        // Q(=0) <=> Lacks(<1): same canonical form, digest, and support.
        let q0 = WorldQuery::Q(CountQuery::Exactly {
            card: card.into(),
            count: 0,
        });
        let lacks1 = WorldQuery::Lacks {
            card: card.into(),
            fewer_than: 1,
        };
        assert_eq!(q0.canonicalize(&space), lacks1.canonicalize(&space));
        assert_eq!(
            q0.canonicalize(&space).digest(),
            lacks1.canonicalize(&space).digest()
        );
        assert_eq!(
            space.condition(&q0).unwrap().worlds,
            space.condition(&lacks1).unwrap().worlds,
        );

        // Not(Q(=0)) <=> Has(>=1): same canonical form, digest, and support.
        let not0 = WorldQuery::Not(CountQuery::Exactly {
            card: card.into(),
            count: 0,
        });
        let has1 = WorldQuery::Has {
            card: card.into(),
            at_least: 1,
        };
        assert_eq!(not0.canonicalize(&space), has1.canonicalize(&space));
        assert_eq!(
            not0.canonicalize(&space).digest(),
            has1.canonicalize(&space).digest()
        );
        assert_eq!(
            space.condition(&not0).unwrap().worlds,
            space.condition(&has1).unwrap().worlds,
        );
    }

    #[test]
    fn empty_support_is_explicit_error() {
        let space = PossibleWorldSpace::from_pool(PlayerId(0), 2, pool(&[("A", 2), ("B", 1)]));
        let impossible = WorldQuery::Has {
            card: "A".into(),
            at_least: 99,
        };
        let receipt = space.support_receipt(&impossible);
        assert_eq!(receipt.support_size, 0);
        assert_eq!(receipt.total_weight, 0);
        match space.condition(&impossible) {
            Err(ConditioningError::EmptySupport {
                query_digest,
                canonical_digest,
            }) => {
                assert_ne!(query_digest, canonical_digest);
            }
            other => panic!("expected EmptySupport, got {other:?}"),
        }
    }

    #[test]
    fn support_weights_partition_total() {
        let space =
            PossibleWorldSpace::from_pool(PlayerId(0), 3, pool(&[("A", 3), ("B", 2), ("C", 2)]));
        let total = space.total_weight;
        // Has(A, >=1) and Lacks(A, >=1) (i.e. Q(A, 0)) partition the space.
        let has = space
            .condition(&WorldQuery::Has {
                card: "A".into(),
                at_least: 1,
            })
            .unwrap();
        let lacks = space
            .condition(&WorldQuery::Lacks {
                card: "A".into(),
                fewer_than: 1,
            })
            .unwrap();
        assert_eq!(has.receipt.total_weight + lacks.receipt.total_weight, total);
        assert_eq!(
            has.receipt.support_size + lacks.receipt.support_size,
            space.worlds.len(),
        );
    }
}
