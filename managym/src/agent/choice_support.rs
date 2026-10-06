//! Constructive support for canonical subset tapes over shared legal offers.
//!
//! One offer token precedes include/exclude tokens in native role/candidate
//! order. Only independent distinct selections are certified. This read-only
//! support can be retained with a training row; executing its answer still
//! requires the original bound offer set and normal Command validation.

use std::collections::BTreeSet;

use super::structured_offer::{
    ChoiceAnswer, ChoiceStep, OfferSubmission, OfferVerb, StructuredOfferProjection,
};

#[derive(Clone, Debug)]
pub struct ChoiceSupport {
    projection: StructuredOfferProjection,
}

pub enum PrefixSupport {
    Alternatives(Vec<bool>),
    Complete(OfferSubmission),
}

impl ChoiceSupport {
    pub fn new(projection: StructuredOfferProjection) -> Result<Self, String> {
        if projection.schema_version != crate::decision::SEMANTIC_DECISION_VERSION
            || projection.factorization_version != 1
        {
            return Err("unsupported choice schema or factorization order".into());
        }
        if projection.offers.is_empty() {
            return Err("projection must contain at least one offer".into());
        }
        let mut offers = BTreeSet::new();
        for offer in &projection.offers {
            if !offers.insert(offer.id) || offer.actor != projection.actor {
                return Err("duplicate offer or inconsistent actor".into());
            }
            let mut roles = BTreeSet::new();
            for choice in &offer.choices {
                let ChoiceStep::Select {
                    role,
                    candidates,
                    min,
                    max,
                    ordered,
                    distinct,
                    ..
                } = choice;
                if !roles.insert(*role)
                    || *ordered
                    || !distinct
                    || !candidates.depends_on.is_empty()
                {
                    return Err(
                        "support requires unique distinct unordered roles without dependencies"
                            .into(),
                    );
                }
                let rows = candidates
                    .initial
                    .as_ref()
                    .ok_or("missing initial candidates")?;
                if min > max || usize::from(*max) > rows.len() {
                    return Err("invalid choice cardinality".into());
                }
                let ids: BTreeSet<_> = rows.iter().map(|row| row.id).collect();
                if ids.len() != rows.len() {
                    return Err("duplicate candidate identity".into());
                }
            }
        }
        Ok(Self { projection })
    }

    /// Disjoint prefix cylinders for the next native Command. Unchosen tails
    /// integrate to one. In payments, earlier excluded taps remain unreachable.
    pub fn command_prefixes(
        &self,
        prefix: &[usize],
        executed: usize,
    ) -> Result<Vec<Vec<usize>>, String> {
        self.query(prefix)?;
        let first = &self.projection.offers[0];
        if prefix.is_empty()
            && !matches!(
                first.verb,
                OfferVerb::DeclareAttackers | OfferVerb::DeclareBlockers | OfferVerb::PayWaterbend
            )
        {
            return Ok((0..self.projection.offers.len()).map(|i| vec![i]).collect());
        }
        let tokens = if prefix.is_empty() {
            vec![0]
        } else {
            prefix.to_vec()
        };
        let offer = &self.projection.offers[tokens[0]];
        let append = |base: &[usize], bit| {
            let mut p = base.to_vec();
            p.push(bit);
            p
        };
        match offer.verb {
            OfferVerb::DeclareAttackers => {
                let PrefixSupport::Alternatives(allowed) = self.query(&tokens)? else {
                    return Err("attacker prefix complete".into());
                };
                Ok(allowed
                    .iter()
                    .enumerate()
                    .filter(|(_, yes)| **yes)
                    .map(|(bit, _)| append(&tokens, bit))
                    .collect())
            }
            OfferVerb::PayWaterbend => {
                let mut routes = Vec::new();
                let mut tail = tokens;
                loop {
                    match self.query(&tail)? {
                        PrefixSupport::Complete(_) => {
                            routes.push(tail);
                            break;
                        }
                        PrefixSupport::Alternatives(allowed) => {
                            if allowed[1] {
                                routes.push(append(&tail, 1));
                            }
                            if !allowed[0] {
                                break;
                            }
                            tail.push(0);
                        }
                    }
                }
                Ok(routes)
            }
            OfferVerb::DeclareBlockers | OfferVerb::Cast => {
                let role_index = if offer.verb == OfferVerb::DeclareBlockers {
                    executed
                } else {
                    0
                };
                let choice = offer
                    .choices
                    .get(role_index)
                    .ok_or("role prefix complete")?;
                let ChoiceStep::Select {
                    candidates,
                    min,
                    max,
                    ..
                } = choice;
                if *max > 1 {
                    return Err("unsupported Command role cardinality".into());
                }
                let expected = 1 + offer.choices[..role_index]
                    .iter()
                    .map(|choice| {
                        let ChoiceStep::Select { candidates, .. } = choice;
                        candidates.initial.as_ref().expect("validated source").len()
                    })
                    .sum::<usize>();
                if tokens.len() != expected {
                    return Err("prefix must end at role boundary".into());
                }
                let count = candidates.initial.as_ref().expect("validated source").len();
                let mut routes: Vec<_> = (0..count)
                    .map(|selected| {
                        let mut p = tokens.clone();
                        p.extend((0..count).map(|i| usize::from(i == selected)));
                        p
                    })
                    .collect();
                if *min == 0 {
                    let mut p = tokens;
                    p.extend(vec![0; count]);
                    routes.push(p);
                }
                for route in &routes {
                    self.query(route)?;
                }
                Ok(routes)
            }
            _ => Err("unsupported prefix-to-Command boundary".into()),
        }
    }

    pub fn query(&self, prefix: &[usize]) -> Result<PrefixSupport, String> {
        let Some(&offer_index) = prefix.first() else {
            return Ok(PrefixSupport::Alternatives(vec![
                true;
                self.projection
                    .offers
                    .len()
            ]));
        };
        let offer = self
            .projection
            .offers
            .get(offer_index)
            .ok_or("invalid offer token")?;
        let mut cursor = 1;
        let mut answers = Vec::new();
        for choice in &offer.choices {
            let ChoiceStep::Select {
                role,
                candidates,
                min,
                max,
                ..
            } = choice;
            let rows = candidates
                .initial
                .as_ref()
                .expect("validated candidate source");
            let mut selected = Vec::new();
            for (ordinal, row) in rows.iter().enumerate() {
                let allowed = [
                    selected.len() + rows.len() - ordinal - 1 >= usize::from(*min),
                    selected.len() < usize::from(*max),
                ];
                let Some(&token) = prefix.get(cursor) else {
                    return Ok(PrefixSupport::Alternatives(allowed.to_vec()));
                };
                if token > 1 || !allowed[token] {
                    return Err("illegal compound token".into());
                }
                if token == 1 {
                    selected.push(row.id);
                }
                cursor += 1;
            }
            answers.push(ChoiceAnswer::Candidates {
                role: *role,
                candidates: selected,
            });
        }
        if cursor != prefix.len() {
            return Err("compound prefix has trailing choices".into());
        }
        Ok(PrefixSupport::Complete(OfferSubmission {
            offer_id: offer.id,
            answers,
        }))
    }
}
