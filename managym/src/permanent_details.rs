//! Academy reads effective battlefield abilities from the same layered objects
//! as combat. Public permanent ids join these details to either viewer's rows.
use std::{collections::BTreeMap, sync::LazyLock};

use crate::{
    agent::ability_description::cost_description,
    experience::{KeywordAbilityView, PermanentDetails, PermanentRestriction},
    state::{
        ability::{
            Ability, Condition, Effect, EffectValue, PaymentCost, Per, TriggerCondition,
            TriggerSubject,
        },
        game_object::PlayerId,
        mana::Color,
    },
    Game,
};

/// A keyword trigger's public name. ActiveIf is evaluated for its controller;
/// intervening "if" and other trigger shapes are distinct rules.
fn keyword_label(ability: &Ability, active: impl FnOnce(&Condition) -> bool) -> Option<String> {
    static PROWESS: LazyLock<Ability> = LazyLock::new(Ability::prowess);
    if ability == &*PROWESS {
        return Some("Prowess".into());
    }
    let Ability::Triggered {
        condition,
        intervening_if: None,
        effects,
    } = ability
    else {
        return None;
    };
    let condition = match condition {
        TriggerCondition::ActiveIf {
            active_if,
            condition,
        } => {
            if !active(active_if) {
                return None;
            }
            condition.as_ref()
        }
        other => other,
    };
    match effects.as_slice() {
        [Effect::AddMana {
            mana,
            amount,
            until_end_of_combat: true,
        }] if condition == &TriggerCondition::attacks(TriggerSubject::This, Per::Object) => {
            let red = mana.mana[Color::Red as usize];
            if red == 0 || mana.classes.is_some() || mana.total() != u16::from(red) {
                return None;
            }
            match amount {
                EffectValue::Fixed(1) => Some(format!("Firebending {red}")),
                EffectValue::SourcePower | EffectValue::BattlefieldMatching { .. } => {
                    Some("Firebending X".into())
                }
                _ => None,
            }
        }
        [Effect::CounterUnlessPays { cost }]
            if matches!(
                condition,
                TriggerCondition::BecomesTargeted {
                    subject: TriggerSubject::This
                }
            ) =>
        {
            Some(match cost {
                PaymentCost::Mana(cost) => format!("Ward {}", cost_description(cost)),
                PaymentCost::Life(n) => format!("Ward—Pay {n} life"),
                PaymentCost::Discard => "Ward—Discard a card".into(),
                PaymentCost::Waterbend { generic } => format!("Ward—Waterbend {{{generic}}}"),
            })
        }
        _ => None,
    }
}

impl Game {
    pub fn permanent_details(&self, viewer: PlayerId) -> BTreeMap<u32, PermanentDetails> {
        let sides = [viewer, PlayerId(1 - viewer.0)];
        let ids = sides
            .into_iter()
            .flat_map(|side| self.iter_battlefield_including_phased_out(side));
        self.characteristics_many(ids)
            .map(|(id, view)| {
                let permanent = self.state.permanents[id].as_ref().unwrap();
                let controller = permanent.controller;
                // Layer 6 starts with the current copiable face's trigger list.
                // Grants append instances; ability loss clears the entire prefix.
                let own = if view.abilities_lost {
                    0
                } else {
                    self.state[permanent.object].abilities.len()
                };
                let keyword_abilities = view
                    .triggered_abilities
                    .iter()
                    .enumerate()
                    .filter_map(|(i, trigger)| {
                        Some(KeywordAbilityView {
                            label: keyword_label(&trigger.ability, |condition| {
                                self.check_static_condition(condition, controller)
                            })?,
                            granted: i >= own,
                            effect_timestamp: trigger.effect_timestamp.map(|timestamp| timestamp.0),
                        })
                    })
                    .collect();
                let restrictions = [
                    (
                        PermanentRestriction::CantBeBlocked,
                        self.cant_be_blocked_unconditionally(id, &view, controller),
                    ),
                    (
                        PermanentRestriction::CantBlock,
                        self.cant_block_unconditionally(id, &view),
                    ),
                    (
                        PermanentRestriction::CantAttackOrBlock,
                        self.cant_attack_or_block(id, &view),
                    ),
                    (PermanentRestriction::DoesntUntap, self.doesnt_untap(id)),
                    (PermanentRestriction::MustAttack, self.must_attack(id)),
                ]
                .into_iter()
                .filter_map(|(rule, active)| active.then_some(rule))
                .collect();
                (
                    permanent.id.0,
                    PermanentDetails {
                        attacks_despite_defender: self.attacks_despite_defender(&view, controller),
                        keyword_abilities,
                        keyword_grants: (view.keywords.iter().next().is_some()).then(|| {
                            view.keyword_grants
                                .iter()
                                .map(|(k, t)| (k.name().to_owned(), t.map(|t| t.0)))
                                .collect()
                        }),
                        restrictions,
                    },
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::content_pack::default_content_pack;

    #[test]
    fn permanent_details_recognize_keyword_declarations_on_every_face_and_token() {
        for face in default_content_pack().definitions() {
            let labels: Vec<_> = face
                .abilities
                .iter()
                .filter_map(|ability| keyword_label(ability, |_| true))
                .collect();
            // Keyword lines can be comma-separated and followed by reminder
            // text. A spell naming a token or granting an ability isn't a declaration.
            let text = face.text_box.to_lowercase();
            let declarations: Vec<_> = text
                .lines()
                .flat_map(|line| line.split(" (").next().unwrap().split(", "))
                .map(|clause| clause.strip_prefix("this creature has ").unwrap_or(clause))
                .collect();
            for keyword in ["prowess", "firebending", "ward"] {
                let printed = declarations.iter().any(|clause| match keyword {
                    "prowess" => *clause == keyword,
                    "firebending" => clause
                        .strip_prefix("firebending ")
                        .and_then(|rest| rest.split_whitespace().next())
                        .is_some_and(|amount| amount == "x" || amount.parse::<u8>().is_ok()),
                    _ => clause.starts_with("ward {") || clause.starts_with("ward—"),
                });
                assert_eq!(
                    labels
                        .iter()
                        .any(|label| label.to_lowercase().starts_with(keyword)),
                    printed,
                    "{}: {keyword} in {}",
                    face.key,
                    face.text_box
                );
            }
        }
    }
}
