//! Small strategic ratings compiled once from the same serialized rules definitions
//! exposed by card_definitions_json. Modal bodies deliberately remain opaque to
//! polarity/damage, matching the original heuristic's approximate strategy.
//! Borrow the projection directly: no alternate effect vocabulary or copied
//! program bodies are needed to extract these ratings.
use crate::cards::content_pack::default_content_pack;
use serde_json::Value;
use std::{collections::HashMap, sync::OnceLock};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Polarity {
    Harmful,
    Beneficial,
}

#[derive(Clone, Debug, Default)]
pub(super) struct Spell {
    pub polarity: Option<Polarity>,
    pub selector: String,
    pub damage: i32,
    pub kicked_damage: i32,
    pub lesson_damage: bool,
    pub pump: (i32, i32),
}

#[derive(Debug, Default)]
pub(super) struct Knowledge {
    pub spell: Option<Spell>,
    pub instant: bool,
    pub fetch: bool,
    pub spare: bool,
    pub pump: bool,
}

pub(super) fn knowledge(name: &str) -> Option<&'static Knowledge> {
    static CARDS: OnceLock<HashMap<String, Knowledge>> = OnceLock::new();
    CARDS
        .get_or_init(|| {
            default_content_pack()
                .definitions()
                .map(|d| {
                    let value = serde_json::to_value(d).expect("card definitions serialize");
                    (d.name.clone(), derive(&value))
                })
                .collect()
        })
        .get(name)
}

fn target(body: &Value) -> Option<&Value> {
    body.get("target").or_else(|| {
        ["what", "on", "from", "player", "affected"]
            .iter()
            .find_map(|key| body.get(key)?.get("Target"))
    })
}
fn spec_name(spec: &Value) -> &str {
    spec.as_str()
        .or_else(|| spec.as_object()?.keys().next().map(String::as_str))
        .unwrap_or("")
}
fn number(value: &Value) -> i32 {
    value.as_i64().unwrap_or(0) as i32
}
fn variant(effect: &Value) -> (&str, &Value) {
    if let Some(name) = effect.as_str() {
        return (name, &Value::Null);
    }
    effect
        .as_object()
        .and_then(|o| o.iter().next())
        .map_or(("", &Value::Null), |(name, body)| (name.as_str(), body))
}
fn children<'a>(name: &str, body: &'a Value) -> impl Iterator<Item = &'a [Value]> {
    body.as_object()
        .filter(|_| name != "Modal")
        .into_iter()
        .flat_map(|o| o.values())
        .filter_map(|v| {
            let a = v.as_array()?;
            a.first().filter(|v| v.is_object() || v.is_string())?;
            Some(a.as_slice())
        })
}
fn target_effects(effects: &[Value], visit: &mut impl FnMut(&str, &Value)) {
    for effect in effects {
        let (name, body) = variant(effect);
        for nested in children(name, body) {
            target_effects(nested, visit);
        }
        if target(body).is_some()
            || matches!(
                name,
                "CounterUnlessPays" | "TargetCreaturesDealPowerDamageToLastTarget"
            )
        {
            visit(name, body);
        }
    }
}
fn first_spec(effects: &[Value]) -> Option<&Value> {
    for effect in effects {
        let (name, body) = variant(effect);
        if name == "TargetClause" {
            return Some(&body["requirement"]["spec"]);
        }
        if let Some(t) = target(body) {
            return Some(t);
        }
        if let Some(o) = body.as_object() {
            for (key, value) in o {
                if let Some(a) = value.as_array() {
                    let nested = if key == "modes" {
                        a.iter().find_map(|mode| first_spec(array(mode)))
                    } else if a.first().is_some_and(Value::is_object) {
                        first_spec(a)
                    } else {
                        None
                    };
                    if nested.is_some() {
                        return nested;
                    }
                }
            }
        }
    }
    None
}
fn array(value: &Value) -> &[Value] {
    value.as_array().map(Vec::as_slice).unwrap_or(&[])
}
fn damage(effects: &[Value], kicked: bool) -> i32 {
    let mut best = 0;
    for effect in effects {
        let (name, body) = variant(effect);
        if name == "If" && body["condition"] == "Kicked" {
            best = best.max(damage(
                array(&body[if kicked { "then" } else { "otherwise" }]),
                kicked,
            ));
            continue;
        }
        for nested in children(name, body) {
            best = best.max(damage(nested, kicked));
        }
        if name == "DealDamage" {
            best = best.max(number(
                body["amount"]
                    .get("Fixed")
                    .unwrap_or(&body["amount"]["GraveyardMatching"]["base"]),
            ));
        }
    }
    best
}
fn fixed_pump(body: &Value) -> Option<(i32, i32)> {
    let change = &body["modification"]["ModifyPt"];
    Some((
        number(change["power"].get("Fixed")?),
        number(change["toughness"].get("Fixed")?),
    ))
}
fn polarity(name: &str, body: &Value) -> Option<Polarity> {
    use Polarity::*;
    match name {
        "DealDamage"
        | "TargetCreaturesDealPowerDamageToLastTarget"
        | "Destroy"
        | "CounterSpell"
        | "CounterUnlessPays"
        | "Discard" => Some(Harmful),
        "Move"
            if body["to"] != "Battlefield"
                && target(body).is_some_and(|t| !spec_name(t).contains("Graveyard")) =>
        {
            Some(Harmful)
        }
        "Untap" | "Earthbend" => Some(Beneficial),
        "ApplyContinuous" => match variant(&body["modification"]).0 {
            "ModifyPt" => match fixed_pump(body) {
                Some((power, toughness)) => match (power + toughness).cmp(&0) {
                    std::cmp::Ordering::Greater => Some(Beneficial),
                    std::cmp::Ordering::Less => Some(Harmful),
                    _ => None,
                },
                None => Some(Beneficial),
            },
            "AddKeywords" | "CantBeBlocked" | "AddSubtypes" => Some(Beneficial),
            _ => None,
        },
        _ => None,
    }
}
fn derive(d: &Value) -> Knowledge {
    let mut out = Knowledge {
        instant: d["types"]["types"]
            .as_array()
            .is_some_and(|a| a.iter().any(|v| v == "Instant")),
        ..Knowledge::default()
    };
    let effects = array(&d["spell_effects"]);
    let first = array(&d["targeting"])
        .first()
        .map(|v| &v["spec"])
        .or_else(|| first_spec(effects));
    if let Some(first) = first {
        let mut signs = vec![];
        let mut spell = Spell {
            selector: spec_name(first).into(),
            damage: damage(effects, false),
            kicked_damage: damage(effects, true),
            ..Spell::default()
        };
        target_effects(effects, &mut |name, body| {
            if let Some(p) = polarity(name, body) {
                if !signs.contains(&p) {
                    signs.push(p);
                }
            }
            if name == "ApplyContinuous" {
                if let Some((power, toughness)) = fixed_pump(body) {
                    spell.pump.0 += power;
                    spell.pump.1 += toughness;
                }
            }
            if name == "DealDamage" {
                if let Some(predicate) =
                    body["amount"]["GraveyardMatching"]["predicate"].as_object()
                {
                    spell.lesson_damage |= predicate.get("subtype").is_some_and(|v| v == "Lesson")
                        && predicate.iter().all(|(k, v)| {
                            k == "subtype"
                                || v.is_null()
                                || v == false
                                || v == 0
                                || v == ""
                                || v.as_array().is_some_and(Vec::is_empty)
                                || v.as_object().is_some_and(serde_json::Map::is_empty)
                        });
                }
            }
        });
        if signs.len() == 1 {
            spell.polarity = Some(signs[0]);
        }
        out.spell = Some(spell);
    }
    for ability in array(&d["activated_abilities"]) {
        if ability.get("zone").is_some_and(|v| v != "Battlefield") {
            continue;
        }
        for effect in array(&ability["effects"]) {
            let (kind, body) = variant(effect);
            out.fetch |= kind == "SelectCards"
                && body["from"]["Zone"] == "Library"
                && body["predicate"]["supertype"] == "basic"
                && body["predicate"]["card_type"] == "Land";
            out.spare |= matches!(kind, "DrawCards" | "GainLife");
            out.pump |= kind == "ApplyContinuous"
                && body["affected"] == "This"
                && fixed_pump(body).is_some_and(|(power, _)| power > 0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn heuristic_knowledge_reads_rules_not_card_names() {
        let spell =
            |effects: Value| json!({"spell_effects":effects,"targeting":[{"spec":"Creature"}]});
        let targeted = |effects: Value| derive(&spell(effects)).spell.unwrap();
        assert_eq!(
            targeted(json!([{"Tap":{"what":{"Target":"Creature"}}}])).polarity,
            None
        );
        let damage = json!({"DealDamage": {
            "what": {"Target": "Creature"}, "amount": {"Fixed": 1}
        }});
        assert_eq!(
            targeted(json!([damage, {"Untap": {"what": {"Target": "Creature"}}}])).polarity,
            None
        );
        assert_eq!(
            targeted(json!([{"Move":{"what":{"Target":"Creature"},"to":"Hand"}}])).polarity,
            Some(Polarity::Harmful)
        );
        assert_eq!(
            targeted(json!([{"Move":{"what":{"Target":"Creature"},"to":"Battlefield"}}])).polarity,
            None
        );
        let shrink = targeted(json!([{"ApplyContinuous": {
            "affected": {"Target": "Creature"},
            "modification": {"ModifyPt": {
                "power": {"Fixed": -2}, "toughness": {"Fixed": -2}
            }}
        }}]));
        assert_eq!(shrink.polarity, Some(Polarity::Harmful));
        assert_eq!(shrink.pump, (-2, -2));
        let modal = derive(&json!({"spell_effects": [{"Modal": {"modes": [[damage]]}}]}))
            .spell
            .unwrap();
        // Targets inside modes are visible, but their polarity/damage is unrated.
        assert_eq!(modal.selector, "Creature");
        assert_eq!(modal.polarity, None);
        assert_eq!(modal.damage, 0);
    }
    #[test]
    fn heuristic_catalog_kicker_lessons_and_abilities() {
        let fire = knowledge("Firebending Lesson")
            .unwrap()
            .spell
            .as_ref()
            .unwrap();
        assert_eq!((fire.damage, fire.kicked_damage), (2, 5));
        let combustion = knowledge("Combustion Technique")
            .unwrap()
            .spell
            .as_ref()
            .unwrap();
        assert_eq!((combustion.damage, combustion.lesson_damage), (2, true));
        assert_eq!(
            knowledge("Fancy Footwork")
                .unwrap()
                .spell
                .as_ref()
                .unwrap()
                .pump,
            (2, 2)
        );
        assert!(knowledge("Fabled Passage").unwrap().fetch);
        assert_eq!(
            knowledge("Counterspell")
                .unwrap()
                .spell
                .as_ref()
                .unwrap()
                .selector,
            "Spell"
        );
    }
}
