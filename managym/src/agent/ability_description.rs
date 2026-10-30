//! Human descriptions derived from the same ability/effect definitions that resolve.
//! Offers, public stack inspection and text capture share this projection. It never
//! evaluates hidden state or predicts an outcome; quantities remain symbolic.

use crate::state::{
    ability::{
        Ability, Effect, EffectValue, LinkedChooser, LinkedGrant, PaymentCost, StaticCondition,
        TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{Card, CardType, Keywords},
    mana::{Mana, ManaCost},
    permanent::{BecomesCreature, CounterKind},
    predicate::CardPredicate,
    zone::ZoneType,
};

/// Inline delayed payloads take precedence: their ability index is not a card slot.
pub(crate) fn trigger_description(card: &Card, index: usize, inline: Option<&[Effect]>) -> String {
    if let Some(effects) = inline {
        return format!(
            "{} — delayed ability: {}",
            card.name,
            effects_description(effects)
        );
    }
    let Ability::Triggered { condition, effects } = &card.abilities[index];
    format!(
        "{} — {}: {}",
        card.name,
        condition_description(condition),
        effects_description(effects)
    )
}

fn condition_description(condition: &TriggerCondition) -> String {
    match condition {
        TriggerCondition::WasCast { condition } => {
            format!("{} (if this was cast)", condition_description(condition))
        }
        TriggerCondition::OnceEachTurn { condition } => {
            format!("{} (once each turn)", condition_description(condition))
        }
        TriggerCondition::ActiveIf {
            active_if,
            condition,
        } => format!(
            "{} (while {})",
            condition_description(condition),
            static_description(active_if)
        ),
        TriggerCondition::Sacrificed {
            predicate,
            controller_only,
            another,
        } => format!(
            "When {}{} {} sacrificed",
            if *another { "another " } else { "" },
            predicate_description(predicate, "permanent"),
            if *controller_only {
                "you control is"
            } else {
                "is"
            }
        ),
        TriggerCondition::YouCastSpell { predicate } => format!(
            "When you cast a {}",
            predicate_description(predicate, "spell")
        ),
        TriggerCondition::EntersTheBattlefield { subject } => format!(
            "When {} enters the battlefield",
            subject_description(subject)
        ),
        TriggerCondition::Dies { subject } => format!("When {} dies", subject_description(subject)),
        TriggerCondition::Attacks {
            subject: TriggerSubject::This,
        } => "When this attacks".into(),
        TriggerCondition::Attacks {
            subject: TriggerSubject::AnotherYouControl(predicate),
        } => format!(
            "When one or more other {} you control attack",
            predicate_description(predicate, "permanents")
        ),
        TriggerCondition::Attacks {
            subject: TriggerSubject::AnyYouControl(predicate),
        } => format!(
            "When one or more {} you control attack",
            predicate_description(predicate, "permanents")
        ),
        TriggerCondition::AttacksAlone { subject } => {
            format!("When {} attacks alone", subject_description(subject))
        }
        TriggerCondition::BecomesTapped { subject } => {
            format!("When {} becomes tapped", subject_description(subject))
        }
        TriggerCondition::TappedForMana { subject } => {
            format!("When {} is tapped for mana", subject_description(subject))
        }
        TriggerCondition::BecomesTargeted { subject } => format!(
            "When {} becomes the target of an opponent's spell",
            subject_description(subject)
        ),
        TriggerCondition::BeginningOfYourUpkeep => "At the beginning of your upkeep".into(),
        TriggerCondition::BeginningOfYourCombat => "At the beginning of your combat".into(),
        TriggerCondition::BeginningOfYourEndStep => "At the beginning of your end step".into(),
        TriggerCondition::YouDrawNthCardThisTurn { n } => {
            format!("When you draw card {n} this turn")
        }
    }
}

fn subject_description(subject: &TriggerSubject) -> String {
    match subject {
        TriggerSubject::This => "this".into(),
        TriggerSubject::AnotherYouControl(p) => format!(
            "another {} you control",
            predicate_description(p, "permanent")
        ),
        TriggerSubject::AnyYouControl(p) => {
            format!("a {} you control", predicate_description(p, "permanent"))
        }
    }
}

fn static_description(condition: &StaticCondition) -> String {
    match condition {
        StaticCondition::YourTurn => "it is your turn".into(),
        StaticCondition::Always => "always".into(),
        StaticCondition::CardsDrawnAtLeast { count } => {
            format!("you have drawn at least {count} cards this turn")
        }
        StaticCondition::GraveyardAtLeast { count, predicate } => format!(
            "your graveyard has at least {count} {} cards",
            predicate_description(predicate, "matching")
        ),
        StaticCondition::ControlsMatching { predicate } => format!(
            "you control a {}",
            predicate_description(predicate, "permanent")
        ),
        StaticCondition::UnspentManaAtLeast { count } => {
            format!("you have {count} or more unspent mana")
        }
    }
}

fn card_type(t: CardType) -> &'static str {
    match t {
        CardType::Creature => "creature",
        CardType::Instant => "instant",
        CardType::Sorcery => "sorcery",
        CardType::Planeswalker => "planeswalker",
        CardType::Land => "land",
        CardType::Enchantment => "enchantment",
        CardType::Artifact => "artifact",
        CardType::Kindred => "kindred",
        CardType::Battle => "battle",
    }
}

fn predicate_description(p: &CardPredicate, noun: &str) -> String {
    let CardPredicate {
        flying,
        no_counters,
        in_combat,
        token,
        with_counter,
        attacking,
        subtypes_any,
        not_subtypes,
        supertype,
        any_of,
        min_power,
        card_type: kind,
        card_types_any,
        not_card_types,
        subtype,
        max_power,
        min_mana_value,
    } = p;
    let mut terms = Vec::new();
    if *flying {
        terms.push("flying".into());
    }
    terms.extend(supertype.iter().cloned());
    if *token == Some(false) {
        terms.push("nontoken".into());
    }
    terms.extend(
        not_card_types
            .iter()
            .map(|t| format!("non{}", card_type(*t))),
    );
    terms.extend(not_subtypes.iter().map(|t| format!("non-{t}")));
    terms.extend(subtype.iter().cloned());
    if !subtypes_any.is_empty() {
        terms.push(format!("({})", subtypes_any.join(" or ")));
    }
    if let Some(t) = kind {
        terms.push(card_type(*t).into());
    }
    if !any_of.is_empty() {
        terms.push(format!(
            "({})",
            any_of
                .iter()
                .map(|alternative| predicate_description(alternative, ""))
                .map(|text| text.trim().to_string())
                .collect::<Vec<_>>()
                .join(" or ")
        ));
    }
    if !card_types_any.is_empty() {
        terms.push(format!(
            "({})",
            card_types_any
                .iter()
                .map(|t| card_type(*t))
                .collect::<Vec<_>>()
                .join(" or ")
        ));
    }
    if kind.is_none() || noun != "permanent" {
        terms.push(noun.into());
    }
    if *token == Some(true) {
        terms.push("token".into());
    }
    if let Some(n) = min_power {
        terms.push(format!("with power at least {n}"));
    }
    if let Some(n) = max_power {
        terms.push(format!("with power at most {n}"));
    }
    if let Some(n) = min_mana_value {
        terms.push(format!("with mana value at least {n}"));
    }
    if *no_counters {
        terms.push("with no counters on it".into());
    }
    if *with_counter {
        terms.push("with a +1/+1 counter on it".into());
    }
    if *attacking {
        terms.insert(0, "attacking".into());
    }
    if *in_combat {
        terms.insert(0, "attacking or blocking".into());
    }
    terms.join(" ")
}

pub(crate) fn counter_description(kind: CounterKind) -> &'static str {
    match kind {
        CounterKind::PlusOnePlusOne => "+1/+1",
        CounterKind::Stun => "stun",
        CounterKind::Fire => "fire",
        CounterKind::DoubleStrike => "double strike",
    }
}

fn target_description(t: &TargetSpec) -> String {
    let noun = match t {
        TargetSpec::PermanentYouControl { predicate } => format!(
            "{} you control",
            predicate_description(predicate, "permanent")
        ),
        TargetSpec::PermanentOpponentControls { predicate } => format!(
            "{} an opponent controls",
            predicate_description(predicate, "permanent")
        ),
        TargetSpec::GraveyardCardYouOwn { predicate } => format!(
            "{} in your graveyard",
            predicate_description(predicate, "card")
        ),
        TargetSpec::Permanent { predicate } => predicate_description(predicate, "permanent"),
        TargetSpec::Player => "player".into(),
        TargetSpec::Opponent => "opponent".into(),
        TargetSpec::Creature => "creature".into(),
        TargetSpec::CreatureOrPlayer => "creature or player".into(),
        TargetSpec::Spell => "spell".into(),
        TargetSpec::SpellOrPermanent { min_mana_value } => {
            format!("spell or permanent with mana value at least {min_mana_value}")
        }
        TargetSpec::CreatureYouControl => "creature you control".into(),
        TargetSpec::CreatureOpponentControls => "creature an opponent controls".into(),
        TargetSpec::LandYouControl => "land you control".into(),
    };
    format!("target {noun}")
}

fn value_description(value: &EffectValue) -> String {
    match value {
        EffectValue::Fixed(value) => value.to_string(),
        EffectValue::X => "the announced value of X".into(),
        EffectValue::CreaturesWithCounters => {
            "the number of creatures you control with +1/+1 counters".into()
        }
        EffectValue::BattlefieldMatching { predicate } => format!(
            "the number of {} you control",
            predicate_description(predicate, "permanents")
        ),
        EffectValue::GraveyardMatching { base, predicate } => format!(
            "{base} plus the number of {} in your graveyard",
            predicate_description(predicate, "cards")
        ),
        EffectValue::GreatestSharedCreatureType => {
            "the greatest number of creatures you control that share a creature type".into()
        }
        EffectValue::CardsDrawnMinus { subtract } => {
            format!("the number of cards you have drawn this turn minus {subtract} (minimum zero)")
        }
        EffectValue::ExcessDamage => "the excess damage dealt this way".into(),
        EffectValue::SourcePower => "this permanent's power".into(),
    }
}

pub(crate) fn mana_description(mana: &Mana) -> String {
    let restricted = mana.restricted_colors();
    let mut text: String = ["W", "U", "B", "R", "G", "C"]
        .into_iter()
        .enumerate()
        .flat_map(|(i, s)| {
            std::iter::repeat_n(
                format!("{{{s}}}"),
                mana.mana[i] as usize + restricted[i] as usize,
            )
        })
        .collect();
    for (restriction, _) in mana.restricted() {
        text.push_str(&format!(" ({} only)", restriction.describe()));
    }
    text
}

fn cost_description(cost: &ManaCost) -> String {
    let mut text = "{X}".repeat(usize::from(cost.x));
    if cost.cost[6] > 0 {
        text.push_str(&format!("{{{}}}", cost.cost[6]));
    }
    for (i, symbol) in ["W", "U", "B", "R", "G", "C"].into_iter().enumerate() {
        for _ in 0..cost.cost[i] {
            text.push_str(&format!("{{{symbol}}}"));
        }
    }
    for [a, b] in &cost.hybrid {
        text.push_str(&format!("{{{}/{}}}", a.symbol(), b.symbol()));
    }
    if text.is_empty() {
        text.push_str("{0}");
    }
    text
}

fn base_pt_description(power: Option<i32>, toughness: Option<i32>) -> String {
    match (power, toughness) {
        (Some(power), Some(toughness)) => format!("power and toughness {power}/{toughness}"),
        (Some(power), None) => format!("power {power}"),
        (None, Some(toughness)) => format!("toughness {toughness}"),
        (None, None) => "power and toughness unchanged".into(),
    }
}

fn keywords_description(k: &Keywords) -> String {
    let Keywords {
        flying,
        reach,
        haste,
        flash,
        vigilance,
        trample,
        first_strike,
        double_strike,
        deathtouch,
        lifelink,
        defender,
        menace,
        hexproof,
        indestructible,
    } = k;
    [
        (flying, "flying"),
        (reach, "reach"),
        (haste, "haste"),
        (flash, "flash"),
        (vigilance, "vigilance"),
        (trample, "trample"),
        (first_strike, "first strike"),
        (double_strike, "double strike"),
        (deathtouch, "deathtouch"),
        (lifelink, "lifelink"),
        (defender, "defender"),
        (menace, "menace"),
        (hexproof, "hexproof"),
        (indestructible, "indestructible"),
    ]
    .into_iter()
    .filter_map(|(on, name)| on.then_some(name))
    .collect::<Vec<_>>()
    .join(", ")
}

fn effects_description(effects: &[Effect]) -> String {
    let parts: Vec<_> = effects.iter().filter_map(effect_description).collect();
    if parts.is_empty() {
        "Do nothing".into()
    } else {
        parts.join(". ")
    }
}

/// Exhaustive matching makes new effect vocabulary require a human description.
/// The internal target-clause cursor is not itself an effect a person chooses.
fn effect_description(effect: &Effect) -> Option<String> {
    use Effect::*;
    Some(match effect {
        SelectTargetRequirement { .. } => return None,
        CreateNonlegendaryCopy => "Create a nonlegendary token copy of this".into(),
        ModifyControlled { predicate, power, toughness } => format!("{} you control get {power:+}/{toughness:+} until end of turn", predicate_description(predicate,"permanents")),
        SetBasePtControlled { predicate, power, toughness } => format!("{} you control have base power and toughness {}/{} until end of turn", predicate_description(predicate,"permanents"), value_description(power), value_description(toughness)),
        AddSubtypesControlled { predicate, subtypes } => format!("{} you control gain {} until end of turn", predicate_description(predicate,"permanents"), subtypes.join(", ")),
        OptionalPayment { cost, if_paid, if_declined } => {
            let cost = match cost { PaymentCost::Mana(m) => cost_description(m), PaymentCost::Waterbend { generic } => format!("waterbend {generic}") };
            format!("You may pay {cost}. If paid: {}. Otherwise: {}", effects_description(if_paid), effects_description(if_declined))
        },
        Sacrifice { predicate } => format!("Sacrifice a {}", predicate_description(predicate,"permanent")),
        DealDamageToController { amount } => format!("Deal {amount} damage to you"),
        DrainLife { amount } => format!("Each opponent loses {amount} life and you gain {amount} life"),
        DrainLifeValue { amount } => format!("Each opponent loses X life and you gain X life, where X is {}",value_description(amount)),
        CantBeBlockedTarget { target } => format!("{} can't be blocked this turn",target_description(target)),
        ReturnGraveyardTarget { target, tapped_onto_battlefield: false } => format!("Return {} to your hand",target_description(target)),
        ReturnGraveyardTarget { target, tapped_onto_battlefield: true } => format!("Put {} onto the battlefield tapped",target_description(target)),
        MillCards { count, target: None } => format!("Mill {} cards", value_description(count)),
        MillCards { count, target: Some(target) } => format!("{} mills {} cards", target_description(target), value_description(count)),
        MillAndSelect { count, predicate, if_declined, if_milled } => {
            let mut text = format!("Mill {count} cards. You may return a {} milled this way to your hand. Otherwise: {}",predicate_description(predicate,"card"),effects_description(if_declined));
            if let Some(bonus) = if_milled {
                text.push_str(&format!(". If a {} is milled this way: {}", predicate_description(&bonus.predicate, "card"), effects_description(&bonus.effects)));
            }
            text
        }
        TargetClause { requirement, effects } => format!("Choose {}–{} {} for this clause: {}",requirement.min,if requirement.max_x { "X".to_string() } else { requirement.max.to_string() },target_description(&requirement.spec),effects_description(effects)),
        SearchBasicLand { subtype, to_hand: true, .. } => format!("Search your library for a {} card, reveal it, put it into your hand, then shuffle",subtype.as_deref().unwrap_or("land")),
        SearchBasicLand { subtype, count, tapped, untap_if_lands, bottom_source, .. } => {
            let mut text = format!("Search your library for up to {} basic {} cards and put them onto the battlefield{}. Then shuffle",value_description(count),subtype.as_deref().unwrap_or("land"),if *tapped { " tapped" } else { "" });
            if let Some(n) = untap_if_lands { text.push_str(&format!(". Untap those lands if you control at least {n} lands")); }
            if *bottom_source { text.push_str(". Put this on the bottom of its owner's library"); }
            text
        },
        OptionalDiscardDraw { count, up_to: 1 } => format!("You may discard a card. If you do, draw {count} cards"),
        OptionalDiscardDraw { count, up_to } => format!("Discard up to {up_to} cards, then draw {count} for each"),
        DiscardTargetPlayer { count, target } => format!("{} discards {count} cards",target_description(target)),
        Destroy { target } => format!("Destroy {}",target_description(target)),
        Exile { target } => format!("Exile {}",target_description(target)),
        Airbend { target } => format!("Airbend {}. Its owner may cast it from exile for {{2}}",target_description(target)),
        AirbendAll { predicate } => format!("Airbend all other {}. Their owners may cast them from exile for {{2}}", predicate_description(predicate,"permanents")),
        OpponentsCastOnlyFromHand => "Until your next turn, your opponents can't cast spells from anywhere other than their hands".to_string(),
        ExileSourceSpell => "Exile this spell".to_string(),
        ExileTopCardPlayThisTurn => "Exile the top card of your library. You may play that card this turn".to_string(),
        GrantFirebendingUntilEot { amount } => format!("Each creature you control gains firebending {amount} until end of turn"),
        GrantFirebendingToTarget { amount, target } => format!("{} gains firebending {amount} until end of turn", target_description(target)),
        ExileLinked { chooser, zone, predicate, up_to, grant, target } => format!("{} {} {} {} from {}; {}", match chooser { LinkedChooser::Controller => "You exile", LinkedChooser::TargetPlayer => "That player exiles" }, if *up_to { "up to one" } else { "a" }, predicate_description(predicate, "card"), match zone { ZoneType::Battlefield => "they control", _ => "" }, target_description(target), match grant { LinkedGrant::WaterbendManaValue => "for as long as you control this, you may cast it during your turn by waterbending {X}, where X is its mana value", LinkedGrant::FlashAnyMana => "during your turn, you may cast it as though it had flash, spending mana of any type" }),
        ExileUntilNonlandCastFree { below } => format!("Exile cards from the top of your library until you exile a nonland card. You may cast that card without paying its mana cost if its mana value is less than {}. If you don't cast it this way, put it into your hand", value_description(below)),
        CastFromHandFree { predicate } => format!("You may cast a {} from your hand without paying its mana cost", predicate_description(predicate, "spell")),
        NoMaximumHandSize => "You have no maximum hand size for the rest of the game".to_string(),
        ShuffleGraveyardIntoLibrary => "Shuffle your graveyard into your library".to_string(),
        RevealHandDiscardChosen { target, predicate } => format!("{} reveals their hand. You choose a {} from it; that player discards it", target_description(target), predicate_description(predicate, "card")),
        SearchLandsOpponentSplits { count, opponent_picks } => format!("Search your library for up to {count} land cards with different names and reveal them. An opponent chooses {opponent_picks}; put those into your graveyard and the rest onto the battlefield tapped, then shuffle"),
        OwnerPutsIntoLibrary { target } => format!("The owner of {} puts it into their library second from the top or on the bottom", target_description(target)),
        PlaceTargetInLibrary { second_from_top: true } => "Put it into its owner's library second from the top".into(),
        PlaceTargetInLibrary { second_from_top: false } => "Put it on the bottom of its owner's library".into(),
        SacrificeSource { then } => format!("Sacrifice this. If you do: {}", effects_description(then)),
        BuffTargetValue { power, toughness, target } => format!("{} gets +X/{toughness:+} until end of turn, where X is {}",target_description(target),value_description(power)),
        ReturnToHandDrawIfControlled { target } => format!("Return {} to its owner's hand. If you controlled it, draw a card",target_description(target)),
        BuffTriggeringPermanentValue { power, toughness } => format!("It gets +X/+Y until end of turn, where X is {} and Y is {}",value_description(power),value_description(toughness)),
        ModifyTriggeringPermanent { counters, keywords } => {
            let mut text = format!("Put {counters} +1/+1 counters on the permanent that caused this trigger");
            let keywords = keywords_description(keywords);
            if !keywords.is_empty() { text.push_str(&format!(". It gains {keywords} until end of turn")); }
            text
        },
        Discard => "Discard a card".into(),
        DealDamageValue { amount, target } => format!("Deal X damage to {}, where X is {}",target_description(target),value_description(amount)),
        PutCountersValue { count, target } => format!("Put X +1/+1 counters on {}, where X is {}",target_description(target),value_description(count)),
        AddSubtypeToTarget { target, subtype } => format!("{} becomes a {subtype} in addition to its other types until end of turn",target_description(target)),
        ExileIfDiesThisTurn { target } => format!("If {} would die this turn, exile it instead",target_description(target)),
        IfCondition { condition: StaticCondition::Always, effects, .. } => effects_description(effects),
        IfCondition { condition, effects, otherwise } if otherwise.is_empty() => format!("If {}: {}",static_description(condition),effects_description(effects)),
        IfCondition { condition, effects, otherwise } => format!("If {}: {}. Otherwise: {}",static_description(condition),effects_description(effects),effects_description(otherwise)),
        ReturnToHand { target } => format!("Return {} to its owner's hand",target_description(target)),
        DealDamage { amount, target } => format!("Deal {amount} damage to {}",target_description(target)),
        CounterSpell { target } => format!("Counter {}",target_description(target)),
        ModifyUntilEot { power_delta, toughness_delta } => format!("This gets {power_delta:+}/{toughness_delta:+} until end of turn"),
        DrawCards { count } => format!("Draw {count} cards"),
        DrawCardsValue { count } => format!("Draw X cards, where X is {}", value_description(count)),
        LoseLife { amount } => format!("You lose {amount} life"),
        DealDamageToEachOpponent { amount } => format!("Deal {amount} damage to each opponent"),
        MassDamage { amount } => format!("Deal {amount} damage to each creature"),
        DestroyCreaturesLosingAbilities { max_mana_value } => format!("Each creature with mana value {} or less loses all abilities until end of turn. Destroy those creatures", value_description(max_mana_value)),
        DamageOpponentCreatures { amount } => format!("Deal {amount} damage to each creature your opponents control"),
        CreateTokensValue { token_name, count } => format!("Create X {token_name} tokens, where X is {}",value_description(count)),
        ExileWithControllerToken { target, token_name } => format!("Exile {}. Its controller creates a {token_name} token",target_description(target)),
        CreateToken { token_name, count, tapped_and_attacking, attach_source } => format!("Create {count} {token_name} token{}{}{}",if *count == 1 { "" } else { "s" },if *tapped_and_attacking { " tapped and attacking" } else { "" },if *attach_source { ", then attach this to it" } else { "" }),
        PutCountersOnSource { count, kind } => format!("Put {count} {} counters on this",counter_description(*kind)),
        PutCounters { count, kind, target } => format!("Put {count} {} counters on {}",counter_description(*kind),target_description(target)),
        TapSource => "Tap this".into(), UntapSource => "Untap this".into(),
        GrantKeywordsToSource { keywords } => format!("This gains {} until end of turn",keywords_description(keywords)),
        ReturnSourceToHand => "Return this card from your graveyard to your hand".into(),
        CantBeBlockedThisTurnSource => "This can't be blocked this turn".into(),
        SetBasePtUntilEot { power, toughness } => format!("This has base {} until end of turn",base_pt_description(*power,*toughness)),
        BecomeCreatureTarget { target, duration } => format!("{} becomes an artifact creature{}",target_description(target),match duration { BecomesCreature::EndOfTurn => " until end of turn", BecomesCreature::Incarnation => "" }),
        SetBasePtTarget { target, power, toughness } => format!("{} has base {} until end of turn",target_description(target),base_pt_description(*power,*toughness)),
        BecomeCreatureSource { duration } => match duration {
            BecomesCreature::EndOfTurn => "This becomes an artifact creature until end of turn".into(),
            BecomesCreature::Incarnation => "This becomes an artifact creature".into(),
        },
        GainLife { amount } => format!("You gain {amount} life"),
        GainLifeValue { amount } => format!("You gain life equal to {}", value_description(amount)),
        AddManaValue { mana, amount, until_end_of_combat } => format!("Add {} for each of {}{}", mana_description(mana), value_description(amount), if *until_end_of_combat { ". This mana lasts until end of combat" } else { "" }),
        KeepUnspentRedMana => "Until end of turn, you don't lose unspent red mana as steps and phases end".into(),
        FightLastTarget => "The first target fights the last target".into(),
        GrantKeywordsControlled { predicate, keywords } => format!("Each {} you control gains {} until end of turn", predicate_description(predicate, "permanent"), keywords_description(keywords)),
        SearchLibraryToHand { predicate } => format!("Search your library for {}, reveal it, put it into your hand, then shuffle", predicate_description(predicate, "card")),
        OnNthResolutionThisTurn { n, effect } => format!("If this is resolution {n} of this ability this turn: {}",effects_description(std::slice::from_ref(effect))),
        Scry { count } => format!("Scry {count}"),
        LookAndSelect { reveal, look, min_select, max_select, predicate } => format!("{} the top {} cards of your library. Put {min_select}–{max_select} {} from among them into your hand and the rest on the bottom in a random order",if *reveal { "Reveal" } else { "Look at" },value_description(look),predicate_description(predicate,"cards")),
        PutTopCardsInHand { count } => format!("Put the top {count} cards of your library into your hand"),
        Learn => "Learn: you may reveal a Lesson you own from outside the game and put it into your hand, or discard a card to draw a card, or decline".into(),
        Modal { modes } => format!("Choose one — {}",modes.iter().enumerate().map(|(i,m)| format!("({}) {}",i+1,effects_description(m))).collect::<Vec<_>>().join("; ")),
        CounterUnlessPays { cost } => format!("Counter the spell unless its controller pays {}",cost_description(cost)),
        IfKicked { then, otherwise } => format!("If kicked: {}. Otherwise: {}",effects_description(then),effects_description(otherwise)),
        IfGraveyardAtLeast { count, predicate, then, otherwise } => format!("If your graveyard has at least {count} {}: {}. Otherwise: {}",predicate_description(predicate,"cards"),effects_description(then),effects_description(otherwise)),
        TargetCreaturesDealPowerDamageToLastTarget => "Each chosen creature except the last target deals damage equal to its power to the last target".into(),
        Earthbend { count, target } => { let count = value_description(count); format!("Earthbend {count} {}: it becomes a 0/0 creature with haste that's still a land; put {count} +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped",target_description(target)) },
        ReturnSourceToBattlefieldTapped => "Return this card to the battlefield tapped".into(),
        ExileUntilSourceLeaves { target } => format!("Exile up to one {} until this leaves the battlefield, only if this is still on the battlefield",target_description(target)),
        AddOneColorManaValue { amount } => format!("Add X mana of any one color, where X is {}",value_description(amount)),
        AddMana { mana, until_end_of_combat } => format!("Add {}{}",mana_description(mana),if *until_end_of_combat { ". This mana lasts until end of combat" } else { "" }),
        BuffTarget { power, toughness, target } => format!("{} gets {power:+}/{toughness:+} until end of turn",target_description(target)),
        UntapTarget { target } => format!("Untap {}",target_description(target)),
        TapTarget { target } => format!("Tap {}",target_description(target)),
        AttachSourceTo { target } => format!("Attach this to {}",target_description(target)),
        ExileAttached => "Exile the creature this is attached to".into(),
        TapAttached => "Tap the creature this is attached to".into(),
        RemoveAllCountersFromAttached => "Remove all counters from the creature this is attached to".into(),
        DistributeCountersToTappedOpponentCreatures { kind, count } => format!("Distribute {count} {} counters among any number of tapped creatures your opponents control",counter_description(*kind)),
        ReturnGrantorToHand => "Return the permanent granting this ability to its owner's hand".into(),
        ShuffleAttachedIntoLibrary => "The attached creature's owner shuffles it into their library".into(),
        RemoveCounters { up_to, chosen: None, target } => format!("Remove up to {up_to} counters from {}",target_description(target)),
        RemoveCounters { chosen: Some(kind), target, .. } => format!("Remove a {} counter from {}",counter_description(*kind),target_description(target)),
        GrantKeywordsToTarget { keywords, target } => format!("{} gains {} until end of turn",target_description(target),keywords_description(keywords)),
        IfTargetMatches { predicate, then } => format!("If that target is a {}: {}",predicate_description(predicate,"permanent"),effects_description(then)),
        ForEachTarget { effects } => format!("For each chosen target: {}",effects_description(effects)),
        PutCountersOnEachMatching { count, predicate, other } => format!("Put {} +1/+1 counters on each {}{} you control",value_description(count),if *other { "other " } else { "" },predicate_description(predicate,"permanent")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ability::TargetRequirement;

    #[test]
    fn descriptions_keep_target_clauses_and_conditional_branches() {
        let description = effects_description(&[
            Effect::TargetClause {
                requirement: TargetRequirement::up_to(2, TargetSpec::CreatureYouControl),
                effects: vec![Effect::ForEachTarget {
                    effects: vec![Effect::UntapTarget {
                        target: TargetSpec::CreatureYouControl,
                    }],
                }],
            },
            Effect::OptionalPayment {
                cost: PaymentCost::Mana(ManaCost::parse("1W")),
                if_paid: vec![Effect::DrawCards { count: 2 }],
                if_declined: vec![Effect::DealDamageToController { amount: 1 }],
            },
        ]);
        assert_eq!(description, "Choose 0–2 target creature you control for this clause: For each chosen target: Untap target creature you control. You may pay {1}{W}. If paid: Draw 2 cards. Otherwise: Deal 1 damage to you");
    }
}
