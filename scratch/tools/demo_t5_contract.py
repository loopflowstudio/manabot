"""Continue Joo Dee's ability through native semantic Commands (no text capture)."""

import json

from managym.decision import Observation

from manabot.semantic.decision_contract import SemanticDecisionContract
from manabot.verify.text_observation import priority_root


def offers(env):
    contract = SemanticDecisionContract.from_env(env)
    obs = Observation.from_json(env.semantic_observation_json(contract.frame.actor))
    return contract, obs


env = priority_root((), ("Swamp", "Joo Dee, One of Many"))
for step in range(12):
    contract, obs = offers(env)
    verbs = [o["verb"] for o in obs.decision.offers]
    print(step, "actor", contract.frame.actor, verbs)
    pick = next((o for o in obs.decision.offers if o["verb"] == "activate_ability"), None) if step == 0 else None
    if pick is None:
        non_pass = [o for o in obs.decision.offers if o["verb"] != "pass_priority"]
        if not non_pass:
            if step > 2 and all(p.get("copy") for p in obs.viewer_state["agent_permanents"] if "Joo" in str(p)):
                pass
            pick = obs.decision.offers[0]
        else:
            pick = non_pass[-1]
    print("   ->", json.dumps(pick)[:200])
    contract.apply(env, pick["id"])
    contract, obs = offers(env)
    state = obs.viewer_state
    joos = [p for p in state["agent_permanents"] + state.get("opponent_permanents", []) if "copy" in p and p["copy"]]
    if joos or state["agent"].get("turn_history"):
        print("   copies:", json.dumps([p["copy"] for p in joos]), "history:", state["agent"].get("turn_history"))
        if state["agent"].get("turn_history", {}).get("permanents_sacrificed"):
            break
