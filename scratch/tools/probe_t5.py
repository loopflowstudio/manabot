import json
from manabot.semantic.text_observation import capture_text_decision, submit_response
from manabot.verify.text_observation import priority_root
env = priority_root((), ("Swamp", "Joo Dee, One of Many"))
for want in ["Activate ability of C2", "Pass priority", "Pass priority"]:
    d = capture_text_decision(env, history_capacity=32)
    a = [a for a in d.actions if want in a.description][0]
    submit_response(env, d, a.reference)
actor = env.current_agent_index()
cap = json.loads(env.capture_decision(actor, 32).to_json())
for f in cap["frames"]:
    print(json.dumps(f)[:600])
print(json.dumps(cap["observation"].get("decision"))[:1500])
