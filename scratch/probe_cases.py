import torch
from manabot.verify.action_cases import decision_case
for fam in ("mode","ability"):
    for p in (0,1):
        c = decision_case(fam, preference=p)
        o = c.observation
        n = int(o["actions_valid"].sum())
        print(fam, p, "focus", o["action_focus"][0,:n].tolist())
        print("  actions", o["actions"][0,:n].tolist())
        print("  agent_player", o["agent_player"][0].tolist())
