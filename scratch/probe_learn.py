import torch
from manabot.verify.action_cases import decision_case
for split, v in (("fit",0),("held_out",1),("fit",2),("held_out",3)):
    for p in (0,1):
        c = decision_case("learn", preference=p, variant=v, split=split)
        o = c.observation
        n = int(o["actions_valid"].sum())
        f = o["action_focus"][0,:n].tolist()
        print(split, v, p, "cands", c.candidates, "focus", [f[i] for i in c.candidates], "pref", c.preferred)
        sc = o["semantic_cards"][0]
        for i in c.candidates:
            for fi in f[i]:
                if fi>=2:
                    k = fi-2
                    print("    focus", fi, "def", sc.flatten()[k].item() if k < sc.numel() else None)
        print("   outcomes", [(x.__dict__) for x in c.outcomes][:2])
