"""Wrap forward-call arguments in admit_observation (test migration helper)."""
import re
import sys
from pathlib import Path

MODELS = r"agent|restored|loaded|baseline|model|teacher\.agent|reference"
for name in sys.argv[1:]:
    p = Path(name)
    s = p.read_text()
    s = re.sub(
        rf"\b({MODELS})\(([A-Za-z_][A-Za-z_0-9]*|reorder_objects\(obs\)|_remove_discriminant\(case, obs\)|\{{[^{{}}\n]*\}})\)",
        lambda m: f"{m.group(1)}(admit_observation({m.group(1)}, {m.group(2)}))"
        if m.group(2) not in ("prepared",) else m.group(0),
        s,
    )
    s = re.sub(
        rf"\b({MODELS})\.(forward_distribution|get_value|get_action_and_value)\(([A-Za-z_][A-Za-z_0-9]*)",
        lambda m: m.group(0) if m.group(3) == "prepared" else f"{m.group(1)}.{m.group(2)}(admit_observation({m.group(1)}, {m.group(3)})",
        s,
    )
    if "admit_observation" in s and "from manabot.model.admission import" not in s:
        anchor = "from manabot.model.actor_critic import ActorCritic\n"
        if anchor in s:
            s = s.replace(anchor, anchor + "from manabot.model.admission import admit_observation\n", 1)
        else:
            lines = s.splitlines(keepends=True)
            idx = next(i for i, l in enumerate(lines) if l.startswith("from manabot."))
            lines.insert(idx, "from manabot.model.admission import admit_observation\n")
            s = "".join(lines)
    p.write_text(s)
