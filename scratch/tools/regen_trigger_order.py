"""Re-record the w17 trigger-order root and content digests in place."""


import json
import sys
from pathlib import Path

sys.path.insert(0, "tests/academy")
from test_trigger_order import root  # noqa: E402

path = Path("academy/fixtures/trigger-order-w17.json")
evidence = json.loads(path.read_text())
env, _ = root()
capture = json.loads(env.capture_decision(0).to_json())
assert capture["world_version"] == evidence["world_version"]
evidence["root_digest"] = env.state_digest()
evidence["content_digest"] = capture["content_digest"]
path.write_text(json.dumps(evidence, indent=2) + "\n")
print(evidence["root_digest"], evidence["content_digest"])
