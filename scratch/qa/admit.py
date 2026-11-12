"""Admit every trophy decklist of the pinned corpus, one event at a time."""
from __future__ import annotations
import collections, json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from experiments.tla_corpus import Corpus
import managym
from managym import DeckImportError

path = Path("experiments/data/mtg213/corpus-v1/corpus.json")
corpus = Corpus.model_validate_json(path.read_text())
waiting = {"Release to Memory", "Lita, Mechanical Engineer", "Heartbeat of Spring"}
results = []
for t in corpus.trophies[:1000]:
    try:
        (imp,) = managym.admit_decklists([(t.event_id, managym.Decklist(t.decklist.source, t.decklist.filename))])
        deck = imp.deck
        main = sum(deck.main.values()) if hasattr(deck, "main") else None
        results.append({"event": t.event_id, "ok": True})
    except DeckImportError as e:
        results.append({"event": t.event_id, "ok": False, "error": str(e)})
ok = sum(r["ok"] for r in results)
print("admitted", ok, "of", len(results))
fails = [r for r in results if not r["ok"]]
by = collections.Counter()
other = []
for r in fails:
    names = [n for n in waiting if n in r["error"]]
    if names:
        by.update(names)
    else:
        other.append(r)
print("failures naming waiting cards:", sum(1 for r in fails if any(n in r["error"] for n in waiting)), dict(by))
print("other failures:", len(other))
for r in other[:20]:
    print(" ", r["event"], r["error"][:400])
if fails:
    print("sample error:", fails[0]["error"][:600])
Path(sys.argv[1]).write_text(json.dumps(results, indent=1))
