# Open questions / assumptions

- Mana display is still deferred. The current GUI slice ships without it because the Python bindings used by `gui/server.py` do not expose mana pool state.
- Live play still relies on frontend redaction for opponent hands. The shared board renders opponent hand backs from counts and ignores any serialized hand contents; a later pass could redact hidden hand contents server-side as well.
