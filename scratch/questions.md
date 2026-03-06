# Open Questions / Assumptions

- Assumed session-expiration traces should be finalized with `end_reason = "session_expired"`.
- Assumed resume failures should return `type: "error"` while keeping the socket open so the client can immediately send `new_game`.
- Assumed `session_id` and `resume_token` only need to be attached to `observation` responses, not `game_over` responses.
