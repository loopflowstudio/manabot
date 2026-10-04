"""Complete typed card programs joined to viewer-safe card and knowledge rows."""

import torch
from torch import nn

from manabot.semantic.learning import BoundSemanticPack
import managym


class SemanticCardEncoder(nn.Module):
    def __init__(self, pack_key: str, hidden_dim: int):
        super().__init__()
        if pack_key != "ur-lessons-vs-gw-allies":
            raise ValueError(f"unsupported policy semantic pack: {pack_key}")
        engine = managym.Env(seed=0)
        engine.reset(
            [
                managym.authored_deck_setup(pack_key, key)
                for key in ("ur_lessons", "gw_allies")
            ]
        )
        pack = BoundSemanticPack.from_env(engine)
        catalog = pack.catalog
        pairs = list(
            zip(
                map(int, catalog.token_kind), map(int, catalog.token_value), strict=True
            )
        )
        vocabulary = {pair: index + 1 for index, pair in enumerate(sorted(set(pairs)))}
        references = dict(
            zip(
                map(int, catalog.definition_ref_source_tokens),
                map(int, catalog.definition_ref_target_rows),
                strict=True,
            )
        )

        def definition_tokens(row, ancestors=()):
            if row in ancestors:
                raise ValueError("cyclic semantic definition reference is not admitted")
            start, stop = catalog.definition_offsets[row : row + 2]
            spans = [(int(start), int(stop))]
            for program in pack.program_rows(row):
                start, stop = catalog.program_offsets[program : program + 2]
                spans.append((int(start), int(stop)))
            tokens = []
            for start, stop in spans:
                for offset in range(start, stop):
                    tokens.append(vocabulary[pairs[offset]])
                    if offset in references:
                        tokens.extend(
                            definition_tokens(references[offset], (*ancestors, row))
                        )
            return tokens

        # Structure, execution order, program boundaries and referenced token
        # definitions come from the checked catalog. Nothing is truncated.
        rows = [definition_tokens(row) for row in range(len(pack.ir.definitions))]
        width = max(map(len, rows))
        self.token_embedding = nn.Embedding(len(vocabulary) + 1, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.register_buffer(
            "tokens",
            torch.tensor(
                [row + [0] * (width - len(row)) for row in rows], dtype=torch.long
            ),
        )
        self.register_buffer("lengths", torch.tensor(list(map(len, rows))))
        # IDs are transport into the complete catalog, never learned identity.
        lookup = torch.full(
            (max(pack.definition_row_by_card_def_id) + 2,), -1, dtype=torch.long
        )
        lookup[0] = 0
        for definition, row in pack.definition_row_by_card_def_id.items():
            lookup[definition + 1] = row + 1
        self.register_buffer("definition_rows", lookup)

    def _rows(self, ids):
        if (
            not torch.equal(ids, ids.floor())
            or bool((ids < 0).any())
            or bool((ids >= len(self.definition_rows)).any())
        ):
            raise ValueError("policy received an unadmitted semantic definition")
        rows = self.definition_rows[ids.long()]
        if bool((rows < 0).any()):
            raise ValueError("policy received an unadmitted semantic definition")
        return rows

    def forward(self, obs):
        tokens = self.token_embedding(self.tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            tokens, self.lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, state = self.encoder(packed)
        definitions = torch.cat((state.new_zeros((1, state.shape[-1])), state[0]))
        cards = definitions[self._rows(obs["semantic_cards"])]
        knowledge = obs["known_hand"]
        known = definitions[self._rows(knowledge[..., 0])]
        known = (known * knowledge[..., 1, None] / 60.0).sum(dim=2)
        return cards, known
