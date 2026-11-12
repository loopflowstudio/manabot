"""Opt-in live-token attention: object attention over valid objects only.

The object table is padded: every observation carries all of its slots, and
padded attention runs every block (projections, attention, norms, feedforward)
over each slot before masking padding to zero. With
`ActorCritic(..., live_tokens=True)`, set by a training stage's
`Execution.live_tokens`, the policy packs a batch's valid objects into one
sequence [L,D] (row-major, so each observation's objects are contiguous and in
slot order), runs the blocks over that sequence and scatters the result back to
[B,N,D] with zero padding, which is what the padded path returns.

Attention is restricted to objects of the same observation by running
`scaled_dot_product_attention` on jagged nested tensors whose offsets are the
cumulative object counts. On CUDA this dispatches to the fused variable-length
kernels (memory-efficient attention in float32, flash in half precision); on
CPU it falls back to per-observation math attention, which is correct but slow
and only meant for tests. The setting changes no parameter, buffer,
configuration field, checkpoint or policy input.

Host reads: sizing the packed sequence needs the batch's total number of valid
objects. Collection measures it on the host while preparing a batch
(`PreparedObservation.live_objects`), so its forward reads nothing back;
learning reads it once per forward through `nonzero`. Offsets stay on the
device, and the jagged tensors carry host bounds on sequence length so the
attention dispatch reads nothing either.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LiveTokens:
    """Packed coordinates of a batch's present slots.

    `index` [L] holds flat positions into [B*S]; `offsets` [B+1] (int64, on
    the device) are the cumulative per-observation counts, so observation b
    owns packed rows offsets[b]:offsets[b+1].
    """

    index: torch.Tensor
    offsets: torch.Tensor
    batch: int
    slots: int

    @classmethod
    def pack(cls, present: torch.Tensor, total: int | None) -> "LiveTokens":
        """Pack `present` [B,S] (bool); `total` is its host-measured count.

        Without `total`, `nonzero` reads the count back once. A wrong `total`
        would mis-pack silently, so only preparation, which counts the same
        CPU tensors, supplies it.
        """
        batch, slots = present.shape
        flat = present.reshape(-1)
        if total is None:
            index = flat.nonzero().squeeze(1)
        else:
            index = torch.nonzero_static(flat, size=total).squeeze(1)
        offsets = F.pad(present.sum(1).cumsum(0), (1, 0))
        return cls(index, offsets, batch, slots)

    def gather(self, values: torch.Tensor) -> torch.Tensor:
        """Packed rows [L,...] of `values` [B,S,...]."""
        return values.reshape(self.batch * self.slots, *values.shape[2:]).index_select(
            0, self.index
        )

    def scatter(self, packed: torch.Tensor) -> torch.Tensor:
        """Packed rows [L,...] back to [B,S,...], zero at absent slots."""
        rest = packed.shape[1:]
        full = packed.new_zeros((self.batch * self.slots, *rest))
        return full.index_copy(0, self.index, packed).view(
            self.batch, self.slots, *rest
        )

    def self_attention(
        self, attention: nn.MultiheadAttention, packed: torch.Tensor
    ) -> torch.Tensor:
        """`attention`'s self-attention [L,D] within each observation.

        Uses the module's own projections, so it is the same function as
        `attention(x, x, x, key_padding_mask=absent)` at present slots.
        """
        length, width = packed.shape
        heads = attention.num_heads
        qkv = F.linear(packed, attention.in_proj_weight, attention.in_proj_bias)
        # [L,3,H,d] -> three [L,H,d] jagged tensors viewed as [B,H,{j},d].
        q, k, v = (
            torch.nested.nested_tensor_from_jagged(
                part.contiguous(),
                self.offsets,
                # Both player rows are always valid, so no sequence is empty;
                # the slot count bounds every length. Passing both keeps the
                # fused-kernel checks from reading them back.
                min_seqlen=1,
                max_seqlen=self.slots,
            ).transpose(1, 2)
            for part in qkv.view(length, 3, heads, width // heads).unbind(1)
        )
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).values().reshape(length, width)
        return F.linear(out, attention.out_proj.weight, attention.out_proj.bias)
