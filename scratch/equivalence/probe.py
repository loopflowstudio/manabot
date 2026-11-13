"""Bitwise probes: NumPy host arithmetic against torch CPU arithmetic."""
import numpy as np
import torch

torch.set_num_threads(1)
rng = np.random.default_rng(0)
x = np.concatenate([rng.integers(-2**31, 2**31, 2_000_000), np.arange(-70000, 70000)]).astype(np.float32)
t = torch.from_numpy(x)
for scale in (60.0, 128.0, 256.0, 20.0, 10.0):
    a = (t / scale).numpy()
    b = x / np.float32(scale)
    print("div", scale, "differ", int((a.view(np.uint32) != b.view(np.uint32)).sum()))
a = (t.sign() * t.abs().log1p() / 10).numpy()
b = (np.sign(x) * np.log1p(np.abs(x)) / np.float32(10)).astype(np.float32)
d = a.view(np.uint32) != b.view(np.uint32)
print("program_number differ", int(d.sum()), "of", len(x), "max abs", float(np.abs(a - b).max()))
small = np.arange(-70000, 70000).astype(np.float32)
a = (torch.from_numpy(small).abs().log1p()).numpy(); b = np.log1p(np.abs(small))
print("log1p small ints differ", int((a.view(np.uint32) != b.view(np.uint32)).sum()))
age = np.arange(0, 65).astype(np.float32)
print("log1p ages differ", int((torch.from_numpy(age).log1p().numpy().view(np.uint32) != np.log1p(age).view(np.uint32)).sum()))

# Embedding gradient: select + index_put (padded learning today) against a
# masked full-shape lookup (bucketed today, and what host rows give).
torch.manual_seed(0)
emb = torch.nn.Embedding(500, 16)
B, F = 24, 96
active = torch.rand(B, F) < 0.3
rows = torch.randint(1, 500, (B, F))
w = torch.randn(B, F, 16)
def grad(fn):
    emb.zero_grad()
    (fn() * w).sum().backward()
    return emb.weight.grad.clone()
def selected():
    zeros = emb.weight.new_zeros((B, F, 16))
    return zeros.index_put(torch.nonzero(active, as_tuple=True), emb(rows[active]))
def masked_where():
    return torch.where(active.unsqueeze(-1), emb(torch.where(active, rows, 7)), 0)
def masked_mul():
    return emb(torch.where(active, rows, 0)) * active.unsqueeze(-1).float()
g0, g1, g2 = grad(selected), grad(masked_where), grad(masked_mul)
bits = lambda g: g.numpy().view(np.uint32)
print("grad select vs where: bit-differ", int((bits(g0) != bits(g1)).sum()), "value-differ", int((g0 != g1).sum()))
print("grad select vs mul:   bit-differ", int((bits(g0) != bits(g2)).sum()), "value-differ", int((g0 != g2).sum()))
with torch.no_grad():
    f0, f1, f2 = selected(), masked_where(), masked_mul()
print("fwd select vs where bit-differ", int((bits(f0) != bits(f1)).sum()), "vs mul", int((bits(f0) != bits(f2)).sum()), "value", int((f0 != f2).sum()))
# precomputed count/60 against (x*count)/60
h = torch.randn(1000, 16); c = torch.randint(0, 5, (1000, 1)).float()
p = (h * c / 60.0); q = h * (c / 60.0)
print("known (h*c)/60 vs h*(c/60) value-differ", int((p != q).sum()), "of", p.numel())
