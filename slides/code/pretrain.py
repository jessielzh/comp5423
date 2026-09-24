"""The training loop on slide 3:1, complete and runnable.

    python3 pretrain.py corpus.txt       # 5,000 steps, prints the loss curve
    python3 pretrain.py corpus.txt 500   # a shorter run

Needs PyTorch and a plain-text corpus; one character is one token, so any
text file works. The Tang-poem corpus used in the lecture is 1.7M characters
of five-character poems, and this is the script that produced the loss curve
and the samples on slide 3:2 (about nine minutes on a laptop GPU).
"""
import math
import re
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

path = sys.argv[1] if len(sys.argv) > 1 else "corpus.txt"
max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# ---- data: one character, one token ----
corpus = open(path).read()
chars = sorted(set(corpus))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in corpus])
n = int(0.9 * len(data))
train, held_out = data[:n], data[n:]
print(f"characters {len(corpus):,}  vocabulary {len(chars)}", flush=True)

# ---- the model of the previous lecture ----
V, T, C, H, L = len(chars), 64, 256, 4, 4


class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(C), nn.LayerNorm(C)
        s.qkv, s.proj = nn.Linear(C, 3 * C), nn.Linear(C, C)
        s.mlp = nn.Sequential(nn.Linear(C, 4 * C), nn.GELU(), nn.Linear(4 * C, C))
        s.drop = nn.Dropout(0.2)
        s.register_buffer("mask", torch.tril(torch.ones(T, T)).bool())

    def forward(s, x):
        B, t, _ = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(C, 2)
        q, k, v = (z.view(B, t, H, C // H).transpose(1, 2) for z in (q, k, v))
        a = (q @ k.transpose(-2, -1)) / math.sqrt(C // H)
        a = a.masked_fill(~s.mask[:t, :t], float("-inf")).softmax(-1)
        x = x + s.drop(s.proj((a @ v).transpose(1, 2).reshape(B, t, C)))
        return x + s.drop(s.mlp(s.ln2(x)))


class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok, s.pos = nn.Embedding(V, C), nn.Embedding(T, C)
        s.blocks = nn.Sequential(*[Block() for _ in range(L)])
        s.ln = nn.LayerNorm(C)
        s.head = nn.Linear(C, V, bias=False)
        s.head.weight = s.tok.weight
        for mod in s.modules():
            if isinstance(mod, (nn.Linear, nn.Embedding)):
                nn.init.normal_(mod.weight, std=0.02)
            if isinstance(mod, nn.Linear) and mod.bias is not None:
                nn.init.zeros_(mod.bias)

    def forward(s, idx):
        x = s.tok(idx) + s.pos(torch.arange(idx.shape[1], device=idx.device))
        return s.head(s.ln(s.blocks(x)))


m = GPT().to(device)
print(f"parameters {sum(p.numel() for p in m.parameters()):,}  device {device}", flush=True)


def batch(d, B=64):
    """B windows of T tokens, drawn at random. y is x shifted one place left."""
    i = torch.randint(len(d) - T - 1, (B,))
    return (torch.stack([d[j:j + T] for j in i]).to(device),
            torch.stack([d[j + 1:j + T + 1] for j in i]).to(device))


@torch.no_grad()
def evaluate():
    m.eval()
    out = [sum(F.cross_entropy(m(x).view(-1, V), y.view(-1)).item()
               for x, y in [batch(d) for _ in range(20)]) / 20
           for d in (train, held_out)]
    m.train()
    return out


@torch.no_grad()
def sample(k=4, temp=0.8):
    """k passages, with any line copied from the corpus flagged."""
    m.eval()
    known = set(re.split(r"[，。\n]", corpus))
    g = torch.Generator().manual_seed(1234)
    for _ in range(k):
        idx = torch.tensor([[stoi["\n"]]], device=device)
        while idx.shape[1] < 60:
            p = (m(idx[:, -T:])[:, -1] / temp).softmax(-1).cpu()
            nxt = torch.multinomial(p, 1, generator=g)
            if chars[nxt.item()] == "\n" and idx.shape[1] > 1:
                break
            idx = torch.cat([idx, nxt.to(device)], 1)
        out = "".join(chars[i] for i in idx[0, 1:].tolist())
        copied = [l for l in re.split(r"[，。]", out) if l and l in known]
        print("   " + out + ("   * copied: " + " ".join(copied) if copied else ""), flush=True)
    m.train()


# ---- the loop: slide 3:1, line for line ----
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1)
t0 = time.time()
for step in range(max_steps + 1):
    if step % 250 == 0 or step == max_steps:
        tr, ho = evaluate()
        print(f"step {step:5d}  train {tr:.3f}  held out {ho:.3f}  "
              f"perplexity {math.exp(ho):7.1f}  {time.time() - t0:6.0f}s", flush=True)
    if step in (0, 100, 1000, max_steps):
        sample()
    for g_ in opt.param_groups:                       # cosine decay to zero
        g_["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * step / max_steps))
    x, y = batch(train)
    loss = F.cross_entropy(m(x).view(-1, V), y.view(-1))
    loss.backward()
    opt.step()
    opt.zero_grad()
print(f"total {time.time() - t0:.0f}s", flush=True)
