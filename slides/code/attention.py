"""Self-attention on a three-word sentence, by hand.

The embedding table E is the one from the neural-network lecture, unchanged:
five words, two numbers each. The sentence is "the cat sat".

Every weight below is chosen by hand so the arithmetic is checkable on paper;
a real model reaches its weights by training.
"""

from math import exp, sqrt, tanh
import random

VOCAB = ["the", "cat", "dog", "sat", "ran"]

E = [[0.5, -0.2],    # the
     [0.9, 0.4],     # cat
     [0.8, 0.5],     # dog
     [-0.3, 0.7],    # sat
     [-0.4, 0.6]]    # ran

W_Q = [[-2.0, 2.0],
       [0.0, 2.0]]
W_K = [[1.0, 1.0],
       [0.0, -2.0]]
W_V = [[1.0, 1.0],
       [1.0, -1.0]]


def matvec(M, v):
    return [sum(row[i] * v[i] for i in range(len(v))) for row in M]


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def softmax(s):
    live = [v for v in s if v is not None]
    top = max(live)
    e = [0.0 if v is None else exp(v - top) for v in s]
    total = sum(e)
    return [v / total for v in e]


def weighted(alpha, vectors):
    return [sum(a * v[i] for a, v in zip(alpha, vectors)) for i in range(len(vectors[0]))]


def attention(X, WQ, WK, WV, causal=True, scale=True):
    Q = [matvec(WQ, x) for x in X]
    K = [matvec(WK, x) for x in X]
    V = [matvec(WV, x) for x in X]
    dk = len(Q[0])
    S = [[dot(q, k) / (sqrt(dk) if scale else 1.0) for k in K] for q in Q]
    A = []
    for i, row in enumerate(S):
        A.append(softmax([v if (j <= i or not causal) else None for j, v in enumerate(row)]))
    O = [weighted(a, V) for a in A]
    return Q, K, V, S, A, O


def vec(v, w=5, p=2):
    return "[" + " ".join(f"{x:{w}.{p}f}" for x in v) + "]"


def table(name, rows, labels, p=2):
    print(f"  {name:<6}" + "".join(f"{l:>8}" for l in labels))
    for l, r in zip(labels, rows):
        print(f"  {l:<6}" + "".join("       -" if x is None else f"{x:8.{p}f}" for x in r))


words = ["the", "cat", "sat"]
X = [E[VOCAB.index(w)] for w in words]

print("$ python attention.py")
print()
print("--- the sentence, one row of E per word ---")
for w, x in zip(words, X):
    print(f"  x_{w:<4} {vec(x)}")

# ---------------------------------------------------------------- recurrence
print()
print("--- a recurrent network reads it one word at a time ---")
print("  h_t = tanh(0.5 h_(t-1) + x_t),  h_0 = [0 0]")
h = [0.0, 0.0]
for w, x in zip(words, X):
    h = [tanh(0.5 * a + b) for a, b in zip(h, x)]
    print(f"  after {w:<4} h = {vec(h)}")

print()
print("--- change only the FIRST word, cat -> dog; how far does the last h move? ---")
filler = ["sat", "the", "dog", "ran", "the", "cat", "sat", "the"] * 4
for T in (2, 4, 8, 16, 32):
    ends = []
    for first in ("cat", "dog"):
        h = [0.0, 0.0]
        for w in [first] + filler[:T - 1]:
            h = [tanh(0.5 * a + b) for a, b in zip(h, E[VOCAB.index(w)])]
        ends.append(h)
    gap = max(abs(a - b) for a, b in zip(*ends))
    print(f"  {T:>2} words   last h moves by {gap:.7f}")

# ---------------------------------------------------------------- no parameters
print()
print("--- attention with no parameters: score = x_i . x_j ---")
S0 = [[dot(a, b) for b in X] for a in X]
table("score", S0, words)
A0 = [softmax(r) for r in S0]
print()
table("alpha", A0, words)
print()
print(f"  o_sat = {A0[2][0]:.2f} x_the + {A0[2][1]:.2f} x_cat + {A0[2][2]:.2f} x_sat"
      f" = {vec(weighted(A0[2], X))}")

# ---------------------------------------------------------------- Q K V
Q, K, V, S, A, O = attention(X, W_Q, W_K, W_V)
print()
print("--- queries, keys and values: q = W_Q x,  k = W_K x,  v = W_V x ---")
for w, q, k, v in zip(words, Q, K, V):
    print(f"  {w:<4} q {vec(q)}   k {vec(k)}   v {vec(v)}")

print()
print("--- scores q_i . k_j, before scaling ---")
raw = [[dot(q, k) for k in K] for q in Q]
table("q.k", raw, words)
print(f"  sat->cat = {raw[2][1]:.2f}    cat->sat = {raw[1][2]:.2f}    not symmetric")

print()
print("--- divide by sqrt(d_k) = sqrt(2) = 1.41, then softmax the sat row ---")
print(f"  sat row   {' '.join(f'{v:6.2f}' for v in raw[2])}   /1.41 ="
      f"   {' '.join(f'{v:6.2f}' for v in S[2])}")
print(f"  alpha     {' '.join(f'{v:6.2f}' for v in A[2])}")

print()
print("--- the output for sat: a weighted sum of the VALUES ---")
print(f"  o_sat = {A[2][0]:.2f} v_the + {A[2][1]:.2f} v_cat + {A[2][2]:.2f} v_sat")
print(f"        = {vec(O[2])}")

# ---------------------------------------------------------------- scaling
print()
print("--- why divide: dot products of random vectors grow with d_k ---")
rng = random.Random(5423)
for dk in (2, 64, 512):
    qs = [[rng.gauss(0, 1) for _ in range(dk)] for _ in range(200)]
    ks = [[rng.gauss(0, 1) for _ in range(dk)] for _ in range(200)]
    sc = [dot(q, k) for q, k in zip(qs, ks)]
    spread = sqrt(sum(v * v for v in sc) / len(sc))
    row = sc[:3]
    unscaled = softmax(row)
    scaled = softmax([v / sqrt(dk) for v in row])
    print(f"  d_k = {dk:>3}   typical score size {spread:6.1f}   "
          f"softmax {vec(unscaled, 4, 2)}   scaled {vec(scaled, 4, 2)}")
    print(f"             three scores {vec(row, 6, 2)}   / sqrt(d_k) = {sqrt(dk):.2f}"
          f" -> {vec([v / sqrt(dk) for v in row], 6, 2)}")

# ---------------------------------------------------------------- matrix form, mask
print()
print("--- every row at once, with the causal mask: each word sees itself and earlier ---")
Sfull = [[dot(q, k) / sqrt(2) for k in K] for q in Q]
masked = [[v if j <= i else None for j, v in enumerate(r)] for i, r in enumerate(Sfull)]
table("score", masked, words)
print()
table("alpha", A, words)
print()
for w, o in zip(words, O):
    print(f"  o_{w:<4} {vec(o)}")

# ---------------------------------------------------------------- two heads
print()
print("--- two heads: head 1 uses row 1 of W_Q, W_K, W_V; head 2 uses row 2 ---")
heads = []
for r in (0, 1):
    q_, k_, v_, s_, a_, o_ = attention(X, [W_Q[r]], [W_K[r]], [W_V[r]])
    heads.append(o_)
    print(f"  head {r + 1}   sat row  alpha = "
          + "  ".join(f"{w} {v:.2f}" for w, v in zip(words, a_[2])))
print(f"  concat for sat: [{heads[0][2][0]:.2f} {heads[1][2][0]:.2f}]"
      f"   then W_O (identity here)")
print(f"  one head of size 2 gave sat:  "
      + "  ".join(f"{w} {v:.2f}" for w, v in zip(words, A[2])))

# ---------------------------------------------------------------- the two limits
print()
print("--- the same three weight matrices on sentences of any length ---")
for sent in (["the", "cat", "sat"],
             ["the", "cat", "sat", "the", "dog", "ran"]):
    Xs = [E[VOCAB.index(w)] for w in sent]
    *_, As, Os = attention(Xs, W_Q, W_K, W_V)
    print(f"  {len(sent)} words, W_Q {len(W_Q)}x{len(W_Q[0])}, last word's alpha:")
    print("    " + "  ".join(f"{w} {a:.2f}" for w, a in zip(sent, As[-1])))

print()
print("--- swap the first two words: what does sat receive? ---")
for sent in (["the", "cat", "sat"], ["cat", "the", "sat"]):
    Xs = [E[VOCAB.index(w)] for w in sent]
    *_, As, Os = attention(Xs, W_Q, W_K, W_V)
    print(f"  {' '.join(sent):<12} o_sat = {vec(Os[-1], 6, 3)}")
