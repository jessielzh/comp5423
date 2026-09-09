"""One forward pass of a fixed-window neural language model, by hand.

Vocabulary, corpus and bigram counts are the ones from the counting lecture:
    the cat sat    the cat ran    the dog sat

Five words, two-dimensional embeddings, a window of two, three hidden units.
Every weight below is chosen by hand so the arithmetic is checkable on paper;
a real model reaches its weights by training, which is the next lecture.
"""

from math import exp, log

VOCAB = ["the", "cat", "dog", "sat", "ran"]

E = [[0.5, -0.2],    # the
     [0.9, 0.4],     # cat
     [0.8, 0.5],     # dog
     [-0.3, 0.7],    # sat
     [-0.4, 0.6]]    # ran

W1 = [[1.0, 0.0, 1.0, 0.0],
      [0.0, 1.0, 0.0, 1.0],
      [0.0, 1.0, -1.0, 0.0]]
b1 = [0.0, 0.0, 0.0]

W2 = [[-1.0, 0.0, 0.5],    # the
      [-1.0, 0.0, 0.5],    # cat
      [-1.0, 0.0, 0.5],    # dog
      [0.5, 1.0, 0.0],     # sat
      [0.5, 0.5, 0.0]]     # ran
b2 = [0.0, 0.0, 0.0, 0.0, 0.0]


def matvec(M, v):
    return [sum(row[i] * v[i] for i in range(len(v))) for row in M]


def add(u, v):
    return [a + b for a, b in zip(u, v)]


def forward(context, E, W1, b1, W2, b2):
    x = E[context[0]] + E[context[1]]

    z = add(matvec(W1, x), b1)
    h = [max(0.0, v) for v in z]

    s = add(matvec(W2, h), b2)

    top = max(s)
    e = [exp(v - top) for v in s]
    total = sum(e)
    p = [v / total for v in e]

    return x, z, h, s, p


def row(name, values, note=""):
    body = " ".join(f"{v:6.2f}" for v in values)
    print(f"{name:<7}{body}   {note}".rstrip())


def words(name, values):
    body = "  ".join(f"{w} {v:.2f}" for w, v in zip(VOCAB, values))
    print(f"{name:<7}{body}")


def run(context, truth):
    ids = [VOCAB.index(w) for w in context]
    x, z, h, s, p = forward(ids, E, W1, b1, W2, b2)
    return ids, x, z, h, s, p, VOCAB.index(truth)


print("$ python forward.py")
print()
print("vocab  " + "  ".join(f"{i}:{w}" for i, w in enumerate(VOCAB)))
print("E      " + "   ".join(f"{w} [{E[i][0]:5.1f} {E[i][1]:5.1f}]"
                             for i, w in enumerate(VOCAB[:3])))
print("       " + "   ".join(f"{w} [{E[i+3][0]:5.1f} {E[i+3][1]:5.1f}]"
                             for i, w in enumerate(VOCAB[3:])))

ids, x, z, h, s, p, gold = run(["the", "cat"], "sat")
print()
print("--- context: the cat ---")
print(f"ids    {ids[0]} {ids[1]}")
row("x", x, "two rows of E, laid end to end")

print()
print("--- layer 1:  z = W1 x + b1,  h = max(0, z) ---")
row("z", z)
row("h", h, "the third unit was negative, so it is off")

print()
print("--- layer 2:  s = W2 h + b2 ---")
row("s", s, "one score per word in the vocabulary")

print()
print("--- softmax ---")
words("P", p)

print()
print("--- loss ---")
print(f"truth  sat     J = -log {p[gold]:.3f} = {-log(p[gold]):.3f}")

ids, x, z, h, s, p, gold = run(["the", "dog"], "ran")
print()
print("--- same weights, context: the dog ---")
row("x", x)
row("h", h)
words("P", p)
print(f"       counting said P(ran | dog) = 0, never seen")

print()
print("--- every training example in the corpus ---")
print("with a window of two, each sentence gives one example")
print()
print("  context      truth    P(truth)   J_t")
CORPUS = [(["the", "cat"], "sat"), (["the", "cat"], "ran"), (["the", "dog"], "sat")]
total = 0.0
for ctx, truth in CORPUS:
    ids = [VOCAB.index(w) for w in ctx]
    _, _, _, _, pr = forward(ids, E, W1, b1, W2, b2)
    j = -log(pr[VOCAB.index(truth)])
    total += j
    print(f"  {' '.join(ctx):<12} {truth:<8} {pr[VOCAB.index(truth)]:.4f}"
          f"     {j:.3f}")
print()
print(f"  J(theta) = average of the three = {total / len(CORPUS):.3f}")

print()
print("--- the same weights, scored against a shuffled corpus ---")
SHUFFLED = [(["cat", "sat"], "the"), (["dog", "the"], "ran"), (["sat", "the"], "cat")]
print()
print("  context      truth    P(truth)   J_t")
total = 0.0
for ctx, truth in SHUFFLED:
    ids = [VOCAB.index(w) for w in ctx]
    _, _, _, _, pr = forward(ids, E, W1, b1, W2, b2)
    j = -log(pr[VOCAB.index(truth)])
    total += j
    print(f"  {' '.join(ctx):<12} {truth:<8} {pr[VOCAB.index(truth)]:.4f}     {j:.3f}")
print()
print(f"  J(theta) = {total / len(SHUFFLED):.3f}   against 0.820 on the real corpus")
