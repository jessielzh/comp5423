"""One backward pass of the same fixed-window neural language model.

The network, the weights and the context are the ones from forward.py:
    the cat sat    the cat ran    the dog sat
    context "the cat", the word that actually came next is "sat".

Every gradient printed here is checked against the slow, obvious method —
change one weight by a tiny amount and see how much the loss moves.
"""

from math import exp, log

VOCAB = ["the", "cat", "dog", "sat", "ran"]

E = [[0.5, -0.2], [0.9, 0.4], [0.8, 0.5], [-0.3, 0.7], [-0.4, 0.6]]

W1 = [[1.0, 0.0, 1.0, 0.0],
      [0.0, 1.0, 0.0, 1.0],
      [0.0, 1.0, -1.0, 0.0]]
b1 = [0.0, 0.0, 0.0]

W2 = [[-1.0, 0.0, 0.5], [-1.0, 0.0, 0.5], [-1.0, 0.0, 0.5],
      [0.5, 1.0, 0.0], [0.5, 0.5, 0.0]]
b2 = [0.0, 0.0, 0.0, 0.0, 0.0]

CONTEXT = [0, 1]      # the cat
TRUTH = 3             # sat


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


def backward(x, z, h, p, truth, W1, W2):
    ds = [p[i] - (1.0 if i == truth else 0.0) for i in range(len(p))]

    dW2 = [[ds[i] * h[j] for j in range(len(h))] for i in range(len(ds))]
    dh = [sum(W2[i][j] * ds[i] for i in range(len(ds))) for j in range(len(h))]

    dz = [dh[j] if z[j] > 0 else 0.0 for j in range(len(z))]

    dW1 = [[dz[j] * x[k] for k in range(len(x))] for j in range(len(dz))]
    dx = [sum(W1[j][k] * dz[j] for j in range(len(dz))) for k in range(len(x))]

    return ds, dW2, dh, dz, dW1, dx


def loss(W1, W2):
    _, _, _, _, p = forward(CONTEXT, E, W1, b1, W2, b2)
    return -log(p[TRUTH])


def wiggle(M, i, j, eps=1e-5):
    """The slow way: nudge one weight, see how much the loss moves."""
    M[i][j] += eps
    up = loss(W1, W2)
    M[i][j] -= 2 * eps
    down = loss(W1, W2)
    M[i][j] += eps
    return (up - down) / (2 * eps)


def row(name, values, note=""):
    body = " ".join(f"{v + 0.0:7.3f}" for v in values)
    print(f"{name:<8}{body}   {note}".rstrip())


x, z, h, s, p = forward(CONTEXT, E, W1, b1, W2, b2)
J = -log(p[TRUTH])

print("$ python backward.py")
print()
print("forward  context 'the cat', truth 'sat'")
row("h", h)
row("P", p)
print(f"J        {J:.3f}")

ds, dW2, dh, dz, dW1, dx = backward(x, z, h, p, TRUTH, W1, W2)

print()
print("--- the slow way: nudge one weight and watch J ---")
print(f"W2[sat][0] = {W2[3][0]:+.3f}          J = {J:.5f}")
W2[3][0] += 0.001
print(f"           = {W2[3][0]:+.3f}   +0.001  J = {loss(W1, W2):.5f}")
W2[3][0] -= 0.001
print(f"slope      {wiggle(W2, 3, 0):+.4f}   J falls, so this weight should rise")

print()
# printed in the order the code computes them, so each quantity is followed by
# the weights it settles before the signal moves on: ds, dW2, dh, dz, dW1
print("--- the fast way: one backward pass ---")
row("dJ/ds", ds, "what you said, minus what was true")
print("dJ/dW2   the 'sat' row")
row("", dW2[3], "= dJ/ds[sat] times h")

print()
row("dJ/dh", dh)
row("dJ/dz", dz, "the third unit was off, so no blame reaches it")
print("dJ/dW1   the first row")
row("", dW1[0], "= dJ/dz[0] times x")

print()
print("--- do they agree? ---")
for (M, i, j, name) in [(W2, 3, 0, "W2[sat][0]"), (W2, 4, 1, "W2[ran][1]"),
                        (W1, 0, 2, "W1[0][2]")]:
    fast = (dW2 if M is W2 else dW1)[i][j]
    print(f"{name:<12} backward {fast:+.4f}    nudging {wiggle(M, i, j):+.4f}")

print()
print("--- one step, learning rate 0.5 ---")
for i in range(len(W2)):
    for j in range(len(W2[0])):
        W2[i][j] -= 0.5 * dW2[i][j]
for i in range(len(W1)):
    for j in range(len(W1[0])):
        W1[i][j] -= 0.5 * dW1[i][j]
_, _, _, _, p2 = forward(CONTEXT, E, W1, b1, W2, b2)
print(f"P(sat)   {p[TRUTH]:.3f}  ->  {p2[TRUTH]:.3f}")
print(f"J        {J:.3f}  ->  {-log(p2[TRUTH]):.3f}")
