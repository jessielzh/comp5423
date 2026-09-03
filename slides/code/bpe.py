"""Byte pair encoding, trained on one short string.

The program on slide 2:1 of L2A, with the printing added: run it and you get
exactly the terminal panel on that slide.

    python bpe.py                                the string below, 8 symbols
    python bpe.py abracadabra --vocab-size 12    any string, any size
"""
import argparse
from collections import Counter


def most_common(pairs, vocab):
    """The most frequent neighbouring pair. Ties go to the pair whose symbols
    are the older ones, so the same text always trains the same tokenizer."""
    top = max(pairs.values())
    tied = [p for p in pairs if pairs[p] == top]
    return min(tied, key=lambda p: max(vocab[p[0]], vocab[p[1]]))


def merge(data, a, b):
    """Rewrite the data with every neighbouring a, b joined into one piece."""
    out, i = [], 0
    while i < len(data):
        if data[i:i + 2] == [a, b]:
            out.append(a + b)
            i += 2
        else:
            out.append(data[i])
            i += 1
    return out


def show(vocab, data):
    print('data   ' + '  '.join(data))
    print('ids    ' + '  '.join(str(vocab[t]).rjust(len(t)) for t in data))


def train(data, vocab_size):
    data = list(data)
    chars = sorted(set(data))
    vocab = {c: i for i, c in enumerate(chars)}

    print(f'target {vocab_size}       the vocabulary may hold {vocab_size} symbols')
    print('vocab  ' + '  '.join(f'{i}:{c}' for c, i in vocab.items()) +
          f'      the {len(chars)} characters')
    show(vocab, data)

    while len(vocab) < vocab_size:
        pairs = Counter(zip(data, data[1:]))
        print(f'\n--- round {len(vocab) - len(chars) + 1} ---')
        print('pairs  ' + '   '.join(f'{a}+{b}:{n}' for (a, b), n in pairs.most_common()))

        if max(pairs.values()) < 2:
            print(f'stop   nothing appears twice  '
                  f'({len(vocab)} of {vocab_size} slots used)')
            break

        a, b = most_common(pairs, vocab)
        tied = [p for p in pairs if pairs[p] == max(pairs.values())]
        if len(tied) > 1:
            print('tie    ' + ' and '.join(f'{x}+{y}' for x, y in tied) +
                  f'  ->  keep the pair of older symbols  ->  {a}+{b}')
        print(f'best   {a}+{b}  ({pairs[(a, b)]} times)')

        vocab[a + b] = len(vocab)
        print(f'add    {len(vocab) - 1}:{a + b}')
        data = merge(data, a, b)
        show(vocab, data)

    print('\nvocab  ' + '  '.join(f'{i}:{c}' for c, i in vocab.items()))
    return vocab


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('text', nargs='?', default='aaabdaaabac')
    ap.add_argument('--vocab-size', type=int, default=8)
    args = ap.parse_args()

    print(f'$ python bpe.py  {args.text}  --vocab-size {args.vocab_size}\n')
    train(args.text, args.vocab_size)
