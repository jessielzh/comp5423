"""Tokenizing a new string with a tokenizer that has already been trained.

Nothing is counted here and nothing is learned: training did that once, and
all that is left is the vocabulary, where every piece has an id. The id is the
order the piece was learned in, so joining the smallest id first replays
training on the new string.

    python encode.py            abaaab
    python encode.py cabbadaaa  any string over the letters a b c d
"""
import argparse

# the vocabulary bpe.py learned from 'aaabdaaabac', copied here
VOCAB = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'aa': 4, 'ab': 5, 'aaab': 6}


def encode(text, vocab):
    data = list(text)

    while True:
        best = None
        for i in range(len(data) - 1):
            pair = data[i] + data[i + 1]
            if pair in vocab:
                if best is None or vocab[pair] < best[0]:
                    best = (vocab[pair], i)

        if best is None:
            return data

        i = best[1]
        data[i:i + 2] = [data[i] + data[i + 1]]


def show(data, vocab):
    print('data   ' + '  '.join(data))
    print('ids    ' + '  '.join(str(vocab[t]).rjust(len(t)) for t in data))


def encode_aloud(text, vocab):
    """encode(), printing every pass."""
    data = list(text)
    show(data, vocab)

    for n in range(1, 99):
        print(f'\n--- pass {n} ---')
        pairs, best = [], None
        for i in range(len(data) - 1):
            pair = data[i] + data[i + 1]
            pairs.append(f'{pair}:{vocab[pair]}' if pair in vocab else f'{pair}:-')
            if pair in vocab:
                if best is None or vocab[pair] < best[0]:
                    best = (vocab[pair], i)
        print('pairs  ' + '   '.join(pairs) + '      (- = not in the vocabulary)')

        if best is None:
            print('stop   no pair is in the vocabulary')
            return data

        i = best[1]
        print(f'best   {data[i] + data[i + 1]}:{best[0]}   smallest id wins')
        data[i:i + 2] = [data[i] + data[i + 1]]
        show(data, vocab)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('text', nargs='?', default='abaaab')
    text = ap.parse_args().text

    print(f'$ python encode.py  {text}\n')
    print('vocab  ' + '  '.join(f'{i}:{p}' for p, i in VOCAB.items()) +
          '      learned by bpe.py')
    out = encode_aloud(text, VOCAB)
    print('\ntokens ' + '  '.join(out))
    print('ids    ' + '  '.join(str(VOCAB[t]) for t in out))
    print("decode ''.join(tokens)  =  " + ''.join(out))
    assert encode(text, VOCAB) == out
