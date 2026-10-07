# Zero-shot and few-shot intent classification with Qwen3-1.7B (COMP5423 L6A).
#
#   pip install torch transformers
#   python prompting.py --shots 0 --n 20      zero-shot, on 20 of the 200 test messages
#   python prompting.py --shots 1             one example per intent, all 200 test messages
#   python prompting.py --shots 1 --wrong     the same examples with deliberately wrong labels
#
# Runs on an NVIDIA GPU, an Apple GPU, or CPU only. The model is 3.4 GB and is
# downloaded once. On a laptop without a GPU, start with --n 20.
import argparse, csv, io, random, time, urllib.request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--shots", type=int, default=0, help="examples per intent in the prompt")
ap.add_argument("--draw", type=int, default=0, help="which random set of examples")
ap.add_argument("--wrong", action="store_true", help="give the examples wrong labels")
ap.add_argument("--n", type=int, default=200, help="how many test messages (max 200)")
args = ap.parse_args()

LABELS = ["card_arrival", "card_delivery_estimate", "lost_or_stolen_card", "declined_card_payment",
          "card_payment_fee_charged", "pending_transfer", "transfer_not_received_by_recipient",
          "exchange_rate", "top_up_failed", "request_refund"]
URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{}.csv"

def load(split):   # Banking77 (Casanueva et al. 2020), only our ten intents
    rows = csv.DictReader(io.StringIO(urllib.request.urlopen(URL.format(split)).read().decode()))
    return [(r["text"], r["category"]) for r in rows if r["category"] in LABELS]

train, test = load("train"), load("test")
test = [x for L in LABELS for x in [t for t in test if t[1] == L][:20]]   # 20 per intent
test = [test[i] for i in range(0, 200, max(1, 200 // args.n))][:args.n]   # spread over all intents

rng = random.Random(args.draw)
examples = [x for L in LABELS for x in rng.sample([t for t in train if t[1] == L], args.shots)]
rng.shuffle(examples)
if args.wrong:
    examples = [(t, rng.choice([l for l in LABELS if l != L])) for t, L in examples]

INSTR = ("Classify the bank customer's message into exactly one of these intents:\n"
         + "\n".join(LABELS) + "\nReply with the intent name only.")

def messages(text):
    m = [{"role": "system", "content": INSTR}]
    for t, l in examples:
        m += [{"role": "user", "content": t}, {"role": "assistant", "content": l}]
    return m + [{"role": "user", "content": text}]

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to(device).eval()

right, lengths, t0 = 0, [], time.time()
for i, (text, label) in enumerate(test):
    prompt = tok.apply_chat_template(messages(text), add_generation_prompt=True,
                                     enable_thinking=False, tokenize=False)
    x = tok(prompt, return_tensors="pt").to(device)
    lengths.append(x["input_ids"].shape[1])
    with torch.no_grad():
        out = model.generate(**x, max_new_tokens=16, do_sample=False)   # greedy
    answer = tok.decode(out[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    right += answer == label
    if answer != label:
        print(f"  {text[:50]!r:54} said {answer:24} truth {label}")

print(f"\n{args.shots} example(s) per intent{' (wrong labels)' if args.wrong else ''}: "
      f"right {right} of {len(test)} = {100 * right / len(test):.1f}%   "
      f"prompt {sum(lengths) / len(lengths):.0f} tokens   {time.time() - t0:.0f} s on {device}")
