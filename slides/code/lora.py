# LoRA fine-tuning of Qwen3-1.7B on ten Banking77 intents (COMP5423 L6B).
#
# Needs a GPU. The free Google Colab GPU (T4, 16 GB) is enough:
#   Runtime > Change runtime type > T4 GPU, then in a cell:
#   !pip install -q peft
#   !python lora.py
#
# Trains 1,605,632 of 1,720,574,976 parameters (r = 8 on q_proj and v_proj),
# 100 steps on 800 messages, then tests on the same 200 messages as L6A.
import csv, io, random, time, urllib.request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

LABELS = ["card_arrival", "card_delivery_estimate", "lost_or_stolen_card", "declined_card_payment",
          "card_payment_fee_charged", "pending_transfer", "transfer_not_received_by_recipient",
          "exchange_rate", "top_up_failed", "request_refund"]
URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{}.csv"

def load(split):
    rows = csv.DictReader(io.StringIO(urllib.request.urlopen(URL.format(split)).read().decode()))
    return [(r["text"], r["category"]) for r in rows if r["category"] in LABELS]

train, test = load("train"), load("test")
test = [x for L in LABELS for x in [t for t in test if t[1] == L][:20]]          # the 200 of L6A
rng = random.Random(0)
train = [x for L in LABELS for x in rng.sample([t for t in train if t[1] == L], 80)]
rng.shuffle(train)                                                                # 800, none in test

INSTR = ("Classify the bank customer's message into exactly one of these intents:\n"
         + "\n".join(LABELS) + "\nReply with the intent name only.")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.bfloat16    # the T4 has no fast bfloat16
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B"); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=dtype).to(device)

def prompt(text):   # the zero-shot prompt: no examples
    return tok.apply_chat_template([{"role": "system", "content": INSTR}, {"role": "user", "content": text}],
                                   add_generation_prompt=True, enable_thinking=False, tokenize=False)

model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                         task_type="CAUSAL_LM"))
trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable {trained:,} of {total:,} = {100 * trained / total:.3f}%")

def batch(items):   # prompt tokens get label -100: masked out of the loss
    ids, labels = [], []
    for text, label in items:
        p = tok(prompt(text))["input_ids"]
        a = tok(label + "<|im_end|>")["input_ids"]
        ids.append(p + a); labels.append([-100] * len(p) + a)
    m = max(map(len, ids))
    pad = lambda x, v: [v] * (m - len(x)) + x
    I = torch.tensor([pad(x, tok.pad_token_id) for x in ids])
    L = torch.tensor([pad(x, -100) for x in labels])
    A = torch.tensor([pad([1] * len(x), 0) for x in ids])
    return I.to(device), L.to(device), A.to(device)

def accuracy():
    model.eval(); right = 0
    with torch.no_grad():
        for i in range(0, len(test), 20):
            b = test[i:i + 20]
            x = tok([prompt(t) for t, _ in b], return_tensors="pt", padding=True).to(device)
            out = model.generate(**x, max_new_tokens=16, do_sample=False)
            right += sum(tok.decode(o[x["input_ids"].shape[1]:], skip_special_tokens=True).strip() == l
                         for (_, l), o in zip(b, out))
    model.train(); return right

print(f"before training: right {accuracy()} of 200")
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
model.train(); t0 = time.time()
for step in range(len(train) // 8):
    I, L, A = batch(train[step * 8:(step + 1) * 8])
    loss = model(input_ids=I, attention_mask=A, labels=L).loss
    loss.backward(); opt.step(); opt.zero_grad()
    if step % 10 == 0:
        print(f"step {step:3d}  loss {loss.item():.3f}  {time.time() - t0:.0f} s")
print(f"trained {len(train) // 8} steps in {time.time() - t0:.0f} s on {device}")
if device == "cuda":
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
print(f"after training: right {accuracy()} of 200")
model.save_pretrained("adapter")   # a few MB: only A and B
