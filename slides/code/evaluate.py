# Evaluating the mortgage assistant (COMP5423 L9B).
#
#   pip install torch transformers
#   python evaluate.py --part test      100 customers, written in different ways; agent and workflow
#   python evaluate.py --part missing   20 customers who forgot to say how many years
#   python evaluate.py --part judge     replies written in 6 ways: a code rule vs the model as judge
#   python evaluate.py --part test --n 5    only the first 5 items
#
# Runs on an NVIDIA GPU, an Apple GPU, or CPU only. The model is 3.4 GB and is downloaded once.
import argparse, json, random, re, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--part", default="test", choices=["test", "missing", "judge"])
ap.add_argument("--n", type=int, default=100, help="how many items")
ap.add_argument("--out", help="save every reply to this JSON file")
ap.add_argument("--cpu", action="store_true", help="run on the CPU even if a GPU is available")
args = ap.parse_args()

RATE = 3.5   # the bank's mortgage rate, % a year

def monthly_payment(principal, annual_rate_percent, years):
    r, n = float(annual_rate_percent) / 100 / 12, round(float(years) * 12)
    if r == 0:
        return round(float(principal) / n, 2)
    return round(float(principal) * r * (1 + r) ** n / ((1 + r) ** n - 1), 2)

# ---- the test set: 100 customers, each written in one of several ways ------------------------
def amount_text(P, k):
    m = P / 1_000_000
    return [f"HK${P:,}", f"HK${m:g} million", f"{m:g}M HKD", f"{P // 10_000}萬港元"][k]

def years_text(y, k):
    words = {10: "ten", 15: "fifteen", 20: "twenty", 25: "twenty-five", 30: "thirty"}
    return [f"{y} years", f"{words[y]} years", f"{y}年", f"{y * 12} months"][k]

def customers(n, seed=0):
    rnd, out = random.Random(seed), []
    for _ in range(n):
        P = rnd.randrange(100, 801) * 10_000              # HK$1,000,000 to HK$8,000,000
        y = rnd.choice([10, 15, 20, 25, 30])
        a, b = rnd.randrange(4), rnd.randrange(4)
        out.append((P, y, f"I want to borrow {amount_text(P, a)} for {years_text(y, b)}, "
                          "paid back monthly. What is my monthly payment?"))
    return out

def matches(answer, target, tol):   # the number in the answer, within tol (a fraction)
    m = re.search(r"\d[\d,]*(?:\.\d+)?", answer.replace("HK$", ""))
    return bool(m) and abs(float(m[0].replace(",", "")) - target) <= tol * target

# ---- the model --------------------------------------------------------------------------------
device = "cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to(device).eval()
TOKENS = {"in": 0, "out": 0}

def generate(messages, tools=None, max_new_tokens=800):
    prompt = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                     enable_thinking=False, tokenize=False)
    x = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False)   # greedy
    TOKENS["in"] += x["input_ids"].shape[1]
    TOKENS["out"] += out.shape[1] - x["input_ids"].shape[1]
    return tok.decode(out[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()

TOOL = {"type": "function", "function": {
    "name": "monthly_payment",
    "description": "Monthly payment of a loan repaid in equal monthly instalments.",
    "parameters": {"type": "object", "required": ["principal", "annual_rate_percent", "years"],
                   "properties": {"principal": {"type": "number", "description": "amount borrowed"},
                                  "annual_rate_percent": {"type": "number", "description": "e.g. 3.5"},
                                  "years": {"type": "number"}}}}}

AGENT = (f"You help bank customers with mortgage questions. The bank's mortgage rate is {RATE}% a year. "
         "If the customer has not given the amount or the number of years, ask for it. "
         "Otherwise end with one line: ANSWER: followed by the monthly payment in HK$, a number only")
EXTRACT = ("Read the customer's message. Reply with one JSON object only: "
           '{"principal": number, "years": number}. If a number is not in the message, write null.')

def agent(q, max_steps=8):   # the L8A agent loop
    messages = [{"role": "system", "content": AGENT}, {"role": "user", "content": q}]
    steps = []
    for _ in range(max_steps):
        reply = generate(messages, [TOOL])
        steps.append({"reply": reply})
        calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", reply, re.S)
        if not calls:
            break
        parsed = []
        for c in calls:
            try:
                parsed.append(json.loads(c))
            except json.JSONDecodeError:
                parsed.append({"name": "?", "arguments": {}})
        messages.append({"role": "assistant", "content": reply.split("<tool_call>")[0].strip(),
                         "tool_calls": [{"type": "function", "function": {"name": p["name"],
                                         "arguments": p["arguments"]}} for p in parsed]})
        for p in parsed:
            a = p.get("arguments", {})
            try:
                result = str(monthly_payment(a["principal"], a["annual_rate_percent"], a["years"]))
            except Exception as e:
                result = f"error: {e}"
            steps[-1].setdefault("tools", []).append({"call": p, "result": result})
            messages.append({"role": "tool", "content": result})
    final = re.findall(r"ANSWER:\s*(.+)", steps[-1]["reply"])
    return (final[-1].strip() if final else steps[-1]["reply"]), steps

def workflow(q):   # the model reads two numbers; code does the rest (L9A)
    reply = generate([{"role": "system", "content": EXTRACT}, {"role": "user", "content": q}])
    try:
        f = json.loads(re.search(r"\{.*\}", reply, re.S)[0])
        if f.get("principal") is None or f.get("years") is None:
            return "ASK: " + ("amount" if f.get("principal") is None else "years"), reply
        return str(monthly_payment(f["principal"], RATE, f["years"])), reply
    except Exception as e:
        return f"error: {e}", reply

log, t0 = [], time.time()

# ---- part 1: the test set ---------------------------------------------------------------------
if args.part == "test":
    items = customers(args.n)
    for system in ["agent", "workflow"]:
        TOKENS.update({"in": 0, "out": 0})
        t1, right = time.time(), {0: 0, 0.001: 0, 0.01: 0}
        for i, (P, y, q) in enumerate(items, 1):
            answer, trace = agent(q) if system == "agent" else workflow(q)
            gold = monthly_payment(P, RATE, y)
            for tol in right:
                right[tol] += matches(answer, gold, tol) if tol else answer.strip() == f"{gold:.2f}"
            log.append({"system": system, "i": i, "message": q, "gold": gold, "answer": answer,
                        "right": matches(answer, gold, 0.001), "trace": trace})
            if i <= 8 or not matches(answer, gold, 0.001):
                print(f"{system:8s} {i:3d} gold {gold:10,.2f}  model: {answer[:28]:28s} "
                      f"{'right' if matches(answer, gold, 0.001) else 'WRONG'}  {q}")
        k = len(items)
        print(f"\n{system}: exact string {right[0]}/{k}, within 0.1% {right[0.001]}/{k}, "
              f"within 1% {right[0.01]}/{k}; tokens in {TOKENS['in']}, out {TOKENS['out']} "
              f"({TOKENS['in'] / k:.0f} + {TOKENS['out'] / k:.0f} per customer); "
              f"{time.time() - t1:.0f} s ({(time.time() - t1) / k:.2f} s per customer)\n")

# ---- part 2: customers who forgot the years -----------------------------------------------------
if args.part == "missing":
    rnd = random.Random(1)
    items = [(rnd.randrange(100, 801) * 10_000, rnd.randrange(4)) for _ in range(min(args.n, 20))]
    for system in ["agent", "workflow"]:
        asked = 0
        for i, (P, a) in enumerate(items, 1):
            q = f"I want to borrow {amount_text(P, a)}, paid back monthly. What is my monthly payment?"
            answer, trace = agent(q) if system == "agent" else workflow(q)
            ok = answer.startswith("ASK") or not (answer.startswith("error") or re.search(r"\d[\d,]*\.\d", answer))
            asked += ok
            log.append({"system": system, "i": i, "message": q, "answer": answer,
                        "asked": ok, "trace": trace})
            print(f"{system:8s} {i:2d}  {'asked' if ok else 'FAILED'}  model: {answer[:60]!r}  {str(trace)[:60]!r}")
        print(f"\n{system}: asked for the years {asked}/{len(items)}\n")

# ---- part 3: the model as a judge ---------------------------------------------------------------
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
       "sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()

def words(n):   # 0 <= n < 1,000,000, in English words
    if n < 20: return ONES[n]
    if n < 100: return TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else "")
    if n < 1000: return ONES[n // 100] + " hundred" + (" and " + words(n % 100) if n % 100 else "")
    return words(n // 1000) + " thousand" + (" " + words(n % 1000) if n % 1000 else "")

STYLES = [lambda x: f"{x:.2f}",
          lambda x: f"HK${x:,.2f} per month",
          lambda x: f"{round(x):,} Hong Kong dollars a month",
          lambda x: f"{x / 10_000:.3f}萬港元",
          lambda x: f"{words(round(x))} dollars",
          lambda x: f"Monthly payment: {x:,.2f}".replace(",", " ")]

if args.part == "judge":
    rnd = random.Random(2)
    items = customers(min(args.n, 20), seed=3)
    score = {"rule": [0] * len(STYLES), "judge": [0] * len(STYLES)}
    for i, (P, y, q) in enumerate(items, 1):
        gold = monthly_payment(P, RATE, y)
        wrong = round(gold * (1 + rnd.choice([-1, 1]) * rnd.uniform(0.03, 0.08)), 2)
        for kind, x in [("right", gold), ("wrong", wrong)]:
            for k, style in enumerate(STYLES):
                reply = style(x)
                rule = matches(reply, gold, 0.001)          # 3:1's rule: the first number, within 0.1%
                prompt = (f"The correct monthly payment is HK${gold:,.2f}.\n"
                          f"An assistant replied: \"{reply}\"\n"
                          "Does the reply give the correct monthly payment? Rounding to the dollar is fine. "
                          "Reply with one word: yes or no.")
                judge = generate([{"role": "user", "content": prompt}], max_new_tokens=10).lower().startswith("yes")
                truth = kind == "right"
                score["rule"][k] += rule == truth
                score["judge"][k] += judge == truth
                log.append({"i": i, "kind": kind, "style": k + 1, "reply": reply, "gold": gold,
                            "rule": rule, "judge": judge})
    k = 2 * len(items)
    for j, style in enumerate(STYLES):
        print(f"style {j + 1}  e.g. {style(18859.88)[:44]:44s}  rule {score['rule'][j]:2d}/{k}  "
              f"judge {score['judge'][j]:2d}/{k}")
    print(f"all     rule {sum(score['rule'])}/{k * len(STYLES)}  judge {sum(score['judge'])}/{k * len(STYLES)}")

print(f"\n({time.time() - t0:.0f} s)")
if args.out:
    json.dump(log, open(args.out, "w"), indent=1, ensure_ascii=False)
