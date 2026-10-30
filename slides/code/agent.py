# A tool-using agent loop with Qwen3-1.7B (COMP5423 L8A).
#
#   pip install torch transformers
#   python agent.py --n 3              the agent with the monthly_payment tool, on 3 of the 8 questions
#   python agent.py --tools calc       a calculator instead: the model must write the formula itself
#   python agent.py --tools none       the model alone, same questions, same instructions
#   python agent.py --show 1           print every step of one question
#   python agent.py --workflow         a workflow instead: the model only writes the numbers, code does the rest
#
# Runs on an NVIDIA GPU, an Apple GPU, or CPU only. The model is 3.4 GB and is
# downloaded once. On a laptop without a GPU, start with --n 3.
import argparse, ast, json, operator, re, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--tools", default="payment", choices=["none", "calc", "payment"],
                help="none; calc = a calculator; payment = the monthly_payment tool")
ap.add_argument("--n", type=int, default=8, help="how many questions (max 8)")
ap.add_argument("--show", type=int, help="run one question (1-8) and print every step")
ap.add_argument("--out", help="save every step of every question to this JSON file")
ap.add_argument("--workflow", action="store_true",
                help="the model writes the three numbers as JSON; our code calls monthly_payment")
args = ap.parse_args()

# ---- the questions: (loan HK$, rate % a year, years) -- question 1 is the running example ---
LOANS = [(4_200_000, 3.5, 30), (3_000_000, 4.0, 20), (5_600_000, 3.75, 25), (2_400_000, 4.25, 15),
         (6_800_000, 3.25, 30), (1_800_000, 4.5, 10), (4_750_000, 3.9, 30), (3_350_000, 4.1, 25)]

def question(P, rate, years):
    return (f"I borrow HK${P:,} for {years} years at {rate}% a year, paid back monthly."
            " What is my monthly payment?")

def gold(P, rate, years):   # the standard mortgage formula, computed exactly
    r, n = rate / 100 / 12, years * 12
    return P * r * (1 + r) ** n / ((1 + r) ** n - 1)

def correct(answer, P, rate, years):   # right to within 0.1%
    m = re.search(r"[\d,]+(?:\.\d+)?", answer)
    return bool(m) and abs(float(m[0].replace(",", "")) - gold(P, rate, years)) <= 0.001 * gold(P, rate, years)

FORMAT = "ANSWER: followed by the monthly payment in HK$, a number only"

# ---- two tools: a calculator, and monthly_payment (the formula, written by us) ------------
TOOLS = [{"type": "function", "function": {
    "name": "calculate",
    "description": "Evaluate an arithmetic expression exactly. Use + - * / ** and parentheses.",
    "parameters": {"type": "object", "required": ["expression"], "properties": {
        "expression": {"type": "string", "description": "for example (1 + 0.05) ** 12"}}}}}]

LOAN_TOOL = {"type": "function", "function": {
    "name": "monthly_payment",
    "description": "Monthly payment of a loan repaid in equal monthly instalments.",
    "parameters": {"type": "object", "required": ["principal", "annual_rate_percent", "years"],
                   "properties": {"principal": {"type": "number", "description": "amount borrowed"},
                                  "annual_rate_percent": {"type": "number", "description": "e.g. 3.5"},
                                  "years": {"type": "number"}}}}}

def monthly_payment(principal, annual_rate_percent, years):
    r, n = float(annual_rate_percent) / 100 / 12, round(float(years) * 12)
    return str(round(float(principal) * r * (1 + r) ** n / ((1 + r) ** n - 1), 2))

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}

def calculate(expression):
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers, + - * / ** and parentheses are allowed")
    try:
        return str(round(ev(ast.parse(expression.replace(",", ""), mode="eval").body), 6))
    except Exception as e:
        return f"error: {e}"

# ---- the agent loop ------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to(device).eval()

def generate(messages, tools):
    prompt = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                     enable_thinking=False, tokenize=False)
    x = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**x, max_new_tokens=800, do_sample=False)   # greedy
    reply = tok.decode(out[0, x["input_ids"].shape[1]:], skip_special_tokens=True)
    return reply.strip(), x["input_ids"].shape[1], out.shape[1] - x["input_ids"].shape[1]

def run(q, tools, max_steps=8):
    system = ("You help bank customers with loan questions. Work it out step by step. "
              f"End with one line: {FORMAT}")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": q}]
    steps = []
    for _ in range(max_steps):
        reply, n_in, n_out = generate(messages, tools)                    # 1. the model acts
        calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", reply, re.S)
        steps.append({"reply": reply, "prompt_tokens": n_in, "output_tokens": n_out})
        if not calls:                                                      # no tool call: done
            break
        text = reply.split("<tool_call>")[0].strip()
        parsed = []
        for c in calls:
            try:
                parsed.append(json.loads(c))
            except json.JSONDecodeError:
                parsed.append({"name": "?", "arguments": {}, "raw": c})
        messages.append({"role": "assistant", "content": text, "tool_calls": [
            {"type": "function", "function": {"name": p["name"], "arguments": p["arguments"]}}
            for p in parsed]})
        for p in parsed:                                                   # 2. run the tool
            a = p.get("arguments", {})
            if p.get("name") == "calculate" and "expression" in a:
                result = calculate(a["expression"])
            elif p.get("name") == "monthly_payment" and tools == [LOAN_TOOL]:
                try:
                    result = monthly_payment(a["principal"], a["annual_rate_percent"], a["years"])
                except Exception as e:
                    result = f"error: {e}"
            else:
                result = f"error: unknown tool or arguments {p.get('raw') or json.dumps(p)}"
            steps[-1].setdefault("tools", []).append({"call": p, "result": result})
            messages.append({"role": "tool", "content": result})          # 3. feed it back
    final = re.findall(r"ANSWER:\s*(.+)", steps[-1]["reply"])
    return (final[-1].strip() if final else ""), steps

# ---- a workflow: the steps are fixed in our code ------------------------------------------
EXTRACT = ("Read the loan question. Reply with one JSON object only: "
           '{"principal": number, "annual_rate_percent": number, "years": number}')

def workflow(q):
    messages = [{"role": "system", "content": EXTRACT}, {"role": "user", "content": q}]
    reply, n_in, n_out = generate(messages, None)                  # step 1: the model writes JSON
    try:
        f = json.loads(re.search(r"\{.*\}", reply, re.S)[0])        # step 2: code reads the JSON
        result = monthly_payment(f["principal"], f["annual_rate_percent"], f["years"])   # step 3
    except Exception as e:
        result = f"error: {e}"
    return result, [{"reply": reply, "prompt_tokens": n_in, "output_tokens": n_out, "result": result}]

tools = {"none": None, "calc": TOOLS, "payment": [LOAN_TOOL]}[args.tools]
items = [args.show] if args.show else range(1, min(args.n, 8) + 1)
log, right, t0 = [], 0, time.time()
for i in items:
    L = LOANS[i - 1]
    answer, steps = workflow(question(*L)) if args.workflow else run(question(*L), tools)
    ok = correct(answer, *L)
    right += ok
    log.append({"q": i, "question": question(*L), "gold": round(gold(*L), 2), "answer": answer,
                "correct": ok, "steps": steps})
    print(f"Q{i} gold {gold(*L):10,.2f}   model: {answer[:40]:40s} "
          f"{'right' if ok else 'WRONG'}   {len(steps)} step(s)")
    if args.show:
        for k, s in enumerate(steps, 1):
            print(f"\n--- step {k}: {s['prompt_tokens']} tokens in, {s['output_tokens']} out ---")
            print(s["reply"])
            for t in s.get("tools", []):
                print(f"[tool result] {t['result']}")
print(f"\n{'workflow' if args.workflow else 'tools = ' + args.tools}: {right}/{len(log)} right "
      f"in {time.time() - t0:.0f} s")
if args.out:
    json.dump(log, open(args.out, "w"), indent=1, ensure_ascii=False)
