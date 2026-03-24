"""
Orion Atlas 1B — Comprehensive Benchmark Evaluation Suite
==========================================================
Usage:
    python benchmark.py --checkpoint path/to/orion_1B_best.pt
    python benchmark.py --checkpoint path/to/checkpoint.pt --output-dir results/

Covers:
    1. Perplexity on held-out val set (val_v3.bin)
    2. Text completion quality (10 diverse prompts)
    3. Tool-calling capability (5 prompts, max 15 pts)
    4. Basic reasoning (5 questions, max 5 pts)

Outputs:
    results/YYYY-MM-DD_results.json
    results/YYYY-MM-DD_results_summary.md
"""

import sys
import os
import json
import time
import math
import struct
import argparse
import datetime
import re
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths (relative to this file's directory)
# ---------------------------------------------------------------------------
EVAL_DIR      = Path(__file__).parent
REPO_ROOT     = EVAL_DIR.parent
RESULTS_DIR   = EVAL_DIR / "results"

DEFAULT_VAL_BIN   = REPO_ROOT / "data" / "val_v3.bin"
DEFAULT_TOKENIZER = REPO_ROOT / "tokenizer" / "orion_tokenizer.model"
DEFAULT_CKPT      = Path("C:/Users/avery/orion-lab/inference/checkpoints/orion_1B_best.pt")

# Baseline perplexities for comparison (on WikiText-103 / PTB – noted where known)
BASELINES = {
    "GPT-2 (1.5B)":    29.41,
    "GPT-Neo 1.3B":    20.0,
    "OPT-1.3B":        20.3,
}

# ---------------------------------------------------------------------------
# Import model from repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(REPO_ROOT))
from model import OrionModel  # noqa: E402  (must come after sys.path insert)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(ckpt_path: Path, device: str):
    """Load checkpoint + tokenizer. Forces fp32 (known inference fix)."""
    import sentencepiece as spm

    print(f"[load] Checkpoint : {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    config = ckpt["config"]
    model = OrionModel(config)
    model.load_state_dict(ckpt["model"])
    model = model.float()          # fp32 required for correct output
    model = model.to(device)
    model.eval()

    print(f"[load] Params      : {model.count_params() / 1e9:.3f}B")
    print(f"[load] Iter        : {ckpt.get('iter', '?')}")
    print(f"[load] Best val loss: {ckpt.get('best_val_loss', '?')}")

    tokenizer_path = str(DEFAULT_TOKENIZER)
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    print(f"[load] Tokenizer   : {sp.get_piece_size()} vocab tokens")

    return model, sp, ckpt


def generate_text(model, sp, device, prompt: str,
                  max_new_tokens: int = 200,
                  temperature: float = 0.7,
                  top_k: int = 50) -> str:
    """Encode prompt, run model.generate, decode only new tokens."""
    tokens = sp.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens,
                             temperature=temperature, top_k=top_k)

    new_tokens = out[0].tolist()[len(tokens):]
    return sp.decode(new_tokens)


# ---------------------------------------------------------------------------
# 1. Perplexity on val_v3.bin
# ---------------------------------------------------------------------------

def eval_perplexity(model, device, val_bin: Path,
                    block_size: int = 512,
                    max_batches: int = 200) -> float:
    """
    Stream val_v3.bin in non-overlapping blocks and compute average cross-entropy.
    Returns perplexity. Caps at max_batches blocks to keep runtime reasonable.
    """
    print("\n[perplexity] Loading val_v3.bin …")

    if not val_bin.exists():
        print(f"[perplexity] WARNING: {val_bin} not found — skipping.")
        return None

    data = torch.from_numpy(
        __import__("numpy").frombuffer(
            val_bin.read_bytes(), dtype=__import__("numpy").uint16
        ).astype(__import__("numpy").int64)
    )

    total_loss = 0.0
    n_batches  = 0
    n_tokens   = len(data)
    step       = block_size + 1          # +1 because targets are shifted by 1

    with torch.no_grad():
        for start in range(0, min(n_tokens - step, max_batches * step), step):
            chunk = data[start : start + step].to(device)
            x = chunk[:-1].unsqueeze(0)   # [1, block_size]
            y = chunk[1:].unsqueeze(0)    # [1, block_size]
            _, loss = model(x, y)
            total_loss += loss.item()
            n_batches  += 1

    avg_loss  = total_loss / n_batches
    perplexity = math.exp(avg_loss)
    print(f"[perplexity] {n_batches} blocks × {block_size} tokens → loss={avg_loss:.4f} → ppl={perplexity:.2f}")
    return perplexity


# ---------------------------------------------------------------------------
# 2. Text completion quality
# ---------------------------------------------------------------------------

COMPLETION_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Translate to Spanish: Hello, how are you?",
    "Summarize: The quick brown fox jumps over the lazy dog.",
    "What is 15 * 7?",
    "Once upon a time in a land far away,",
    "The most important thing about neural networks is",
    "import torch\nimport torch.nn as nn\n\nclass",
    "The Python function to sort a list is",
    "In 1969, humans",
]


def eval_completions(model, sp, device) -> dict:
    print("\n[completions] Running 10 diverse prompts …")
    results = {}
    for i, prompt in enumerate(COMPLETION_PROMPTS):
        t0 = time.time()
        output = generate_text(model, sp, device, prompt)
        elapsed = time.time() - t0
        key = f"prompt_{i+1:02d}"
        results[key] = {
            "prompt":  prompt,
            "output":  output,
            "elapsed_s": round(elapsed, 2),
        }
        short_prompt = prompt.replace("\n", "\\n")[:60]
        print(f"  [{i+1:02d}] {short_prompt!r}")
        print(f"       → {output[:120].replace(chr(10), ' ')!r}")
    return results


# ---------------------------------------------------------------------------
# 3. Tool-calling capability
# ---------------------------------------------------------------------------

TOOL_PROMPTS = [
    {
        "messages": [
            ("system", "You are an AI assistant. When the user asks you to do something, call the appropriate tool using JSON."),
            ("user",   "Search the web for the latest news about AI."),
        ],
        "expected_tool": "web_search",
        "expected_param_keys": ["query"],
    },
    {
        "messages": [
            ("system", "You are an AI assistant. When the user asks you to do something, call the appropriate tool using JSON."),
            ("user",   "Get the current weather in New York City."),
        ],
        "expected_tool": "get_weather",
        "expected_param_keys": ["location"],
    },
    {
        "messages": [
            ("system", "You are an AI assistant. When the user asks you to do something, call the appropriate tool using JSON."),
            ("user",   "Calculate the square root of 144."),
        ],
        "expected_tool": "calculator",
        "expected_param_keys": ["expression"],
    },
    {
        "messages": [
            ("system", "You are an AI assistant. When the user asks you to do something, call the appropriate tool using JSON."),
            ("user",   "Send an email to john@example.com with subject 'Meeting' and body 'See you at 3pm'."),
        ],
        "expected_tool": "send_email",
        "expected_param_keys": ["to", "subject", "body"],
    },
    {
        "messages": [
            ("system", "You are an AI assistant. When the user asks you to do something, call the appropriate tool using JSON."),
            ("user",   "Open the file report.pdf and summarize its contents."),
        ],
        "expected_tool": "read_file",
        "expected_param_keys": ["filename"],
    },
]

TOOL_SYNONYMS = {
    "web_search":  {"search", "web_search", "search_web", "browser_search"},
    "get_weather": {"get_weather", "weather", "fetch_weather", "weather_lookup"},
    "calculator":  {"calculator", "calc", "evaluate", "compute", "math"},
    "send_email":  {"send_email", "email", "compose_email", "send_message"},
    "read_file":   {"read_file", "open_file", "file_read", "load_file"},
}


def _build_prompt(messages) -> str:
    """Convert message list to the Orion chat template."""
    parts = []
    for role, content in messages:
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|assistant|>")
    return "\n".join(parts)


def _extract_json(text: str):
    """Try to pull the first JSON object out of a string."""
    # Try straight parse first
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find first {...} block
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


def _score_tool_call(output: str, expected_tool: str, expected_param_keys: list) -> dict:
    """
    Score one tool-call attempt:
        1 pt  valid JSON
        1 pt  correct tool name (flexible matching)
        1 pt  at least one expected param key present
    Returns dict with scores and parsed object.
    """
    parsed  = _extract_json(output)
    valid_json = parsed is not None

    correct_tool = False
    has_params   = False

    if valid_json:
        tool_val = (
            parsed.get("tool") or
            parsed.get("name") or
            parsed.get("function") or
            parsed.get("action") or ""
        ).lower()

        synonyms = TOOL_SYNONYMS.get(expected_tool, {expected_tool})
        correct_tool = tool_val in synonyms

        params_val = (
            parsed.get("params") or
            parsed.get("parameters") or
            parsed.get("args") or
            parsed.get("arguments") or {}
        )
        if isinstance(params_val, dict):
            has_params = any(k in params_val for k in expected_param_keys)
        elif isinstance(params_val, str):
            has_params = any(k in params_val for k in expected_param_keys)

    return {
        "valid_json":    int(valid_json),
        "correct_tool":  int(correct_tool),
        "has_params":    int(has_params),
        "score":         int(valid_json) + int(correct_tool) + int(has_params),
        "parsed":        parsed,
    }


def eval_tool_calling(model, sp, device) -> dict:
    print("\n[tool-calling] Running 5 prompts (max 15 pts) …")
    results  = []
    total    = 0

    for i, spec in enumerate(TOOL_PROMPTS):
        prompt = _build_prompt(spec["messages"])
        output = generate_text(model, sp, device, prompt, max_new_tokens=200)
        scores = _score_tool_call(output, spec["expected_tool"], spec["expected_param_keys"])
        total += scores["score"]

        results.append({
            "prompt_idx":       i + 1,
            "expected_tool":    spec["expected_tool"],
            "raw_output":       output,
            **scores,
        })
        print(f"  [{i+1}] tool={spec['expected_tool']:12s}  "
              f"json={scores['valid_json']}  "
              f"tool={scores['correct_tool']}  "
              f"params={scores['has_params']}  "
              f"→ {scores['score']}/3")

    print(f"  TOTAL: {total}/15")
    return {"prompts": results, "total_score": total, "max_score": 15}


# ---------------------------------------------------------------------------
# 4. Basic reasoning
# ---------------------------------------------------------------------------

REASONING_QUESTIONS = [
    {
        "question": "If I have 3 apples and give away 1, how many do I have?",
        "answer_keywords": ["2", "two"],
    },
    {
        "question": "What comes after Monday?",
        "answer_keywords": ["tuesday"],
    },
    {
        "question": "Is 7 a prime number?",
        "answer_keywords": ["yes", "prime", "it is"],
    },
    {
        "question": "What color do you get when you mix red and blue?",
        "answer_keywords": ["purple", "violet"],
    },
    {
        "question": "Complete the pattern: 2, 4, 6, 8, ?",
        "answer_keywords": ["10", "ten"],
    },
]


def eval_reasoning(model, sp, device) -> dict:
    print("\n[reasoning] Running 5 questions (max 5 pts) …")
    results = []
    total   = 0

    for i, q in enumerate(REASONING_QUESTIONS):
        prompt = f"Question: {q['question']}\nAnswer:"
        output = generate_text(model, sp, device, prompt, max_new_tokens=60,
                               temperature=0.3, top_k=20)
        output_lower = output.lower()
        correct = any(kw in output_lower for kw in q["answer_keywords"])
        total += int(correct)

        results.append({
            "question":  q["question"],
            "output":    output.strip(),
            "correct":   correct,
            "expected":  q["answer_keywords"],
        })
        mark = "[PASS]" if correct else "[FAIL]"
        print(f"  [{i+1}] {mark}  Q: {q['question']}")
        print(f"         A: {output.strip()[:100]}")

    print(f"  TOTAL: {total}/5")
    return {"questions": results, "total_score": total, "max_score": 5}


# ---------------------------------------------------------------------------
# 5. Save results
# ---------------------------------------------------------------------------

def save_results(results: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.date.today().isoformat()

    # --- JSON ---
    json_path = output_dir / f"{date_str}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[save] JSON → {json_path}")

    # --- Markdown summary ---
    md_path = output_dir / f"{date_str}_results_summary.md"
    _write_markdown(results, md_path)
    print(f"[save] MD   → {md_path}")

    return json_path, md_path


def _ppl_comparison_table(ppl) -> str:
    lines = [
        "| Model | Perplexity |",
        "|-------|-----------|",
    ]
    if ppl is not None:
        lines.append(f"| **Orion Atlas 1B** | **{ppl:.2f}** |")
    for name, val in BASELINES.items():
        lines.append(f"| {name} | {val} |")
    return "\n".join(lines)


def _write_markdown(r: dict, path: Path):
    ppl  = r.get("perplexity")
    tc   = r.get("tool_call_score", "?/15")
    rs   = r.get("reasoning_score", "?/5")
    date = r.get("date", "unknown")
    ckpt = r.get("checkpoint", "unknown")
    it   = r.get("iter", "unknown")

    ppl_str = f"{ppl:.2f}" if ppl is not None else "N/A"

    md = f"""# Orion Atlas 1B — Benchmark Results

**Date:** {date}
**Checkpoint:** `{ckpt}`
**Training iterations:** {it}

---

## Summary

| Metric | Score |
|--------|-------|
| Perplexity (val set) | {ppl_str} |
| Tool-calling | {tc} |
| Basic reasoning | {rs} |

---

## 1. Perplexity

{_ppl_comparison_table(ppl)}

> Perplexity measured on held-out `val_v3.bin` validation set.
> Lower is better.

---

## 2. Text Completions

"""

    completions = r.get("completions", {})
    for key, val in completions.items():
        prompt = val.get("prompt", "").replace("\n", "\\n")
        output = val.get("output", "").strip().replace("\n", "\n> ")
        md += f"### `{prompt[:80]}`\n\n> {output[:400]}\n\n"

    md += "---\n\n## 3. Tool-Calling\n\n"
    tc_data = r.get("tool_call_data", {})
    md += f"**Score: {tc}**\n\n"
    md += "| # | Expected Tool | JSON | Tool Match | Params | Score |\n"
    md += "|---|--------------|------|-----------|--------|-------|\n"
    for p in tc_data.get("prompts", []):
        md += (f"| {p['prompt_idx']} | `{p['expected_tool']}` | "
               f"{p['valid_json']} | {p['correct_tool']} | "
               f"{p['has_params']} | {p['score']}/3 |\n")

    md += "\n---\n\n## 4. Basic Reasoning\n\n"
    md += f"**Score: {rs}**\n\n"
    reason_data = r.get("reasoning_data", {})
    for q in reason_data.get("questions", []):
        mark = "[PASS]" if q["correct"] else "[FAIL]"
        md += f"- {mark} **{q['question']}**\n  - *Output:* {q['output'][:120]}\n\n"

    md += "---\n\n*Generated by `eval/benchmark.py` — Orion Atlas 1B Benchmark Suite*\n"

    path.write_text(md, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Orion Atlas 1B — Benchmark Evaluation Suite")
    parser.add_argument(
        "--checkpoint", "-c",
        type=Path,
        default=DEFAULT_CKPT,
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--val-bin",
        type=Path,
        default=DEFAULT_VAL_BIN,
        help="Path to val_v3.bin for perplexity eval",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to save results (default: eval/results/)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device (cpu / cuda / cuda:0). Auto-detects if not set.",
    )
    parser.add_argument(
        "--skip-perplexity", action="store_true",
        help="Skip perplexity eval (saves time on large val sets)",
    )
    parser.add_argument(
        "--skip-completions", action="store_true",
        help="Skip text completion eval",
    )
    parser.add_argument(
        "--skip-tool-calling", action="store_true",
        help="Skip tool-calling eval",
    )
    parser.add_argument(
        "--skip-reasoning", action="store_true",
        help="Skip basic reasoning eval",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=200,
        help="Max blocks to use for perplexity (default 200 × 512 = 102k tokens)",
    )
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bench] Device: {device}")

    # Load
    t_start = time.time()
    model, sp, ckpt = load_model_and_tokenizer(args.checkpoint, device)

    results = {
        "model":      "orion-atlas-1b",
        "checkpoint": args.checkpoint.name,
        "date":       datetime.datetime.now().isoformat(),
        "iter":       ckpt.get("iter", None),
    }

    # --- Perplexity ---
    if not args.skip_perplexity:
        ppl = eval_perplexity(model, device, args.val_bin,
                              max_batches=args.max_val_batches)
        results["perplexity"] = ppl
    else:
        results["perplexity"] = None
        print("[perplexity] Skipped.")

    # --- Completions ---
    if not args.skip_completions:
        completions = eval_completions(model, sp, device)
        results["completions"] = completions
    else:
        results["completions"] = {}
        print("[completions] Skipped.")

    # --- Tool-calling ---
    if not args.skip_tool_calling:
        tc_data = eval_tool_calling(model, sp, device)
        results["tool_call_data"]  = tc_data
        results["tool_call_score"] = f"{tc_data['total_score']}/{tc_data['max_score']}"
    else:
        results["tool_call_data"]  = {}
        results["tool_call_score"] = "skipped"
        print("[tool-calling] Skipped.")

    # --- Reasoning ---
    if not args.skip_reasoning:
        reason_data = eval_reasoning(model, sp, device)
        results["reasoning_data"]  = reason_data
        results["reasoning_score"] = f"{reason_data['total_score']}/{reason_data['max_score']}"
    else:
        results["reasoning_data"]  = {}
        results["reasoning_score"] = "skipped"
        print("[reasoning] Skipped.")

    # --- Save ---
    json_path, md_path = save_results(results, args.output_dir)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Orion Atlas 1B — Benchmark Complete")
    print(f"  Elapsed       : {elapsed:.1f}s")
    print(f"  Perplexity    : {results.get('perplexity', 'N/A')}")
    print(f"  Tool-calling  : {results.get('tool_call_score', 'N/A')}")
    print(f"  Reasoning     : {results.get('reasoning_score', 'N/A')}")
    print(f"  Results JSON  : {json_path}")
    print(f"  Results MD    : {md_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
