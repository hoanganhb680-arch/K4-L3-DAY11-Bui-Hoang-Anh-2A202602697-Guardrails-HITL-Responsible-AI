"""Run Phase 4 attacks with spacing + retry to respect Gemini free-tier
RPM (requests/minute) and transient 429/503 errors."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from agents.agent import create_red_agent_default
from agents.guards_agent import create_red_agent_advance
from attacks.attacks import (
    adversarial_prompts,
    classify_attack_outcome,
    write_run_attack_json,
    save_attack_results,
)
from core.utils import chat_with_agent


async def call_with_retry(agent, runner, text, attempt_limit=6):
    last_err = None
    for attempt in range(attempt_limit):
        try:
            response, _ = await chat_with_agent(agent, runner, text)
            return response, None
        except Exception as e:
            last_err = e
            msg = str(e)
            retriable = (
                "429" in msg or "503" in msg
                or "UNAVAILABLE" in msg.upper()
                or "RESOURCE" in msg.upper()
                or "quota" in msg.lower()
            )
            wait = 20 * (attempt + 1) if retriable else 8 * (attempt + 1)
            print(f"    [retry {attempt + 1}/{attempt_limit - 1}] {type(e).__name__}, cho {wait}s ...", flush=True)
            await asyncio.sleep(wait)

    return None, last_err


async def run_list(agent, runner, target, delay):
    results = []
    for attack in adversarial_prompts:
        print(f"[{target}] Attack #{attack['id']} ({attack['category']}) ...", flush=True)
        response, err = await call_with_retry(agent, runner, attack["input"])

        if err is not None:
            outcome = {
                "leaked": False, "blocked_input": False, "blocked": False,
                "layer": "error", "blocked_at": f"ERROR - {type(err).__name__}",
                "response_preview": f"Error: {err}"[:300],
            }
        else:
            outcome = classify_attack_outcome(attack["input"], response, target_name=target)
            outcome["response_preview"] = (response or "")[:300]

        row = {
            "id": attack["id"],
            "name": attack.get("category") or f"Attack #{attack['id']}",
            "category": attack["category"],
            "input": attack["input"],
            "response_preview": outcome.get("response_preview", ""),
            "leaked": outcome["leaked"],
            "blocked_input": outcome["blocked_input"],
            "blocked": outcome["blocked"],
            "layer": outcome["layer"],
            "blocked_at": outcome["blocked_at"],
            "error": None,
            "target": target,
        }
        results.append(row)
        leak_tag = "  <<< LEAKED" if outcome["leaked"] else ""
        print(f"    -> {outcome['blocked_at']}{leak_tag}", flush=True)
        await asyncio.sleep(delay)
    return results


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=15.0)
    args = parser.parse_args()

    print("Tao Red (default) ...", flush=True)
    red, red_runner = create_red_agent_default()
    unsafe = await run_list(red, red_runner, "red_default", args.delay)
    write_run_attack_json(unsafe, target_name="red_default")

    print("\nTao Red Advance ...", flush=True)
    adv, adv_runner = create_red_agent_advance()
    guards = await run_list(adv, adv_runner, "red_advance", args.delay)
    write_run_attack_json(guards, target_name="red_advance")

    save_attack_results(unsafe_results=unsafe, guards_results=guards, ai_attacks=None)

    unsafe_leaks = sum(1 for r in unsafe if r.get("leaked"))
    guards_leaks = sum(1 for r in guards if r.get("leaked"))
    print("\n" + "=" * 60, flush=True)
    print(f"Red leaks: {unsafe_leaks}/5 | Red Advance leaks: {guards_leaks}/5", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    asyncio.run(main())