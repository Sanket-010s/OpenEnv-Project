"""
Baseline inference script for the OpenEnv Email Triage environment.
Runs an LLM agent against all three tasks using the OpenAI client.

Reads credentials from environment variables:
    API_BASE_URL  — LLM API endpoint (provided by validator)
    API_KEY       — API key (provided by validator, fallback to OPENAI_API_KEY)
    MODEL_NAME    — Model identifier
    HF_TOKEN      — Hugging Face token (fallback if no API_KEY)
    ENV_BASE_URL  — Base URL of the running OpenEnv server (default: http://localhost:7860)

Usage:
    python inference.py
    python inference.py --task easy
    python inference.py --task all
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
from typing import Any, Dict, List, Optional

from openai import OpenAI
import httpx

from config import (
    API_BASE_URL, MODEL_NAME, ENV_BASE_URL,
    API_KEY, HF_TOKEN, USE_OPENAI, USE_HF,
    DEBUG, MAX_STEPS, TEMPERATURE, MAX_TOKENS, EMAILS_PER_TASK, TASKS
)
from models import Action

# ── Client Initialization ─────────────────────────────────────────────────────

DEMO_MODE = API_KEY == "demo-mode"

if DEMO_MODE:
    client = None
    print(f"[Inference] Using DEMO MODE with model: {MODEL_NAME}")
    print("[Inference] This will simulate realistic agent responses for demonstration")
elif USE_OPENAI:
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    print(f"[Inference] Using OpenAI client with model: {MODEL_NAME}")
    print(f"[Inference] API_BASE_URL: {API_BASE_URL}")
elif USE_HF:
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=HF_TOKEN)
    print(f"[Inference] Using Hugging Face client with model: {MODEL_NAME}")
else:
    print("[Inference] ERROR: No API key found. Set API_KEY or HF_TOKEN")
    sys.exit(1)

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert email triage agent.
Categorize each email into exactly ONE of these categories:
- SPAM: Unsolicited bulk email, phishing, scams, fake promotions
- BILLING: Payment issues, invoices, subscription problems
- COMPLAINT: Customer expressing dissatisfaction, demanding refunds
- SUPPORT: Technical help requests, account issues, how-to questions
- URGENT: Critical business matters needing immediate attention
- LATER: Normal communication that can wait

Reply ONLY with the category name, nothing else.
Example: SPAM"""


def build_user_prompt(observation: dict) -> str:
    """Build the user prompt from observation."""
    return f"""Subject: {observation.get('subject', '')}
Sender: {observation.get('sender', '')}
Body: {observation.get('body', '')}

What category is this email? Reply with ONE word only."""


def parse_action(response_text: str) -> Optional[str]:
    """Parse the LLM response to extract the category."""
    if not response_text:
        return None

    text = response_text.strip().upper()

    # Remove markdown and extra text
    text = text.replace("```", "").strip()

    # Try to extract category from response
    valid_categories = ["SPAM", "BILLING", "COMPLAINT", "SUPPORT", "LATER", "URGENT"]

    # Check if response is exactly a valid category
    if text in valid_categories:
        return text

    # Try to find a category word in the response
    for category in valid_categories:
        if category in text:
            return category

    return None


# ── Logging Helpers ───────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line for validator."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    """Emit [STEP] line for validator."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

    if DEBUG:
        print(f"  [DEBUG] Step {step}: action={action}, reward={reward:.2f}, done={done}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float], task: str = "") -> None:
    """Emit [END] line for validator."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

    if DEBUG:
        avg = sum(rewards) / len(rewards) if rewards else 0.0
        print(f"  [DEBUG] Task={task}, success={success}, avg_reward={avg:.3f}", flush=True)


# ── HTTP Client Helpers ───────────────────────────────────────────────────────

def api_reset(http: httpx.Client, task_id: str) -> dict:
    """Call environment reset endpoint."""
    resp = http.post(f"{ENV_BASE_URL}/reset", params={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_step(http: httpx.Client, action: dict) -> dict:
    """Call environment step endpoint."""
    resp = http.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Task Runner ───────────────────────────────────────────────────────────────

def run_task(
    client: Optional[OpenAI],
    http: httpx.Client,
    task_id: str,
    emails_count: int = EMAILS_PER_TASK,
) -> Dict[str, Any]:
    """Run the agent on emails for a given task. Returns aggregate scores."""

    print(f"\n[Inference] Starting task={task_id}", flush=True)

    all_rewards: List[float] = []
    success_count = 0
    total_steps = 0

    # Reset environment for task
    try:
        reset_result = api_reset(http, task_id)
        observation = reset_result.get("observation", {})
        done = reset_result.get("done", False)
    except Exception as exc:
        print(f"[Inference] ERROR: reset() failed: {exc}")
        return {
            "task_id": task_id,
            "emails_evaluated": 0,
            "avg_score": 0.0,
            "success_rate": 0.0,
        }

    rewards: List[float] = []
    steps_taken = 0

    log_start(task=task_id, env="email-triage", model=MODEL_NAME)

    # Run episode
    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        # Build prompt and call LLM
        user_prompt = build_user_prompt(observation)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            if DEMO_MODE:
                # Smart demo responses based on email content
                subject = observation.get("subject", "").upper()
                body = observation.get("body", "").upper()
                text = subject + " " + body

                if any(w in text for w in ["SPAM", "LOTTERY", "VIAGRA", "SCAM", "PRINCE", "WINNER", "CONGRATULATIONS"]):
                    llm_action = "SPAM"
                elif any(w in text for w in ["URGENT", "CEO", "CRITICAL", "ASAP", "NOW"]):
                    llm_action = "URGENT"
                elif any(w in text for w in ["BILL", "INVOICE", "PAYMENT", "DECLINED"]):
                    llm_action = "BILLING"
                elif any(w in text for w in ["COMPLAIN", "UNHAPPY", "TERRIBLE", "REFUND"]):
                    llm_action = "COMPLAINT"
                elif any(w in text for w in ["HELP", "SUPPORT", "ISSUE", "ERROR", "DOWN"]):
                    llm_action = "SUPPORT"
                else:
                    llm_action = "LATER"
            elif client:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
                llm_action = parse_action(response_text)
                if llm_action is None:
                    llm_action = "LATER"  # fallback
            else:
                llm_action = "LATER"
        except Exception as exc:
            print(f"[Inference] LLM request failed at step {step}: {exc}")
            llm_action = "LATER"

        if DEBUG:
            print(f"[DEBUG] LLM action: {llm_action}", flush=True)

        # Step environment
        try:
            step_result = api_step(http, {"category": llm_action})
        except Exception as exc:
            print(f"[Inference] ERROR: step() failed: {exc}")
            break

        observation = step_result.get("observation", {})
        reward = float(step_result.get("reward", 0.0))
        done = step_result.get("done", False)
        error = step_result.get("info", {}).get("error", None)

        rewards.append(reward)
        steps_taken = step

        log_step(step=step, action=llm_action, reward=reward, done=done, error=error)

        if done:
            break

    # Calculate final score
    score = sum(rewards) / len(rewards) if rewards else 0.0
    score = max(0.001, min(0.999, score))  # clamp to (0, 1) exclusive
    success = score >= 0.5

    all_rewards = rewards
    total_steps = steps_taken

    log_end(success=success, steps=total_steps, score=score, rewards=all_rewards, task=task_id)

    print(f"[Inference] Task {task_id} complete — score={score:.3f}", flush=True)

    return {
        "task_id": task_id,
        "avg_score": score,
        "success": success,
        "steps": total_steps,
        "rewards": all_rewards,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="OpenEnv Email Triage baseline inference")
    parser.add_argument("--task", default="all", help="Task ID or 'all'")
    parser.add_argument("--env-url", default=ENV_BASE_URL, help="Environment server URL")
    args = parser.parse_args()

    env_base_url = args.env_url

    print(f"[Inference] API Provider: {'OpenAI' if USE_OPENAI else 'HuggingFace' if USE_HF else 'Demo'}")
    print(f"[Inference] API_BASE_URL={API_BASE_URL}")
    print(f"[Inference] MODEL_NAME={MODEL_NAME}")
    print(f"[Inference] ENV_BASE_URL={env_base_url}")

    # Check environment health
    with httpx.Client() as http:
        try:
            health = http.get(f"{env_base_url}/health", timeout=10)
            health.raise_for_status()
            print(f"[Inference] Environment health: {health.json()}")
        except Exception as exc:
            print(f"[Inference] WARNING: Could not reach environment at {env_base_url}: {exc}")
            print("[Inference] Make sure the server is running: python -m server.app")
            sys.exit(1)

        # Determine tasks to run
        tasks_to_run = TASKS if args.task == "all" else [args.task]

        all_results: List[Dict] = []
        start_time = time.time()

        for task_id in tasks_to_run:
            result = run_task(client, http, task_id)
            all_results.append(result)

        elapsed = time.time() - start_time

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  INFERENCE SUMMARY")
    print("=" * 60)
    print(f"  Model:   {MODEL_NAME}")
    print(f"  Runtime: {elapsed:.1f}s")
    print()

    for r in all_results:
        print(f"  Task: {r['task_id']}")
        print(f"    Score:  {r['avg_score']:.3f}")
        print(f"    Success: {r['success']}")
        print(f"    Steps:   {r['steps']}")
        print()

    overall_avg = sum(r["avg_score"] for r in all_results) / len(all_results) if all_results else 0.0
    print(f"  Overall average score: {overall_avg:.3f}")
    print("=" * 60)

    # Save results to JSON
    results_path = "inference_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "api_base_url": API_BASE_URL,
                "runtime_seconds": elapsed,
                "overall_avg_score": overall_avg,
                "task_results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\n[Inference] Results saved to {results_path}")


if __name__ == "__main__":
    main()
