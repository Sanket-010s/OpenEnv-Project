import os
import time
from openai import OpenAI
from models import Action
from client import PlayClient

# ── Hackathon required variables ──────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-3.5-turbo")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "dummy_token")

client     = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
env_client = PlayClient("http://localhost:7860")

VALID_LABELS = ["SPAM", "BILLING", "COMPLAINT", "SUPPORT", "LATER", "URGENT"]


def get_action(obs: dict) -> str:
    prompt = f"""You are an expert email triage AI agent.
Categorize the email into exactly one of: SPAM, BILLING, COMPLAINT, SUPPORT, LATER, URGENT.
Reply ONLY with the label name, nothing else.

Subject: {obs.get("subject", "")}
Sender:  {obs.get("sender", "")}
Body:    {obs.get("body", "")}"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        label = response.choices[0].message.content.strip().upper()
        return label if label in VALID_LABELS else "LATER"

    except Exception:
        text = (
            str(obs.get("subject", "")) + " " +
            str(obs.get("body",    "")) + " " +
            str(obs.get("sender",  ""))
        ).upper()

        if any(w in text for w in ["SPAM", "LOTTERY", "VIAGRA", "SCAM", "PRINCE"]):
            return "SPAM"
        if any(w in text for w in ["URGENT", "CEO", "CRITICAL"]):
            return "URGENT"
        if any(w in text for w in ["BILL", "INVOICE", "PAYMENT"]):
            return "BILLING"
        if any(w in text for w in ["COMPLAIN", "UNHAPPY"]):
            return "COMPLAINT"
        if any(w in text for w in ["HELP", "SUPPORT", "ISSUE", "ERROR"]):
            return "SUPPORT"
        return "LATER"


def run_task(task_id: str):
    data = env_client.reset(task_id)
    obs  = data.get("observation", {})
    done = False
    step_num    = 0
    all_rewards = []

    # ── [START] ───────────────────────────────────────────────────
    print(f"[START] task={task_id} env=email-triage model={MODEL_NAME}", flush=True)

    while not done and step_num < 10:
        step_num += 1

        llm_action = get_action(obs)

        step_data = env_client.step(Action(category=llm_action))

        obs    = step_data.get("observation", {})
        reward = float(step_data.get("reward", 0.0))
        done   = step_data.get("done", True)
        error  = step_data.get("info", {}).get("error", None)

        all_rewards.append(reward)

        # ── [STEP] ────────────────────────────────────────────────
        print(
            f"[STEP] step={step_num} "
            f"action={llm_action} "
            f"reward={reward:.2f} "
            f"done={str(done).lower()} "
            f"error={error}",
            flush=True,
        )

    # ── [END] ─────────────────────────────────────────────────────
    success     = any(r >= 1.0 for r in all_rewards)
    rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={step_num} "
        f"rewards={rewards_str}",
        flush=True,
    )


if __name__ == "__main__":
    for t in ["easy", "medium", "hard"]:
        run_task(t)
        time.sleep(1)