import os
import time
from openai import OpenAI
from models import Action
from client import PlayClient

# Hackathon required variables
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.environ.get("HF_TOKEN", "dummy_token")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

env_client = PlayClient("http://localhost:8000")

def run_task(task_id: str):
    print(f"[START] Task: {task_id}")
    data = env_client.reset(task_id)
    obs = data.get("observation", {})
    done = False
    step_num = 0
    total_reward = 0.0
    final_score = 0.0
    
    while not done:
        step_num += 1
        prompt = f"""
You are an expert email triage AI agent.
Please categorize the following email into exactly one of these labels: URGENT, LATER, SPAM.
Reply ONLY with the label name.

Subject: {obs.get("subject")}
Sender: {obs.get("sender")}
Body: {obs.get("body")}
"""
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0
            )
            llm_action = response.choices[0].message.content.strip().upper()
            if llm_action not in ["URGENT", "LATER", "SPAM"]:
                llm_action = "LATER" # default fallback
        except Exception as e:
            # gracefully degrade to a safe guess if api call fails
            llm_action = "LATER"
            
        step_data = env_client.step(Action(category=llm_action))
        
        obs = step_data.get("observation", {})
        reward = step_data.get("reward", 0.0)
        done = step_data.get("done", True)
        info = step_data.get("info", {})
        
        total_reward += float(reward)
        if done:
            final_score = info.get("task_score", 0.0)
            
        # Ensure correct stdout requirement from hackathon rule
        print(f"[STEP] StepNum: {step_num} | Action: {llm_action} | Reward: {reward} | Done: {done}")
        
    print(f"[END] Task: {task_id} | Total Reward: {total_reward} | Final Score: {final_score}")

if __name__ == "__main__":
    for t in ["easy", "medium", "hard"]:
        run_task(t)
        time.sleep(1)
