<<<<<<< HEAD
# Email Triage OpenEnv 📧

A complete, real-world OpenEnv environment adhering to the Meta PyTorch Hackathon constraints.

##  The Environment
This RL environment simulates an **Email Triage System**. The agent is tasked with evaluating incoming organizational emails and routing them into the following specific operational buckets:
- **SPAM**: Solicitations or phishing attempts.
- **BILLING**: Issues with suspended accounts or payment declines.
- **COMPLAINT**: Customers expressing extreme dissatisfaction or demanding refunds.
- **SUPPORT**: Major system downtime reports or data breach notifications.
- **URGENT**: Emergency internal organizational tasks.
- **LATER**: Standard communication that does not disrupt current workflows.

###  Why this task?
Email overloads are a massive bottleneck in modern organizations. An efficient AI agent securely making priority judgments saves countless operational hours and improves response times for critical anomalies.

## Features
- **OpenEnv Spec Compliance:** Uses strict `pydantic` types for Action, Observation, and State.
- **Graduating Tasks:** Features `easy`, `medium`, and `hard` task suites.
    - Easy: Recognizing obvious spam.
    - Medium: Catching standard urgent business operations.
    - Hard: Mixed bag with nuanced professional tone parsing.
- **Built-in Inference Script:** Ships with standard `[START]`, `[STEP]`, `[END]` evaluation logs, compatible with any LLM via the `openai` python package.
- **Scalable Hosting:** FastAPI-based and encapsulated tightly within a Docker container for prompt Hugging Face Spaces deployments.

## Setup & Testing Local Version

### Prerequisites:
- Python 3.10+
- Docker (Optional, for building HF space container)

### 1) Install the dependencies
```bash
pip install -r requirements.txt
```

### 2) Run the Environment API
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 3) Run the Baseline Inference Agent (in another terminal)
Ensure your environment variables are set before execution to authorize the AI agent:
```bash
export API_BASE_URL="https://api.openai.com/v1" 
export MODEL_NAME="gpt-3.5-turbo"
export HF_TOKEN="your_token_here"
python inference.py
```

## 🏗 Schema
- **Action Space (`models.Action`):** String Literal `category` (`SPAM`, `BILLING`, `COMPLAINT`, `SUPPORT`, `URGENT`, `LATER`)
- **Observation Space (`models.Observation`):** Evaluates `subject`, `body`, and `sender` attributes representing the current message sequence.
- **State Space (`models.State`):** Internal tracker containing the `current_index`, `task_id`, and scalar `score`.
=======
---
title: Email-triage OpenEnv
emoji: 📊
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> fe1c5eca4f8a7362c56bab6ee74fd93c9d1b0972
