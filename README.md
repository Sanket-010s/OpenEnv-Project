---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# Email Triage OpenEnv 📧

A complete, real-world OpenEnv environment adhering to the Meta PyTorch Hackathon constraints.

## The Environment
This RL environment simulates an **Email Triage System**. The agent is tasked with evaluating incoming organizational emails and routing them into the following specific operational buckets:
- **SPAM**: Solicitations or phishing attempts.
- **BILLING**: Issues with suspended accounts or payment declines.
- **COMPLAINT**: Customers expressing extreme dissatisfaction or demanding refunds.
- **SUPPORT**: Major system downtime reports or data breach notifications.
- **URGENT**: Emergency internal organizational tasks.
- **LATER**: Standard communication that does not disrupt current workflows.

### Why this task?
Email overloads are a massive bottleneck in modern organizations. An efficient AI agent securely making priority judgments saves countless operational hours and improves response times for critical anomalies.

## Features
- **OpenEnv Spec Compliance:** Uses strict `pydantic` types for Action, Observation, and State.
- **Graduating Tasks:** Features `easy`, `medium`, and `hard` task suites.
    - Easy: Recognizing obvious spam (5 emails).
    - Medium: Mixed spam with varying difficulty (4 emails).
    - Hard: Nuanced business emails requiring context (5 emails).
- **LLM-Powered Inference:** All LLM calls use the OpenAI Client with configurable endpoint and model.
- **Structured Output Logs:** Emits `[START]`, `[STEP]`, `[END]` format for hackathon validation.
- **Scalable Hosting:** FastAPI-based, Docker-ready for Hugging Face Spaces deployment.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | No | `https://api.openai.com/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `gpt-3.5-turbo` | Model identifier for inference |
| `API_KEY` | No | — | Primary API key (preferred) |
| `OPENAI_API_KEY` | No | — | Fallback API key for OpenAI |
| `HF_TOKEN` | No | — | Hugging Face token (fallback) |
| `ENV_BASE_URL` | No | `http://localhost:7860` | Environment server URL |
| `DEBUG` | No | `0` | Enable debug logging (`1` to enable) |
| `MAX_STEPS` | No | `10` | Maximum steps per episode |
| `EMAILS_PER_TASK` | No | `5` | Number of emails per task |

**Note:** At least one of `API_KEY`, `OPENAI_API_KEY`, or `HF_TOKEN` must be set.
Set `ALLOW_DEMO=1` to run in demo mode without credentials.

## Setup & Testing

### Prerequisites
- Python 3.10+
- Docker (optional, for HF Spaces deployment)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Environment API
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 3. Run the Inference Agent (in another terminal)
```bash
# Set environment variables (choose one API key method)
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-3.5-turbo"
export API_KEY="your_api_key_here"
# OR: export OPENAI_API_KEY="your_key"
# OR: export HF_TOKEN="your_hf_token"

# Run all tasks (easy, medium, hard)
python inference.py

# Run a single task
python inference.py --task easy

# Enable debug logging
export DEBUG=1
python inference.py
```

## Output Format

The inference script emits structured stdout logs:

```
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
```

### Example Output
```
[START] task=easy env=email-triage model=gpt-3.5-turbo
[STEP] step=1 action=SPAM reward=0.90 done=false error=null
[STEP] step=2 action=SPAM reward=0.90 done=false error=null
[STEP] step=3 action=SPAM reward=0.90 done=false error=null
[STEP] step=4 action=SPAM reward=0.90 done=false error=null
[STEP] step=5 action=SPAM reward=0.90 done=true error=null
[END] success=true steps=5 score=0.900 rewards=0.90,0.90,0.90,0.90,0.90
```

## Schema

| Space | Model | Fields |
|-------|-------|--------|
| **Action** | `models.Action` | `category`: str (`SPAM`, `BILLING`, `COMPLAINT`, `SUPPORT`, `URGENT`, `LATER`) |
| **Observation** | `models.Observation` | `subject`, `body`, `sender`, `metadata` |
| **State** | `models.State` | `current_index`, `score`, `task_id`, `completed` |

## Project Structure

```
.
├── inference.py          # Inference script (root directory)
├── client.py             # HTTP client for environment API
├── models.py             # Pydantic models (Action, Observation, State)
├── server/
│   ├── app.py            # FastAPI server
│   └── environment.py    # EmailTriageEnv implementation
├── openenv.yaml          # OpenEnv specification
├── Dockerfile            # Docker build for HF Spaces
└── requirements.txt      # Python dependencies
```

