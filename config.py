"""
Configuration module for OpenEnv Email Triage.
Loads environment variables and initializes API clients.
"""

import os
import sys

# ── Environment Variables ─────────────────────────────────────────────────────
# API_BASE_URL and MODEL_NAME have defaults; API_KEY/HF_TOKEN is mandatory
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-3.5-turbo")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

# API key: prefer API_KEY, fallback to HF_TOKEN or OPENAI_API_KEY
API_KEY  = os.environ.get("API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Resolve API key with priority: API_KEY > OPENAI_API_KEY > HF_TOKEN
if API_KEY:
    USE_OPENAI = True
    USE_HF = False
elif OPENAI_API_KEY:
    API_KEY = OPENAI_API_KEY
    USE_OPENAI = True
    USE_HF = False
elif HF_TOKEN:
    API_KEY = HF_TOKEN
    USE_OPENAI = False
    USE_HF = True
else:
    # Demo mode for testing without credentials
    API_KEY = "demo-mode"
    USE_OPENAI = False
    USE_HF = False

# Validation
if not os.environ.get("ALLOW_DEMO") and API_KEY == "demo-mode":
    print("[Config] WARNING: No API key found. Set API_KEY, OPENAI_API_KEY, or HF_TOKEN")
    print("[Config] Set ALLOW_DEMO=1 to run in demo mode")

# Runtime settings
DEBUG = os.environ.get("DEBUG", "0") == "1"
MAX_STEPS = int(os.environ.get("MAX_STEPS", "10"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "50"))
EMAILS_PER_TASK = int(os.environ.get("EMAILS_PER_TASK", "5"))

# Task configuration
TASKS = ["easy", "medium", "hard"]
