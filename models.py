"""
Pydantic models for OpenEnv Email Triage environment.
Defines Action, Observation, and State spaces.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Literal


# ── Action Space ──────────────────────────────────────────────────────────────
# Valid email categories for classification
VALID_CATEGORIES = Literal["SPAM", "BILLING", "COMPLAINT", "SUPPORT", "LATER", "URGENT"]


class Action(BaseModel):
    """
    Action model for email triage.
    The agent selects a category for the current email.
    """
    category: str = Field(
        ...,
        description="Email category: SPAM, BILLING, COMPLAINT, SUPPORT, LATER, or URGENT"
    )

    def model_dump(self) -> Dict[str, Any]:
        return {"category": self.category.upper()}


# ── Observation Space ─────────────────────────────────────────────────────────

class Observation(BaseModel):
    """
    Observation model representing an email to triage.
    """
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body content")
    sender: str = Field(..., description="Sender email address")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (task_id, emails_remaining, etc.)"
    )


# ── State Space ───────────────────────────────────────────────────────────────

class State(BaseModel):
    """
    State model tracking the environment progress.
    """
    current_index: int = Field(..., description="Current email index in dataset")
    score: float = Field(..., description="Cumulative score")
    task_id: str = Field(..., description="Current task identifier (easy/medium/hard)")
    completed: bool = Field(..., description="Whether the episode is complete")
