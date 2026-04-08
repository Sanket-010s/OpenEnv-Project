from pydantic import BaseModel
from typing import Dict, Any

class Action(BaseModel):
    category: str

class Observation(BaseModel):
    subject: str
    body: str
    sender: str
    metadata: Dict[str, Any]

class State(BaseModel):
    current_index: int
    score: float
    task_id: str
    completed: bool
