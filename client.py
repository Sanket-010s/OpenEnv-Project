import requests
from models import Action

class PlayClient:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url

    def reset(self, task_id: str = "easy"):
        res = requests.post(f"{self.base_url}/reset", params={"task_id": task_id})
        res.raise_for_status()
        return res.json()

    def step(self, action: Action):
        res = requests.post(f"{self.base_url}/step", json=action.dict())
        res.raise_for_status()
        return res.json()
        
    def state(self):
        res = requests.get(f"{self.base_url}/state")
        res.raise_for_status()
        return res.json()
