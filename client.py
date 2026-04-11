import httpx
from models import Action

class PlayClient:
    """HTTP client for interacting with the OpenEnv environment server."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url
        self._client = httpx.Client(base_url=base_url, timeout=30)

    def reset(self, task_id: str = "easy"):
        """Reset the environment for a given task."""
        res = self._client.post("/reset", params={"task_id": task_id})
        res.raise_for_status()
        return res.json()

    def step(self, action: Action):
        """Take a step in the environment."""
        res = self._client.post("/step", json=action.model_dump())
        res.raise_for_status()
        return res.json()

    def state(self):
        """Get the current environment state."""
        res = self._client.get("/state")
        res.raise_for_status()
        return res.json()

    def health(self):
        """Check environment health."""
        res = self._client.get("/health")
        res.raise_for_status()
        return res.json()

    def close(self):
        """Close the HTTP client."""
        self._client.close()
