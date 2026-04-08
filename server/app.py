from fastapi import FastAPI
from models import Action
from server.environment import EmailTriageEnv

app = FastAPI(title="Email Triage OpenEnv")

env = EmailTriageEnv()

@app.get("/")
def root():
    return {"status": "ok", "name": "email-triage_OpenEnv"}

@app.post("/reset")
def reset(task_id: str = "easy"):
    obs = env.reset(task_id=task_id)
    return {"observation": obs.dict()}

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def get_state():
    return env.state_obj().dict()

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()