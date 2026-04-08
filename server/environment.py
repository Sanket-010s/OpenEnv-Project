from models import Action, Observation, State

EMAILS_DATABASE = {
    "easy": [
        {"subject": "You won a lottery!", "body": "Click here to claim your $1000000.", "sender": "scam@lottery.com", "label": "SPAM"},
        {"subject": "Free Viagra", "body": "Cheap pills here.", "sender": "rx@spam.com", "label": "SPAM"},
        {"subject": "Prince of Nigeria", "body": "I need your help transferring funds.", "sender": "prince@nigeria.ng", "label": "SPAM"},
    ],
    "medium": [
        {"subject": "Urgent: Server is down", "body": "Production server API-1 is unreachable.", "sender": "alerts@company.com", "label": "URGENT"},
        {"subject": "Hey", "body": "Want to get lunch later?", "sender": "bob@company.com", "label": "LATER"},
        {"subject": "Urgent: Invoice overdue", "body": "Please pay immediately to avoid service disruption.", "sender": "billing@vendor.com", "label": "URGENT"},
    ],
    "hard": [
        {"subject": "FW: Project update", "body": "Here is the weekly update.", "sender": "alice@company.com", "label": "LATER"},
        {"subject": "Critical Security Vulnerability", "body": "CVE-2024-XXXX requires immediate patching.", "sender": "security@company.com", "label": "URGENT"},
        {"subject": "Limited Time Offer!!!", "body": "Buy one get one free", "sender": "marketing@fake.com", "label": "SPAM"},
        {"subject": "Can we reschedule?", "body": "Let's move our meeting to Thursday.", "sender": "charlie@company.com", "label": "LATER"},
        {"subject": "CEO Request", "body": "I need the Q3 report by EOD.", "sender": "ceo@company.com", "label": "URGENT"},
    ]
}

class EmailTriageEnv:
    def __init__(self):
        self.state = State(current_index=0, score=0.0, task_id="easy", completed=False)
        self.dataset = []

    def reset(self, task_id: str = "easy"):
        if task_id not in EMAILS_DATABASE:
            task_id = "easy"
        self.dataset = EMAILS_DATABASE[task_id]
        self.state = State(current_index=0, score=0.0, task_id=task_id, completed=False)
        return self._get_observation()

    def _get_observation(self) -> Observation:
        if self.state.completed or self.state.current_index >= len(self.dataset):
            return Observation(subject="", body="", sender="", metadata={"info": "No more emails", "score": self.state.score})
        email = self.dataset[self.state.current_index]
        return Observation(
            subject=email["subject"],
            body=email["body"],
            sender=email["sender"],
            metadata={
                "task_id": self.state.task_id,
                "emails_remaining": len(self.dataset) - self.state.current_index,
            }
        )

    def step(self, action: Action):
        if self.state.completed:
            return self._get_observation(), 0.0, True, {"info": "Episode completed"}

        current_email = self.dataset[self.state.current_index]
        correct_label = current_email["label"]
        
        is_correct = action.category.upper() == correct_label
        reward = 1.0 if is_correct else 0.0
        
        self.state.score += reward
        self.state.current_index += 1
        done = self.state.current_index >= len(self.dataset)
        
        if done:
            self.state.completed = True
        
        task_score = self.state.score / len(self.dataset) if done else 0.0
        info = {"is_correct": is_correct, "task_score": task_score}
        
        return self._get_observation(), reward, done, info
        
    def state_obj(self):
        return self.state
