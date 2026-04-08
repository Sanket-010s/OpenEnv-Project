from models import Action, Observation, State

EMAILS_DATABASE = {
    "easy": [
        {
        "subject": "Congratulations! You won $1,000,000!!!",
        "body": "Click this link NOW to claim your prize. Send your bank details immediately!!!",
        "sender": "noreply@totallylegit.xyz",
        "correct_priority": "low",
        "correct_category": "spam",
        "difficulty": "easy",
    },
    {
        "subject": "FREE GIFT CARDS - Limited offer inside!!!",
        "body": "You have been selected for a special promotion. Get $500 gift cards. Act NOW before offer expires!!!",
        "sender": "offers@promo-deals.net",
        "correct_priority": "low",
        "correct_category": "spam",
        "difficulty": "easy",
    },
    {
        "subject": "You are a WINNER! Claim your Tesla now",
        "body": "Dear lucky winner, click here and enter your credit card to verify your identity and claim prize.",
        "sender": "winner@scam-alert.ru",
        "correct_priority": "low",
        "correct_category": "spam",
        "difficulty": "easy",
    },
    {
        "subject": "Make $5000 a day working from home!!!",
        "body": "No experience needed. Join thousands already making money. Limited spots. Sign up now!",
        "sender": "jobs@workfromhome-scam.com",
        "correct_priority": "low",
        "correct_category": "spam",
        "difficulty": "easy",
    },
    {
        "subject": "URGENT: Your account needs verification!!!",
        "body": "Click the link below and enter your password to avoid account suspension. Act immediately!",
        "sender": "security@fake-bank.xyz",
        "correct_priority": "low",
        "correct_category": "spam",
        "difficulty": "easy",
    },
    ],
    "medium": [
        {
        "subject": "Congratulations! You won $1,000,000!!!",
        "body": "Click this link to claim your prize. Send your bank details now!!!",
        "sender": "noreply@totallylegit.xyz",
        "correct_priority": "low",
        "correct_category": "spam",
        "difficulty": "easy",
    },
    {
        "subject": "FREE GIFT CARDS - Limited offer inside",
        "body": "You have been selected for a special promotion. Get free gift cards worth $500. Act now!!!",
        "sender": "offers@promo-deals.net",
        "correct_priority": "low",
        "correct_category": "spam",
        "difficulty": "easy",
    },
    {
        "subject": "You are a WINNER! Claim your Tesla now",
        "body": "Dear lucky winner, you have been randomly selected. Click here and enter your credit card to verify identity.",
        "sender": "winner@scam-alert.ru",
        "correct_priority": "low",
        "correct_category": "spam",
        "difficulty": "easy",
    },
    {
        "subject": "Make $5000 a day working from home!!!",
        "body": "No experience needed. Join thousands of people already making money. Limited spots available. Sign up now!",
        "sender": "jobs@workfromhome-scam.com",
        "correct_priority": "low",
        "correct_category": "spam",
        "difficulty": "easy",
    },
    ],
    "hard": [
        {
        "subject": "URGENT: Payment declined - account suspended",
        "body": "My payment was declined and I cannot access my account. I have a meeting in 2 hours. Fix this NOW!",
        "sender": "angry.customer@gmail.com",
        "correct_priority": "urgent",
        "correct_category": "billing",
        "difficulty": "hard",
    },
    {
        "subject": "My order arrived damaged - need replacement ASAP",
        "body": "I ordered a birthday gift for my daughter and it arrived completely broken. Her birthday is tomorrow!",
        "sender": "upset.mom@yahoo.com",
        "correct_priority": "urgent",
        "correct_category": "complaint",
        "difficulty": "hard",
    },
    {
        "subject": "System is completely down - losing money every minute",
        "body": "Our entire payment system is down. We are losing thousands per minute. Need emergency support NOW.",
        "sender": "cto@bigcorporate.com",
        "correct_priority": "urgent",
        "correct_category": "support",
        "difficulty": "hard",
    },
    {
        "subject": "Terrible service - I want to speak to a manager",
        "body": "I have been waiting 3 weeks for my refund. Every time I call I get transferred. Filing a complaint.",
        "sender": "frustrated.user@hotmail.com",
        "correct_priority": "urgent",
        "correct_category": "complaint",
        "difficulty": "hard",
    },
    {
        "subject": "Data breach - my account was hacked",
        "body": "Someone accessed my account from another country. I see transactions I did not make. Help immediately!",
        "sender": "victim@gmail.com",
        "correct_priority": "urgent",
        "correct_category": "support",
        "difficulty": "hard",
    },
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
        
        # Safely handle your new dataset properties!
        correct_label = current_email.get("label", current_email.get("correct_category", "LATER")).upper()
        
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
