class ConversationMemory:
    """Manages conversation history for context-aware interactions."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages = []

    def add_message(self, role: str, content: str):
        self.messages.append({'role': role, 'content': content})
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-(self.max_turns * 2):]

    def get_context(self) -> str:
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.messages])

    def clear(self):
        self.messages = []

    def get_messages(self) -> list:
        return self.messages
