"""
ConversationMemory — Uses LangChain's ChatMessageHistory for conversation tracking.
"""
from langchain_community.chat_message_histories import ChatMessageHistory


class ConversationMemory:
    """Manages conversation history using LangChain's ChatMessageHistory."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history = ChatMessageHistory()
        self._messages = []  # Parallel list for UI compatibility

    def add_message(self, role: str, content: str):
        self._messages.append({'role': role, 'content': content})
        if len(self._messages) > self.max_turns * 2:
            self._messages = self._messages[-(self.max_turns * 2):]

        # Also store in LangChain's message history
        if role == 'user':
            self.history.add_user_message(content)
        else:
            self.history.add_ai_message(content)

    def get_context(self) -> str:
        return "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in self._messages]
        )

    def get_langchain_messages(self) -> list:
        """Return LangChain message objects for use in chains."""
        return self.history.messages

    def clear(self):
        self._messages = []
        self.history.clear()

    def get_messages(self) -> list:
        return self._messages
