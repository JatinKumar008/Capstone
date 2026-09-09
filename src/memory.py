from src.config import client, LLM_MODEL


class ConversationMemory:
    """Summary-buffer conversation memory (like LangChain's ConversationSummaryBufferMemory).

    Keeps the recent turns verbatim in a buffer. When the total token estimate
    exceeds max_tokens, the oldest turns are rolled into a summary via the LLM
    and dropped from the buffer.
    """

    def __init__(self, max_tokens: int = 1200, keep_recent: int = 2):
        self.buffer = []
        self.summary = ""
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent

    @staticmethod
    def _approx_tokens(text: str) -> int:
        return len(text) // 4

    def save_context(self, user_input: str, assistant_output: str) -> None:
        self.buffer.append({"input": user_input, "output": assistant_output})
        total = sum(
            self._approx_tokens(m["input"]) + self._approx_tokens(m["output"])
            for m in self.buffer
        )
        if total > self.max_tokens and len(self.buffer) > self.keep_recent:
            self._summarize_oldest()

    def _summarize_oldest(self) -> None:
        oldest = self.buffer[:-self.keep_recent]
        if not oldest:
            return
        text = "\n".join(
            f"User: {m['input']}\nAssistant: {m['output']}" for m in oldest
        )
        prompt = (
            "Summarize the following banking Q&A conversation into a brief recap "
            "(max ~80 words). Keep important products, fees and numbers.\n\n"
            f"{text}"
        )
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        recap = response.choices[0].message.content.strip()
        self.summary = (self.summary + "\n" + recap).strip() if self.summary else recap
        self.buffer = self.buffer[-self.keep_recent:]

    def load_memory(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"Prior summary: {self.summary}")
        parts += [
            f"User: {m['input']}\nAssistant: {m['output']}" for m in self.buffer
        ]
        return "\n\n".join(parts)

    def clear(self) -> None:
        self.buffer = []
        self.summary = ""