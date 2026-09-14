import os
import time as _time
from functools import wraps
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

LLM_PROVIDER = "openrouter"
LLM_MODEL    = "openai/gpt-oss-120b"

if LLM_PROVIDER == "groq":
    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    print(f"✅ Groq client initialized | model: {LLM_MODEL}")

elif LLM_PROVIDER == "openrouter":
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "https://github.com/your-repo"},
    )
    print(f"✅ OpenRouter client initialized | model: {LLM_MODEL}")

else:
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

_orig_create = client.chat.completions.create

@wraps(_orig_create)
def _create_with_retry(*args, **kwargs):
    max_retries = 8
    backoff = 1.5
    for attempt in range(max_retries):
        try:
            return _orig_create(*args, **kwargs)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = backoff * (attempt + 1)
            print(f"⚠️  Rate limit hit, retrying in {wait:.1f}s (attempt {attempt+2}/{max_retries})")
            _time.sleep(wait)

client.chat.completions.create = _create_with_retry
print("✅ Rate-limit retry wrapper enabled")
