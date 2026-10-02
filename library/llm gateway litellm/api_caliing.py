import os

os.environ["OPENAI_API_KEY"] = "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"
os.environ["GEMINI_API_KEY"] = "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"
os.environ["GROQ_API_KEY"] = "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"
"""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"""
base_url = "https://openrouter.ai/api/v1"

PROVIDER_KEYS = {
    "open_ai" : os.getenv("OPENAI_API_KEY"),
    "google": os.getenv("GEMINI_API_KEY"),
    "xai": os.getenv("GROQ_API_KEY")
}

def get_api_key(model : str ) -> str :
    provider = model.split("/")[0]
    return PROVIDER_KEYS.get(provider)