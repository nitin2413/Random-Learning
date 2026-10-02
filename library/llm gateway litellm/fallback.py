import asyncio

from api_caliing import get_api_key
import litellm


litellm.verbose_logger = False # set True to see retry logs

fallback_chain = [
    "openai/gpt-4o",
    "anthropic/claude-3-5-sonnet-20241022",   # fallback 1 — different provider
    "openai/gpt-4o-mini",                     # fallback 2 — cheaper, very reliable
    "gemini/gemini-1.5-flash",
]

async def completion_with_feedback(
        prompt : str ,
        fallback : list[str] = fallback_chain,
        max_tokens = 1024
    )-> dict:
    """
        Tries each model in the chain until one succeeds.
        Returns both the response and which model actually answered.
    """
    primary = fallback[0]
    fallbacks = fallback[1:]

    response = await litellm.acompletion(
        model = primary,
        messages= [{"role":"user" , "content":prompt}],
        fallbacks=fallbacks,
        base_url = "https://openrouter.ai/api/v1",
        # Which HTTP errors should trigger a fallback (not a crash)
        fallback_on_status_codes=[429, 500, 502, 503, 504],
    )

    model_used = response.model
    print(f"[Fallback] Responded by: {model_used}")
    return {
        "content": response.choices[0].message.content,
        "model_used": model_used,
        "primary_was_used": model_used == primary,
    }

prompt = "what in capital of norway"
a = asyncio.run(completion_with_feedback(prompt,fallback_chain))
print(a)