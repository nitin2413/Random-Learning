import litellm
from enum import Enum
from api_caliing import get_api_key


class TaskType(Enum):
    CODE = "code"
    CHAT = "chat"
    SUMMARY = "summary"
    ANALYSIS = "analysis"

ROUTING_TABLE = {
    TaskType.CODE : "openai/gpt-4o-mini",
    TaskType.SUMMARY : "openai/gpt-3.5-turbo",
    TaskType.CHAT : "openai/gpt-4o-mini",
    TaskType.ANALYSIS : "openai/gpt-3.5-turbo"
}

def detect_task(prompt : str) -> TaskType:
    prompt = prompt.lower()
    if any(kw in prompt for kw in ["code", "function", "bug", "debug", "python", "implement"]):
        return TaskType.CODE
    if any(kw in prompt for kw in ["summarize", "tldr", "brief", "shorten"]):
        return TaskType.SUMMARY
    if any(kw in prompt for kw in ["analyze", "compare", "evaluate", "explain why"]):
        return TaskType.ANALYSIS
    return TaskType.CHAT

def route(prompt : str ) -> str :
    task = detect_task(prompt)
    model = ROUTING_TABLE[task]
    print(f"[Router] Task detected: {task.value} → model: {model}")
    return model

async def routed_completion(prompt : str ) -> str :
    model = route(prompt)
    response = await litellm.acompletion(
        model = model ,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens = 1024,
        api_key= get_api_key(model),
        base_url="https://openrouter.ai/api/v1"
    )
    return response.choices[0].message.content

