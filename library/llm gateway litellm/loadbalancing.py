import asyncio
import litellm
from litellm import Router


model_pool = [
    {
        "model_name": "gpt-4o-mini",
        "litellm_params":{
            "model" : "openai/gpt-4o-mini",
            "api_key" : "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",
            "api_base": "https://openrouter.ai/api/v1"
        },
        "tpm" : 10000,
        "rpm" : 100,
    },
    {
        "model_name": "gpt-4o-mini",
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "api_key": "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",
            "api_base": "https://openrouter.ai/api/v1"
        },
        "tpm": 10000,
        "rpm": 100,
    },
]
router_roundrobin = Router(
    model_list=model_pool,
    routing_strategy="simple-shuffle"
)

router_latency = Router(
    model_list= model_pool,
    routing_strategy="latency-based-routing",
    enable_pre_call_checks= True, # skip keys that are known to be rate limited
)


router_usage = Router(
    model_list=model_pool,
    routing_strategy="usage-based-routing"
)

async def loadbalancing(prompt:str , strategy:str = "latency") ->dict:
    router = {
        "latency" : router_latency,
        "round_robin" : router_roundrobin,
        "usage" : router_usage,
    }.get(strategy , router_latency)

    response = await router.acompletion(
        model="gpt-4o-mini",
        messages= [{"role":"user" , "content":prompt}],
        max_tokens=1024,
    )
    return {
        "content": response.choices[0].message.content,
        "deployment": response._hidden_params.get("api_key", "unknown")[:8] + "...",
    }
prompt = "capital of paris"
a = asyncio.run(loadbalancing(prompt , strategy= "latency"))
print(a)