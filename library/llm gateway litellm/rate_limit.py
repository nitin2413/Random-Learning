'''
asyncio.Lock inside TokenBucket is not optional — without it, two concurrent requests can both read tokens=1,
both pass the check, both subtract 1, and you end up at -1. The lock makes the read-check-subtract atomic.

inbound_limiter rejects immediately with a 429 — the user gets a fast error and can retry later.
outbound_limiter uses wait_for_slot which blocks instead of rejecting — because your own code retrying is fine, but returning a 429 from a provider to your user looks bad.

retry_after_seconds in the 429 response is important — good clients use this to know exactly how long to wait before retrying. Don't make them guess.

The X-RateLimit-Remaining header on every response (like GitHub does) lets API consumers build smarter clients that slow themselves down before hitting the limit.
'''



import time
import asyncio
from dataclasses import dataclass, field
from collections import defaultdict
from fastapi import HTTPException, Request
from functools import wraps

@dataclass
class TokenBucket :
    """
       A bucket that holds up to `capacity` tokens.
       Tokens refill at `refill_rate` per second.
       Each request consumes 1 token.
       If the bucket is empty → request is rejected (429).
    """
    capacity : int
    refill_rate : int
    _tokens : float = field(init=False)
    _last_refill : int = field(init=False)
    _lock : asyncio.Lock = field(init=False)

    def __post_init__(self):
        self._tokens : float(self.capacity)
        self._last_refill : time.monotonic()
        self._lock : asyncio.Lock()

    async def consume(self, n: int = 1)-> bool:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self.tokens = min(self.capacity , self._tokens + elapsed * self.refill_rate)
            self._last_refill = now
            if self._tokens >= n:
                self._tokens -= 1
                return True
            return False

@property
def tokens_remaining(self) -> float :
    return round(self._tokens, 2)

class InboundRateLimiter :
    """
    Tracks one TokenBucket per API key.
    Different tiers get different limits.
    """

    TIER_CONFIG = {
        "free":       {"capacity": 5,   "refill_rate": 0.083},  # 5 req burst, 5/min
        "pro":        {"capacity": 20,  "refill_rate": 0.5},    # 20 req burst, 30/min
        "enterprise": {"capacity": 100, "refill_rate": 10.0},   # 100 req burst, 600/min
    }

    def __init__(self):
        self._bucket : dict[str , TokenBucket] = {}

    def get_bucket(self, api_key : str , tier : str = "free") -> TokenBucket:
        if api_key not in self._bucket:
            config = self.TIER_CONFIG.get(tier , self.TIER_CONFIG["free"])
            # The ** operator unpacks the dictionary into keyword arguments,
            self._bucket[api_key] = TokenBucket(**config)
        return self._bucket[api_key]

    async def check(self , api_key : str , tier : str = "free"):
        bucket = self._bucket(api_key , tier)
        allowed = await bucket.consume()
        if not allowed:
            remaining = bucket.tokens_remaining
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "tier": tier,
                    "tokens_remaining": remaining,
                    "retry_after_seconds": round(1 / bucket.refill_rate, 1),
                }
            )

        print(f"[RateLimit] {api_key[:8]}... allowed | tokens left: {bucket.tokens_remaining}")
# Singleton — one limiter for the whole app
inbound_limiter = InboundRateLimiter()

class OutboundRateLimiter :
    """
    Slows down YOUR calls to the provider so you don't
    get 429s from them. Sits between your code and LiteLLM.
    """

    PROVIDER_LIMITS = {
        "openai": {"capacity": 3, "refill_rate": 0.05},  # free: 3 RPM
        "anthropic": {"capacity": 5, "refill_rate": 0.083},  # free: 5 RPM
        "gemini": {"capacity": 15, "refill_rate": 0.25},  # free: 15 RPM
    }

    def __init__(self):
        self._buckets : dict [str , TokenBucket] = {}

    def _get_bucket(self, provider : str )-> TokenBucket:
        if  provider not in self._buckets:
            config = self.PROVIDER_LIMITS.get(provider , {"capacity": 5, "refill_rate": 0.083})
            self._buckets[provider] = TokenBucket(**config)
        return self._buckets[provider]

    async def wait_for_slot(self, provider : str ):
        bucket = self._get_bucket[provider]
        while True:
            allowed = await bucket.consume()
            if allowed:
                break
            wait = round(1 / bucket.refill_rate , 2) # round till 2 decimal point
            print(f"[OutboundRL] {provider} quota full, waiting {wait}s...")
            await asyncio.sleep(wait)

        outbound_limiter = OutboundRateLimiter()

