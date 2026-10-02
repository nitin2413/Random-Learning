from fundamental import routed_completion
import asyncio

async def main():
    prompts = [
        "Write a Python function to reverse a linked list",
        "Hey, how are you doing today?",
        "Summarize generate 3 bullet dog",
        "Compare transformer vs LSTM architectures",
    ]

    for prompt in prompts:
        result = await routed_completion(prompt)
        print(f"\nPrompt : {prompt[:50]}")
        print(f"Answer : {result[:100]}...")

asyncio.run(main())