# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
"""Extended thinking: read back Claude's reasoning trace alongside its answer.

There's no dedicated Converse parameter for this — it goes through the
escape hatch, additionalModelRequestFields, as {"thinking": {"type":
"enabled", "budget_tokens": N}}. budget_tokens caps the reasoning tokens
Claude may spend before answering, and maxTokens must be set higher than
budget_tokens since the final answer needs room on top of that budget.

Reasoning tokens are billed as output tokens at the standard rate, whether
or not budget_tokens is fully used, so a large budget on every call is a
real cost lever, not a free quality knob.
"""

import sys

import botocore.exceptions

from config import MODEL_ID, REGION, bedrock_client

BUDGET_TOKENS = 1024
MAX_TOKENS = BUDGET_TOKENS + 512  # must exceed budget_tokens


def ask_with_thinking(prompt: str) -> dict:
    client = bedrock_client()
    response = client.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": MAX_TOKENS},
        additionalModelRequestFields={
            "thinking": {"type": "enabled", "budget_tokens": BUDGET_TOKENS}
        },
    )
    reasoning = ""
    answer = ""
    for block in response["output"]["message"]["content"]:
        if "reasoningContent" in block:
            reasoning = block["reasoningContent"]["reasoningText"]["text"]
        elif "text" in block:
            answer = block["text"]
    return {"reasoning": reasoning, "answer": answer, "usage": response["usage"]}


if __name__ == "__main__":
    user_prompt = " ".join(sys.argv[1:]) or (
        "A snail is at the bottom of a 10-meter well. Each day it climbs 3 "
        "meters, and each night it slides back 2 meters. On which day does "
        "it reach the top?"
    )
    print(f"[{MODEL_ID} @ {REGION}]\n")
    try:
        result = ask_with_thinking(user_prompt)
        print("--- reasoning ---")
        print(result["reasoning"] or "(model returned no reasoning block)")
        print("\n--- answer ---")
        print(result["answer"])
        usage = result["usage"]
        print(
            f"\ntokens: {usage['inputTokens']} in, {usage['outputTokens']} out"
            f" ({usage['totalTokens']} total, includes reasoning tokens)",
            file=sys.stderr,
        )
    except botocore.exceptions.ClientError as error:
        code = error.response["Error"]["Code"]
        print(f"{code}: {error.response['Error']['Message']}", file=sys.stderr)
        sys.exit(1)
