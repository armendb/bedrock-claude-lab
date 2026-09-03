# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
"""Prompt caching: cache a static context once, reuse it across questions.

Bedrock charges the standard input rate on a cache write and a reduced rate
on a cache read, and a cache hit responds faster since the model skips
recomputing that prefix. A cachePoint marks where the cacheable prefix ends;
each model has a minimum token count before a checkpoint actually takes —
Claude Haiku 4.5 needs at least 4,096 tokens per checkpoint, smaller than
that and the call still succeeds but nothing gets cached.

When caching is active, `usage.inputTokens` only counts the NON-cached
tokens in that request — cacheReadInputTokens and cacheWriteInputTokens are
reported separately and must be added in for the true input token count.
"""

import sys
import time

import botocore.exceptions

from config import MODEL_ID, REGION, bedrock_client

# A static "house style guide" the model references on every question.
# Repeated to clear Claude Haiku 4.5's 4,096-token-per-checkpoint minimum —
# a shorter block would still work, it just wouldn't get cached.
_STYLE_RULE = (
    "Rule: prefer plain language over jargon. When a technical term is "
    "unavoidable, define it on first use in the same sentence, not in a "
    "footnote or a separate glossary entry. Numbers under ten are spelled "
    "out in prose; numbers ten and above use digits, except at the start "
    "of a sentence, which is always spelled out regardless of size. "
    "Headings are sentence case, not title case. Oxford commas are "
    "required in every list of three or more items. Passive voice is "
    "permitted only when the actor is unknown or irrelevant to the point "
    "being made.\n\n"
)
STYLE_GUIDE = _STYLE_RULE * 45  # ~5,400 tokens, comfortably over the 4,096 minimum

SYSTEM_WITH_CACHE_POINT = [
    {"text": STYLE_GUIDE},
    {"cachePoint": {"type": "default"}},
]


def ask_cached(question: str) -> dict:
    client = bedrock_client()
    start = time.monotonic()
    response = client.converse(
        modelId=MODEL_ID,
        system=SYSTEM_WITH_CACHE_POINT,
        messages=[{"role": "user", "content": [{"text": question}]}],
        inferenceConfig={"maxTokens": 256, "temperature": 0.3},
    )
    elapsed = time.monotonic() - start
    answer = response["output"]["message"]["content"][0]["text"]
    usage = response["usage"]
    return {"answer": answer, "usage": usage, "elapsed": elapsed}


def print_usage(label: str, result: dict) -> None:
    usage = result["usage"]
    cache_read = usage.get("cacheReadInputTokens", 0)
    cache_write = usage.get("cacheWriteInputTokens", 0)
    total_input = usage["inputTokens"] + cache_read + cache_write
    print(
        f"[{label}] {result['elapsed']:.2f}s | "
        f"input {usage['inputTokens']} (total incl. cache: {total_input}), "
        f"cache read {cache_read}, cache write {cache_write}, "
        f"output {usage['outputTokens']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    print(f"[{MODEL_ID} @ {REGION}]\n")
    try:
        first = ask_cached("In one sentence, why does this style guide exist?")
        print(first["answer"], "\n")
        print_usage("call 1, cache write", first)

        second = ask_cached("In one sentence, what's the rule on the Oxford comma?")
        print("\n" + second["answer"], "\n")
        print_usage("call 2, cache read", second)
    except botocore.exceptions.ClientError as error:
        code = error.response["Error"]["Code"]
        print(f"{code}: {error.response['Error']['Message']}", file=sys.stderr)
        sys.exit(1)
