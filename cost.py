# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
"""Per-call cost from the usage block.

Rates come from the environment, in USD per million tokens, copied from
https://aws.amazon.com/bedrock/pricing/ for your model and Region. They are
not looked up automatically: the AWS Price List API does not list current
Claude models, so any built-in number here would be a guess.

Bedrock bills four token types separately. With prompt caching on,
usage.inputTokens excludes cached tokens, so all four have to be summed.
"""

import os
import sys

import botocore.exceptions

from config import MODEL_ID, REGION, bedrock_client

RATE_ENV = {
    "inputTokens": "BEDROCK_PRICE_INPUT",
    "outputTokens": "BEDROCK_PRICE_OUTPUT",
    "cacheReadInputTokens": "BEDROCK_PRICE_CACHE_READ",
    "cacheWriteInputTokens": "BEDROCK_PRICE_CACHE_WRITE",
}


def load_rates() -> dict[str, float]:
    missing = [name for name in RATE_ENV.values() if name not in os.environ]
    if missing:
        sys.exit(f"set per-million-token USD rates first: {', '.join(missing)}")
    return {field: float(os.environ[name]) for field, name in RATE_ENV.items()}


def usage_cost(usage: dict, rates: dict[str, float]) -> float:
    return sum(usage.get(field, 0) * rate / 1_000_000 for field, rate in rates.items())


if __name__ == "__main__":
    token_rates = load_rates()
    user_prompt = " ".join(sys.argv[1:]) or "In one sentence, what is Amazon Bedrock?"
    print(f"[{MODEL_ID} @ {REGION}]\n")
    try:
        response = bedrock_client().converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": 512, "temperature": 0.3},
        )
        print(response["output"]["message"]["content"][0]["text"])
        usage = response["usage"]
        print(
            f"\ntokens: {usage['inputTokens']} in, {usage['outputTokens']} out"
            f" | cost: ${usage_cost(usage, token_rates):.6f}",
            file=sys.stderr,
        )
    except botocore.exceptions.ClientError as error:
        code = error.response["Error"]["Code"]
        print(f"{code}: {error.response['Error']['Message']}", file=sys.stderr)
        sys.exit(1)
