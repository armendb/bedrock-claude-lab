# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
"""Error handling: retry with backoff, then fall back to another Region.

The client from config.py already retries throttling and transient errors
with exponential backoff (botocore "standard" retry mode). This script adds
the next layer: once one Region's retries are exhausted, try the same model
in another Region. Errors that retrying can't fix (bad request, no model
access) are raised immediately instead of being retried elsewhere.
"""

import sys

import botocore.exceptions

from config import MODEL_ID, REGION, bedrock_client

FALLBACK_REGIONS = [REGION, "eu-central-1"]

RETRYABLE_CODES = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelNotReadyException",
}


def converse_with_fallback(prompt: str, regions: list[str] = FALLBACK_REGIONS) -> tuple[str, str, dict]:
    last_error = None
    for region in regions:
        try:
            response = bedrock_client(region).converse(
                modelId=MODEL_ID,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 512, "temperature": 0.3},
            )
            return response["output"]["message"]["content"][0]["text"], region, response["usage"]
        except botocore.exceptions.ClientError as error:
            code = error.response["Error"]["Code"]
            if code not in RETRYABLE_CODES:
                raise
            print(f"[{region}] {code}, trying next Region", file=sys.stderr)
            last_error = error
    raise last_error


if __name__ == "__main__":
    user_prompt = " ".join(sys.argv[1:]) or "In one sentence, what is exponential backoff?"
    print(f"[{MODEL_ID} @ {' -> '.join(FALLBACK_REGIONS)}]\n")
    try:
        answer, served_by, usage = converse_with_fallback(user_prompt)
        print(answer)
        print(
            f"\nserved by {served_by} | tokens: {usage['inputTokens']} in,"
            f" {usage['outputTokens']} out ({usage['totalTokens']} total)",
            file=sys.stderr,
        )
    except botocore.exceptions.ClientError as error:
        code = error.response["Error"]["Code"]
        print(f"{code}: {error.response['Error']['Message']}", file=sys.stderr)
        sys.exit(1)
