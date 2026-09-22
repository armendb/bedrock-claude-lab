# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
"""Apply a Bedrock Guardrail to a Converse call.

The guardrail is a separate resource, created once (console or
`aws bedrock create-guardrail`) and referenced by ID. When it intervenes,
Converse still succeeds: stopReason becomes "guardrail_intervened" and the
reply is the guardrail's configured blocked message, not model output.
"""

import os
import sys

import botocore.exceptions

from config import MODEL_ID, REGION, bedrock_client

GUARDRAIL_ID = os.environ.get("BEDROCK_GUARDRAIL_ID")
GUARDRAIL_VERSION = os.environ.get("BEDROCK_GUARDRAIL_VERSION", "DRAFT")


def ask_guarded(prompt: str) -> tuple[str, str, dict]:
    response = bedrock_client().converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 512, "temperature": 0.3},
        guardrailConfig={
            "guardrailIdentifier": GUARDRAIL_ID,
            "guardrailVersion": GUARDRAIL_VERSION,
        },
    )
    text = response["output"]["message"]["content"][0]["text"]
    return text, response["stopReason"], response["usage"]


if __name__ == "__main__":
    if not GUARDRAIL_ID:
        sys.exit("set BEDROCK_GUARDRAIL_ID to an existing guardrail's ID")
    user_prompt = " ".join(sys.argv[1:]) or "Which stock should I buy this week?"
    print(f"[{MODEL_ID} @ {REGION}, guardrail {GUARDRAIL_ID}:{GUARDRAIL_VERSION}]\n")
    try:
        answer, stop_reason, usage = ask_guarded(user_prompt)
        print(answer)
        print(
            f"\nstopReason: {stop_reason} | tokens: {usage['inputTokens']} in,"
            f" {usage['outputTokens']} out ({usage['totalTokens']} total)",
            file=sys.stderr,
        )
    except botocore.exceptions.ClientError as error:
        code = error.response["Error"]["Code"]
        print(f"{code}: {error.response['Error']['Message']}", file=sys.stderr)
        sys.exit(1)
