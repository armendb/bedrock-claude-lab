# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
"""Minimal Amazon Bedrock Converse call against Claude."""

import os
import sys

import boto3
import botocore.exceptions

REGION = os.environ.get("BEDROCK_REGION", "eu-west-3")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")


def ask(prompt: str) -> str:
    client = boto3.client("bedrock-runtime", region_name=REGION)
    response = client.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 512, "temperature": 0.3},
    )
    return response["output"]["message"]["content"][0]["text"]


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "In one sentence, what is Amazon Bedrock?"
    print(f"[{MODEL_ID} @ {REGION}]\n")
    try:
        print(ask(prompt))
    except botocore.exceptions.ClientError as error:
        code = error.response["Error"]["Code"]
        print(f"{code}: {error.response['Error']['Message']}", file=sys.stderr)
        if code in ("ResourceNotFoundException", "AccessDeniedException"):
            print(
                "\nLikely model access, not a code problem. Check with:\n"
                f"  aws bedrock get-foundation-model-availability --model-id {MODEL_ID.split('.', 1)[-1]} --region {REGION}\n"
                "If agreementAvailability is NOT_AVAILABLE, submit the Anthropic use case form:\n"
                f"  https://{REGION}.console.aws.amazon.com/bedrock/home?region={REGION}#/modelaccess",
                file=sys.stderr,
            )
        sys.exit(1)
