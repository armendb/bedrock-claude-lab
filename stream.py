# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
"""Amazon Bedrock ConverseStream call, printing text as it arrives."""

import sys

import botocore.exceptions

from config import MODEL_ID, REGION, bedrock_client


def ask_stream(prompt: str) -> dict:
    client = bedrock_client()
    response = client.converse_stream(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 512, "temperature": 0.3},
    )
    token_usage = {}
    for event in response["stream"]:
        if "contentBlockDelta" in event:
            print(event["contentBlockDelta"]["delta"]["text"], end="", flush=True)
        elif "metadata" in event and "usage" in event["metadata"]:
            token_usage = event["metadata"]["usage"]
    print()
    return token_usage


if __name__ == "__main__":
    user_prompt = " ".join(sys.argv[1:]) or "In one sentence, what is Amazon Bedrock?"
    print(f"[{MODEL_ID} @ {REGION}]\n")
    try:
        usage = ask_stream(user_prompt)
        print(
            f"\ntokens: {usage['inputTokens']} in, {usage['outputTokens']} out"
            f" ({usage['totalTokens']} total)",
            file=sys.stderr,
        )
    except botocore.exceptions.ClientError as error:
        code = error.response["Error"]["Code"]
        print(f"{code}: {error.response['Error']['Message']}", file=sys.stderr)
        sys.exit(1)
