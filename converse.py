# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
"""Minimal Amazon Bedrock Converse call against Claude."""
import json
import sys

import botocore.exceptions

from config import MODEL_ID, REGION, bedrock_client


def ask(prompt: str) -> tuple[str, dict]:
    client = bedrock_client()
    response = client.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 512, "temperature": 0.3},
    )
    print(json.dumps(response, indent=2, default=str))
    return response["output"]["message"]["content"][0]["text"], response["usage"]


if __name__ == "__main__":
    user_prompt = " ".join(sys.argv[1:]) or "In one sentence, what is Amazon Bedrock?"
    print(f"[{MODEL_ID} @ {REGION}]\n")
    try:
        answer, usage = ask(user_prompt)
        print(answer)
        print(
            f"\ntokens: {usage['inputTokens']} in, {usage['outputTokens']} out"
            f" ({usage['totalTokens']} total)",
            file=sys.stderr,
        )
    except botocore.exceptions.ClientError as error:
        code = error.response["Error"]["Code"]
        print(f"{code}: {error.response['Error']['Message']}", file=sys.stderr)
        sys.exit(1)
