# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
"""Multi-turn Amazon Bedrock chat, holding conversation history across turns."""

import os
import sys

import boto3
import botocore.exceptions

REGION = os.environ.get("BEDROCK_REGION", "eu-west-3")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "eu.anthropic.claude-haiku-4-5-20251001-v1:0")


def chat_turn(bedrock_client, messages: list[dict], user_prompt: str) -> str:
    messages.append({"role": "user", "content": [{"text": user_prompt}]})
    response = bedrock_client.converse(
        modelId=MODEL_ID,
        messages=messages,
        inferenceConfig={"maxTokens": 512, "temperature": 0.3},
    )
    model_message = response["output"]["message"]
    messages.append(model_message)
    return model_message["content"][0]["text"]


if __name__ == "__main__":
    client = boto3.client("bedrock-runtime", region_name=REGION)
    history: list[dict] = []
    print(f"[{MODEL_ID} @ {REGION}] type 'exit' to quit\n")
    try:
        while True:
            try:
                prompt = input("you> ").strip()
            except EOFError:
                break
            if not prompt or prompt.lower() == "exit":
                break
            reply = chat_turn(client, history, prompt)
            print(f"model> {reply}\n")
    except botocore.exceptions.ClientError as error:
        code = error.response["Error"]["Code"]
        print(f"{code}: {error.response['Error']['Message']}", file=sys.stderr)
        sys.exit(1)
