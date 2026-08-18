# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
"""Multi-turn Amazon Bedrock chat, holding conversation history across turns."""

import sys

import botocore.exceptions

from config import MODEL_ID, REGION, bedrock_client


def chat_turn(client, messages: list[dict], user_prompt: str) -> dict:
    messages.append({"role": "user", "content": [{"text": user_prompt}]})
    response = client.converse_stream(
        modelId=MODEL_ID,
        messages=messages,
        inferenceConfig={"maxTokens": 512, "temperature": 0.3},
    )
    reply_text = ""
    token_usage = {}
    for event in response["stream"]:
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]["text"]
            print(delta, end="", flush=True)
            reply_text += delta
        elif "metadata" in event and "usage" in event["metadata"]:
            token_usage = event["metadata"]["usage"]
    print()
    messages.append({"role": "assistant", "content": [{"text": reply_text}]})
    return token_usage


if __name__ == "__main__":
    chat_client = bedrock_client()
    history: list[dict] = []
    total_input_tokens = 0
    total_output_tokens = 0
    print(f"[{MODEL_ID} @ {REGION}] type 'exit' to quit\n")
    try:
        while True:
            try:
                prompt = input("you> ").strip()
            except EOFError:
                break
            if not prompt or prompt.lower() == "exit":
                break
            print("model> ", end="")
            turn_usage = chat_turn(chat_client, history, prompt)
            total_input_tokens += turn_usage.get("inputTokens", 0)
            total_output_tokens += turn_usage.get("outputTokens", 0)
    except botocore.exceptions.ClientError as error:
        code = error.response["Error"]["Code"]
        print(f"{code}: {error.response['Error']['Message']}", file=sys.stderr)
        sys.exit(1)
    print(
        f"\ntokens: {total_input_tokens} in, {total_output_tokens} out"
        f" ({total_input_tokens + total_output_tokens} total)",
        file=sys.stderr,
    )
