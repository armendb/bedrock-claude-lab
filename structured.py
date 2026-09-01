# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
"""Schema-constrained output from Bedrock's Converse API.

Converse has no output_config.format parameter — that only exists on the
OpenAI-compatible Responses/Chat Completions APIs served by bedrock-mantle,
and Anthropic models reject it there with a 400. On Converse, structured
output means forcing a single tool call (toolChoice) whose strict input
schema is the shape you want, then reading the args back as structured data
instead of a free-text answer.
"""

import json
import sys

import botocore.exceptions

from config import MODEL_ID, REGION, bedrock_client

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "A short title for the text"},
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Up to 3 key points from the text",
        },
    },
    "required": ["title", "sentiment", "key_points"],
    "additionalProperties": False,
}

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "extract_summary",
                "description": "Extract a structured summary of the given text.",
                "strict": True,
                "inputSchema": {"json": EXTRACT_SCHEMA},
            }
        }
    ],
    "toolChoice": {"tool": {"name": "extract_summary"}},
}


def extract(text: str) -> tuple[dict, dict]:
    client = bedrock_client()
    response = client.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": text}]}],
        toolConfig=TOOL_CONFIG,
        inferenceConfig={"maxTokens": 512, "temperature": 0.0},
    )
    for block in response["output"]["message"]["content"]:
        if "toolUse" in block:
            return block["toolUse"]["input"], response["usage"]
    raise RuntimeError(f"model did not call the forced tool: {response['output']}")


if __name__ == "__main__":
    input_text = " ".join(sys.argv[1:]) or (
        "The new coffee maker arrived on time and looks great on the counter, "
        "but the water tank leaks a little after every brew, which is disappointing "
        "for a product at this price point."
    )
    print(f"[{MODEL_ID} @ {REGION}]\n")
    try:
        summary, usage = extract(input_text)
        print(json.dumps(summary, indent=2))
        print(
            f"\ntokens: {usage['inputTokens']} in, {usage['outputTokens']} out"
            f" ({usage['totalTokens']} total)",
            file=sys.stderr,
        )
    except botocore.exceptions.ClientError as error:
        code = error.response["Error"]["Code"]
        print(f"{code}: {error.response['Error']['Message']}", file=sys.stderr)
        sys.exit(1)
