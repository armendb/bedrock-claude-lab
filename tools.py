# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
"""Amazon Bedrock function calling: model requests a tool, we run it, model uses the result."""

import json
import sys

import botocore.exceptions

from config import MODEL_ID, REGION, bedrock_client

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                        },
                        "required": ["city"],
                    }
                },
            }
        }
    ]
}


def get_weather(city: str) -> dict:
    """Fake weather lookup, standing in for a real API call."""
    return {"city": city, "condition": "sunny", "temperature_c": 22}


def run_tool(name: str, tool_input: dict) -> dict:
    if name == "get_weather":
        return get_weather(tool_input["city"])
    raise ValueError(f"unknown tool: {name}")


def ask_with_tools(prompt: str) -> str:
    client = bedrock_client()
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    while True:
        response = client.converse(
            modelId=MODEL_ID,
            messages=messages,
            toolConfig=TOOL_CONFIG,
            inferenceConfig={"maxTokens": 512, "temperature": 0.3},
        )
        output_message = response["output"]["message"]
        messages.append(output_message)

        if response["stopReason"] != "tool_use":
            return output_message["content"][0]["text"]

        tool_results = []
        for block in output_message["content"]:
            if "toolUse" not in block:
                continue
            tool_use = block["toolUse"]
            print(
                f"[calling {tool_use['name']}({json.dumps(tool_use['input'])})]",
                file=sys.stderr,
            )
            result = run_tool(tool_use["name"], tool_use["input"])
            tool_results.append(
                {
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": [{"json": result}],
                    }
                }
            )
        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    user_prompt = " ".join(sys.argv[1:]) or "What's the weather like in Paris?"
    print(f"[{MODEL_ID} @ {REGION}]\n")
    try:
        print(ask_with_tools(user_prompt))
    except botocore.exceptions.ClientError as error:
        code = error.response["Error"]["Code"]
        print(f"{code}: {error.response['Error']['Message']}", file=sys.stderr)
        sys.exit(1)
