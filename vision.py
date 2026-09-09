# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
"""Image input: send a local image alongside a text prompt."""

import sys
from pathlib import Path

import botocore.exceptions

from config import MODEL_ID, REGION, bedrock_client

FORMATS = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".gif": "gif", ".webp": "webp"}


def describe(image_path: Path, prompt: str) -> tuple[str, dict]:
    image_format = FORMATS.get(image_path.suffix.lower())
    if image_format is None:
        raise ValueError(f"unsupported image type {image_path.suffix}, expected one of {sorted(FORMATS)}")
    client = bedrock_client()
    response = client.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": {"format": image_format, "source": {"bytes": image_path.read_bytes()}}},
                    {"text": prompt},
                ],
            }
        ],
        inferenceConfig={"maxTokens": 512, "temperature": 0.3},
    )
    return response["output"]["message"]["content"][0]["text"], response["usage"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: uv run vision.py <image-path> [prompt]")
    path = Path(sys.argv[1])
    user_prompt = " ".join(sys.argv[2:]) or "Describe this image in two sentences."
    print(f"[{MODEL_ID} @ {REGION}]\n")
    try:
        answer, usage = describe(path, user_prompt)
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
