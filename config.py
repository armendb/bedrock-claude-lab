"""Shared Bedrock client setup: region/model resolution and retry config."""

import os

import boto3
from botocore.config import Config

REGION = os.environ.get("BEDROCK_REGION", "eu-west-3")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "eu.anthropic.claude-haiku-4-5-20251001-v1:0")

MAX_ATTEMPTS = int(os.environ.get("BEDROCK_MAX_ATTEMPTS", "3"))


def bedrock_client(region: str = REGION):
    # noinspection PyTypeChecker
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(retries={"max_attempts": MAX_ATTEMPTS, "mode": "standard"}),
    )
