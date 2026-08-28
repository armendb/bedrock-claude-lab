# bedrock-claude-lab

Runnable samples for calling models on Amazon Bedrock from Python. Each script is standalone, no project setup step is
required.

## Why this exists

Bedrock's docs default to API-key authentication, which is the wrong choice on a developer machine that already has AWS
credentials. Everything here uses the normal credential chain (IAM Identity Center / SSO), so there is no long-lived
secret anywhere in the repo or in your shell profile.

## Prerequisites

- AWS CLI v2 and an authenticated session: `aws login` (or `aws sso login --profile <name>`)
- [uv](https://docs.astral.sh/uv/) for running the scripts
- Bedrock model access in your Region (see [Model access](#model-access))

## Run

Dependencies are declared inline per script ([PEP 723](https://peps.python.org/pep-0723/)), so `uv` resolves them on the
first run:

```bash
uv run converse.py
uv run converse.py "Explain Bedrock inference profiles in two sentences."

uv run stream.py "count from 1 to 5, one number per line"
```

Override the defaults with environment variables (see `.env.example`):

```bash
BEDROCK_REGION=us-east-1 \
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
  uv run converse.py "hello"
```

## Model access

Two things bite on a fresh account:

**1. Use the geo-prefixed model ID.** Current Claude models are on-demand only through inference profiles. In
`eu-west-3` that means `eu.anthropic.claude-haiku-4-5-20251001-v1:0`, not the bare
`anthropic.claude-haiku-4-5-20251001-v1:0`. Check which IDs your Region exposes:

```bash
aws bedrock list-foundation-models --by-provider Anthropic \
  --query 'modelSummaries[].modelId' --region eu-west-3
```

**2. Submit the Anthropic use case form.** Until you do, calls fail with
`ResourceNotFoundException: Model use case details have not been submitted for
this account`. Diagnose it without guessing:

```bash
aws bedrock get-foundation-model-availability \
  --model-id anthropic.claude-haiku-4-5-20251001-v1:0 --region eu-west-3
```

`agreementAvailability.status: NOT_AVAILABLE` means the form is outstanding. Submit it from the model catalog in the
Bedrock console, then allow ~15 minutes. `authorizationStatus`, `entitlementAvailability`, and
`regionAvailability` being fine is not enough on its own.

## Cost

On-demand pricing is per token, billed separately for input and output. Every script prints token usage to stderr after
each call. Set a budget alert before running anything at scale, from Billing → Budgets in the console or via
`aws budgets create-budget`.

## Tests

```bash
uv run --with pytest pytest
```

## Layout

```
config.py      shared client factory, region/model resolution, retry config
converse.py    single-turn Converse call
stream.py      ConverseStream call, printing text as it arrives
chat.py        interactive multi-turn chat, holding history, streamed
tools.py       function calling via toolConfig
test_tools.py  unit tests for the tool-call loop, against a mocked client
README.md      this file
```

## License

MIT
