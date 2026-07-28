# bedrock-claude-lab

Hands-on samples for calling Claude models on Amazon Bedrock from Python. Each
script is standalone and runnable with a single command, no project setup step.

## Why this exists

Bedrock's docs default to API-key authentication, which is the wrong choice on a
developer machine that already has AWS credentials. Everything here uses the
normal credential chain (IAM Identity Center / SSO), so there is no long-lived
secret anywhere in the repo or in your shell profile.

## Prerequisites

- AWS CLI v2 and an authenticated session: `aws login` (or `aws sso login --profile <name>`)
- [uv](https://docs.astral.sh/uv/) for running the scripts
- Bedrock model access in your Region (see [Model access](#model-access))

## Run

Dependencies are declared inline per script ([PEP 723](https://peps.python.org/pep-0723/)),
so `uv` resolves them on first run:

```bash
uv run converse.py
uv run converse.py "Explain Bedrock inference profiles in two sentences."
```

Override the defaults with environment variables:

```bash
BEDROCK_REGION=us-east-1 \
BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0 \
  uv run converse.py "hello"
```

## Model access

Two things bite on a fresh account:

**1. Use the geo-prefixed model ID.** Current Claude models are on-demand only
through inference profiles. In `eu-west-3` that means `eu.anthropic.claude-sonnet-4-5-20250929-v1:0`,
not the bare `anthropic.claude-sonnet-4-5-20250929-v1:0`. Check which IDs your
Region exposes:

```bash
aws bedrock list-foundation-models --by-provider Anthropic \
  --query 'modelSummaries[].modelId' --region eu-west-3
```

**2. Submit the Anthropic use case form.** Until you do, calls fail with
`ResourceNotFoundException: Model use case details have not been submitted for
this account`. Diagnose it without guessing:

```bash
aws bedrock get-foundation-model-availability \
  --model-id anthropic.claude-sonnet-4-5-20250929-v1:0 --region eu-west-3
```

`agreementAvailability.status: NOT_AVAILABLE` means the form is outstanding.
Submit it under **Model access** in the [Bedrock console](https://console.aws.amazon.com/bedrock/home#/modelaccess),
then allow ~15 minutes. `authorizationStatus`, `entitlementAvailability`, and
`regionAvailability` being fine is not sufficient on its own.

## Roadmap

Ordered so each step builds on the one before it.

**Foundations**

1. `converse.py` — single-turn Converse call ✅
2. `stream.py` — `converse_stream`, incremental token output
3. `chat.py` — multi-turn loop holding conversation history
4. `config.py` — shared client factory, Region and model resolution, retry config

**Model capabilities**

5. `tools.py` — function calling via the `toolConfig` parameter
6. `structured.py` — schema-constrained output, including the `bedrock-runtime`
   vs `bedrock-mantle` difference on `output_config.format`
7. `vision.py` — image input
8. `caching.py` — prompt caching and its effect on cost and latency
9. `thinking.py` — extended thinking, and reading the reasoning blocks back

**Production concerns**

10. `guardrails.py` — apply a Guardrail to a Converse call
11. `observability.py` — CloudWatch metrics and model invocation logging
12. `cost.py` — token accounting from the `usage` block, priced per model
13. `errors.py` — throttling, retries with backoff, cross-Region fallback

**Beyond single calls**

14. `knowledge_base.py` — retrieval-augmented generation over a Knowledge Base
15. `agentcore/` — a minimal agent on Bedrock AgentCore
16. `infra/` — CDK stack provisioning the Guardrail, Knowledge Base, and IAM roles

## Layout

```
converse.py    minimal Converse call, with model-access diagnostics on failure
README.md      this file
```

## License

MIT
