"""Unit tests for the tool-call loop in tools.py, against a mocked Bedrock client."""

from unittest.mock import MagicMock, patch

import tools


def _usage(input_tokens: int, output_tokens: int) -> dict:
    return {"inputTokens": input_tokens, "outputTokens": output_tokens}


def test_direct_answer_skips_tool_call():
    client = MagicMock()
    client.converse.return_value = {
        "stopReason": "end_turn",
        "usage": _usage(10, 5),
        "output": {"message": {"role": "assistant", "content": [{"text": "4"}]}},
    }

    with patch("tools.bedrock_client", return_value=client):
        answer, usage = tools.ask_with_tools("what is 2+2?")

    assert answer == "4"
    assert usage == {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
    assert client.converse.call_count == 1


def test_tool_call_result_feeds_back_into_final_answer():
    client = MagicMock()
    client.converse.side_effect = [
        {
            "stopReason": "tool_use",
            "usage": _usage(20, 8),
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "t1",
                                "name": "get_weather",
                                "input": {"city": "Paris"},
                            }
                        }
                    ],
                }
            },
        },
        {
            "stopReason": "end_turn",
            "usage": _usage(30, 12),
            "output": {
                "message": {"role": "assistant", "content": [{"text": "It's sunny in Paris."}]}
            },
        },
    ]

    with patch("tools.bedrock_client", return_value=client):
        answer, usage = tools.ask_with_tools("What's the weather in Paris?")

    assert answer == "It's sunny in Paris."
    assert usage == {"inputTokens": 50, "outputTokens": 20, "totalTokens": 70}
    assert client.converse.call_count == 2

    # `messages` is mutated in place across the loop, so by the time we inspect
    # it here it also holds the final assistant reply appended after this call.
    second_call_messages = client.converse.call_args_list[1].kwargs["messages"]
    tool_result_message = second_call_messages[-2]
    assert tool_result_message["role"] == "user"
    assert tool_result_message["content"][0]["toolResult"]["toolUseId"] == "t1"


def test_run_tool_raises_on_unknown_tool_name():
    try:
        tools.run_tool("not_a_real_tool", {})
        raised = False
    except ValueError:
        raised = True
    assert raised
