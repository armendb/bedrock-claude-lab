"""Unit tests for the Region fallback in errors.py, against mocked clients."""

from unittest.mock import MagicMock, patch

import botocore.exceptions
import pytest

import errors


def _client_error(code: str) -> botocore.exceptions.ClientError:
    return botocore.exceptions.ClientError({"Error": {"Code": code, "Message": code}}, "Converse")


def _ok_response(text: str) -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8},
    }


def test_falls_back_to_next_region_on_throttling():
    throttled, healthy = MagicMock(), MagicMock()
    throttled.converse.side_effect = _client_error("ThrottlingException")
    healthy.converse.return_value = _ok_response("ok")
    clients = {"a": throttled, "b": healthy}

    with patch("errors.bedrock_client", side_effect=clients.__getitem__):
        answer, region, _ = errors.converse_with_fallback("hi", regions=["a", "b"])

    assert (answer, region) == ("ok", "b")


def test_non_retryable_error_is_raised_without_fallback():
    denied, healthy = MagicMock(), MagicMock()
    denied.converse.side_effect = _client_error("AccessDeniedException")
    clients = {"a": denied, "b": healthy}

    with patch("errors.bedrock_client", side_effect=clients.__getitem__):
        with pytest.raises(botocore.exceptions.ClientError):
            errors.converse_with_fallback("hi", regions=["a", "b"])

    healthy.converse.assert_not_called()


def test_raises_last_error_when_every_region_fails():
    client = MagicMock()
    client.converse.side_effect = _client_error("ServiceUnavailableException")

    with patch("errors.bedrock_client", return_value=client):
        with pytest.raises(botocore.exceptions.ClientError) as caught:
            errors.converse_with_fallback("hi", regions=["a", "b"])

    assert caught.value.response["Error"]["Code"] == "ServiceUnavailableException"
    assert client.converse.call_count == 2
