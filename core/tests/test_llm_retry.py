"""Tests for LLM retry logic with exponential backoff.

Run with:
    cd core
    pytest tests/test_llm_retry.py -v

These tests verify:
- Retry behavior on various error types (429, 500, 502, 503, timeout)
- No retry on non-retryable errors (400, 401, 403, 404)
- Exponential backoff delay calculation
- Jitter randomization
- Custom configuration
- Logging of retry attempts
"""

import logging
import time
from unittest.mock import MagicMock, call, patch

import pytest

# Check if litellm is available for integration tests
try:
    import litellm

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

from framework.llm.retry import (
    MaxRetriesExceededError,
    RetryConfig,
    RetryableError,
    calculate_delay,
    is_retryable,
    retry_call,
    with_retry,
)


class TestRetryConfig:
    """Test RetryConfig dataclass initialization and validation."""

    def test_default_values(self):
        """Test RetryConfig has sensible defaults."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.jitter_factor == 0.25
        assert config.retryable_status_codes == (429, 500, 502, 503)
        assert config.retry_on_timeout is True

    def test_custom_values(self):
        """Test RetryConfig accepts custom values."""
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=False,
            jitter_factor=0.1,
            retryable_status_codes=(429, 503),
            retry_on_timeout=False,
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 3.0
        assert config.jitter is False
        assert config.jitter_factor == 0.1
        assert config.retryable_status_codes == (429, 503)
        assert config.retry_on_timeout is False

    def test_disable_retries(self):
        """Test RetryConfig can disable retries entirely."""
        config = RetryConfig(max_retries=0)
        assert config.max_retries == 0

    def test_validation_negative_max_retries(self):
        """Test that negative max_retries raises ValueError."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            RetryConfig(max_retries=-1)

    def test_validation_negative_base_delay(self):
        """Test that negative base_delay raises ValueError."""
        with pytest.raises(ValueError, match="base_delay must be non-negative"):
            RetryConfig(base_delay=-1.0)

    def test_validation_max_delay_less_than_base(self):
        """Test that max_delay < base_delay raises ValueError."""
        with pytest.raises(ValueError, match="max_delay must be >= base_delay"):
            RetryConfig(base_delay=10.0, max_delay=5.0)

    def test_validation_exponential_base_less_than_one(self):
        """Test that exponential_base < 1 raises ValueError."""
        with pytest.raises(ValueError, match="exponential_base must be >= 1"):
            RetryConfig(exponential_base=0.5)

    def test_validation_jitter_factor_out_of_range(self):
        """Test that jitter_factor outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="jitter_factor must be between 0 and 1"):
            RetryConfig(jitter_factor=1.5)
        with pytest.raises(ValueError, match="jitter_factor must be between 0 and 1"):
            RetryConfig(jitter_factor=-0.1)


class TestCalculateDelay:
    """Test exponential backoff delay calculation."""

    def test_exponential_backoff_no_jitter(self):
        """Test basic exponential backoff without jitter."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)

        # Attempt 0: 1 * 2^0 = 1.0
        assert calculate_delay(0, config) == 1.0
        # Attempt 1: 1 * 2^1 = 2.0
        assert calculate_delay(1, config) == 2.0
        # Attempt 2: 1 * 2^2 = 4.0
        assert calculate_delay(2, config) == 4.0
        # Attempt 3: 1 * 2^3 = 8.0
        assert calculate_delay(3, config) == 8.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(base_delay=1.0, max_delay=5.0, jitter=False)

        # Attempt 3 would be 8.0, but capped at 5.0
        assert calculate_delay(3, config) == 5.0
        # Attempt 10 would be 1024.0, but capped at 5.0
        assert calculate_delay(10, config) == 5.0

    def test_jitter_adds_randomness(self):
        """Test that jitter adds randomness to delay."""
        config = RetryConfig(base_delay=1.0, jitter=True, jitter_factor=0.25)

        # Run many times and verify variance
        delays = [calculate_delay(0, config) for _ in range(100)]

        # With jitter_factor=0.25, delay should be between 0.75 and 1.25
        assert all(0.75 <= d <= 1.25 for d in delays)

        # Should have some variance (not all the same)
        unique_delays = set(delays)
        assert len(unique_delays) > 1, "Jitter should produce varying delays"

    def test_jitter_disabled(self):
        """Test that disabling jitter produces consistent delays."""
        config = RetryConfig(base_delay=1.0, jitter=False)

        delays = [calculate_delay(0, config) for _ in range(10)]

        # All delays should be identical
        assert all(d == 1.0 for d in delays)

    def test_custom_exponential_base(self):
        """Test exponential backoff with custom base."""
        config = RetryConfig(base_delay=1.0, exponential_base=3.0, jitter=False)

        # Attempt 0: 1 * 3^0 = 1.0
        assert calculate_delay(0, config) == 1.0
        # Attempt 1: 1 * 3^1 = 3.0
        assert calculate_delay(1, config) == 3.0
        # Attempt 2: 1 * 3^2 = 9.0
        assert calculate_delay(2, config) == 9.0


class TestIsRetryable:
    """Test exception classification for retry eligibility."""

    def test_rate_limit_429_is_retryable(self):
        """Test that 429 rate limit errors are retryable."""
        config = RetryConfig()

        # Create mock exception with status_code attribute
        exc = Exception("Rate limit exceeded")
        exc.status_code = 429

        retryable, status_code, reason = is_retryable(exc, config)
        assert retryable is True
        assert status_code == 429
        assert "Rate limit" in reason

    def test_server_error_500_is_retryable(self):
        """Test that 500 server errors are retryable."""
        config = RetryConfig()

        exc = Exception("Internal server error")
        exc.status_code = 500

        retryable, status_code, reason = is_retryable(exc, config)
        assert retryable is True
        assert status_code == 500
        assert "server error" in reason.lower()

    def test_bad_gateway_502_is_retryable(self):
        """Test that 502 bad gateway errors are retryable."""
        config = RetryConfig()

        exc = Exception("Bad gateway")
        exc.status_code = 502

        retryable, status_code, reason = is_retryable(exc, config)
        assert retryable is True
        assert status_code == 502
        assert "gateway" in reason.lower()

    def test_service_unavailable_503_is_retryable(self):
        """Test that 503 service unavailable errors are retryable."""
        config = RetryConfig()

        exc = Exception("Service unavailable")
        exc.status_code = 503

        retryable, status_code, reason = is_retryable(exc, config)
        assert retryable is True
        assert status_code == 503
        assert "unavailable" in reason.lower()

    def test_timeout_is_retryable(self):
        """Test that timeout errors are retryable by default."""
        config = RetryConfig(retry_on_timeout=True)

        # Exception with 'timeout' in class name
        class TimeoutError(Exception):
            pass

        exc = TimeoutError("Connection timed out")

        retryable, status_code, reason = is_retryable(exc, config)
        assert retryable is True
        assert "Timeout" in reason

    def test_timeout_not_retryable_when_disabled(self):
        """Test that timeout errors are not retryable when disabled."""
        config = RetryConfig(retry_on_timeout=False)

        class TimeoutError(Exception):
            pass

        exc = TimeoutError("Connection timed out")

        retryable, status_code, reason = is_retryable(exc, config)
        assert retryable is False

    def test_client_error_400_not_retryable(self):
        """Test that 400 client errors are not retryable."""
        config = RetryConfig()

        exc = Exception("Bad request")
        exc.status_code = 400

        retryable, status_code, reason = is_retryable(exc, config)
        assert retryable is False
        assert status_code == 400

    def test_unauthorized_401_not_retryable(self):
        """Test that 401 unauthorized errors are not retryable."""
        config = RetryConfig()

        exc = Exception("Unauthorized")
        exc.status_code = 401

        retryable, status_code, reason = is_retryable(exc, config)
        assert retryable is False
        assert status_code == 401

    def test_forbidden_403_not_retryable(self):
        """Test that 403 forbidden errors are not retryable."""
        config = RetryConfig()

        exc = Exception("Forbidden")
        exc.status_code = 403

        retryable, status_code, reason = is_retryable(exc, config)
        assert retryable is False
        assert status_code == 403

    def test_not_found_404_not_retryable(self):
        """Test that 404 not found errors are not retryable."""
        config = RetryConfig()

        exc = Exception("Not found")
        exc.status_code = 404

        retryable, status_code, reason = is_retryable(exc, config)
        assert retryable is False
        assert status_code == 404

    def test_connection_error_is_retryable(self):
        """Test that connection errors are retryable."""
        config = RetryConfig()

        class ConnectionError(Exception):
            pass

        exc = ConnectionError("Connection refused")

        retryable, status_code, reason = is_retryable(exc, config)
        assert retryable is True
        assert "Connection" in reason

    def test_custom_retryable_status_codes(self):
        """Test custom retryable status codes configuration."""
        # Only retry on 429, not 500
        config = RetryConfig(retryable_status_codes=(429,))

        exc_429 = Exception("Rate limit")
        exc_429.status_code = 429

        exc_500 = Exception("Server error")
        exc_500.status_code = 500

        retryable_429, _, _ = is_retryable(exc_429, config)
        retryable_500, _, _ = is_retryable(exc_500, config)

        assert retryable_429 is True
        assert retryable_500 is False


class TestWithRetryDecorator:
    """Test the with_retry decorator."""

    def test_success_on_first_try(self):
        """Test that successful calls don't trigger retry logic."""
        config = RetryConfig()
        call_count = 0

        @with_retry(config)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()

        assert result == "success"
        assert call_count == 1

    @patch("framework.llm.retry.time.sleep")
    def test_retry_on_429_then_success(self, mock_sleep):
        """Test retry on 429 followed by success."""
        config = RetryConfig(max_retries=3, jitter=False)
        call_count = 0

        @with_retry(config)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                exc = Exception("Rate limit exceeded")
                exc.status_code = 429
                raise exc
            return "success"

        result = flaky_func()

        assert result == "success"
        assert call_count == 3
        # Should have slept twice (after attempt 1 and 2)
        assert mock_sleep.call_count == 2

    @patch("framework.llm.retry.time.sleep")
    def test_max_retries_exceeded(self, mock_sleep):
        """Test that MaxRetriesExceededError is raised after max retries."""
        config = RetryConfig(max_retries=2, jitter=False)
        call_count = 0

        @with_retry(config)
        def always_fails():
            nonlocal call_count
            call_count += 1
            exc = Exception("Server error")
            exc.status_code = 500
            raise exc

        with pytest.raises(MaxRetriesExceededError) as exc_info:
            always_fails()

        # Should have tried 3 times (1 initial + 2 retries)
        assert call_count == 3
        assert exc_info.value.attempts == 3
        assert "Max retries (2) exceeded" in str(exc_info.value)

    def test_non_retryable_error_raises_immediately(self):
        """Test that non-retryable errors are raised immediately."""
        config = RetryConfig()
        call_count = 0

        @with_retry(config)
        def auth_error_func():
            nonlocal call_count
            call_count += 1
            exc = Exception("Unauthorized")
            exc.status_code = 401
            raise exc

        with pytest.raises(Exception, match="Unauthorized"):
            auth_error_func()

        # Should have only tried once
        assert call_count == 1

    @patch("framework.llm.retry.time.sleep")
    def test_exponential_backoff_delays(self, mock_sleep):
        """Test that delays follow exponential backoff pattern."""
        config = RetryConfig(max_retries=3, base_delay=1.0, jitter=False)
        call_count = 0

        @with_retry(config)
        def always_fails():
            nonlocal call_count
            call_count += 1
            exc = Exception("Server error")
            exc.status_code = 500
            raise exc

        with pytest.raises(MaxRetriesExceededError):
            always_fails()

        # Verify sleep delays: 1.0, 2.0, 4.0
        assert mock_sleep.call_count == 3
        calls = mock_sleep.call_args_list
        assert calls[0] == call(1.0)  # First retry: 1 * 2^0 = 1
        assert calls[1] == call(2.0)  # Second retry: 1 * 2^1 = 2
        assert calls[2] == call(4.0)  # Third retry: 1 * 2^2 = 4

    def test_retries_disabled(self):
        """Test that max_retries=0 disables retry logic."""
        config = RetryConfig(max_retries=0)
        call_count = 0

        @with_retry(config)
        def fails_once():
            nonlocal call_count
            call_count += 1
            exc = Exception("Server error")
            exc.status_code = 500
            raise exc

        with pytest.raises(MaxRetriesExceededError):
            fails_once()

        # Should have only tried once
        assert call_count == 1


class TestRetryCall:
    """Test the retry_call function."""

    def test_retry_call_success(self):
        """Test retry_call with successful function."""

        def success():
            return "result"

        result = retry_call(success)
        assert result == "result"

    @patch("framework.llm.retry.time.sleep")
    def test_retry_call_with_args(self, mock_sleep):
        """Test retry_call passes arguments correctly."""
        call_count = 0

        def func_with_args(a, b, c=None):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                exc = Exception("Error")
                exc.status_code = 500
                raise exc
            return f"{a}-{b}-{c}"

        result = retry_call(func_with_args, RetryConfig(jitter=False), None, "x", "y", c="z")

        assert result == "x-y-z"
        assert call_count == 2


class TestRetryLogging:
    """Test that retry attempts are properly logged."""

    @patch("framework.llm.retry.time.sleep")
    def test_retry_logs_warning_on_retry(self, mock_sleep, caplog):
        """Test that retry attempts are logged at WARNING level."""
        config = RetryConfig(max_retries=2, jitter=False)
        call_count = 0

        @with_retry(config)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                exc = Exception("Rate limit")
                exc.status_code = 429
                raise exc
            return "success"

        with caplog.at_level(logging.WARNING, logger="framework.llm.retry"):
            result = flaky_func()

        assert result == "success"
        # Should have 2 warning log messages for retry attempts
        warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_logs) == 2
        assert "Attempt 1" in warning_logs[0].message
        assert "429" in warning_logs[0].message
        assert "Attempt 2" in warning_logs[1].message

    @patch("framework.llm.retry.time.sleep")
    def test_retry_logs_error_on_exhaustion(self, mock_sleep, caplog):
        """Test that exhausted retries are logged at ERROR level."""
        config = RetryConfig(max_retries=1, jitter=False)

        @with_retry(config)
        def always_fails():
            exc = Exception("Server error")
            exc.status_code = 500
            raise exc

        with caplog.at_level(logging.ERROR, logger="framework.llm.retry"):
            with pytest.raises(MaxRetriesExceededError):
                always_fails()

        error_logs = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_logs) >= 1
        assert "exhausted" in error_logs[0].message.lower()


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="litellm not installed")
class TestLiteLLMIntegration:
    """Test retry integration with LiteLLM provider."""

    @patch("litellm.completion")
    def test_litellm_provider_uses_retry(self, mock_completion):
        """Test that LiteLLMProvider uses retry logic."""
        from framework.llm.litellm import LiteLLMProvider
        from framework.llm.retry import RetryConfig

        # First call fails with 429, second succeeds
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o-mini"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        rate_limit_error = Exception("Rate limit exceeded")
        rate_limit_error.status_code = 429

        mock_completion.side_effect = [rate_limit_error, mock_response]

        with patch("framework.llm.retry.time.sleep"):
            provider = LiteLLMProvider(
                model="gpt-4o-mini",
                api_key="test-key",
                retry_config=RetryConfig(max_retries=3, jitter=False),
            )
            result = provider.complete(messages=[{"role": "user", "content": "Hi"}])

        assert result.content == "Hello"
        assert mock_completion.call_count == 2

    @patch("litellm.completion")
    def test_litellm_provider_no_retry_on_auth_error(self, mock_completion):
        """Test that LiteLLMProvider doesn't retry on 401 errors."""
        from framework.llm.litellm import LiteLLMProvider
        from framework.llm.retry import RetryConfig

        auth_error = Exception("Invalid API key")
        auth_error.status_code = 401
        mock_completion.side_effect = auth_error

        provider = LiteLLMProvider(
            model="gpt-4o-mini",
            api_key="bad-key",
            retry_config=RetryConfig(max_retries=3),
        )

        with pytest.raises(Exception, match="Invalid API key"):
            provider.complete(messages=[{"role": "user", "content": "Hi"}])

        # Should have only tried once
        assert mock_completion.call_count == 1

    @patch("litellm.completion")
    def test_litellm_provider_max_retries_exceeded(self, mock_completion):
        """Test that LiteLLMProvider raises after max retries."""
        from framework.llm.litellm import LiteLLMProvider
        from framework.llm.retry import RetryConfig

        server_error = Exception("Internal server error")
        server_error.status_code = 500
        mock_completion.side_effect = server_error

        with patch("framework.llm.retry.time.sleep"):
            provider = LiteLLMProvider(
                model="gpt-4o-mini",
                api_key="test-key",
                retry_config=RetryConfig(max_retries=2, jitter=False),
            )

            with pytest.raises(MaxRetriesExceededError):
                provider.complete(messages=[{"role": "user", "content": "Hi"}])

        # Should have tried 3 times (1 initial + 2 retries)
        assert mock_completion.call_count == 3

    @patch("litellm.completion")
    def test_litellm_provider_default_retry_config(self, mock_completion):
        """Test that LiteLLMProvider uses default retry config when none provided."""
        from framework.llm.litellm import LiteLLMProvider

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o-mini"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_completion.return_value = mock_response

        provider = LiteLLMProvider(model="gpt-4o-mini", api_key="test-key")

        # Should have default retry config
        assert provider.retry_config is not None
        assert provider.retry_config.max_retries == 3

    @patch("litellm.completion")
    def test_litellm_provider_disable_retry(self, mock_completion):
        """Test that retry can be disabled in LiteLLMProvider."""
        from framework.llm.litellm import LiteLLMProvider
        from framework.llm.retry import RetryConfig

        server_error = Exception("Server error")
        server_error.status_code = 500
        mock_completion.side_effect = server_error

        provider = LiteLLMProvider(
            model="gpt-4o-mini",
            api_key="test-key",
            retry_config=RetryConfig(max_retries=0),
        )

        with pytest.raises(MaxRetriesExceededError):
            provider.complete(messages=[{"role": "user", "content": "Hi"}])

        # Should have only tried once
        assert mock_completion.call_count == 1


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="litellm not installed")
class TestAnthropicProviderRetry:
    """Test retry integration with Anthropic provider."""

    @patch("litellm.completion")
    def test_anthropic_provider_passes_retry_config(self, mock_completion):
        """Test that AnthropicProvider passes retry_config to LiteLLM."""
        from framework.llm.anthropic import AnthropicProvider
        from framework.llm.retry import RetryConfig

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "claude-haiku-4-5-20251001"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_completion.return_value = mock_response

        custom_config = RetryConfig(max_retries=5, base_delay=2.0)
        provider = AnthropicProvider(api_key="test-key", retry_config=custom_config)

        # Verify config was passed through
        assert provider._provider.retry_config.max_retries == 5
        assert provider._provider.retry_config.base_delay == 2.0


class TestRetryableErrorClass:
    """Test RetryableError exception class."""

    def test_retryable_error_attributes(self):
        """Test RetryableError stores all attributes correctly."""
        original = ValueError("Original error")
        error = RetryableError(
            message="Retry needed",
            original_exception=original,
            status_code=429,
        )

        assert str(error) == "Retry needed"
        assert error.message == "Retry needed"
        assert error.original_exception is original
        assert error.status_code == 429


class TestMaxRetriesExceededErrorClass:
    """Test MaxRetriesExceededError exception class."""

    def test_max_retries_exceeded_attributes(self):
        """Test MaxRetriesExceededError stores all attributes correctly."""
        last_exc = Exception("Final failure")
        error = MaxRetriesExceededError(
            message="All retries failed",
            attempts=4,
            last_exception=last_exc,
        )

        assert str(error) == "All retries failed"
        assert error.attempts == 4
        assert error.last_exception is last_exc
