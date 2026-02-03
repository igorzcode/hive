"""
Retry logic with exponential backoff for LLM API calls.

This module provides a configurable retry mechanism to handle transient API failures
such as rate limits (429), server errors (500, 502, 503), and timeouts.

Example:
    from framework.llm.retry import RetryConfig, with_retry

    # Use default configuration
    config = RetryConfig()

    # Or customize
    config = RetryConfig(
        max_retries=5,
        base_delay=0.5,
        jitter=True,
    )

    # Apply to a function
    @with_retry(config)
    def call_api():
        return litellm.completion(...)
"""

import logging
import random
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, TypeVar

# LiteLLM exception imports - handle gracefully if not installed
try:
    import litellm
    from litellm.exceptions import (
        APIConnectionError,
        APIError,
        RateLimitError,
        ServiceUnavailableError,
        Timeout,
    )

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    # Define placeholder classes for type checking when litellm not installed
    APIConnectionError = Exception  # type: ignore
    APIError = Exception  # type: ignore
    RateLimitError = Exception  # type: ignore
    ServiceUnavailableError = Exception  # type: ignore
    Timeout = Exception  # type: ignore

# OpenAI exception imports - handle gracefully if not installed
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

T = TypeVar("T")

# Default logger for retry operations
_default_logger = logging.getLogger("framework.llm.retry")


@dataclass
class RetryConfig:
    """
    Configuration for LLM API retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3).
            Set to 0 to disable retries.
        base_delay: Initial delay in seconds before first retry (default: 1.0).
        max_delay: Maximum delay cap in seconds (default: 60.0).
            Prevents excessively long waits.
        exponential_base: Base for exponential backoff calculation (default: 2.0).
            Delay formula: base_delay * (exponential_base ** attempt)
        jitter: Whether to add random jitter to delays (default: True).
            Helps prevent thundering herd problem.
        jitter_factor: Jitter range as fraction of delay (default: 0.25).
            A factor of 0.25 means +/- 25% randomization.
        retryable_status_codes: HTTP status codes that should trigger retry.
            Default: (429, 500, 502, 503) - rate limits and server errors.
        retry_on_timeout: Whether to retry on timeout errors (default: True).

    Example:
        # Default configuration (3 retries, 1s base delay)
        config = RetryConfig()

        # Aggressive retry for critical operations
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
        )

        # Conservative retry for non-critical operations
        config = RetryConfig(
            max_retries=2,
            base_delay=2.0,
        )

        # Disable retries
        config = RetryConfig(max_retries=0)
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.25
    retryable_status_codes: tuple[int, ...] = field(
        default_factory=lambda: (429, 500, 502, 503)
    )
    retry_on_timeout: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if self.exponential_base < 1:
            raise ValueError("exponential_base must be >= 1")
        if not 0 <= self.jitter_factor <= 1:
            raise ValueError("jitter_factor must be between 0 and 1")


class RetryableError(Exception):
    """
    Exception wrapper indicating an error is retryable.

    Attributes:
        original_exception: The underlying exception that triggered retry.
        status_code: HTTP status code if available.
        message: Human-readable error message.
    """

    def __init__(
        self,
        message: str,
        original_exception: Exception | None = None,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.original_exception = original_exception
        self.status_code = status_code
        self.message = message


class MaxRetriesExceededError(Exception):
    """
    Raised when all retry attempts have been exhausted.

    Attributes:
        attempts: Number of attempts made.
        last_exception: The final exception that caused failure.
    """

    def __init__(
        self,
        message: str,
        attempts: int,
        last_exception: Exception | None = None,
    ):
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


def _get_status_code_from_exception(exc: Exception) -> int | None:
    """
    Extract HTTP status code from various exception types.

    Handles exceptions from litellm, openai, and httpx libraries.

    Args:
        exc: The exception to extract status code from.

    Returns:
        HTTP status code if found, None otherwise.
    """
    # Check for status_code attribute (litellm exceptions)
    if hasattr(exc, "status_code"):
        code = getattr(exc, "status_code")
        if isinstance(code, int):
            return code

    # Check for response.status_code (openai/httpx exceptions)
    if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        code = exc.response.status_code
        if isinstance(code, int):
            return code

    # Check for http_status attribute
    if hasattr(exc, "http_status"):
        code = getattr(exc, "http_status")
        if isinstance(code, int):
            return code

    # Known exception type mappings
    if LITELLM_AVAILABLE:
        if isinstance(exc, RateLimitError):
            return 429
        if isinstance(exc, ServiceUnavailableError):
            return 503

    return None


def _is_timeout_error(exc: Exception) -> bool:
    """
    Check if an exception represents a timeout error.

    Args:
        exc: The exception to check.

    Returns:
        True if the exception is a timeout error.
    """
    # Check exception type name (covers various timeout classes)
    exc_name = type(exc).__name__.lower()
    if "timeout" in exc_name:
        return True

    # Check litellm Timeout
    if LITELLM_AVAILABLE and isinstance(exc, Timeout):
        return True

    # Check for timeout in message
    exc_msg = str(exc).lower()
    if "timeout" in exc_msg or "timed out" in exc_msg:
        return True

    return False


def _is_connection_error(exc: Exception) -> bool:
    """
    Check if an exception represents a connection error.

    Args:
        exc: The exception to check.

    Returns:
        True if the exception is a connection error.
    """
    exc_name = type(exc).__name__.lower()
    if "connection" in exc_name:
        return True

    if LITELLM_AVAILABLE and isinstance(exc, APIConnectionError):
        return True

    exc_msg = str(exc).lower()
    if "connection" in exc_msg:
        return True

    return False


def is_retryable(
    exc: Exception,
    config: RetryConfig,
) -> tuple[bool, int | None, str]:
    """
    Determine if an exception should trigger a retry.

    Args:
        exc: The exception to evaluate.
        config: Retry configuration with retryable status codes.

    Returns:
        Tuple of (is_retryable, status_code, reason_string).
    """
    # Check for timeout errors
    if _is_timeout_error(exc):
        if config.retry_on_timeout:
            return True, None, "Timeout"
        return False, None, "Timeout (retries disabled)"

    # Check for connection errors (typically retryable)
    if _is_connection_error(exc):
        return True, None, "Connection error"

    # Check status code
    status_code = _get_status_code_from_exception(exc)
    if status_code is not None:
        if status_code in config.retryable_status_codes:
            reason = _status_code_to_reason(status_code)
            return True, status_code, reason
        return False, status_code, f"Status {status_code} not retryable"

    # Default: not retryable for unknown exceptions
    return False, None, "Unknown error type"


def _status_code_to_reason(status_code: int) -> str:
    """Convert status code to human-readable reason."""
    reasons = {
        429: "Rate limit exceeded",
        500: "Internal server error",
        502: "Bad gateway",
        503: "Service unavailable",
        504: "Gateway timeout",
    }
    return reasons.get(status_code, f"HTTP {status_code}")


def calculate_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """
    Calculate delay before next retry attempt.

    Uses exponential backoff with optional jitter.

    Args:
        attempt: Current attempt number (0-indexed).
        config: Retry configuration.

    Returns:
        Delay in seconds before next retry.

    Example:
        # With base_delay=1.0, exponential_base=2.0:
        # attempt 0: 1.0s (+ jitter)
        # attempt 1: 2.0s (+ jitter)
        # attempt 2: 4.0s (+ jitter)
        # attempt 3: 8.0s (+ jitter)
    """
    # Calculate base exponential delay
    delay = config.base_delay * (config.exponential_base**attempt)

    # Cap at max_delay
    delay = min(delay, config.max_delay)

    # Add jitter if enabled
    if config.jitter and config.jitter_factor > 0:
        jitter_range = delay * config.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)
        delay = max(0, delay + jitter)  # Ensure non-negative

    return delay


def with_retry(
    config: RetryConfig | None = None,
    logger: logging.Logger | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to add retry logic with exponential backoff to a function.

    The decorated function will automatically retry on retryable errors
    (rate limits, server errors, timeouts) with exponential backoff delays.

    Args:
        config: Retry configuration. Uses defaults if not provided.
        logger: Logger for retry messages. Uses module logger if not provided.

    Returns:
        Decorator function.

    Example:
        @with_retry(RetryConfig(max_retries=3))
        def call_api():
            return litellm.completion(model="gpt-4", messages=[...])

        # Or with custom logger
        @with_retry(config, logger=my_logger)
        def call_api():
            ...

    Raises:
        MaxRetriesExceededError: When all retry attempts are exhausted.
        Exception: Re-raises non-retryable exceptions immediately.
    """
    if config is None:
        config = RetryConfig()

    log = logger or _default_logger

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as exc:
                    last_exception = exc

                    # Check if error is retryable
                    retryable, status_code, reason = is_retryable(exc, config)

                    if not retryable:
                        # Not retryable - re-raise immediately
                        log.debug(
                            f"[LLM Retry] Non-retryable error: {reason}. "
                            f"Exception: {type(exc).__name__}: {exc}"
                        )
                        raise

                    # Check if we have retries left
                    if attempt >= config.max_retries:
                        log.error(
                            f"[LLM Retry] All {config.max_retries} retries exhausted. "
                            f"Last error: {reason}. Raising exception."
                        )
                        raise MaxRetriesExceededError(
                            f"Max retries ({config.max_retries}) exceeded. "
                            f"Last error: {reason} - {exc}",
                            attempts=attempt + 1,
                            last_exception=exc,
                        )

                    # Calculate delay
                    delay = calculate_delay(attempt, config)

                    # Log retry attempt
                    status_info = f" ({status_code})" if status_code else ""
                    log.warning(
                        f"[LLM Retry] Attempt {attempt + 1}/{config.max_retries + 1} "
                        f"failed: {reason}{status_info}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    # Wait before retry
                    time.sleep(delay)

            # Should not reach here, but handle edge case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected state in retry logic")

        return wrapper

    return decorator


def retry_call(
    func: Callable[..., T],
    config: RetryConfig | None = None,
    logger: logging.Logger | None = None,
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Execute a function with retry logic.

    Alternative to the decorator for cases where decoration isn't convenient.

    Args:
        func: Function to call with retries.
        config: Retry configuration. Uses defaults if not provided.
        logger: Logger for retry messages.
        *args: Positional arguments to pass to func.
        **kwargs: Keyword arguments to pass to func.

    Returns:
        Return value from func.

    Example:
        result = retry_call(
            litellm.completion,
            config=RetryConfig(max_retries=3),
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

    Raises:
        MaxRetriesExceededError: When all retry attempts are exhausted.
        Exception: Re-raises non-retryable exceptions immediately.
    """
    if config is None:
        config = RetryConfig()

    @with_retry(config=config, logger=logger)
    def _wrapped() -> T:
        return func(*args, **kwargs)

    return _wrapped()
